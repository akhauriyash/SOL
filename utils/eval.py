import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import copy



import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

import numpy as np
from tqdm import tqdm
from itertools import islice

from .masks import build_recency_mask_2d, build_sparse_attention_bias
from .cache import select_cache_by_indices, merge_cache_by_indices
from .probe import probe_losses_with_lookahead
from .cache import detach_cache_to_tuple
from .actions import build_action_spec
from .masks import enable_structured_controls, set_structured_action, clear_structured_action, clear_relevancy_keep


import torch.backends.cuda as sdp
sdp.enable_flash_sdp(False)
sdp.enable_math_sdp(False)
sdp.enable_mem_efficient_sdp(True)  # SDPA only

@torch.no_grad()
def evaluate_stateful_policy_rollout(
    cfg,
    model,
    policy,
    dl,
    Ts,
    Tw,
    keep_fracs,
    context_len,
    rollout_len,
    device,
    greedy: bool = True,
    temperature: float = 1.0,
    lambda_keep: float = 0.0,
    lambda_prune: float = 0.0,
    lambda_quant: float = 0.0,
    sparsity_bias: float = 0.0,
    prune_bias: float = 0.0,
    quant_bias: float = 0.0,
):
    """
    Greedy (or temperature-scaled) rollout with a *recurrent* policy that owns its
    autoregressive KV cache and policy-local positions.

    Differences vs the stateless evaluator:
      - Uses policy.init_state(B, device) and policy.step(..., state).
      - Feeds the same 8D scalar vector as training:
            [t_frac, eff_flag, lambda_keep, lambda_prune, lambda_quant,
             dev_keep, dev_prune, dev_qratio]
      - Keeps LM frozen; all LM-derived features are detached.
    """
    model.eval()

    # If policy is DDP-wrapped, access the underlying module for .init_state/.step
    pol = getattr(policy, "module", policy)
    pol.eval()
    m = getattr(model, "module", model)

    # Build composite action spec
    spec = build_action_spec(
        keep_fracs=keep_fracs,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
    )

    A = int(spec.n_actions)
    # Per-action arrays (length A) – this is what build_action_spec() returns.
    KEEP_PER_ACTION  = torch.tensor(spec.token_keep, device=device, dtype=torch.float32)  # [A]
    PRUNE_PER_ACTION = torch.tensor(spec.prune_keep, device=device, dtype=torch.float32)  # [A]
    QBITS_PER_ACTION = torch.tensor(spec.q_bits,     device=device, dtype=torch.int64)    # [A]
    # Values for the "dense" composite action to use on non-effective steps.
    p_dense_val = float(spec.prune_keep[spec.dense_idx])
    q_dense_val = int(spec.q_bits[spec.dense_idx])

    P = len(spec.prune_keep)
    Q = len(spec.q_bits)
    P_MAX = float(max(spec.prune_keep)) if len(spec.prune_keep) > 0 else 1.0

    thr = Ts + Tw + 1
    # Match training precedence for targets/tolerances
    C_tok   = float(getattr(cfg, "C_target_token", getattr(cfg, "C_target", getattr(cfg, "keep_target", 1.0))))
    tol_tok = float(getattr(cfg, "tol_token", getattr(cfg, "budget_tolerance", getattr(cfg, "keep_tolerance", 0.01))))
    C_pru   = float(getattr(cfg, "C_target_prune", 0.70))
    tol_pru = float(getattr(cfg, "tol_prune", 0.05))
    C_qbits = float(getattr(cfg, "C_target_quant_bits", 8.0))
    C_q     = C_qbits / 16.0
    tol_q   = float(getattr(cfg, "tol_quant_bits", 1.0)) / 16.0
 
    # index for "dense" κ to force on non-effective steps (matches teacher)
    # dense_idx = keep_fracs.index(1.0) if 1.0 in keep_fracs else int(torch.argmax(KEEP).item())
    dense_idx = int(spec.dense_idx)
    win = float(Ts + Tw + 1)
    total_nll = 0.0
    total_tok = 0
    total_keep_all = 0.0
    total_keep_eff = 0.0
    eff_tok = 0
    action_hist = torch.zeros(A, device=device)
    # Track structural averages to sanity-check budgets
    total_prune_eff = 0.0
    total_qratio_eff = 0.0


    emb_layer = m.get_input_embeddings()
    try:
        m_dtype = next(m.parameters()).dtype
    except StopIteration:
        m_dtype = torch.float32

    # --- Generalized per-dimension logit bias (token keep, prune keep, quant bits) ---
    # Normalize each per-action attribute to [0,1] with 0 = most aggressive, 1 = densest/highest bits.
    # Positive biases push toward more aggressive settings; negative toward denser/high-precision.
    def _norm01(x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        xmin = torch.min(x)
        xmax = torch.max(x)
        denom = torch.clamp(xmax - xmin, min=1e-8)
        return (x - xmin) / denom

    logit_bias = None  # [1, A]
    if (sparsity_bias != 0.0) or (prune_bias != 0.0) or (quant_bias != 0.0):
        dens_keep  = _norm01(KEEP_PER_ACTION)                   # 0 = lowest keep, 1 = highest keep
        dens_prune = _norm01(PRUNE_PER_ACTION)                  # 0 = most pruned, 1 = s100
        dens_qbits = _norm01(QBITS_PER_ACTION.to(torch.float32))# 0 = lowest bits, 1 = max bits
        bias_vec = (
            float(sparsity_bias) * dens_keep
            + float(prune_bias)  * dens_prune
            + float(quant_bias)  * dens_qbits
        )  # [A]
        logit_bias = bias_vec.unsqueeze(0)  # [1, A]
    enable_structured_controls(model)
    if str(getattr(cfg, "sparsity_criteria", "recency")) == "relevancy":
        clear_relevancy_keep(model)
    clear_structured_action(model)  # ensure prefill/teacher stays dense
    for batch in tqdm(dl, desc="eval policy (stateful)"):
        batch = batch.to(device)
        B, _ = batch.shape

        prefill_ids = batch[:, :context_len]
        step_inputs = batch[:, context_len : context_len + rollout_len]
        step_labels = batch[:, context_len + 1 : context_len + rollout_len + 1]

        # Dense prefill to seed LM cache and last hidden; LM is frozen & detached for policy.
        out = model(
            input_ids=prefill_ids,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
        # past_kv = out.past_key_values
        past_kv = detach_cache_to_tuple(out.past_key_values)  # ensure tuple-of-(k,v)
        kv_len = torch.full((B,), context_len + 1, device=device, dtype=torch.long)
        state_lm = out.hidden_states[-1][:, -1, :].detach()  # h_{t-1} for t=0

        # Initialize recurrent policy state (policy-local time starts at 0; prev_action = BOS)
        pi_state = pol.init_state(B, device=device)
        # Per‑sequence running stats over effective tokens
        cum_keep   = torch.zeros(B, device=device)   # sum_t eff_t * kappa_t
        cum_eff    = torch.zeros(B, device=device)   # sum_t eff_t
        cum_prune  = torch.zeros(B, device=device)   # sum_t eff_t * (prune_t / P_MAX)
        cum_qratio = torch.zeros(B, device=device)   # sum_t eff_t * qratio_t
        for t in range(rollout_len):
            cur = step_inputs[:, t]
            labels_t = step_labels[:, t]
            # Scalars & features (all LM-derived tensors are detached)
            eff_mask = (kv_len > thr)                                        # [B] bool
            tok_embed = emb_layer(cur).detach()                              # [B, E]

            # === 8D scalar feature vector (must match training) ===
            # 0: t_frac        in [0,1]
            # 1: eff_flag      in {0,1}
            # 2: lambda_keep
            # 3: lambda_prune
            # 4: lambda_quant
            # 5: dev_keep      = mean_keep_prev   - C_tok
            # 6: dev_prune     = mean_prune_prev  - C_pru
            # 7: dev_qratio    = mean_qratio_prev - C_q
            t_frac = torch.full(
                (B, 1),
                (t + 1) / float(rollout_len),
                device=device,
                dtype=torch.float32,
            )
            eff_flag = eff_mask.float().unsqueeze(1)

            lambda_keep_now  = torch.full_like(t_frac, float(lambda_keep))
            lambda_prune_now = torch.full_like(t_frac, float(lambda_prune))
            lambda_quant_now = torch.full_like(t_frac, float(lambda_quant))

            mean_keep_prev = torch.where(
                cum_eff > 0,
                cum_keep / cum_eff,
                torch.full_like(cum_keep, C_tok),
            )
            mean_prune_prev = torch.where(
                cum_eff > 0,
                cum_prune / cum_eff,
                torch.full_like(cum_prune, C_pru),
            )
            mean_qratio_prev = torch.where(
                cum_eff > 0,
                cum_qratio / cum_eff,
                torch.full_like(cum_qratio, C_q),
            )

            dev_keep   = mean_keep_prev   - C_tok
            dev_prune  = mean_prune_prev  - C_pru
            dev_qratio = mean_qratio_prev - C_q

            scalars = torch.cat(
                [
                    t_frac,
                    eff_flag,
                    lambda_keep_now,
                    lambda_prune_now,
                    lambda_quant_now,
                    dev_keep.unsqueeze(1),
                    dev_prune.unsqueeze(1),
                    dev_qratio.unsqueeze(1),
                ],
                dim=-1,
            ).to(torch.float32)  # [B, 8]
            # One recurrent policy step
            logits, _, pi_state_next = pol.step(
                h_lm=state_lm.to(torch.float32),
                e_tok=tok_embed.to(torch.float32),
                scalars=scalars,
                state=pi_state,
                temperature=temperature,
            )

            if logit_bias is not None:
                logits = logits - logit_bias.to(logits.dtype)

            # Action selection
            if greedy:
                a = torch.argmax(logits, dim=-1)
            else:
                dist = Categorical(logits=logits)  # temperature already applied in step()
                a = dist.sample()
            # After choosing a
            dense = torch.full_like(a, dense_idx)
            a_eff = torch.where(eff_mask, a, dense)

            # Use a_eff everywhere downstream:
            pi_state = pi_state_next
            pi_state.last_action = a_eff.detach()  # feed back dense on non‑effective steps

            # Derive controls from a_eff (this keeps everything self‑consistent)
            kappa_now = KEEP_PER_ACTION[a_eff]
            prune_now = PRUNE_PER_ACTION[a_eff]
            qbits_now = QBITS_PER_ACTION[a_eff]
            qratio_now = qbits_now.to(torch.float32).clamp_(min=1.0) / 16.0

            # Histogram should also reflect the effective action:
            action_hist.index_add_(0, a_eff, torch.ones_like(a_eff, dtype=torch.float32))

            # ---- Group by (prune, quant) to set model-global controls per forward ----
            # Build grouping keys; use float for uniqueness (bits cast to float)
            pq = torch.stack([prune_now, qbits_now.to(torch.float32)], dim=-1)  # [B, 2]
            uniq, inv = torch.unique(pq, dim=0, return_inverse=True)

            logits_step = None
            new_state = torch.empty_like(state_lm)
            # new_cache = past_kv
            new_cache = None 
            pos_ids_all = (kv_len - 1).clamp_min(0).unsqueeze(1)

            # helper to allocate an expanded cache with batch=B using subgroup shapes
            def _init_cache_container_like(sub_cache, B_total: int):
                container = []
                for (k_src, v_src) in sub_cache:
                    k_shape = list(k_src.shape); k_shape[0] = B_total
                    v_shape = list(v_src.shape); v_shape[0] = B_total
                    container.append((
                        torch.empty(k_shape, dtype=k_src.dtype, device=k_src.device),
                        torch.empty(v_shape, dtype=v_src.dtype, device=v_src.device),
                    ))
                return tuple(container)

            for g, (p_val, q_val) in enumerate(uniq.tolist()):
                sel = (inv == g).nonzero(as_tuple=False).squeeze(-1)  # [Bg]
                if sel.numel() == 0:
                    continue
                p_scalar = float(p_val)
                q_scalar = int(q_val)
                set_structured_action(model, p_scalar, q_scalar)

                cur_g      = cur.index_select(0, sel)
                pos_ids_g  = pos_ids_all.index_select(0, sel)
                kappa_g    = kappa_now.index_select(0, sel)
                bias_g = build_sparse_attention_bias(
                    model=model,
                    past_kv_lens=kv_len.index_select(0, sel),
                    keep_fracs=kappa_g,
                    Ts=Ts, Tw=Tw,
                    device=device, dtype=m_dtype,
                    criteria=getattr(cfg, "sparsity_criteria", "recency"),
                    tier=getattr(cfg, "relevancy_tier", "per_head"),
                )
                cache_g = select_cache_by_indices(past_kv, sel)
                out_g = model(
                    input_ids=cur_g.unsqueeze(1),
                    use_cache=True,
                    past_key_values=cache_g,
                    position_ids=pos_ids_g,
                    attention_mask=bias_g,
                    return_dict=True,
                    output_hidden_states=True,
                )
                if logits_step is None:
                    logits_step = torch.empty(
                        (B, out_g.logits.size(-1)), device=device, dtype=out_g.logits.dtype
                    )
                logits_step.index_copy_(0, sel, out_g.logits[:, -1, :])
                if new_cache is None:
                    new_cache = _init_cache_container_like(out_g.past_key_values, B)
                # Merge subgroup caches into expanded destination along batch dim
                for li, (k_src, v_src) in enumerate(out_g.past_key_values):
                    k_dst, v_dst = new_cache[li]
                    k_dst.index_copy_(0, sel, k_src)  # src has L+1; dst already L+1
                    v_dst.index_copy_(0, sel, v_src)
                kv_len.index_add_(0, sel, torch.ones_like(sel, device=device, dtype=kv_len.dtype))
                new_state.index_copy_(0, sel, out_g.hidden_states[-1][:, -1, :].detach())

            clear_structured_action(model)

            assert new_cache is not None, "new_cache must be allocated in subgroup loop"
            past_kv = new_cache
            state_lm = new_state
            # NLL for perplexity
            nll_t = F.cross_entropy(logits_step, labels_t, reduction="none")
            total_nll += nll_t.sum().item()
            total_tok += B

            action_hist.index_add_(0, a, torch.ones_like(a, dtype=torch.float32))
            has_old = eff_mask.float()
            eff_tok += int(has_old.sum().item())
            total_keep_all += kappa_now.sum().item()
            total_keep_eff += (kappa_now * has_old).sum().item()
            # Structural running averages over effective steps
            total_prune_eff  += (prune_now  * has_old).sum().item()
            total_qratio_eff += (qratio_now * has_old).sum().item()

            # === Update running budget state AFTER applying action (for next step's features) ===
            eff = has_old  # [B]
            cum_eff_next    = cum_eff + eff
            cum_keep_next   = cum_keep + eff * kappa_now
            cum_prune_next  = cum_prune + eff * (prune_now / P_MAX)
            cum_qratio_next = cum_qratio + eff * qratio_now

            cum_eff, cum_keep, cum_prune, cum_qratio = (
                cum_eff_next,
                cum_keep_next,
                cum_prune_next,
                cum_qratio_next,
            )
    ppl = math.exp(total_nll / max(1, total_tok))
    avg_keep_all = total_keep_all / max(1, total_tok)
    avg_keep_eff = (total_keep_eff / max(1, eff_tok)) if eff_tok > 0 else 0.0
    action_probs = (action_hist / action_hist.sum().clamp_min(1)).tolist()
    avg_prune_keep  = total_prune_eff  / max(1, eff_tok)
    avg_quant_ratio = total_qratio_eff / max(1, eff_tok)
    clear_structured_action(model)
    return {
        "ppl": ppl,
        "avg_keep_all": avg_keep_all,
        "avg_keep_effective": avg_keep_eff,
        "avg_prune_keep": avg_prune_keep,
        "avg_quant_ratio": avg_quant_ratio,
        "action_hist": action_hist.tolist(),
        "action_probs": action_probs,
        "tokens": total_tok,
        "tokens_effective": eff_tok,
    }

@torch.no_grad()
def evaluate_sft_teacher_matched_keep(
    cfg,
    model,
    dl,
    Ts: int,
    Tw: int,
    keep_fracs: Tuple[float, ...],
    target_keep_effective: float,
    context_len: int,
    rollout_len: int,
    device: str,
    return_assignments: bool = False,
    collect_policy_tensors: bool = False,
    lambda_keep_value: Optional[float] = None,
    initial_prev_action: Optional[int] = None,
):
    """
    Per-sequence 'required keep' steering (greedy, horizon-aware):

      - Track per-sequence (cum_eff, cum_keep).
      - At step t, estimate remaining effective steps R_i from (kv_len, Ts, Tw, rollout_len).
      - Compute c_req_i = clip((C_target * (cum_eff_i + R_i) - cum_keep_i) / max(R_i,1), kappa_min, kappa_max).
      - Choose a_i* = argmin_k [ d_{i,k} + beta * (kappa_k - c_req_i)^2 ],
        where d_{i,k} is lookahead loss from `probe_losses_with_lookahead`.

    Notes:
      - Non-effective steps are forced to dense (κ=1 if available).
      - Still returns the same output dict keys as the original function for compatibility.
      - If collect_policy_tensors=True, emits soft teacher targets over ALL actions using
        a softmax over negative scores with temperature `score_soft_tau`.
    """

    model.eval()

    # --- κ setup (keep list & neighbors of target for compatibility metrics) ---
    ks_pairs = sorted([(float(v), i) for i, v in enumerate(keep_fracs)], key=lambda x: x[0])
    ks_vals = [v for v, _ in ks_pairs]
    map_sorted_to_orig = torch.tensor([i for _, i in ks_pairs], device=device, dtype=torch.long)
    KEEP = torch.tensor(keep_fracs, device=device, dtype=torch.float32)   # [A]
    A = len(keep_fracs)

    # two neighbors around target (for compatibility fields in the output)
    if target_keep_effective <= ks_vals[0]:
        lo_s, hi_s, p_hi = 0, 0, 0.0
    elif target_keep_effective >= ks_vals[-1]:
        lo_s, hi_s, p_hi = len(ks_vals) - 1, len(ks_vals) - 1, 1.0
    else:
        lo_s = max(i for i, v in enumerate(ks_vals) if v <= target_keep_effective)
        hi_s = min(i for i, v in enumerate(ks_vals) if v >= target_keep_effective)
        lo_v, hi_v = ks_vals[lo_s], ks_vals[hi_s]
        p_hi = 0.0 if hi_v == lo_v else (target_keep_effective - lo_v) / (hi_v - lo_v)

    lo_idx = int(map_sorted_to_orig[lo_s].item())
    hi_idx = int(map_sorted_to_orig[hi_s].item())

    # dense idx (for non-effective steps)
    dense_idx = keep_fracs.index(1.0) if 1.0 in keep_fracs else int(torch.argmax(KEEP).item())

    # --- steering/penalty controls ---
    penalty_mode = getattr(cfg, "budget_penalty", "linear")  # affects only phi_* in scalars/features
    tol = float(getattr(cfg, "budget_tolerance", getattr(cfg, "keep_tolerance", 0.1)))
    if lambda_keep_value is None:
        lambda_keep_value = float(getattr(cfg, "lambda_init", 0.0))

    # beta = float(getattr(cfg, "budget_steer_beta", getattr(cfg, "steer_beta", 0.5)))
    beta = float(getattr(cfg, "budget_steer_beta", getattr(cfg, "steer_beta", 0.5)))
    tau_soft_default = float(getattr(cfg, "margin_soft_tau", 0.5))
    score_soft_tau = float(getattr(cfg, "score_soft_tau", tau_soft_default))

    kappa_min = float(KEEP.min().item())
    kappa_max = float(KEEP.max().item())
    thr = Ts + Tw + 1

    # --- accumulators (metrics) ---
    total_nll = 0.0
    total_tok = 0
    total_keep_all = 0.0
    total_keep_eff = 0.0
    eff_tok = 0
    action_hist = torch.zeros(A, device=device)
    step_action_hist = torch.zeros(A, rollout_len, device=device, dtype=torch.long)

    # optional collectors
    assignments_batches = [] if return_assignments else None
    policy_batches = [] if collect_policy_tensors else None

    # embedding layer only if we have to collect policy tensors
    if collect_policy_tensors:
        try:
            emb_layer = unwrap(model).get_input_embeddings()
        except NameError:
            emb_layer = model.get_input_embeddings()

    for batch in tqdm(dl, desc="eval teacher (per-seq keep steering)"):
        batch = batch.to(device)
        B, _ = batch.shape

        prefill_ids  = batch[:, :context_len]
        step_inputs  = batch[:, context_len : context_len + rollout_len]
        step_labels  = batch[:, context_len + 1 : context_len + rollout_len + 1]

        # Dense prefill to build cache (and optionally last hidden)
        out = model(
            input_ids=prefill_ids,
            use_cache=True,
            return_dict=True,
            output_hidden_states=collect_policy_tensors,
        )
        past_kv = out.past_key_values
        kv_len = torch.full((B,), context_len + 1, device=device, dtype=torch.long)  # includes current
        if collect_policy_tensors:
            last_h = out.hidden_states[-1][:, -1, :].detach().to(torch.float32)

        # Per-sequence budget trackers across the rollout
        cum_keep = torch.zeros(B, device=device)  # sum of κ on effective steps so far
        cum_eff  = torch.zeros(B, device=device)  # count of effective steps so far

        # Buffers for optional outputs
        teacher_actions_seq_buf = []  # [T,B]
        margins_seq_buf = []          # we'll log "score margin" (2nd best - best)

        if collect_policy_tensors:
            if initial_prev_action is None:
                prev_action_ids = torch.full((B,), dense_idx, device=device, dtype=torch.long)
            else:
                prev_action_ids = torch.full((B,), int(initial_prev_action), device=device, dtype=torch.long)

            h_seq_buf, e_seq_buf, scalars_seq_buf, prev_actions_seq_buf = [], [], [], []
            soft_seq_buf = []
        for t in range(rollout_len):
            cur = step_inputs[:, t]          # [B]
            labels_t = step_labels[:, t]     # [B]

            kv_before = kv_len
            # Effective now if we already have at least one truly "old" token
            eff_mask = (kv_before > thr)                # [B]
            has_old = eff_mask.float()

            # =======================
            # PRE-DECISION FEATURES (policy-view, 8D scalars)
            # =======================
            if collect_policy_tensors:
                tok_embed = emb_layer(cur).detach().to(torch.float32)  # [B,E]

                # Same scalar schema as GRPO:
                # [t_frac, eff_flag, lambda_keep, lambda_prune, lambda_quant,
                #  dev_keep, dev_prune, dev_qratio]
                t_frac = torch.full(
                    (B, 1),
                    (t + 1) / float(rollout_len),
                    device=device,
                    dtype=torch.float32,
                )
                eff_flag = has_old.unsqueeze(1)

                lambda_keep_now  = torch.full_like(t_frac, float(lambda_keep_value))
                lambda_prune_now = torch.zeros_like(t_frac)  # no structured pruning in this teacher
                lambda_quant_now = torch.zeros_like(t_frac)  # no quantization in this teacher

                mean_keep_prev = torch.where(
                    cum_eff > 0,
                    cum_keep / cum_eff,
                    torch.full_like(cum_keep, target_keep_effective),
                )
                dev_keep   = mean_keep_prev - target_keep_effective
                dev_prune  = torch.zeros_like(dev_keep)
                dev_qratio = torch.zeros_like(dev_keep)

                scalars = torch.cat(
                    [
                        t_frac,
                        eff_flag,
                        lambda_keep_now,
                        lambda_prune_now,
                        lambda_quant_now,
                        dev_keep.unsqueeze(1),
                        dev_prune.unsqueeze(1),
                        dev_qratio.unsqueeze(1),
                    ],
                    dim=-1,
                ).to(torch.float32)

                h_seq_buf.append(last_h)
                e_seq_buf.append(tok_embed)
                scalars_seq_buf.append(scalars)
                prev_actions_seq_buf.append(prev_action_ids)
            # =======================
            # PER-SEQUENCE DECISION
            # =======================
            a_star = torch.full((B,), dense_idx, device=device, dtype=torch.long)

            # T_rem = steps remaining including current: t..rollout_len-1
            T_rem = rollout_len - t
            # Non-effective steps remaining starting now:
            neff_rem = (thr - kv_before + 1).clamp_min(0)  # [B]
            # Remaining effective steps (including current if already effective):
            R = (T_rem - neff_rem).clamp_min(0).to(torch.float32)  # [B]

            # Required keep to finish on target (clipped to [kappa_min, kappa_max])
            c_req = (target_keep_effective * (cum_eff + R) - cum_keep) / torch.clamp(R, min=1.0)
            c_req = torch.clamp(c_req, kappa_min, kappa_max)  # [B]

            if eff_mask.any():
                # Probe lookahead losses for all actions (dense_kl or CE) on the (still) full batch
                losses_a = probe_losses_with_lookahead(
                    model,
                    past_kv_live=past_kv,
                    kv_len=kv_before,
                    cur=cur,
                    step_inputs=step_inputs,
                    step_labels=step_labels,
                    t=t,
                    keep_fracs=keep_fracs,
                    Ts=Ts, Tw=Tw,
                    device=device, dtype=model.dtype,
                    horizon=getattr(cfg, "horizon", 1),
                    future_mask_rule=getattr(cfg, "future_mask_rule", "dense"),
                    metric=getattr(cfg, "probe_metric", "dense_kl"),
                    sparsity_criteria=getattr(cfg, "sparsity_criteria", "recency"),
                    relevancy_tier=getattr(cfg, "relevancy_tier", "per_head"),
                )  # [B, A]

                # Score with per-sequence steering: loss + beta * (kappa - c_req)^2
                # Broadcast: c_req[:,None] vs KEEP[None,:]
                steer_pen = (KEEP.view(1, -1) - c_req.view(-1, 1)) ** 2  # [B,A]
                scores = losses_a + beta * steer_pen

                # Choose argmin per sequence, but only for those effective now
                idx_eff = torch.nonzero(eff_mask, as_tuple=False).squeeze(-1)
                a_star[idx_eff] = torch.argmin(scores[idx_eff, :], dim=-1)

                with torch.no_grad():
                    sorted_scores, _ = torch.sort(scores, dim=-1)
                    score_margin_all = (sorted_scores[:, 1] - sorted_scores[:, 0]).detach()
                    m_t = torch.full((B,), float('nan'), device=device)
                    m_t[eff_mask] = score_margin_all[eff_mask]
                    margins_seq_buf.append(m_t)

            # ---------- Build soft teacher over actions for step t ----------
            if collect_policy_tensors:
                soft_t = torch.zeros(B, A, device=device, dtype=torch.float32)
                idx_non = torch.nonzero(~eff_mask, as_tuple=False).squeeze(-1)
                if idx_non.numel() > 0:
                    soft_t[idx_non, dense_idx] = 1.0
                if eff_mask.any():
                    idx_eff = torch.nonzero(eff_mask, as_tuple=False).squeeze(-1)
                    scores_eff = scores[idx_eff, :]  # [B_eff, A]
                    probs_eff = torch.softmax(-scores_eff / max(score_soft_tau, 1e-6), dim=-1)
                    soft_t[idx_eff, :] = probs_eff
                soft_seq_buf.append(soft_t)

            kappa_now = KEEP[a_star]                       # [B]
            counts_t = torch.bincount(a_star, minlength=A) # [A]
            step_action_hist[:, t] += counts_t

            pos_ids = (kv_before - 1).clamp_min(0).unsqueeze(1)
            inference_mask = build_sparse_attention_bias(
                model=model,
                past_kv_lens=kv_before,
                keep_fracs=kappa_now,
                Ts=Ts,
                Tw=Tw,
                device=device,
                dtype=model.dtype,
                criteria=getattr(cfg, "sparsity_criteria", "recency"),
                tier=getattr(cfg, "relevancy_tier", "per_head"),
            )
            out_step = model(
                input_ids=cur.unsqueeze(1),
                use_cache=True,
                past_key_values=past_kv,
                position_ids=pos_ids,
                attention_mask=inference_mask,
                return_dict=True,
                output_hidden_states=collect_policy_tensors,
            )
            logits_step = out_step.logits[:, -1, :]
            past_kv = out_step.past_key_values
            kv_len = kv_len + 1

            if collect_policy_tensors:
                last_h = out_step.hidden_states[-1][:, -1, :].detach().to(torch.float32)

            # CE against labels for ppl metric
            nll_t = F.cross_entropy(logits_step, labels_t, reduction="none")
            total_nll += nll_t.sum().item()
            total_tok += B

            # Stats/budget tracking
            action_hist.index_add_(0, a_star, torch.ones_like(a_star, dtype=torch.float32))
            eff_tok += int(has_old.sum().item())
            total_keep_all += kappa_now.sum().item()
            total_keep_eff += (kappa_now * has_old).sum().item()

            eff = has_old  # [B] float in {0,1}
            cum_eff  = cum_eff  + eff
            cum_keep = cum_keep + eff * kappa_now

            if return_assignments or collect_policy_tensors:
                teacher_actions_seq_buf.append(a_star.detach())

            if collect_policy_tensors:
                prev_action_ids = a_star.detach()
        if return_assignments:
            teacher_actions_seq = torch.stack(teacher_actions_seq_buf, dim=0)  # [T,B]
            assignments_batches.append(teacher_actions_seq)

        if collect_policy_tensors:
            h_seq = torch.stack(h_seq_buf, dim=0)
            e_seq = torch.stack(e_seq_buf, dim=0)
            scalars_seq = torch.stack(scalars_seq_buf, dim=0)
            prev_actions_seq = torch.stack(prev_actions_seq_buf, dim=0)
            teacher_actions_seq = torch.stack(teacher_actions_seq_buf, dim=0)
            teacher_soft_seq = torch.stack(soft_seq_buf, dim=0)  # [T,B,A]
            margins_seq = torch.stack(margins_seq_buf, dim=0)    # [T,B] score margins
            policy_batches.append({
                "h_seq": h_seq,
                "e_seq": e_seq,
                "scalars_seq": scalars_seq,
                "prev_actions_seq": prev_actions_seq,
                "teacher_actions_seq": teacher_actions_seq,
                "teacher_soft_seq": teacher_soft_seq,
                "margins_seq": margins_seq,
            })

    ppl = math.exp(total_nll / max(1, total_tok))
    avg_keep_all = total_keep_all / max(1, total_tok)
    avg_keep_eff = (total_keep_eff / max(1, eff_tok)) if eff_tok > 0 else 0.0
    action_probs = (action_hist / action_hist.sum().clamp_min(1)).tolist()
    step_totals = step_action_hist.sum(dim=0, keepdim=True).clamp_min(1)
    step_action_probs = (step_action_hist.float() / step_totals).tolist()

    out = {
        "ppl": ppl,
        "avg_keep_all": avg_keep_all,
        "avg_keep_effective": avg_keep_eff,
        "action_hist": action_hist.tolist(),
        "action_probs": action_probs,
        "tokens": total_tok,
        "tokens_effective": eff_tok,
        "mix_lo_k": ks_vals[lo_s],
        "mix_hi_k": ks_vals[hi_s],
        "mix_p_hi": float(p_hi),
        "step_action_probs": step_action_probs,
    }
    if return_assignments:
        out["assignments"] = assignments_batches  # List[Tensor[T,B]]
    if collect_policy_tensors:
        out["policy_batches"] = policy_batches   # List[Dict[str, Tensor]]
    return out
