import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import copy



import math
from typing import Tuple, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

import math
from typing import Tuple, List
import torch
import torch.nn.functional as F
from torch import Tensor
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
import wandb
from tqdm import tqdm
from itertools import islice

from .masks import build_recency_bias_4d, build_recency_mask_2d, build_sparse_attention_bias
from .probe import probe_losses_with_lookahead
from .cache import detach_cache_to_tuple
from .actions import build_action_spec
from .masks import enable_structured_controls, set_structured_action, clear_structured_action, clear_relevancy_keep
from .cache import select_cache_by_indices, merge_cache_by_indices

import torch.backends.cuda as sdp
sdp.enable_flash_sdp(False)
sdp.enable_math_sdp(False)
sdp.enable_mem_efficient_sdp(True)  # SDPA only


def _unique_float_axis(values, tol: float = 1e-6):
    axis = []
    for v in values:
        fv = float(v)
        if not any(abs(fv - existing) <= tol for existing in axis):
            axis.append(fv)
    return axis


def _unique_int_axis(values):
    axis = []
    for v in values:
        iv = int(v)
        if iv not in axis:
            axis.append(iv)
    return axis

@torch.no_grad()
def evaluate_fixed_matched_keep(
    cfg,
    model,
    dl,
    Ts: int,
    Tw: int,
    keep_fracs: Tuple[float, ...],
    prune_choices: Tuple[str, ...],
    quant_choices: Tuple[str, ...],
    target_keep_effective: float,
    target_prune_keep: float,
    target_quant_ratio: float,           # ratio in [0,1], e.g. 16->1.0, 8->0.5
    context_len: int,
    rollout_len: int,
    device: str,
    struct_on_non_eff: bool = False,     # if True, also apply prune/quant on non-effective steps
):
    """
    Deterministic matched baseline for *three* axes at once:
      - Token keep kappa (κ)       : match target_keep_effective on effective tokens.
      - Channel keep prune (ρ)     : match target_prune_keep (by fraction kept).
      - Quantization ratio q_ratio : match target_quant_ratio (bits/16).

    At each effective step, we assign a round–robin mixture between the two nearest
    discrete choices on *each axis* to hit the per-step targets as closely as possible.
    Structural controls are applied via set_structured_action() per (ρ, bits) subgroup
    so the LM and KV cache reflect those choices.
    """
    model.eval()
    m = getattr(model, "module", model)
    try:
        m_dtype = next(m.parameters()).dtype
    except StopIteration:
        m_dtype = torch.float32

    # Build composite action spec so we reuse your exact choice sets
    spec = build_action_spec(
        keep_fracs=keep_fracs,
        prune_choices=prune_choices,
        quant_choices=quant_choices,
    )
    KEEP  = torch.tensor(keep_fracs,            device=device, dtype=torch.float32)      # [K]
    prune_axis = _unique_float_axis(spec.prune_keep)
    quant_axis = _unique_int_axis(spec.q_bits)
    PRUNE = torch.tensor(prune_axis,            device=device, dtype=torch.float32)      # [P]
    QBITS = torch.tensor(quant_axis,            device=device, dtype=torch.int64)        # [Q]
    QRAT  = QBITS.to(torch.float32).clamp(min=1.0) / 16.0                                # [Q]
    K, P, Q = len(keep_fracs), len(prune_axis), len(quant_axis)

    # Densest indices (used for non-effective steps)
    dense_k_idx = (keep_fracs.index(1.0) if 1.0 in keep_fracs else int(torch.argmax(KEEP).item()))
    dense_p_idx = int(torch.argmax(PRUNE).item())  # expect 1.0 in choices
    dense_q_idx = int(torch.argmax(QBITS).item())  # expect 16 in choices
    # --- NEW: batch‑wide residual trackers for structured axes (ρ, q_ratio) ---
    cum_prune_steps: int = 0
    cum_quant_steps: int = 0
    cum_prune_sum: float = 0.0
    cum_quant_sum: float = 0.0
    p_axis_min, p_axis_max = float(PRUNE.min().item()), float(PRUNE.max().item())
    q_axis_min, q_axis_max = float(QRAT.min().item()),  float(QRAT.max().item())
    def _mix_for_target(vals: List[float], target: float):
        pairs = sorted([(float(v), i) for i, v in enumerate(vals)], key=lambda x: x[0])
        vs = [v for v, _ in pairs]
        map_sorted_to_orig = [i for _, i in pairs]
        if target <= vs[0]:
            lo_s, hi_s, p_hi = 0, 0, 0.0
        elif target >= vs[-1]:
            lo_s, hi_s, p_hi = len(vs)-1, len(vs)-1, 1.0
        else:
            lo_s = max(i for i, v in enumerate(vs) if v <= target)
            hi_s = min(i for i, v in enumerate(vs) if v >= target)
            lo_v, hi_v = vs[lo_s], vs[hi_s]
            p_hi = 0.0 if hi_v == lo_v else (target - lo_v) / (hi_v - lo_v)
        lo_idx = int(map_sorted_to_orig[lo_s])
        hi_idx = int(map_sorted_to_orig[hi_s])
        return lo_idx, hi_idx, float(p_hi), vs[lo_s], vs[hi_s]

    # Precompute per-axis mixing against targets
    k_lo, k_hi, k_p_hi, k_lo_v, k_hi_v = _mix_for_target(list(KEEP.tolist()),  float(target_keep_effective))
    p_lo, p_hi, p_p_hi, p_lo_v, p_hi_v = _mix_for_target(list(PRUNE.tolist()), float(target_prune_keep))
    q_lo, q_hi, q_p_hi, q_lo_v, q_hi_v = _mix_for_target(list(QRAT.tolist()),  float(target_quant_ratio))

    thr = Ts + Tw + 1
    total_nll = 0.0
    total_tok = 0
    eff_tok   = 0
    total_keep_all = 0.0
    total_keep_eff = 0.0
    total_prune_eff = 0.0
    total_qratio_eff = 0.0

    # Optional: composite action histogram over K×P×Q
    action_hist = torch.zeros(K * P * Q, device=device)

    enable_structured_controls(model)
    if str(getattr(cfg, "sparsity_criteria", "recency")) == "relevancy": clear_relevancy_keep(model)
    clear_structured_action(model)  # teacher/prefill dense

    for batch in tqdm(dl, desc="eval fixed matched structured"):
        batch = batch.to(device)
        B, _ = batch.shape

        prefill_ids = batch[:, :context_len]
        step_inputs = batch[:, context_len : context_len + rollout_len]
        step_labels = batch[:, context_len + 1 : context_len + rollout_len + 1]

        out = model(input_ids=prefill_ids, use_cache=True, return_dict=True)
        past_kv = detach_cache_to_tuple(out.past_key_values)
        kv_len = torch.full((B,), context_len + 1, device=device, dtype=torch.long)

        for t in range(rollout_len):
            cur = step_inputs[:, t]
            labels_t = step_labels[:, t]

            eff_mask = (kv_len > thr)  # [B]
            idx_eff = torch.nonzero(eff_mask, as_tuple=False).squeeze(-1)  # [Beff]
            Beff = int(idx_eff.numel())

            # Defaults (non-effective steps -> densest settings)
            a_k = torch.full((B,), dense_k_idx, device=device, dtype=torch.long)
            a_p = torch.full((B,), dense_p_idx, device=device, dtype=torch.long)
            a_q = torch.full((B,), dense_q_idx, device=device, dtype=torch.long)

            if Beff > 0:
                # Token keep: choose n_hi_k of Beff to use k_hi, rest k_lo
                if k_lo != k_hi:
                    n_hi_k = int(round(k_p_hi * Beff))
                    n_hi_k = max(0, min(Beff, n_hi_k))
                    a_k[idx_eff] = k_lo
                    if n_hi_k > 0:
                        start_k = (t * max(n_hi_k, 1)) % Beff
                        rr_k = (start_k + torch.arange(n_hi_k, device=device)) % Beff
                        a_k[idx_eff[rr_k]] = k_hi
                else:
                    a_k[idx_eff] = k_lo

            # --- NEW: structured axes assignment uses batch‑wide residuals ---
            struct_mask = (eff_mask if not struct_on_non_eff else torch.ones_like(eff_mask))
            idx_struct = torch.nonzero(struct_mask, as_tuple=False).squeeze(-1)
            S_struct = int(idx_struct.numel())

            # Prune keep (ρ): choose mix so that the running batch‑wide average tracks target_prune_keep
            if S_struct > 0:
                if p_lo != p_hi:
                    p_req = (float(target_prune_keep) * (cum_prune_steps + S_struct) - cum_prune_sum) / float(S_struct)
                    p_req = float(max(p_axis_min, min(p_axis_max, p_req)))
                    share_hi = 0.0 if (p_hi_v == p_lo_v) else (p_req - p_lo_v) / (p_hi_v - p_lo_v)
                    n_hi_p = int(round(share_hi * S_struct))
                    n_hi_p = max(0, min(S_struct, n_hi_p))
                    a_p[idx_struct] = p_lo
                    if n_hi_p > 0:
                        start_p = ((t + 17) * max(n_hi_p, 1)) % S_struct  # decorrelate from κ
                        rr_p = (start_p + torch.arange(n_hi_p, device=device)) % S_struct
                        a_p[idx_struct[rr_p]] = p_hi
                else:
                    a_p[idx_struct] = p_lo

            # Quant ratio (q_ratio via bits): same residual idea versus target_quant_ratio
            if S_struct > 0:
                if q_lo != q_hi:
                    q_req = (float(target_quant_ratio) * (cum_quant_steps + S_struct) - cum_quant_sum) / float(S_struct)
                    q_req = float(max(q_axis_min, min(q_axis_max, q_req)))
                    share_hi_q = 0.0 if (q_hi_v == q_lo_v) else (q_req - q_lo_v) / (q_hi_v - q_lo_v)
                    n_hi_q = int(round(share_hi_q * S_struct))
                    n_hi_q = max(0, min(S_struct, n_hi_q))
                    a_q[idx_struct] = q_lo
                    if n_hi_q > 0:
                        start_q = ((t + 37) * max(n_hi_q, 1)) % S_struct
                        rr_q = (start_q + torch.arange(n_hi_q, device=device)) % S_struct
                        a_q[idx_struct[rr_q]] = q_hi
                else:
                    a_q[idx_struct] = q_lo
            kappa_now  = KEEP[a_k]                 # [B]
            prune_now  = PRUNE[a_p]                # [B]
            qbits_now  = QBITS[a_q]                # [B]
            qratio_now = QRAT[a_q]                 # [B]

            # Histogram over composite (k,p,q) for visibility
            flat_idx = a_k * (P * Q) + a_p * Q + a_q
            action_hist.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))

            # Apply structural controls per (prune, qbits) subgroup
            pq = torch.stack([prune_now, qbits_now.to(torch.float32)], dim=-1)  # [B,2]
            uniq, inv = torch.unique(pq, dim=0, return_inverse=True)

            logits_step = None
            new_cache = None
            pos_ids_all = (kv_len - 1).clamp_min(0).unsqueeze(1)

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
                sel = (inv == g).nonzero(as_tuple=False).squeeze(-1)
                if sel.numel() == 0:
                    continue
                p_scalar = float(p_val)
                q_scalar = int(q_val)
                # Apply subgroup structural controls
                set_structured_action(model, p_scalar, q_scalar)

                cur_g     = cur.index_select(0, sel)
                pos_ids_g = pos_ids_all.index_select(0, sel)
                kappa_g   = kappa_now.index_select(0, sel)

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
                )
                if logits_step is None:
                    logits_step = torch.empty((B, out_g.logits.size(-1)),
                                              device=device, dtype=out_g.logits.dtype)
                logits_step.index_copy_(0, sel, out_g.logits[:, -1, :])
                if new_cache is None:
                    new_cache = _init_cache_container_like(out_g.past_key_values, B)
                # merge subgroup caches (already L+1) into expanded destination
                for li, (k_src, v_src) in enumerate(out_g.past_key_values):
                    k_dst, v_dst = new_cache[li]
                    k_dst.index_copy_(0, sel, k_src)
                    v_dst.index_copy_(0, sel, v_src)
                kv_len.index_add_(0, sel, torch.ones_like(sel, device=device, dtype=kv_len.dtype))

            past_kv = new_cache
            clear_structured_action(model)  # clear for next loop (we’ll set again per subgroup)

            # Stats
            nll_t = F.cross_entropy(logits_step, labels_t, reduction="none")
            total_nll += nll_t.sum().item()
            total_tok += B

            has_old = eff_mask.float()
            eff_tok += int(has_old.sum().item())
            total_keep_all += kappa_now.sum().item()
            total_keep_eff += (kappa_now * has_old).sum().item()
            # For structural metrics we report effective-step averages by default
            gate = has_old if not struct_on_non_eff else torch.ones_like(has_old)
            total_prune_eff  += (prune_now  * gate).sum().item()
            total_qratio_eff += (qratio_now * gate).sum().item()

            # --- NEW: advance batch‑wide residual trackers for structured axes ---
            s_gate = (eff_mask if not struct_on_non_eff else torch.ones_like(eff_mask)).float()
            S_this = int(s_gate.sum().item())
            if S_this > 0:
                cum_prune_sum  += float((prune_now  * s_gate).sum().item())
                cum_quant_sum  += float((qratio_now * s_gate).sum().item())
                cum_prune_steps += S_this
                cum_quant_steps += S_this
    clear_structured_action(model)
    ppl = math.exp(total_nll / max(1, total_tok))
    avg_keep_all = total_keep_all / max(1, total_tok)
    avg_keep_eff = (total_keep_eff / max(1, eff_tok)) if eff_tok > 0 else 0.0
    denom_struct = (eff_tok if not struct_on_non_eff else total_tok)
    avg_prune_keep  = (total_prune_eff  / max(1, denom_struct))
    avg_quant_ratio = (total_qratio_eff / max(1, denom_struct))

    action_probs = (action_hist / action_hist.sum().clamp_min(1)).tolist()
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
        "mix_keep":  {"lo": k_lo_v, "hi": k_hi_v, "p_hi": k_p_hi},
        "mix_prune": {"lo": p_lo_v, "hi": p_hi_v, "p_hi": p_p_hi},
        "mix_quant": {"lo": q_lo_v, "hi": q_hi_v, "p_hi": q_p_hi},
    }


@torch.no_grad()
def evaluate_dense_full(model, dl, context_len, rollout_len, device):
    """One-shot teacher-forcing baseline over the whole block (upper bound)."""
    model.eval()
    total_nll = 0.0
    total_tok = 0
    for batch in tqdm(dl, desc="eval dense full"):
        batch = batch.to(device)
        B, _ = batch.shape
        out = model(input_ids=batch[:, :context_len + rollout_len], use_cache=False, return_dict=True)
        logits = out.logits  # [B, L, V]
        logit_slice = logits[:, context_len : context_len + rollout_len, :]      # predicts [x_{M+1}..x_{M+W}]
        labels      = batch[:,  context_len+1 : context_len + rollout_len + 1]   # targets  [x_{M+1}..x_{M+W}]


        nll = F.cross_entropy(logit_slice.reshape(-1, logit_slice.size(-1)),
                              labels.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tok += B * rollout_len

    ppl = math.exp(total_nll / max(1, total_tok))
    return {"ppl": ppl, "tokens": total_tok}


@torch.no_grad()
def evaluate_randomized_matched_sparsity(
    cfg,
    model,
    dl,
    Ts: int,
    Tw: int,
    keep_fracs: Tuple[float, ...],
    target_keep_effective: float,
    prune_choices: Tuple[str, ...],
    quant_choices: Tuple[str, ...],
    target_prune_keep: float,
    target_quant_ratio: float,   # ratio in [0,1], e.g., 16->1.0, 8->0.5
    context_len: int,
    rollout_len: int,
    device: str,
    struct_on_non_eff: bool = False,
):
    """
    Per-sequence randomized matched-keep:
      - For each SEQUENCE, compute the number of effective steps R and a randomized
        schedule consisting of the two nearest discrete κ’s so that the per-sequence
        average keep over its effective steps ≈ target_keep_effective.
      - Non-effective steps are forced dense (κ=1.0).
    """
    # sort κ values but keep mapping back to original indices for histograms
    ks_pairs = sorted([(float(v), i) for i, v in enumerate(keep_fracs)], key=lambda x: x[0])
    ks_vals = [v for v, _ in ks_pairs]
    # pure-tensor maps (avoid CPU .tolist() + sync each step)
    map_sorted_to_orig = torch.tensor([i for _, i in ks_pairs], device=device, dtype=torch.long)
    KEEP = torch.tensor(keep_fracs, device=device, dtype=torch.float32)

    # find neighbors
    if target_keep_effective <= ks_vals[0]:
        lo_s, hi_s, p_hi = 0, 0, 0.0
    elif target_keep_effective >= ks_vals[-1]:
        lo_s, hi_s, p_hi = len(ks_vals) - 1, len(ks_vals) - 1, 1.0
    else:
        lo_s = max(i for i, v in enumerate(ks_vals) if v <= target_keep_effective)
        hi_s = min(i for i, v in enumerate(ks_vals) if v >= target_keep_effective)
        lo_v, hi_v = ks_vals[lo_s], ks_vals[hi_s]
        p_hi = 0.0 if hi_v == lo_v else (target_keep_effective - lo_v) / (hi_v - lo_v)

    total_nll = 0.0
    total_tok = 0
    total_keep_all = 0.0
    total_keep_eff = 0.0
    eff_tok = 0
    # accumulate structural stats across the entire eval (not per-batch)
    total_prune_eff = 0.0
    total_qratio_eff = 0.0

    m = getattr(model, "module", model)
    try:
        m_dtype = next(m.parameters()).dtype
    except StopIteration:
        m_dtype = torch.float32
    dense_idx = keep_fracs.index(1.0) if 1.0 in keep_fracs else int(torch.argmax(KEEP).item())
    thr = Ts + Tw + 1
    # ---- Structured axes (ρ for prune, q=bits/16 for quant) ----
    spec = build_action_spec(
        keep_fracs=keep_fracs,
        prune_choices=prune_choices,
        quant_choices=quant_choices,
    )
    prune_axis = _unique_float_axis(spec.prune_keep)
    quant_axis = _unique_int_axis(spec.q_bits)
    PRUNE = torch.tensor(prune_axis, device=device, dtype=torch.float32)        # [P]
    QBITS = torch.tensor(quant_axis, device=device, dtype=torch.int64)          # [Q]
    QRAT  = (QBITS.to(torch.float32).clamp(min=1.0)) / 16.0                      # [Q]
    dense_p_idx = int(torch.argmax(PRUNE).item())                                # expect ρ=1.0 present
    dense_q_idx = int(torch.argmax(QBITS).item())                                # expect 16 present
    K = KEEP.numel()
    P, Q = PRUNE.numel(), QBITS.numel()
    action_hist = torch.zeros(K * P * Q, device=device)

    def _two_point_mix(vals: torch.Tensor, target: float):
        pairs = sorted([(float(v), i) for i, v in enumerate(vals.tolist())], key=lambda x: x[0])
        vs = [v for v, _ in pairs]
        map_sorted_to_orig = torch.tensor([i for _, i in pairs], device=device, dtype=torch.long)
        if target <= vs[0]:
            lo_s, hi_s, p_hi = 0, 0, 0.0
        elif target >= vs[-1]:
            lo_s, hi_s, p_hi = len(vs) - 1, len(vs) - 1, 1.0
        else:
            lo_s = max(i for i, v in enumerate(vs) if v <= target)
            hi_s = min(i for i, v in enumerate(vs) if v >= target)
            lo_v, hi_v = vs[lo_s], vs[hi_s]
            p_hi = 0.0 if hi_v == lo_v else (target - lo_v) / (hi_v - lo_v)
        return map_sorted_to_orig[lo_s].item(), map_sorted_to_orig[hi_s].item(), float(p_hi)

    p_lo_idx, p_hi_idx, p_p_hi = _two_point_mix(PRUNE, float(target_prune_keep))
    q_lo_idx, q_hi_idx, q_p_hi = _two_point_mix(QRAT,  float(target_quant_ratio))

    enable_structured_controls(model)
    if str(getattr(cfg, "sparsity_criteria", "recency")) == "relevancy": clear_relevancy_keep(model)
    clear_structured_action(model)
    for batch in tqdm(dl, desc="eval sparse randomized"):
        batch = batch.to(device)
        B, _ = batch.shape
        prefill_ids = batch[:, :context_len]
        step_inputs = batch[:, context_len : context_len + rollout_len]
        step_labels = batch[:, context_len + 1 : context_len + rollout_len + 1]

        out = model(input_ids=prefill_ids, use_cache=True, return_dict=True)
        past_kv = detach_cache_to_tuple(out.past_key_values)
        kv_len = torch.full((B,), context_len + 1, device=device, dtype=torch.long)

        # ---- Precompute per-sequence schedules over effective steps ----
        # Number of effective steps for each sequence across the upcoming rollout.
        kv0 = kv_len  # [B]
        neff_rem0 = (thr - kv0 + 1).clamp_min(0)              # [B]
        R_per_b = (rollout_len - neff_rem0).clamp_min(0).to(torch.long)  # [B]
        R_max = int(R_per_b.max().item())

        # Build per-sequence randomized schedules
        #   κ schedule over effective steps: [B, R_max]
        schedule_k = torch.full((B, R_max), lo_s if lo_s == hi_s else lo_s,
                                device=device, dtype=torch.long)
        #   ρ,q schedules over effective or all steps (depending on struct_on_non_eff)
        S_max = R_max if not struct_on_non_eff else rollout_len
        schedule_p = torch.full((B, S_max), p_lo_idx if p_lo_idx == p_hi_idx else p_lo_idx,
                                device=device, dtype=torch.long)
        schedule_q = torch.full((B, S_max), q_lo_idx if q_lo_idx == q_hi_idx else q_lo_idx,
                                device=device, dtype=torch.long)
        for b in range(B):
            Rb = int(R_per_b[b].item())
            if Rb == 0:
                continue
            if lo_s == hi_s:
                seq = torch.full((Rb,), map_sorted_to_orig[lo_s].item(), device=device, dtype=torch.long)
            else:
                n_hi = int(round(float(p_hi) * Rb))
                n_hi = max(0, min(Rb, n_hi))
                n_lo = Rb - n_hi
                seq = torch.cat([
                    torch.full((n_hi,), map_sorted_to_orig[hi_s].item(), device=device, dtype=torch.long),
                    torch.full((n_lo,), map_sorted_to_orig[lo_s].item(), device=device, dtype=torch.long),
                ], dim=0)
                perm = torch.randperm(Rb, device=device)
                seq = seq[perm]
            schedule_k[b, :Rb] = seq
            # ρ schedule (two‑point mix, per‑sequence, like κ)
            Sb = Rb if not struct_on_non_eff else rollout_len
            if Sb > 0:
                if p_lo_idx == p_hi_idx:
                    seq_p = torch.full((Sb,), p_lo_idx, device=device, dtype=torch.long)
                else:
                    lo_v = PRUNE[p_lo_idx].item(); hi_v = PRUNE[p_hi_idx].item()
                    d = max(hi_v - lo_v, 1e-8)
                    n_hi_real = (float(target_prune_keep) * Sb - lo_v * Sb) / d
                    n_hi = int(round(n_hi_real))
                    n_hi = max(0, min(Sb, n_hi))
                    seq_p = torch.cat([
                        torch.full((n_hi,), p_hi_idx, device=device, dtype=torch.long),
                        torch.full((Sb - n_hi,), p_lo_idx, device=device, dtype=torch.long),
                    ], dim=0)
                    perm_p = torch.randperm(Sb, device=device)
                    seq_p = seq_p[perm_p]
                schedule_p[b, :Sb] = seq_p

                # q schedule (two‑point mix, per‑sequence, like κ; using ratios)
                if q_lo_idx == q_hi_idx:
                    seq_q = torch.full((Sb,), q_lo_idx, device=device, dtype=torch.long)
                else:
                    lo_q = QRAT[q_lo_idx].item(); hi_q = QRAT[q_hi_idx].item()
                    dq = max(hi_q - lo_q, 1e-8)
                    n_hi_real_q = (float(target_quant_ratio) * Sb - lo_q * Sb) / dq
                    n_hi_q = int(round(n_hi_real_q))
                    n_hi_q = max(0, min(Sb, n_hi_q))
                    seq_q = torch.cat([
                        torch.full((n_hi_q,), q_hi_idx, device=device, dtype=torch.long),
                        torch.full((Sb - n_hi_q,), q_lo_idx, device=device, dtype=torch.long),
                    ], dim=0)
                    perm_q = torch.randperm(Sb, device=device)
                    seq_q = seq_q[perm_q]
                schedule_q[b, :Sb] = seq_q

        eff_ptrs = torch.zeros(B, device=device, dtype=torch.long)  # κ cursor (effective)
        p_ptrs   = torch.zeros(B, device=device, dtype=torch.long)  # ρ cursor
        q_ptrs   = torch.zeros(B, device=device, dtype=torch.long)  # q cursor
        for t in range(rollout_len):
            kv_before = kv_len
            eff_mask = (kv_before > thr)

            # Choose per-sequence actions from schedules
            a_star = torch.full((B,), dense_idx, device=device, dtype=torch.long)      # κ indices (orig order)
            a_p    = torch.full((B,), dense_p_idx, device=device, dtype=torch.long)    # ρ indices
            a_q    = torch.full((B,), dense_q_idx, device=device, dtype=torch.long)    # q indices
            
            if eff_mask.any():
                idx_eff = torch.nonzero(eff_mask, as_tuple=False).squeeze(-1)
                pos_eff = eff_ptrs[idx_eff]
                chosen = schedule_k[idx_eff, pos_eff]
                a_star[idx_eff] = chosen
                eff_ptrs[idx_eff] = pos_eff + 1
            # Structured axes may apply on effective only or on all steps
            struct_mask = eff_mask if not struct_on_non_eff else torch.ones_like(eff_mask)
            if struct_mask.any():
                idx_struct = torch.nonzero(struct_mask, as_tuple=False).squeeze(-1)
                pos_p = p_ptrs[idx_struct]
                pos_q = q_ptrs[idx_struct]
                a_p[idx_struct] = schedule_p[idx_struct, pos_p]
                a_q[idx_struct] = schedule_q[idx_struct, pos_q]
                p_ptrs[idx_struct] = pos_p + 1
                q_ptrs[idx_struct] = pos_q + 1

            # action histogram over composite (κ, ρ, q)
            flat_idx = a_star * (P * Q) + a_p * Q + a_q
            action_hist.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))

            kappa_vals = KEEP[a_star]      # [B]
            prune_now  = PRUNE[a_p]        # [B]
            qbits_now  = QBITS[a_q]        # [B]
            qratio_now = QRAT[a_q]         # [B]


            # Group by (ρ, bits) and run sub-batches so KV reflects structured choices
            enable_structured_controls(model)
            pq = torch.stack([prune_now, qbits_now.to(torch.float32)], dim=-1)  # [B,2]
            uniq, inv = torch.unique(pq, dim=0, return_inverse=True)

            logits_step = None
            new_cache = None
            pos_ids_all = (kv_before - 1).clamp_min(0).unsqueeze(1)
            cur = step_inputs[:, t]

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
                sel = (inv == g).nonzero(as_tuple=False).squeeze(-1)
                if sel.numel() == 0:
                    continue
                set_structured_action(model, float(p_val), int(q_val))
                cur_g     = cur.index_select(0, sel)
                pos_ids_g = pos_ids_all.index_select(0, sel)
                kappa_g   = kappa_vals.index_select(0, sel)
                bias_g = build_sparse_attention_bias(
                    model=model,
                    past_kv_lens=kv_len.index_select(0, sel),
                    keep_fracs=kappa_g,
                    Ts=Ts, Tw=Tw, device=device, dtype=m_dtype,
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
                )
                if logits_step is None:
                    logits_step = torch.empty((B, out_g.logits.size(-1)),
                                              device=device, dtype=out_g.logits.dtype)
                logits_step.index_copy_(0, sel, out_g.logits[:, -1, :])
                if new_cache is None:
                    new_cache = _init_cache_container_like(out_g.past_key_values, B)
                for li, (k_src, v_src) in enumerate(out_g.past_key_values):
                    k_dst, v_dst = new_cache[li]
                    k_dst.index_copy_(0, sel, k_src)
                    v_dst.index_copy_(0, sel, v_src)
                kv_len.index_add_(0, sel, torch.ones_like(sel, device=device, dtype=kv_len.dtype))

            past_kv = new_cache
            clear_structured_action(model)

            # CE and stats
            nll_t = F.cross_entropy(logits_step, step_labels[:, t], reduction="none")
            total_nll += nll_t.sum().item()
            total_tok += B

            has_old = eff_mask.float()
            struct_gate = has_old if not struct_on_non_eff else torch.ones_like(has_old)
            eff_tok += int(has_old.sum().item())
            total_keep_all += kappa_vals.sum().item()
            total_keep_eff += (kappa_vals * has_old).sum().item()
            # Structured metrics
            total_prune = (prune_now  * struct_gate).sum().item()
            total_qrat  = (qratio_now * struct_gate).sum().item()
            total_prune_eff  += total_prune
            total_qratio_eff += total_qrat

    ppl = math.exp(total_nll / max(1, total_tok))
    avg_keep_all = total_keep_all / max(1, total_tok)
    avg_keep_eff = (total_keep_eff / max(1, eff_tok)) if eff_tok > 0 else 0.0
    action_probs = (action_hist / action_hist.sum().clamp_min(1)).tolist()
    denom_struct = (eff_tok if not struct_on_non_eff else total_tok)
    avg_prune_keep  = total_prune_eff  / max(1, denom_struct)
    avg_quant_ratio = total_qratio_eff / max(1, denom_struct)
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
        "mix_lo_k": ks_vals[lo_s],
        "mix_hi_k": ks_vals[hi_s],
        "mix_p_hi": float(p_hi),
    }

@torch.no_grad()
def evaluate_drift_aware_matched_keep(
    cfg,
    model,
    dl,
    Ts: int,
    Tw: int,
    keep_fracs: Tuple[float, ...],
    target_keep_effective: float,
    prune_choices: Tuple[str, ...],
    quant_choices: Tuple[str, ...],
    target_prune_keep: float,
    target_quant_ratio: float,
    context_len: int,
    rollout_len: int,
    device: str,
    struct_on_non_eff: bool = False,
):
    """
    Drift-Aware Controller (DAC): non-cheating baseline.

    Signal for step t uses *observed representation drift* up to t-1:

      - At t = 0: embedding drift between the prefill's last token and x_t.
      - For t >= 1: cosine drift between last-layer hidden states at t-1 and t-2.
        (We request output_hidden_states=True for the decode steps.)

    High drift => bias toward larger κ; low drift => bias toward smaller κ.

    Budget tracking is identical to EMC: c_req steering + feasibility guardrails.
    """

    import math
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()

    # ----- κ setup: sorted view + map back to original indices for histograms -----
    ks_pairs = sorted([(float(v), i) for i, v in enumerate(keep_fracs)], key=lambda x: x[0])
    ks_vals = torch.tensor([v for v, _ in ks_pairs], device=device, dtype=torch.float32)  # [A] sorted
    map_sorted_to_orig = torch.tensor([i for _, i in ks_pairs], device=device, dtype=torch.long)
    KEEP = torch.tensor(keep_fracs, device=device, dtype=torch.float32)   # [A] original order
    A = len(keep_fracs)

    # Report neighbors of the target
    if target_keep_effective <= ks_vals[0].item():
        lo_s, hi_s, p_hi_target = 0, 0, 0.0
    elif target_keep_effective >= ks_vals[-1].item():
        lo_s, hi_s, p_hi_target = A - 1, A - 1, 1.0
    else:
        lo_s = max(i for i, v in enumerate(ks_vals.tolist()) if v <= target_keep_effective)
        hi_s = min(i for i, v in enumerate(ks_vals.tolist()) if v >= target_keep_effective)
        lo_v, hi_v = ks_vals[lo_s].item(), ks_vals[hi_s].item()
        p_hi_target = 0.0 if hi_v == lo_v else (target_keep_effective - lo_v) / (hi_v - lo_v)

    dense_idx = keep_fracs.index(1.0) if 1.0 in keep_fracs else int(torch.argmax(KEEP).item())
    thr = Ts + Tw + 1
    kappa_min = float(ks_vals.min().item())
    kappa_max = float(ks_vals.max().item())

    # Controller knobs (can be overridden in cfg)
    dac_gamma = float(getattr(cfg, "dac_gamma", 0.35))  # bias strength
    dac_ema   = float(getattr(cfg, "dac_ema", 0.0))    # optional EMA smoothing on drift (0.0 disables)

    # Model dtype for attention bias / quest/relevancy paths
    m = getattr(model, "module", model)
    try:
        m_dtype = next(m.parameters()).dtype
    except StopIteration:
        m_dtype = torch.float32

    # ----- accumulators -----
    total_nll = 0.0
    total_tok = 0
    eff_tok = 0
    total_keep_all = 0.0
    total_keep_eff = 0.0
    # accumulate structured axes across all batches
    total_prune_eff  = 0.0
    total_qratio_eff = 0.0
    # ---- Structured axes (ρ and q) setup ----
    spec = build_action_spec(
        keep_fracs=keep_fracs,
        prune_choices=prune_choices,
        quant_choices=quant_choices,
    )
    prune_axis = _unique_float_axis(spec.prune_keep)
    quant_axis = _unique_int_axis(spec.q_bits)
    PRUNE = torch.tensor(prune_axis, device=device, dtype=torch.float32)        # [P]
    QBITS = torch.tensor(quant_axis, device=device, dtype=torch.int64)          # [Q]
    QRAT  = (QBITS.to(torch.float32).clamp(min=1.0)) / 16.0
    P, Q = PRUNE.numel(), QBITS.numel()
    action_hist = torch.zeros(A * P * Q, device=device)
    dense_p_idx = int(torch.argmax(PRUNE).item())
    dense_q_idx = int(torch.argmax(QBITS).item())
    p_vals_sorted, p_map_sorted_to_orig = torch.sort(PRUNE)                      # sorted values + idx map
    q_vals_sorted, q_map_sorted_to_orig = torch.sort(QRAT)
    p_min, p_max = float(p_vals_sorted[0].item()), float(p_vals_sorted[-1].item())
    q_min, q_max = float(q_vals_sorted[0].item()), float(q_vals_sorted[-1].item())

    emb_layer = model.get_input_embeddings()

    def _cosine_drift(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # Returns value in [0,1]: 0=no change, 1=opposite direction
        return 0.5 * (1.0 - F.cosine_similarity(a, b, dim=-1, eps=eps)).clamp(0.0, 1.0)

    def _choose_with_bias_and_guards(c_req: torch.Tensor,
                                     signal01: torch.Tensor,
                                     cum_keep: torch.Tensor,
                                     cum_eff: torch.Tensor,
                                     R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # As in EMC, but with 'signal01' = drift in [0,1]
        c = c_req.clamp(ks_vals[0], ks_vals[-1])                            # [B_eff]
        hi_sorted = torch.searchsorted(ks_vals, c, right=False).clamp(0, A-1)
        lo_sorted = (hi_sorted - 1).clamp(0, A-1)

        lo_k = ks_vals[lo_sorted]
        hi_k = ks_vals[hi_sorted]
        denom = (hi_k - lo_k).clamp_min(1e-8)
        p_hi_base = torch.where((hi_k - lo_k) > 1e-8, (c - lo_k) / denom, torch.ones_like(c))

        delta = dac_gamma * (signal01 - 0.5)
        p_hi_mod = (p_hi_base + delta).clamp(0.0, 1.0)

        choose_hi = (p_hi_mod >= 0.5) & (hi_sorted != lo_sorted)
        chosen_sorted = torch.where(choose_hi, hi_sorted, lo_sorted)
        chosen_k = ks_vals[chosen_sorted]

        # Feasibility guardrails
        R_post = (R - 1.0).clamp_min(0.0)
        target_total = target_keep_effective * (cum_eff + 1.0 + R_post)
        allowed_min = (target_total - cum_keep - kappa_max * R_post).clamp(kappa_min, kappa_max)
        allowed_max = (target_total - cum_keep - kappa_min * R_post).clamp(kappa_min, kappa_max)

        lo_feas = torch.searchsorted(ks_vals, allowed_min, right=False)
        hi_feas = torch.searchsorted(ks_vals, allowed_max, right=True) - 1
        lo_feas = torch.minimum(lo_feas, hi_feas).clamp(0, A-1)
        hi_feas = torch.maximum(lo_feas, hi_feas).clamp(0, A-1)

        chosen_sorted = torch.maximum(chosen_sorted, lo_feas)
        chosen_sorted = torch.minimum(chosen_sorted, hi_feas)
        chosen_k = ks_vals[chosen_sorted]

        chosen_orig = map_sorted_to_orig[chosen_sorted]
        return chosen_orig, chosen_k

    def _choose_axis(
        c_req: torch.Tensor,
        signal01: torch.Tensor,
        cum_val: torch.Tensor,
        cum_steps: torch.Tensor,
        R: torch.Tensor,
        vals_sorted: torch.Tensor,
        map_sorted_to_orig: torch.Tensor,
        vmin: float, vmax: float,
        gamma: float = 0.0,   # no bias for prune/quant by default
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        A = vals_sorted.numel()
        c = c_req.clamp(vals_sorted[0], vals_sorted[-1])
        hi_sorted = torch.searchsorted(vals_sorted, c, right=False).clamp(0, A-1)
        lo_sorted = (hi_sorted - 1).clamp(0, A-1)
        lo_v = vals_sorted[lo_sorted]
        hi_v = vals_sorted[hi_sorted]
        denom = (hi_v - lo_v).clamp_min(1e-8)
        p_hi_base = torch.where((hi_v - lo_v) > 1e-8, (c - lo_v) / denom, torch.ones_like(c))
        delta = gamma * (signal01 - 0.5)
        p_hi_mod = (p_hi_base + delta).clamp(0.0, 1.0)
        choose_hi = (p_hi_mod >= 0.5) & (hi_sorted != lo_sorted)
        chosen_sorted = torch.where(choose_hi, hi_sorted, lo_sorted)
        chosen_v = vals_sorted[chosen_sorted]
        R_post = (R - 1.0).clamp_min(0.0)
        target_total = c_req * (0.0)  # unused placeholder to keep shapes
        # Feasibility bounds using target (passed via closure)
        return map_sorted_to_orig[chosen_sorted], chosen_v


    for batch in tqdm(dl, desc="eval Drift-Aware matched keep"):
        batch = batch.to(device)
        B, _ = batch.shape

        prefill_ids  = batch[:, :context_len]
        step_inputs  = batch[:, context_len : context_len + rollout_len]
        step_labels  = batch[:, context_len + 1 : context_len + rollout_len + 1]

        # Dense prefill (need last hidden for initial drift reference)
        out = model(input_ids=prefill_ids, use_cache=True, return_dict=True, output_hidden_states=True)
        past_kv = detach_cache_to_tuple(out.past_key_values)
        kv_len = torch.full((B,), context_len + 1, device=device, dtype=torch.long)
        last_h_prev = out.hidden_states[-1][:, -1, :].detach().to(torch.float32)  # h_{prefill_last}
        last_h_prevprev = last_h_prev.clone()

        # Per-sequence budget trackers
        cum_keep = torch.zeros(B, device=device)
        cum_eff  = torch.zeros(B, device=device)
        cum_pru_steps = torch.zeros(B, device=device)   # how many structured decisions made so far
        cum_q_steps   = torch.zeros(B, device=device)
        cum_pru_val   = torch.zeros(B, device=device)   # sum of chosen ρ
        cum_q_val     = torch.zeros(B, device=device)   # sum of chosen q_ratio
        # Initial drift (t = 0) via embeddings vs. prefill's last token
        prev_tok = prefill_ids[:, -1]
        prev_emb = emb_layer(prev_tok).detach().to(torch.float32)  # [B, E]
        drift_prev = torch.full((B,), 0.5, device=device, dtype=torch.float32)  # default
        # Will compute true embedding drift right before choosing at t=0

        # Optional EMA smoother over drift
        drift_ema = None
        if dac_ema > 0.0:
            drift_ema = torch.full((B,), 0.5, device=device, dtype=torch.float32)

        for t in range(rollout_len):
            cur = step_inputs[:, t]
            labels_t = step_labels[:, t]

            kv_before = kv_len
            eff_mask = (kv_before > thr)
            has_old = eff_mask.to(torch.float32)
            a_star = torch.full((B,), dense_idx, device=device, dtype=torch.long)
            a_p    = torch.full((B,), dense_p_idx, device=device, dtype=torch.long)
            a_q    = torch.full((B,), dense_q_idx, device=device, dtype=torch.long)

            # Prepare drift signal in [0,1] for this decision
            if t == 0:
                cur_emb = emb_layer(cur).detach().to(torch.float32)
                drift_now = _cosine_drift(prev_emb, cur_emb)                 # [B]
            else:
                drift_now = _cosine_drift(last_h_prev, last_h_prevprev)      # [B]

            if drift_ema is not None:
                drift_ema = dac_ema * drift_ema + (1.0 - dac_ema) * drift_now
                signal01 = drift_ema
            else:
                signal01 = drift_now

            if eff_mask.any():
                idx_eff = torch.nonzero(eff_mask, as_tuple=False).squeeze(-1)

                # Remaining effective steps including current
                T_rem = rollout_len - t
                neff_rem = (thr - kv_before + 1).clamp_min(0)
                R = (T_rem - neff_rem).clamp_min(0).to(torch.float32)        # [B]
                R_eff = R[idx_eff]

                # Per-sequence required keep
                c_req = (target_keep_effective * (cum_eff[idx_eff] + R_eff) - cum_keep[idx_eff]) / R_eff.clamp_min(1.0)
                c_req = c_req.clamp(kappa_min, kappa_max)

                chosen_orig, chosen_k = _choose_with_bias_and_guards(
                    c_req=c_req,
                    signal01=signal01[idx_eff],
                    cum_keep=cum_keep[idx_eff],
                    cum_eff=cum_eff[idx_eff],
                    R=R_eff,
                )
                a_star[idx_eff] = chosen_orig

            # Structured axes decisions (no drift bias; feasibility via running sums)
            struct_mask = eff_mask if not struct_on_non_eff else torch.ones_like(eff_mask)
            if struct_mask.any():
                idx_struct = torch.nonzero(struct_mask, as_tuple=False).squeeze(-1)
                # Remaining structured steps including current
                if not struct_on_non_eff:
                    T_rem = rollout_len - t
                    neff_rem = (thr - kv_before + 1).clamp_min(0)
                    R_vec_full = (T_rem - neff_rem).clamp_min(0).to(torch.float32)
                else:
                    R_vec_full = torch.full((B,), float(rollout_len - t), device=device, dtype=torch.float32)
                R_vec = R_vec_full[idx_struct]

                # Prune required value per remaining step
                c_req_pru = (float(target_prune_keep) * (cum_pru_steps[idx_struct] + R_vec) -
                             cum_pru_val[idx_struct]) / R_vec.clamp_min(1.0)
                c_req_pru = c_req_pru.clamp(p_min, p_max)
                # Choose prune index (neutral signal 0.5)
                hi_sorted = torch.searchsorted(p_vals_sorted, c_req_pru, right=False).clamp(0, P-1)
                lo_sorted = (hi_sorted - 1).clamp(0, P-1)
                lo_v = p_vals_sorted[lo_sorted]; hi_v = p_vals_sorted[hi_sorted]
                denom = (hi_v - lo_v).clamp_min(1e-8)
                p_hi_base = torch.where((hi_v - lo_v) > 1e-8, (c_req_pru - lo_v) / denom, torch.ones_like(c_req_pru))
                choose_hi = (p_hi_base >= 0.5) & (hi_sorted != lo_sorted)
                chosen_p_sorted = torch.where(choose_hi, hi_sorted, lo_sorted)
                # --- NEW: feasibility guardrails like κ ---
                R_post_p = (R_vec - 1.0).clamp_min(0.0)
                target_total_p = float(target_prune_keep) * (cum_pru_steps[idx_struct] + 1.0 + R_post_p)
                allowed_min_p = (target_total_p - cum_pru_val[idx_struct] - p_max * R_post_p).clamp(p_min, p_max)
                allowed_max_p = (target_total_p - cum_pru_val[idx_struct] - p_min * R_post_p).clamp(p_min, p_max)
                lo_feas_p = torch.searchsorted(p_vals_sorted, allowed_min_p, right=False)
                hi_feas_p = torch.searchsorted(p_vals_sorted, allowed_max_p, right=True) - 1
                lo_feas_p = torch.minimum(lo_feas_p, hi_feas_p).clamp(0, P-1)
                hi_feas_p = torch.maximum(lo_feas_p, hi_feas_p).clamp(0, P-1)
                chosen_p_sorted = torch.maximum(chosen_p_sorted, lo_feas_p)
                chosen_p_sorted = torch.minimum(chosen_p_sorted, hi_feas_p)
                a_p[idx_struct] = p_map_sorted_to_orig[chosen_p_sorted]

                # Quant required ratio per remaining step
                c_req_q = (float(target_quant_ratio) * (cum_q_steps[idx_struct] + R_vec) -
                           cum_q_val[idx_struct]) / R_vec.clamp_min(1.0)
                c_req_q = c_req_q.clamp(q_min, q_max)
                hi_sorted_q = torch.searchsorted(q_vals_sorted, c_req_q, right=False).clamp(0, Q-1)
                lo_sorted_q = (hi_sorted_q - 1).clamp(0, Q-1)
                lo_qv = q_vals_sorted[lo_sorted_q]; hi_qv = q_vals_sorted[hi_sorted_q]
                denom_q = (hi_qv - lo_qv).clamp_min(1e-8)
                p_hi_base_q = torch.where((hi_qv - lo_qv) > 1e-8, (c_req_q - lo_qv) / denom_q, torch.ones_like(c_req_q))
                choose_hi_q = (p_hi_base_q >= 0.5) & (hi_sorted_q != lo_sorted_q)
                chosen_q_sorted = torch.where(choose_hi_q, hi_sorted_q, lo_sorted_q)
                # --- NEW: feasibility guardrails like κ ---
                R_post_q = (R_vec - 1.0).clamp_min(0.0)
                target_total_q = float(target_quant_ratio) * (cum_q_steps[idx_struct] + 1.0 + R_post_q)
                allowed_min_q = (target_total_q - cum_q_val[idx_struct] - q_max * R_post_q).clamp(q_min, q_max)
                allowed_max_q = (target_total_q - cum_q_val[idx_struct] - q_min * R_post_q).clamp(q_min, q_max)
                lo_feas_q = torch.searchsorted(q_vals_sorted, allowed_min_q, right=False)
                hi_feas_q = torch.searchsorted(q_vals_sorted, allowed_max_q, right=True) - 1
                lo_feas_q = torch.minimum(lo_feas_q, hi_feas_q).clamp(0, Q-1)
                hi_feas_q = torch.maximum(lo_feas_q, hi_feas_q).clamp(0, Q-1)
                chosen_q_sorted = torch.maximum(chosen_q_sorted, lo_feas_q)
                chosen_q_sorted = torch.minimum(chosen_q_sorted, hi_feas_q)
                a_q[idx_struct] = q_map_sorted_to_orig[chosen_q_sorted]
            # Decode (request hidden states to measure future drift)
            kappa_now = KEEP[a_star]  # [B]
            prune_now  = PRUNE[a_p]   # [B]
            qbits_now  = QBITS[a_q]   # [B]
            qratio_now = QRAT[a_q]    # [B]
            pos_ids = (kv_before - 1).clamp_min(0).unsqueeze(1)

            enable_structured_controls(model)
            pq = torch.stack([prune_now, qbits_now.to(torch.float32)], dim=-1)
            uniq, inv = torch.unique(pq, dim=0, return_inverse=True)
            logits_step = None
            new_cache = None
            new_state = torch.empty_like(last_h_prev)  # collect per-sample last hidden
            pos_ids_all = pos_ids
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
                sel = (inv == g).nonzero(as_tuple=False).squeeze(-1)
                if sel.numel() == 0:
                    continue
                set_structured_action(model, float(p_val), int(q_val))
                cur_g     = cur.index_select(0, sel)
                pos_ids_g = pos_ids_all.index_select(0, sel)
                kappa_g   = kappa_now.index_select(0, sel)
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
                    logits_step = torch.empty((B, out_g.logits.size(-1)),
                                              device=device, dtype=out_g.logits.dtype)
                logits_step.index_copy_(0, sel, out_g.logits[:, -1, :])
                if new_cache is None:
                    new_cache = _init_cache_container_like(out_g.past_key_values, B)
                for li, (k_src, v_src) in enumerate(out_g.past_key_values):
                    k_dst, v_dst = new_cache[li]
                    k_dst.index_copy_(0, sel, k_src)
                    v_dst.index_copy_(0, sel, v_src)
                kv_len.index_add_(0, sel, torch.ones_like(sel, device=device, dtype=kv_len.dtype))
                new_state.index_copy_(0, sel, out_g.hidden_states[-1][:, -1, :].detach().to(torch.float32))
            clear_structured_action(model)

            past_kv = new_cache
            last_h = new_state

            # Loss and stats
            nll_t = F.cross_entropy(logits_step, labels_t, reduction="none")
            total_nll += nll_t.sum().item()
            total_tok += B

            flat_idx = a_star * (P * Q) + a_p * Q + a_q
            action_hist.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
            eff_tok += int(has_old.sum().item())
            total_keep_all += kappa_now.sum().item()
            total_keep_eff += (kappa_now * has_old).sum().item()

            cum_eff  = cum_eff  + has_old
            cum_keep = cum_keep + has_old * kappa_now
            struct_gate = has_old if not struct_on_non_eff else torch.ones_like(has_old)

            cum_pru_steps = cum_pru_steps + struct_gate
            cum_q_steps   = cum_q_steps   + struct_gate
            cum_pru_val   = cum_pru_val   + struct_gate * prune_now
            cum_q_val     = cum_q_val     + struct_gate * qratio_now
            # global accumulators for final averages
            total_prune_eff  += (prune_now  * struct_gate).sum().item()
            total_qratio_eff += (qratio_now * struct_gate).sum().item()
            # Advance drift state
            last_h_prevprev = last_h_prev
            last_h_prev = last_h

    ppl = math.exp(total_nll / max(1, total_tok))
    avg_keep_all = total_keep_all / max(1, total_tok)
    avg_keep_eff = (total_keep_eff / max(1, eff_tok)) if eff_tok > 0 else 0.0
    action_probs = (action_hist / action_hist.sum().clamp_min(1)).tolist()
    denom_struct = (eff_tok if not struct_on_non_eff else total_tok)
    avg_prune_keep  = total_prune_eff  / max(1, denom_struct)
    avg_quant_ratio = total_qratio_eff / max(1, denom_struct)
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
        "mix_lo_k": ks_vals[lo_s].item(),
        "mix_hi_k": ks_vals[hi_s].item(),
        "mix_p_hi": float(p_hi_target),
    }


@torch.no_grad()
def evaluate_emc_matched_keep(
    cfg,
    model,
    dl,
    Ts: int,
    Tw: int,
    keep_fracs: Tuple[float, ...],
     target_keep_effective: float,
    prune_choices: Tuple[str, ...],
    quant_choices: Tuple[str, ...],
    target_prune_keep: float,
    target_quant_ratio: float,
    context_len: int,
    rollout_len: int,
    device: str,
    struct_on_non_eff: bool = False,
    use_emc_mix: float = None,
):
    """
    Entropy–Margin Controller (EMC): non-cheating baseline.

    Picks κ_t using *previous-step* uncertainty u_{t-1}, where
      u = mix * H_norm + (1 - mix) * (1 - margin)
    H_norm is entropy/log(V), margin is (p_top1 - p_top2).
    High u => bias toward larger κ; low u => bias toward smaller κ.

    Budget tracking:
      - Per-sequence required keep c_req steers actions so the final keep matches the target.
      - Feasibility guardrails ensure we never overshoot/undershoot in a way that
        makes the target unattainable with the remaining steps.

    No lookahead or dense teacher use.
    """

    import math
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()

    # ----- κ setup: sorted view + map back to original indices for histograms -----
    ks_pairs = sorted([(float(v), i) for i, v in enumerate(keep_fracs)], key=lambda x: x[0])
    ks_vals = torch.tensor([v for v, _ in ks_pairs], device=device, dtype=torch.float32)  # [A] sorted
    map_sorted_to_orig = torch.tensor([i for _, i in ks_pairs], device=device, dtype=torch.long)
    KEEP = torch.tensor(keep_fracs, device=device, dtype=torch.float32)   # [A] original order
    A = len(keep_fracs)

    # Report neighbors of the *target* (for consistency with other evaluators)
    if target_keep_effective <= ks_vals[0].item():
        lo_s, hi_s, p_hi_target = 0, 0, 0.0
    elif target_keep_effective >= ks_vals[-1].item():
        lo_s, hi_s, p_hi_target = A - 1, A - 1, 1.0
    else:
        lo_s = max(i for i, v in enumerate(ks_vals.tolist()) if v <= target_keep_effective)
        hi_s = min(i for i, v in enumerate(ks_vals.tolist()) if v >= target_keep_effective)
        lo_v, hi_v = ks_vals[lo_s].item(), ks_vals[hi_s].item()
        p_hi_target = 0.0 if hi_v == lo_v else (target_keep_effective - lo_v) / (hi_v - lo_v)

    # Dense action index (for non-effective steps)
    dense_idx = keep_fracs.index(1.0) if 1.0 in keep_fracs else int(torch.argmax(KEEP).item())
    thr = Ts + Tw + 1
    kappa_min = float(ks_vals.min().item())
    kappa_max = float(ks_vals.max().item())

    # Controller knobs (safe defaults; can be overridden in cfg)
    # emc_mix   = float(getattr(cfg, "emc_mix", 0.5))     # weight for entropy vs margin
    emc_mix   = float(getattr(cfg, "emc_mix", 1.0))     # weight for entropy vs margin
    if use_emc_mix is not None:
        emc_mix = use_emc_mix
    emc_gamma = float(getattr(cfg, "emc_gamma", 0.35))  # bias strength toward hi/lo κ

    # Model dtype for attention bias / quest/relevancy paths
    m = getattr(model, "module", model)
    try:
        m_dtype = next(m.parameters()).dtype
    except StopIteration:
        m_dtype = torch.float32

    # ----- accumulators -----
    total_nll = 0.0
    total_tok = 0
    eff_tok = 0
    total_keep_all = 0.0
    total_keep_eff = 0.0
    # accumulate structured axes across all batches
    total_prune_eff  = 0.0
    total_qratio_eff = 0.0
    # ---- Structured axes (ρ and q) setup ----
    spec = build_action_spec(
        keep_fracs=keep_fracs,
        prune_choices=prune_choices,
        quant_choices=quant_choices,
    )
    prune_axis = _unique_float_axis(spec.prune_keep)
    quant_axis = _unique_int_axis(spec.q_bits)
    PRUNE = torch.tensor(prune_axis, device=device, dtype=torch.float32)        # [P]
    QBITS = torch.tensor(quant_axis, device=device, dtype=torch.int64)          # [Q]
    QRAT  = (QBITS.to(torch.float32).clamp(min=1.0)) / 16.0
    P, Q = PRUNE.numel(), QBITS.numel()
    action_hist = torch.zeros(A * P * Q, device=device)
    dense_p_idx = int(torch.argmax(PRUNE).item())
    dense_q_idx = int(torch.argmax(QBITS).item())
    p_vals_sorted, p_map_sorted_to_orig = torch.sort(PRUNE)
    q_vals_sorted, q_map_sorted_to_orig = torch.sort(QRAT)
    p_min, p_max = float(p_vals_sorted[0].item()), float(p_vals_sorted[-1].item())
    q_min, q_max = float(q_vals_sorted[0].item()), float(q_vals_sorted[-1].item())

    def _compute_uncert_from_logits(logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, V]
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        # normalized entropy
        H = -(p * logp).sum(dim=-1)                           # [B]
        H_norm = H / math.log(logits.size(-1) + 1e-12)
        # top-2 margin
        top2 = torch.topk(p, k=2, dim=-1).values              # [B, 2]
        margin = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)    # [B]
        u = (emc_mix * H_norm + (1.0 - emc_mix) * (1.0 - margin)).clamp(0.0, 1.0)
        return u

    def _choose_with_bias_and_guards(c_req: torch.Tensor,
                                     signal01: torch.Tensor,
                                     cum_keep: torch.Tensor,
                                     cum_eff: torch.Tensor,
                                     R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          chosen_orig_idx: [B_eff] indices in ORIGINAL action order
          chosen_kappa:    [B_eff] chosen κ values
        """
        # Find local [lo,hi] around c_req in the *sorted* action list
        c = c_req.clamp(ks_vals[0], ks_vals[-1])                            # [B_eff]
        hi_sorted = torch.searchsorted(ks_vals, c, right=False).clamp(0, A-1)
        lo_sorted = (hi_sorted - 1).clamp(0, A-1)

        lo_k = ks_vals[lo_sorted]
        hi_k = ks_vals[hi_sorted]

        # base interpolation to match c_req with {lo,hi}
        denom = (hi_k - lo_k).clamp_min(1e-8)
        p_hi_base = torch.where((hi_k - lo_k) > 1e-8, (c - lo_k) / denom, torch.ones_like(c))

        # uncertainty bias: delta in [-gamma/2, +gamma/2] around 0.5 threshold
        delta = emc_gamma * (signal01 - 0.5)
        p_hi_mod = (p_hi_base + delta).clamp(0.0, 1.0)

        # initial deterministic choice (no randomness): threshold at 0.5
        choose_hi = (p_hi_mod >= 0.5) & (hi_sorted != lo_sorted)
        chosen_sorted = torch.where(choose_hi, hi_sorted, lo_sorted)         # [B_eff]
        chosen_k = ks_vals[chosen_sorted]

        # ---- Feasibility guardrails (make target attainable with remaining steps) ----
        R_post = (R - 1.0).clamp_min(0.0)                                    # [B_eff]
        target_total = target_keep_effective * (cum_eff + 1.0 + R_post)      # at end
        # allowed interval for current κ:
        allowed_min = (target_total - cum_keep - kappa_max * R_post).clamp(kappa_min, kappa_max)
        allowed_max = (target_total - cum_keep - kappa_min * R_post).clamp(kappa_min, kappa_max)

        # clamp chosen index to feasible band in sorted κ
        lo_feas = torch.searchsorted(ks_vals, allowed_min, right=False)
        hi_feas = torch.searchsorted(ks_vals, allowed_max, right=True) - 1
        lo_feas = torch.minimum(lo_feas, hi_feas).clamp(0, A-1)
        hi_feas = torch.maximum(lo_feas, hi_feas).clamp(0, A-1)

        chosen_sorted = torch.maximum(chosen_sorted, lo_feas)
        chosen_sorted = torch.minimum(chosen_sorted, hi_feas)
        chosen_k = ks_vals[chosen_sorted]

        # map back to original action indices
        chosen_orig = map_sorted_to_orig[chosen_sorted]
        return chosen_orig, chosen_k

    enable_structured_controls(model)
    if str(getattr(cfg, "sparsity_criteria", "recency")) == "relevancy": clear_relevancy_keep(model)
    clear_structured_action(model)
    for batch in tqdm(dl, desc="eval EMC matched keep"):
        batch = batch.to(device)
        B, _ = batch.shape

        prefill_ids  = batch[:, :context_len]
        step_inputs  = batch[:, context_len : context_len + rollout_len]
        step_labels  = batch[:, context_len + 1 : context_len + rollout_len + 1]

        # Dense prefill to build cache (no hidden states required here)
        out = model(input_ids=prefill_ids, use_cache=True, return_dict=True)
        past_kv = detach_cache_to_tuple(out.past_key_values)
        kv_len = torch.full((B,), context_len + 1, device=device, dtype=torch.long)

        # Track per-sequence budget
        cum_keep = torch.zeros(B, device=device)
        cum_eff  = torch.zeros(B, device=device)
        cum_pru_steps = torch.zeros(B, device=device)
        cum_q_steps   = torch.zeros(B, device=device)
        cum_pru_val   = torch.zeros(B, device=device)
        cum_q_val     = torch.zeros(B, device=device)

        # previous uncertainty (for t=0, neutral 0.5)
        prev_uncert = torch.full((B,), 0.5, device=device, dtype=torch.float32)

        for t in range(rollout_len):
            cur = step_inputs[:, t]                     # [B]
            labels_t = step_labels[:, t]               # [B]

            kv_before = kv_len
            eff_mask = (kv_before > thr)               # [B] bool
            has_old = eff_mask.to(torch.float32)

            a_star = torch.full((B,), dense_idx, device=device, dtype=torch.long)
            a_p    = torch.full((B,), dense_p_idx, device=device, dtype=torch.long)
            a_q    = torch.full((B,), dense_q_idx, device=device, dtype=torch.long)

            if eff_mask.any():
                idx_eff = torch.nonzero(eff_mask, as_tuple=False).squeeze(-1)

                # Remaining effective steps including current
                T_rem = rollout_len - t
                neff_rem = (thr - kv_before + 1).clamp_min(0)
                R = (T_rem - neff_rem).clamp_min(0).to(torch.float32)        # [B]
                R_eff = R[idx_eff]

                # Per-sequence required keep
                c_req = (target_keep_effective * (cum_eff[idx_eff] + R_eff) - cum_keep[idx_eff]) / R_eff.clamp_min(1.0)
                c_req = c_req.clamp(kappa_min, kappa_max)

                # Choose action using uncertainty from previous step
                chosen_orig, chosen_k = _choose_with_bias_and_guards(
                    c_req=c_req,
                    signal01=prev_uncert[idx_eff],
                    cum_keep=cum_keep[idx_eff],
                    cum_eff=cum_eff[idx_eff],
                    R=R_eff,
                )
                a_star[idx_eff] = chosen_orig

            # Decode one step with the chosen κ
            # Structured choices (neutral wrt uncertainty; guard via running averages)
            struct_mask = eff_mask if not struct_on_non_eff else torch.ones_like(eff_mask)
            if struct_mask.any():
                idx_struct = torch.nonzero(struct_mask, as_tuple=False).squeeze(-1)
                # Remaining structured steps per sequence (match κ logic when struct_on_non_eff=False)
                if not struct_on_non_eff:
                    T_rem = rollout_len - t
                    neff_rem = (thr - kv_before + 1).clamp_min(0)
                    R_vec_full = (T_rem - neff_rem).clamp_min(0).to(torch.float32)
                else:
                    R_vec_full = torch.full((B,), float(rollout_len - t), device=device, dtype=torch.float32)
                R_vec = R_vec_full[idx_struct]
                # Prune
                c_req_pru = (float(target_prune_keep) * (cum_pru_steps[idx_struct] + R_vec) -
                             cum_pru_val[idx_struct]) / R_vec.clamp_min(1.0)
                c_req_pru = c_req_pru.clamp(p_min, p_max)
                hi_sorted = torch.searchsorted(p_vals_sorted, c_req_pru, right=False).clamp(0, P-1)
                lo_sorted = (hi_sorted - 1).clamp(0, P-1)
                lo_v = p_vals_sorted[lo_sorted]; hi_v = p_vals_sorted[hi_sorted]
                denom = (hi_v - lo_v).clamp_min(1e-8)
                p_hi_base = torch.where((hi_v - lo_v) > 1e-8, (c_req_pru - lo_v) / denom, torch.ones_like(c_req_pru))
                choose_hi = (p_hi_base >= 0.5) & (hi_sorted != lo_sorted)
                chosen_p_sorted = torch.where(choose_hi, hi_sorted, lo_sorted)
                # --- NEW: feasibility guardrails for prune like κ ---
                R_post_p = (R_vec - 1.0).clamp_min(0.0)
                target_total_p = float(target_prune_keep) * (cum_pru_steps[idx_struct] + 1.0 + R_post_p)
                allowed_min_p = (target_total_p - cum_pru_val[idx_struct] - p_max * R_post_p).clamp(p_min, p_max)
                allowed_max_p = (target_total_p - cum_pru_val[idx_struct] - p_min * R_post_p).clamp(p_min, p_max)
                lo_feas_p = torch.searchsorted(p_vals_sorted, allowed_min_p, right=False)
                hi_feas_p = torch.searchsorted(p_vals_sorted, allowed_max_p, right=True) - 1
                lo_feas_p = torch.minimum(lo_feas_p, hi_feas_p).clamp(0, P-1)
                hi_feas_p = torch.maximum(lo_feas_p, hi_feas_p).clamp(0, P-1)
                chosen_p_sorted = torch.maximum(chosen_p_sorted, lo_feas_p)
                chosen_p_sorted = torch.minimum(chosen_p_sorted, hi_feas_p)
                a_p[idx_struct] = p_map_sorted_to_orig[chosen_p_sorted]
                # Quant
                c_req_q = (float(target_quant_ratio) * (cum_q_steps[idx_struct] + R_vec) -
                           cum_q_val[idx_struct]) / R_vec.clamp_min(1.0)
                c_req_q = c_req_q.clamp(q_min, q_max)
                hi_sorted_q = torch.searchsorted(q_vals_sorted, c_req_q, right=False).clamp(0, Q-1)
                lo_sorted_q = (hi_sorted_q - 1).clamp(0, Q-1)
                lo_qv = q_vals_sorted[lo_sorted_q]; hi_qv = q_vals_sorted[hi_sorted_q]
                denom_q = (hi_qv - lo_qv).clamp_min(1e-8)
                p_hi_base_q = torch.where((hi_qv - lo_qv) > 1e-8, (c_req_q - lo_qv) / denom_q, torch.ones_like(c_req_q))
                choose_hi_q = (p_hi_base_q >= 0.5) & (hi_sorted_q != lo_sorted_q)
                chosen_q_sorted = torch.where(choose_hi_q, hi_sorted_q, lo_sorted_q)
                # --- NEW: feasibility guardrails for quant ratio like κ ---
                R_post_q = (R_vec - 1.0).clamp_min(0.0)
                target_total_q = float(target_quant_ratio) * (cum_q_steps[idx_struct] + 1.0 + R_post_q)
                allowed_min_q = (target_total_q - cum_q_val[idx_struct] - q_max * R_post_q).clamp(q_min, q_max)
                allowed_max_q = (target_total_q - cum_q_val[idx_struct] - q_min * R_post_q).clamp(q_min, q_max)
                lo_feas_q = torch.searchsorted(q_vals_sorted, allowed_min_q, right=False)
                hi_feas_q = torch.searchsorted(q_vals_sorted, allowed_max_q, right=True) - 1
                lo_feas_q = torch.minimum(lo_feas_q, hi_feas_q).clamp(0, Q-1)
                hi_feas_q = torch.maximum(lo_feas_q, hi_feas_q).clamp(0, Q-1)
                chosen_q_sorted = torch.maximum(chosen_q_sorted, lo_feas_q)
                chosen_q_sorted = torch.minimum(chosen_q_sorted, hi_feas_q)
                a_q[idx_struct] = q_map_sorted_to_orig[chosen_q_sorted]

            # Decode one step with the chosen κ/ρ/q
            kappa_now = KEEP[a_star]                                # [B]
            prune_now  = PRUNE[a_p]                                 # [B]
            qbits_now  = QBITS[a_q]                                 # [B]
            qratio_now = QRAT[a_q]                                  # [B]
            pos_ids = (kv_before - 1).clamp_min(0).unsqueeze(1)

            enable_structured_controls(model)
            pq = torch.stack([prune_now, qbits_now.to(torch.float32)], dim=-1)
            uniq, inv = torch.unique(pq, dim=0, return_inverse=True)
            logits_step = None
            new_cache = None
            pos_ids_all = pos_ids
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
                sel = (inv == g).nonzero(as_tuple=False).squeeze(-1)
                if sel.numel() == 0:
                    continue
                set_structured_action(model, float(p_val), int(q_val))
                cur_g     = cur.index_select(0, sel)
                pos_ids_g = pos_ids_all.index_select(0, sel)
                kappa_g   = kappa_now.index_select(0, sel)
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
                )
                if logits_step is None:
                    logits_step = torch.empty((B, out_g.logits.size(-1)),
                                              device=device, dtype=out_g.logits.dtype)
                logits_step.index_copy_(0, sel, out_g.logits[:, -1, :])
                if new_cache is None:
                    new_cache = _init_cache_container_like(out_g.past_key_values, B)
                for li, (k_src, v_src) in enumerate(out_g.past_key_values):
                    k_dst, v_dst = new_cache[li]
                    k_dst.index_copy_(0, sel, k_src)
                    v_dst.index_copy_(0, sel, v_src)
                kv_len.index_add_(0, sel, torch.ones_like(sel, device=device, dtype=kv_len.dtype))

            clear_structured_action(model)
            past_kv = new_cache
            # Loss and stats
            nll_t = F.cross_entropy(logits_step, labels_t, reduction="none")
            total_nll += nll_t.sum().item()
            total_tok += B

            # Update budget trackers
            flat_idx = a_star * (P * Q) + a_p * Q + a_q
            action_hist.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
            eff_tok += int(has_old.sum().item())
            total_keep_all += kappa_now.sum().item()
            total_keep_eff += (kappa_now * has_old).sum().item()
            cum_eff  = cum_eff  + has_old
            cum_keep = cum_keep + has_old * kappa_now
            struct_gate = has_old if not struct_on_non_eff else torch.ones_like(has_old)
            cum_pru_steps = cum_pru_steps + struct_gate
            cum_q_steps   = cum_q_steps   + struct_gate
            cum_pru_val   = cum_pru_val   + struct_gate * prune_now
            cum_q_val     = cum_q_val     + struct_gate * qratio_now
            # global accumulators for final averages
            total_prune_eff  += (prune_now  * struct_gate).sum().item()
            total_qratio_eff += (qratio_now * struct_gate).sum().item()
            # Update uncertainty for NEXT step
            prev_uncert = _compute_uncert_from_logits(logits_step)

    ppl = math.exp(total_nll / max(1, total_tok))
    avg_keep_all = total_keep_all / max(1, total_tok)
    avg_keep_eff = (total_keep_eff / max(1, eff_tok)) if eff_tok > 0 else 0.0
    action_probs = (action_hist / action_hist.sum().clamp_min(1)).tolist()
    denom_struct = (eff_tok if not struct_on_non_eff else total_tok)
    avg_prune_keep  = total_prune_eff  / max(1, denom_struct)
    avg_quant_ratio = total_qratio_eff / max(1, denom_struct)

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
        "mix_lo_k": ks_vals[lo_s].item(),
        "mix_hi_k": ks_vals[hi_s].item(),
        "mix_p_hi": float(p_hi_target),
    }


@torch.no_grad()
def evaluate_lrm_tokens_matched_keep(
    cfg,
    model,
    dl,
    Ts: int,
    Tw: int,
    keep_fracs: Tuple[float, ...],
    target_keep_effective: float,
    prune_choices: Tuple[str, ...],
    quant_choices: Tuple[str, ...],
    target_prune_keep: float,
    target_quant_ratio: float,
    context_len: int,
    rollout_len: int,
    device: str,
    struct_on_non_eff: bool = False,
):
    """
    % https://chatgpt.com/c/6903a41c-ac20-8329-add0-49bca9e72177
    TS‑1. Long‑Range Mass (LRM): non-cheating baseline (token-sparsity controller).

    Idea:
      Use the *previous-step* attention distribution over keys to estimate how much
      mass lies on *distant* positions:
          LRM = mass(dist > tail_threshold)
      Larger LRM => choose larger κ (keep more history) on the next step.

    Notes:
      * Causal: relies only on attention from the step that just ran (t), to
        choose κ for the next step (t+1).
      * Works with recency / relevancy / quest, since we read the *actual*
        attention weights via output_attentions=True.
      * Budget-matched: same feasibility guardrails used in your EMC evaluator.
      * Pruning/quant axes are steered by per-axis running averages (same as EMC).
    """

    import math
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()

    # === κ action axis setup (sorted view + map to original indices) ===
    ks_pairs = sorted([(float(v), i) for i, v in enumerate(keep_fracs)], key=lambda x: x[0])
    ks_vals = torch.tensor([v for v, _ in ks_pairs], device=device, dtype=torch.float32)  # [A] sorted
    map_sorted_to_orig = torch.tensor([i for _, i in ks_pairs], device=device, dtype=torch.long)
    KEEP = torch.tensor(keep_fracs, device=device, dtype=torch.float32)   # [A] original order
    A = len(keep_fracs)

    # For reporting: neighbors of target keep
    if target_keep_effective <= ks_vals[0].item():
        lo_s, hi_s, p_hi_target = 0, 0, 0.0
    elif target_keep_effective >= ks_vals[-1].item():
        lo_s, hi_s, p_hi_target = A - 1, A - 1, 1.0
    else:
        lo_s = max(i for i, v in enumerate(ks_vals.tolist()) if v <= target_keep_effective)
        hi_s = min(i for i, v in enumerate(ks_vals.tolist()) if v >= target_keep_effective)
        lo_v, hi_v = ks_vals[lo_s].item(), ks_vals[hi_s].item()
        p_hi_target = 0.0 if hi_v == lo_v else (target_keep_effective - lo_v) / (hi_v - lo_v)

    # Dense index for non-effective steps
    dense_idx = keep_fracs.index(1.0) if 1.0 in keep_fracs else int(torch.argmax(KEEP).item())
    thr = Ts + Tw + 1
    kappa_min = float(ks_vals.min().item())
    kappa_max = float(ks_vals.max().item())

    # --- Controller knobs ---
    # Bias strength for signaling: delta = gamma * (signal - 0.5)
    lrm_gamma = float(getattr(cfg, "lrm_gamma", 0.35))
    # Tail threshold in tokens (distance from current). If not set, default to max(64, Ts+Tw+1).
    lrm_tail_tokens = int(getattr(cfg, "lrm_tail_tokens", max(64, Ts + Tw + 1)))
    # Reduce heads via "mean" (default) or "median"
    lrm_head_reduce = str(getattr(cfg, "lrm_head_reduce", "mean")).lower()

    # Model dtype for attention bias construction
    m = getattr(model, "module", model)
    try:
        m_dtype = next(m.parameters()).dtype
    except StopIteration:
        m_dtype = torch.float32

    # === Accumulators ===
    total_nll = 0.0
    total_tok = 0
    eff_tok = 0
    total_keep_all = 0.0
    total_keep_eff = 0.0
    # accumulate structured axes across all batches
    total_prune_eff  = 0.0
    total_qratio_eff = 0.0

    # === Structured axes (pruning ρ, quant q) setup (same as EMC) ===
    spec = build_action_spec(
        keep_fracs=keep_fracs,
        prune_choices=prune_choices,
        quant_choices=quant_choices,
    )
    prune_axis = _unique_float_axis(spec.prune_keep)
    quant_axis = _unique_int_axis(spec.q_bits)
    PRUNE = torch.tensor(prune_axis, device=device, dtype=torch.float32)        # [P]
    QBITS = torch.tensor(quant_axis, device=device, dtype=torch.int64)          # [Q]
    QRAT  = (QBITS.to(torch.float32).clamp(min=1.0)) / 16.0
    P, Q = PRUNE.numel(), QBITS.numel()
    action_hist = torch.zeros(A * P * Q, device=device)
    dense_p_idx = int(torch.argmax(PRUNE).item())
    dense_q_idx = int(torch.argmax(QBITS).item())
    p_vals_sorted, p_map_sorted_to_orig = torch.sort(PRUNE)
    q_vals_sorted, q_map_sorted_to_orig = torch.sort(QRAT)
    p_min, p_max = float(p_vals_sorted[0].item()), float(p_vals_sorted[-1].item())
    q_min, q_max = float(q_vals_sorted[0].item()), float(q_vals_sorted[-1].item())

    # --- Helpers ---

    def _reduce_heads(attn: torch.Tensor) -> torch.Tensor:
        """
        attn: [B, H, q_len(=1), K] → returns [B, K] by reducing heads and query dim.
        """
        if attn is None:
            return None
        # reduce over heads
        if lrm_head_reduce == "median":
            w = attn.median(dim=1).values
        else:
            w = attn.mean(dim=1)
        # take the last query position (q_len=1 in decode)
        if w.dim() == 3:
            w = w[:, -1, :]  # [B, K]
        return w

    def _compute_lrm_from_attn(attn_last: torch.Tensor, tail_tokens: int) -> torch.Tensor:
        """
        attn_last: [B, H, 1, K] attention over keys for the just-decoded token.
        Returns: [B] signal in [0,1], mass on positions with distance > tail_tokens.
        """
        w = _reduce_heads(attn_last)  # [B, K]
        if w is None:
            return None
        w = w.clamp_min(0)
        K = w.size(-1)
        if K <= 1:
            return torch.zeros(w.size(0), device=w.device, dtype=torch.float32)
        # distance: last key (self) has dist=0; earlier keys large dist
        dist = torch.arange(K, device=w.device).view(1, K)
        dist = (K - 1) - dist
        mask = (dist > tail_tokens).to(w.dtype)
        num = (w * mask).sum(dim=-1)
        den = w.sum(dim=-1).clamp_min(1e-12)
        return (num / den).clamp(0.0, 1.0).to(torch.float32)

    def _choose_with_bias_and_guards(c_req: torch.Tensor,
                                     signal01: torch.Tensor,
                                     cum_keep: torch.Tensor,
                                     cum_eff: torch.Tensor,
                                     R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map required-keep and a [0,1] signal to a discrete κ with feasibility guardrails.
        Returns:
          chosen_orig_idx: [B_eff] ORIGINAL action indices
          chosen_kappa:    [B_eff] chosen κ values
        """
        c = c_req.clamp(ks_vals[0], ks_vals[-1])                            # [B_eff]
        hi_sorted = torch.searchsorted(ks_vals, c, right=False).clamp(0, A-1)
        lo_sorted = (hi_sorted - 1).clamp(0, A-1)

        lo_k = ks_vals[lo_sorted]
        hi_k = ks_vals[hi_sorted]

        denom = (hi_k - lo_k).clamp_min(1e-8)
        p_hi_base = torch.where((hi_k - lo_k) > 1e-8, (c - lo_k) / denom, torch.ones_like(c))

        # signal-based bias
        delta = lrm_gamma * (signal01 - 0.5)
        p_hi_mod = (p_hi_base + delta).clamp(0.0, 1.0)

        choose_hi = (p_hi_mod >= 0.5) & (hi_sorted != lo_sorted)
        chosen_sorted = torch.where(choose_hi, hi_sorted, lo_sorted)
        chosen_k = ks_vals[chosen_sorted]

        # --- Feasibility guardrails ---
        R_post = (R - 1.0).clamp_min(0.0)
        target_total = target_keep_effective * (cum_eff + 1.0 + R_post)
        allowed_min = (target_total - cum_keep - kappa_max * R_post).clamp(kappa_min, kappa_max)
        allowed_max = (target_total - cum_keep - kappa_min * R_post).clamp(kappa_min, kappa_max)

        lo_feas = torch.searchsorted(ks_vals, allowed_min, right=False)
        hi_feas = torch.searchsorted(ks_vals, allowed_max, right=True) - 1
        lo_feas = torch.minimum(lo_feas, hi_feas).clamp(0, A-1)
        hi_feas = torch.maximum(lo_feas, hi_feas).clamp(0, A-1)

        chosen_sorted = torch.maximum(chosen_sorted, lo_feas)
        chosen_sorted = torch.minimum(chosen_sorted, hi_feas)
        chosen_k = ks_vals[chosen_sorted]
        chosen_orig = map_sorted_to_orig[chosen_sorted]
        return chosen_orig, chosen_k

    # === Main loop ===
    enable_structured_controls(model)
    if str(getattr(cfg, "sparsity_criteria", "recency")) == "relevancy":
        clear_relevancy_keep(model)
    clear_structured_action(model)

    for batch in tqdm(dl, desc="eval LRM matched keep"):
        batch = batch.to(device)
        B, _ = batch.shape

        prefill_ids  = batch[:, :context_len]
        step_inputs  = batch[:, context_len : context_len + rollout_len]
        step_labels  = batch[:, context_len + 1 : context_len + rollout_len + 1]

        # Dense prefill to build cache (no attentions needed here)
        out = model(input_ids=prefill_ids, use_cache=True, return_dict=True)
        past_kv = detach_cache_to_tuple(out.past_key_values)
        kv_len = torch.full((B,), context_len + 1, device=device, dtype=torch.long)

        # Budget trackers
        cum_keep = torch.zeros(B, device=device)
        cum_eff  = torch.zeros(B, device=device)
        cum_pru_steps = torch.zeros(B, device=device)
        cum_q_steps   = torch.zeros(B, device=device)
        cum_pru_val   = torch.zeros(B, device=device)
        cum_q_val     = torch.zeros(B, device=device)

        # previous LRM signal (neutral 0.5 for t=0)
        prev_lrm = torch.full((B,), 0.5, device=device, dtype=torch.float32)

        for t in range(rollout_len):
            cur = step_inputs[:, t]             # [B]
            labels_t = step_labels[:, t]        # [B]

            kv_before = kv_len
            eff_mask = (kv_before > thr)        # [B] bool
            has_old = eff_mask.to(torch.float32)

            a_star = torch.full((B,), dense_idx, device=device, dtype=torch.long)
            a_p    = torch.full((B,), dense_p_idx, device=device, dtype=torch.long)
            a_q    = torch.full((B,), dense_q_idx, device=device, dtype=torch.long)

            if eff_mask.any():
                idx_eff = torch.nonzero(eff_mask, as_tuple=False).squeeze(-1)

                # Remaining effective steps incl. current
                T_rem = rollout_len - t
                neff_rem = (thr - kv_before + 1).clamp_min(0)
                R = (T_rem - neff_rem).clamp_min(0).to(torch.float32)        # [B]
                R_eff = R[idx_eff]

                # Per-sequence required keep
                c_req = (target_keep_effective * (cum_eff[idx_eff] + R_eff) - cum_keep[idx_eff]) / R_eff.clamp_min(1.0)
                c_req = c_req.clamp(kappa_min, kappa_max)

                # Choose κ using previous-step LRM signal
                chosen_orig, chosen_k = _choose_with_bias_and_guards(
                    c_req=c_req,
                    signal01=prev_lrm[idx_eff],
                    cum_keep=cum_keep[idx_eff],
                    cum_eff=cum_eff[idx_eff],
                    R=R_eff,
                )
                a_star[idx_eff] = chosen_orig

            # Structured axes (same as EMC). Optionally apply even on non-effective steps.
            struct_mask = eff_mask if not struct_on_non_eff else torch.ones_like(eff_mask)
            if struct_mask.any():
                idx_struct = torch.nonzero(struct_mask, as_tuple=False).squeeze(-1)
                # match κ logic when struct_on_non_eff=False
                if not struct_on_non_eff:
                    T_rem = rollout_len - t
                    neff_rem = (thr - kv_before + 1).clamp_min(0)
                    R_vec_full = (T_rem - neff_rem).clamp_min(0).to(torch.float32)
                else:
                    R_vec_full = torch.full((B,), float(rollout_len - t), device=device, dtype=torch.float32)
                R_vec = R_vec_full[idx_struct]

                # Prune
                c_req_pru = (float(target_prune_keep) * (cum_pru_steps[idx_struct] + R_vec) -
                             cum_pru_val[idx_struct]) / R_vec.clamp_min(1.0)
                c_req_pru = c_req_pru.clamp(p_min, p_max)
                hi_sorted = torch.searchsorted(p_vals_sorted, c_req_pru, right=False).clamp(0, P-1)
                lo_sorted = (hi_sorted - 1).clamp(0, P-1)
                lo_v = p_vals_sorted[lo_sorted]; hi_v = p_vals_sorted[hi_sorted]
                denom = (hi_v - lo_v).clamp_min(1e-8)
                p_hi_base = torch.where((hi_v - lo_v) > 1e-8, (c_req_pru - lo_v) / denom, torch.ones_like(c_req_pru))
                choose_hi = (p_hi_base >= 0.5) & (hi_sorted != lo_sorted)
                chosen_p_sorted = torch.where(choose_hi, hi_sorted, lo_sorted)

                # Feasibility guardrails (prune)
                R_post_p = (R_vec - 1.0).clamp_min(0.0)
                target_total_p = float(target_prune_keep) * (cum_pru_steps[idx_struct] + 1.0 + R_post_p)
                allowed_min_p = (target_total_p - cum_pru_val[idx_struct] - p_max * R_post_p).clamp(p_min, p_max)
                allowed_max_p = (target_total_p - cum_pru_val[idx_struct] - p_min * R_post_p).clamp(p_min, p_max)
                lo_feas_p = torch.searchsorted(p_vals_sorted, allowed_min_p, right=False)
                hi_feas_p = torch.searchsorted(p_vals_sorted, allowed_max_p, right=True) - 1
                lo_feas_p = torch.minimum(lo_feas_p, hi_feas_p).clamp(0, P-1)
                hi_feas_p = torch.maximum(lo_feas_p, hi_feas_p).clamp(0, P-1)
                chosen_p_sorted = torch.maximum(chosen_p_sorted, lo_feas_p)
                chosen_p_sorted = torch.minimum(chosen_p_sorted, hi_feas_p)
                a_p[idx_struct] = p_map_sorted_to_orig[chosen_p_sorted]

                # Quant
                c_req_q = (float(target_quant_ratio) * (cum_q_steps[idx_struct] + R_vec) -
                           cum_q_val[idx_struct]) / R_vec.clamp_min(1.0)
                c_req_q = c_req_q.clamp(q_min, q_max)
                hi_sorted_q = torch.searchsorted(q_vals_sorted, c_req_q, right=False).clamp(0, Q-1)
                lo_sorted_q = (hi_sorted_q - 1).clamp(0, Q-1)
                lo_qv = q_vals_sorted[lo_sorted_q]; hi_qv = q_vals_sorted[hi_sorted_q]
                denom_q = (hi_qv - lo_qv).clamp_min(1e-8)
                p_hi_base_q = torch.where((hi_qv - lo_qv) > 1e-8, (c_req_q - lo_qv) / denom_q, torch.ones_like(c_req_q))
                choose_hi_q = (p_hi_base_q >= 0.5) & (hi_sorted_q != lo_sorted_q)
                chosen_q_sorted = torch.where(choose_hi_q, hi_sorted_q, lo_sorted_q)

                # Feasibility guardrails (quant)
                R_post_q = (R_vec - 1.0).clamp_min(0.0)
                target_total_q = float(target_quant_ratio) * (cum_q_steps[idx_struct] + 1.0 + R_post_q)
                allowed_min_q = (target_total_q - cum_q_val[idx_struct] - q_max * R_post_q).clamp(q_min, q_max)
                allowed_max_q = (target_total_q - cum_q_val[idx_struct] - q_min * R_post_q).clamp(q_min, q_max)
                lo_feas_q = torch.searchsorted(q_vals_sorted, allowed_min_q, right=False)
                hi_feas_q = torch.searchsorted(q_vals_sorted, allowed_max_q, right=True) - 1
                lo_feas_q = torch.minimum(lo_feas_q, hi_feas_q).clamp(0, Q-1)
                hi_feas_q = torch.maximum(lo_feas_q, hi_feas_q).clamp(0, Q-1)
                chosen_q_sorted = torch.maximum(chosen_q_sorted, lo_feas_q)
                chosen_q_sorted = torch.minimum(chosen_q_sorted, hi_feas_q)
                a_q[idx_struct] = q_map_sorted_to_orig[chosen_q_sorted]

            # ---- Decode one step with chosen κ/ρ/q; request attentions ----
            kappa_now = KEEP[a_star]                         # [B]
            prune_now  = PRUNE[a_p]                          # [B]
            qbits_now  = QBITS[a_q]                          # [B]
            qratio_now = QRAT[a_q]                           # [B]
            pos_ids = (kv_before - 1).clamp_min(0).unsqueeze(1)

            enable_structured_controls(model)
            pq = torch.stack([prune_now, qbits_now.to(torch.float32)], dim=-1)
            uniq, inv = torch.unique(pq, dim=0, return_inverse=True)
            logits_step = None
            lrm_step = torch.full((B,), 0.5, device=device, dtype=torch.float32)  # will be updated per group
            new_cache = None
            pos_ids_all = pos_ids

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
                sel = (inv == g).nonzero(as_tuple=False).squeeze(-1)
                if sel.numel() == 0:
                    continue
                set_structured_action(model, float(p_val), int(q_val))

                cur_g     = cur.index_select(0, sel)
                pos_ids_g = pos_ids_all.index_select(0, sel)
                kappa_g   = kappa_now.index_select(0, sel)

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
                    output_attentions=True,           # <--- needed for LRM
                    return_dict=True,
                )

                # Collect logits
                if logits_step is None:
                    logits_step = torch.empty((B, out_g.logits.size(-1)),
                                              device=device, dtype=out_g.logits.dtype)
                logits_step.index_copy_(0, sel, out_g.logits[:, -1, :])

                # Collect last-layer attentions to compute LRM signal for NEXT step
                attns = getattr(out_g, "attentions", None)
                if attns is not None and len(attns) > 0 and attns[-1] is not None:
                    # last layer attention: [B_g, H, q_len(=1), K]
                    attn_last = attns[-1]
                    # Some backends return a tuple per layer; ensure tensor
                    if isinstance(attn_last, (tuple, list)):
                        attn_last = attn_last[0]
                    lrm_vals = _compute_lrm_from_attn(attn_last, lrm_tail_tokens)  # [B_g] or None
                    if lrm_vals is not None:
                        lrm_step.index_copy_(0, sel, lrm_vals)

                # Merge caches for the next iteration
                if new_cache is None:
                    new_cache = _init_cache_container_like(out_g.past_key_values, B)
                for li, (k_src, v_src) in enumerate(out_g.past_key_values):
                    k_dst, v_dst = new_cache[li]
                    k_dst.index_copy_(0, sel, k_src)
                    v_dst.index_copy_(0, sel, v_src)

                kv_len.index_add_(0, sel, torch.ones_like(sel, device=device, dtype=kv_len.dtype))

            clear_structured_action(model)
            past_kv = new_cache

            # Loss and stats
            nll_t = F.cross_entropy(logits_step, labels_t, reduction="none")
            total_nll += nll_t.sum().item()
            total_tok += B

            # Budget accounting
            flat_idx = a_star * (P * Q) + a_p * Q + a_q
            action_hist.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))

            eff_tok += int(has_old.sum().item())
            total_keep_all += kappa_now.sum().item()
            total_keep_eff += (kappa_now * has_old).sum().item()

            cum_eff  = cum_eff  + has_old
            cum_keep = cum_keep + has_old * kappa_now

            struct_gate = has_old if not struct_on_non_eff else torch.ones_like(has_old)
            cum_pru_steps = cum_pru_steps + struct_gate
            cum_q_steps   = cum_q_steps   + struct_gate
            cum_pru_val   = cum_pru_val   + struct_gate * prune_now
            cum_q_val     = cum_q_val     + struct_gate * qratio_now
            # global accumulators for final averages
            total_prune_eff  += (prune_now  * struct_gate).sum().item()
            total_qratio_eff += (qratio_now * struct_gate).sum().item()

            # Update LRM signal for NEXT step
            prev_lrm = lrm_step

    ppl = math.exp(total_nll / max(1, total_tok))
    avg_keep_all = total_keep_all / max(1, total_tok)
    avg_keep_eff = (total_keep_eff / max(1, eff_tok)) if eff_tok > 0 else 0.0
    action_probs = (action_hist / action_hist.sum().clamp_min(1)).tolist()
    denom_struct = (eff_tok if not struct_on_non_eff else total_tok)
    avg_prune_keep  = total_prune_eff  / max(1, denom_struct)
    avg_quant_ratio = total_qratio_eff / max(1, denom_struct)

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
        "mix_lo_k": ks_vals[lo_s].item(),
        "mix_hi_k": ks_vals[hi_s].item(),
        "mix_p_hi": float(p_hi_target),
    }


@torch.no_grad()
def evaluate_qnr_quant_matched_keep(
    cfg,
    model,
    dl,
    quant_choices: Tuple[str, ...],
    target_quant_ratio: float,           # ratio in [0,1], e.g. 16->1.0, 8->0.5
    context_len: int,
    rollout_len: int,
    device: str,
    Ts: int = 4,
    Tw: int = 2,
    struct_on_non_eff: bool = False,     # if True, apply quant on all steps; else only after thr
):
    """
    % https://chatgpt.com/c/6903a4a1-89a8-832b-8ed3-4001187e4e4a
    Q‑2. Quantization Noise Ratio (QNR) – "SNR guardrail from activation stats".

    Causal baseline that *only* switches quantization bits. Token keep/prune are fixed (dense).
    Decision at step t uses *previous-step* activation stats to pick bits for step t.

    Mechanics:
      - Capture the input to the *last block* MLP.down_proj at each decode step (this is the
        post-gate hidden vector `hidden = silu(gate) * up` as used by your runtime controls).
      - For each candidate bits b in quant_choices, estimate QNR_b = (s_b^2/12) / Var(hidden),
        where s_b = amax(hidden)/qmax and qmax = 2^(b-1)-1.
      - Find the *smallest* b with QNR_b <= tau (cfg.qnr_tau, default 0.02) -> convert to a
        monotone signal in [0,1] (low -> fewer bits OK, high -> need more bits).
      - Use your matched-target chooser with feasibility guardrails on the *quant ratio* axis,
        and bias hi/lo decisions via `gamma * (signal - 0.5)` (cfg.qnr_gamma, default 0.35).

    Returns:
      dict with ppl, avg_quant_ratio (over structured steps), tokens, tokens_effective,
      action histogram/probs over quant bit choices, and the mixing neighborhood for reference.
    """
    import math
    import re
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()
    m = getattr(model, "module", model)
    try:
        m_dtype = next(m.parameters()).dtype
    except StopIteration:
        m_dtype = torch.float32

    # ----------------- Parse & set up quant axis -----------------
    # Allow strings like "int8"/"8" or ints
    def _to_bits(qc: Tuple[str, ...]) -> List[int]:
        bits = []
        for s in qc:
            if isinstance(s, (int, float)):
                bits.append(int(s))
            else:
                found = re.findall(r"\d+", str(s))
                if not found:
                    raise ValueError(f"Cannot parse bits from quant_choice: {s}")
                bits.append(int(found[0]))
        bits = sorted(set(bits))
        return bits

    QBITS_list = _to_bits(quant_choices)
    if len(QBITS_list) == 0:
        raise ValueError("quant_choices must contain at least one option")
    QBITS = torch.tensor(QBITS_list, device=device, dtype=torch.int64)     # [Q]
    QRAT  = QBITS.to(torch.float32).clamp(min=1.0) / 16.0                   # [Q]
    Q = len(QBITS_list)

    # For non-structured steps we default to densest (largest bits)
    dense_q_idx = int(torch.argmax(QBITS).item())
    q_axis_min, q_axis_max = float(QRAT.min().item()), float(QRAT.max().item())

    # Report neighbors of the *target* (for reference; not used in decisions directly)
    def _mix_neighbors(vals: torch.Tensor, target: float):
        vs = vals.tolist()
        if target <= vs[0]:
            return vs[0], vs[0], 0.0
        if target >= vs[-1]:
            return vs[-1], vs[-1], 1.0
        lo = max(i for i, v in enumerate(vs) if v <= target)
        hi = min(i for i, v in enumerate(vs) if v >= target)
        lo_v, hi_v = vs[lo], vs[hi]
        p_hi = 0.0 if hi_v == lo_v else (target - lo_v) / (hi_v - lo_v)
        return lo_v, hi_v, float(p_hi)

    q_lo_v, q_hi_v, q_p_hi = _mix_neighbors(QRAT, float(target_quant_ratio))

    # ----------------- QNR hyper-parameters -----------------
    qnr_tau   = float(getattr(cfg, "qnr_tau", 0.02))     # threshold for acceptable QNR
    qnr_gamma = float(getattr(cfg, "qnr_gamma", 0.35))   # bias strength toward larger bits

    # ----------------- Hook to capture hidden before down_proj -----------------
    last_down_proj = None
    try:
        # Primary: HF Llama-style path
        last_down_proj = m.model.layers[-1].mlp.down_proj
    except Exception:
        last_down_proj = None
    if last_down_proj is None:
        # Robust fallback: search by name anywhere in the module tree
        for name, mod in m.named_modules():
            if name.endswith("down_proj"):
                last_down_proj = mod
    if last_down_proj is None:
        # Hard fail instead of silently continuing with a neutral controller
        raise RuntimeError(
            "evaluate_qnr_quant_matched_keep: could not locate the last MLP 'down_proj'. "
            "This baseline expects that module to exist for activation capture."
        )

    act_hidden_buf = {"tensor": None, "sel": None}  # filled per subgroup call
    def _down_hook(mod, inputs, output):
        # inputs[0] can be [B_sub, 1, D_ff] or [B_sub, D_ff] depending on MLP impl.
        if act_hidden_buf["sel"] is None or act_hidden_buf["tensor"] is None:
            return
        x = inputs[0]
        # Normalize to [B_sub, D_ff] (take the last token when seq_len==1)
        if x.dim() == 3:
            x = x[:, -1, :]
        elif x.dim() == 2:
            pass
        else:
            # Fallback: flatten trailing dims
            x = x.reshape(x.size(0), -1)
        # Store using the buffer's dtype (float32) for stable QNR stats
        act_hidden_buf["tensor"].index_copy_(0, act_hidden_buf["sel"],
                                             x.detach().to(act_hidden_buf["tensor"].dtype))

    hook_handle = None
    if last_down_proj is not None:
        hook_handle = last_down_proj.register_forward_hook(_down_hook)

    # ----------------- Utilities: per-seq QNR→signal & chooser with guardrails -----------------
    def _min_bits_index_from_qnr(amax: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        amax, var: [B_struct] (float32)
        returns: idx_smallest_acceptable_bits in [0, Q-1] per sequence
        """
        # Compute QNR for each candidate bits: broadcast over B×Q
        qmax = ((2 ** (QBITS.to(torch.float32) - 1)) - 1.0).clamp_min(1.0)     # [Q]
        s = (amax.unsqueeze(-1) / qmax.unsqueeze(0))                           # [B, Q]
        q_mse = (s * s) / 12.0                                                 # [B, Q]
        qnr = q_mse / var.unsqueeze(-1).clamp_min(1e-8)                        # [B, Q]
        ok = (qnr <= qnr_tau)
        # For sequences where none are ok, choose last (densest)
        # For sequences where some are ok, choose first index where ok
        first_ok = ok.float().argmax(dim=-1)                                    # [B]
        has_ok = ok.any(dim=-1)
        idx = torch.where(has_ok, first_ok, torch.full_like(first_ok, Q-1))
        return idx

    def _signal_from_prev_hidden(h_prev: torch.Tensor) -> torch.Tensor:
        """
        h_prev: [B, D] float32 hidden captured at step t (input to down_proj)
        returns: [B] signal in [0,1], monotone with required bits (low->few bits OK, high->need more)
        """
        if h_prev is None:
            return None
        amax = h_prev.abs().amax(dim=-1)                            # [B]
        var  = h_prev.var(dim=-1, unbiased=False).clamp_min(1e-8)   # [B]
        idx_min = _min_bits_index_from_qnr(amax, var)               # [B] in [0,Q-1]
        if Q == 1:
            # Neutral when there is only one bits choice
            sig = torch.full_like(amax, 0.5)
        else:
            sig = idx_min.to(torch.float32) / float(Q - 1)          # normalize to [0,1]
        return sig.clamp(0.0, 1.0)

    # Chooser on QRAT axis with feasibility guardrails (vectorized over sequences)
    q_vals_sorted, q_map_sorted_to_orig = torch.sort(QRAT)           # [Q], [Q]
    q_min, q_max = float(q_vals_sorted[0].item()), float(q_vals_sorted[-1].item())

    def _choose_q_with_bias_and_guards(
        c_req: torch.Tensor,          # [B_struct] desired ratio for this step
        signal01: torch.Tensor,       # [B_struct] in [0,1], higher -> more bits
        cum_q_steps: torch.Tensor,    # [B_struct]
        cum_q_val: torch.Tensor,      # [B_struct]
        R: torch.Tensor,              # [B_struct] remaining structured steps incl. current
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          chosen_orig_idx: [B_struct] indices into original QBITS/QRAT axis
          chosen_qrat:     [B_struct] chosen ratio (bits/16)
        """
        # clamp c_req to axis range
        c = c_req.clamp(q_vals_sorted[0], q_vals_sorted[-1])         # [B]
        hi_sorted = torch.searchsorted(q_vals_sorted, c, right=False).clamp(0, Q-1)
        lo_sorted = (hi_sorted - 1).clamp(0, Q-1)

        lo_v = q_vals_sorted[lo_sorted]
        hi_v = q_vals_sorted[hi_sorted]
        denom = (hi_v - lo_v).clamp_min(1e-8)
        p_hi_base = torch.where((hi_v - lo_v) > 1e-8, (c - lo_v) / denom, torch.ones_like(c))

        # bias by QNR signal
        delta = qnr_gamma * (signal01 - 0.5)
        p_hi_mod = (p_hi_base + delta).clamp(0.0, 1.0)

        # deterministic choice (no randomness) via threshold
        choose_hi = (p_hi_mod >= 0.5) & (hi_sorted != lo_sorted)
        chosen_sorted = torch.where(choose_hi, hi_sorted, lo_sorted) # [B]
        chosen_q = q_vals_sorted[chosen_sorted]

        # feasibility guardrails: ensure target remains attainable
        R_post = (R - 1.0).clamp_min(0.0)
        target_total = float(target_quant_ratio) * (cum_q_steps + 1.0 + R_post)

        allowed_min = (target_total - cum_q_val - q_max * R_post).clamp(q_min, q_max)
        allowed_max = (target_total - cum_q_val - q_min * R_post).clamp(q_min, q_max)

        lo_feas = torch.searchsorted(q_vals_sorted, allowed_min, right=False)
        hi_feas = torch.searchsorted(q_vals_sorted, allowed_max, right=True) - 1
        lo_feas = torch.minimum(lo_feas, hi_feas).clamp(0, Q-1)
        hi_feas = torch.maximum(lo_feas, hi_feas).clamp(0, Q-1)

        chosen_sorted = torch.maximum(chosen_sorted, lo_feas)
        chosen_sorted = torch.minimum(chosen_sorted, hi_feas)
        chosen_q = q_vals_sorted[chosen_sorted]

        chosen_orig = q_map_sorted_to_orig[chosen_sorted]
        return chosen_orig, chosen_q

    # ----------------- Accumulators -----------------
    total_nll = 0.0
    total_tok = 0
    eff_tok   = 0

    # average quant ratio (over structured steps by default)
    total_qratio_eff = 0.0

    # histogram over Q only (quant-only evaluator)
    action_hist = torch.zeros(Q, device=device)

    enable_structured_controls(model)
    clear_structured_action(model)

    thr = Ts + Tw + 1

    for batch in tqdm(dl, desc="eval QNR quant-only matched keep"):
        batch = batch.to(device)
        B, _ = batch.shape

        prefill_ids = batch[:, :context_len]
        step_inputs = batch[:, context_len : context_len + rollout_len]
        step_labels = batch[:, context_len + 1 : context_len + rollout_len + 1]

        # Build cache via dense prefill
        out = model(input_ids=prefill_ids, use_cache=True, return_dict=True)
        past_kv = detach_cache_to_tuple(out.past_key_values)
        kv_len = torch.full((B,), context_len + 1, device=device, dtype=torch.long)

        # per-sequence residual trackers for quant axis
        cum_q_steps = torch.zeros(B, device=device, dtype=torch.float32)
        cum_q_val   = torch.zeros(B, device=device, dtype=torch.float32)

        # previous-step signal; neutral 0.5 at t=0
        prev_signal = torch.full((B,), 0.5, device=device, dtype=torch.float32)

        for t in range(rollout_len):
            cur = step_inputs[:, t]                   # [B]
            labels_t = step_labels[:, t]             # [B]

            kv_before = kv_len.clone()
            eff_mask = (kv_before > thr)             # [B] bool
            has_old  = eff_mask.float()

            # Where to apply quant this step?
            struct_mask = eff_mask if not struct_on_non_eff else torch.ones_like(eff_mask)
            idx_struct = torch.nonzero(struct_mask, as_tuple=False).squeeze(-1)
            S_struct = int(idx_struct.numel())

            # Default action: densest bits on non-structured steps
            a_q = torch.full((B,), dense_q_idx, device=device, dtype=torch.long)

            if S_struct > 0:
                # Remaining structured steps per sequence (match κ logic when struct_on_non_eff=False)
                if not struct_on_non_eff:
                    T_rem = rollout_len - t
                    neff_rem = (thr - kv_before + 1).clamp_min(0)
                    R_vec_full = (T_rem - neff_rem).clamp_min(0).to(torch.float32)
                else:
                    R_vec_full = torch.full((B,), float(rollout_len - t), device=device, dtype=torch.float32)
                R_vec = R_vec_full.index_select(0, idx_struct)

                # Per-seq required quant ratio to finish on budget
                q_req = (float(target_quant_ratio) * (cum_q_steps.index_select(0, idx_struct) + R_vec) -
                         cum_q_val.index_select(0, idx_struct)) / R_vec.clamp_min(1.0)
                q_req = q_req.clamp(q_min, q_max)

                # Choose action using prev-step QNR-derived signal
                chosen_orig, chosen_q = _choose_q_with_bias_and_guards(
                    c_req=q_req,
                    signal01=prev_signal.index_select(0, idx_struct),
                    cum_q_steps=cum_q_steps.index_select(0, idx_struct),
                    cum_q_val=cum_q_val.index_select(0, idx_struct),
                    R=R_vec,
                )
                a_q.index_copy_(0, idx_struct, chosen_orig)

            # Bits & ratios for this step
            qbits_now  = QBITS[a_q]                  # [B]
            qratio_now = QRAT[a_q]                   # [B]

            # Histogram over bits
            action_hist.index_add_(0, a_q, torch.ones_like(a_q, dtype=torch.float32))

            # --- Decode one step, grouped by bits ---
            logits_step = None
            new_cache = None
            pos_ids_all = (kv_len - 1).clamp_min(0).unsqueeze(1)

            # Prepare activation capture buffer for this step (allocate before any subgroup)
            act_hidden_buf["sel"] = None
            if last_down_proj is not None:
                D_hidden = last_down_proj.in_features
                act_hidden_buf["tensor"] = torch.empty((B, D_hidden), device=device, dtype=torch.float32)
            else:
                act_hidden_buf["tensor"] = None

            # Group by bits
            uniq_bits, inv = torch.unique(qbits_now, return_inverse=True)
            for g, q_val in enumerate(uniq_bits.tolist()):
                sel = (inv == g).nonzero(as_tuple=False).squeeze(-1)
                if sel.numel() == 0:
                    continue

                # Apply subgroup structural control (prune_keep fixed at 1.0)
                set_structured_action(model, prune_keep=1.0, quant_bits=int(q_val))

                cur_g     = cur.index_select(0, sel)
                pos_ids_g = pos_ids_all.index_select(0, sel)

                # enable capture for this subgroup
                act_hidden_buf["sel"] = sel
                # Will lazily allocate [B, D] act buffer on first call
                # To do so, we need D; we discover it inside the hook from inputs[0] shape.

                cache_g = select_cache_by_indices(past_kv, sel)
                out_g = model(
                    input_ids=cur_g.unsqueeze(1),
                    use_cache=True,
                    past_key_values=cache_g,
                    position_ids=pos_ids_g,
                    return_dict=True,
                )

                # merge logits
                if logits_step is None:
                    logits_step = torch.empty((B, out_g.logits.size(-1)),
                                              device=device, dtype=out_g.logits.dtype)
                logits_step.index_copy_(0, sel, out_g.logits[:, -1, :])

                # merge subgroup caches (already L+1) into expanded destination
                if new_cache is None:
                    # create empty container with expanded batch dim
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
                    new_cache = _init_cache_container_like(out_g.past_key_values, B)

                for li, (k_src, v_src) in enumerate(out_g.past_key_values):
                    k_dst, v_dst = new_cache[li]
                    k_dst.index_copy_(0, sel, k_src)
                    v_dst.index_copy_(0, sel, v_src)

                kv_len.index_add_(0, sel, torch.ones_like(sel, device=device, dtype=kv_len.dtype))

            # switch off capture target for safety
            act_hidden_buf["sel"] = None

            past_kv = new_cache
            clear_structured_action(model)

            # Loss & stats
            nll_t = F.cross_entropy(logits_step, labels_t, reduction="none")
            total_nll += nll_t.sum().item()
            total_tok += B

            eff_tok += int(has_old.sum().item())
            # report avg quant ratio over structured steps (or all if struct_on_non_eff)
            gate = has_old if not struct_on_non_eff else torch.ones_like(has_old)
            total_qratio_eff += (qratio_now * gate).sum().item()

            # advance per-seq residual trackers
            s_gate = (struct_mask).float()
            cum_q_steps = cum_q_steps + s_gate
            cum_q_val   = cum_q_val   + s_gate * qratio_now

            # ---- Compute prev_signal for NEXT step from captured hidden at THIS step ----
            if last_down_proj is not None and act_hidden_buf["tensor"] is not None:
                prev_signal_new = _signal_from_prev_hidden(act_hidden_buf["tensor"])
                if prev_signal_new is not None:
                    prev_signal = prev_signal_new.to(device=device, dtype=torch.float32)
            # else keep previous prev_signal

    # cleanup
    clear_structured_action(model)
    if hook_handle is not None:
        hook_handle.remove()

    ppl = math.exp(total_nll / max(1, total_tok))
    denom_struct = (eff_tok if not struct_on_non_eff else total_tok)
    avg_quant_ratio = (total_qratio_eff / max(1, denom_struct))

    action_probs = (action_hist / action_hist.sum().clamp_min(1)).tolist()
    return {
        "ppl": ppl,
        "avg_quant_ratio": avg_quant_ratio,
        "tokens": total_tok,
        "tokens_effective": eff_tok,
        "action_hist": action_hist.tolist(),
        "action_probs": action_probs,
        "mix_quant": {"lo": q_lo_v, "hi": q_hi_v, "p_hi": q_p_hi},
        "qnr_tau": qnr_tau,
        "qnr_gamma": qnr_gamma,
    }



@torch.no_grad()
def evaluate_dynr_quant_matched_keep(
    cfg,
    model,
    dl,
    Ts: int,
    Tw: int,
    quant_choices: Tuple[str, ...],     # e.g., ("4","8","16") or ("4", "16")
    target_quant_ratio: float,          # desired average bits/16 over (effective|all) steps
    context_len: int,
    rollout_len: int,
    device: str,
    struct_on_non_eff: bool = False,    # if True, apply quant on non-effective steps too
    keep_fracs: Tuple[float, ...] = (1.0,),  # default dense attention; kept for compatibility
):
    """
    Dynamic-Range quantization controller (causal):

    - At each step t, we use the PREVIOUS step's last-layer hidden vector h_{t-1}
      to compute a simple dynamic-range + tail-fraction signal in [0,1].
    - We bias the discrete quant bits choice for step t around the per-sequence
      required ratio to hit the global target, with feasibility guardrails.
    - Token sparsity and pruning are assumed to be singletons (i.e., dense).
      Only quantization is switched.

    Returns a dict with ppl, avg_quant_ratio, action histogram, etc.
    """

    # -------------------- Helpers --------------------

    def _parse_quant_axis(qchoices: Tuple[str, ...]) -> Tensor:
        """Turn a tuple of strings into a sorted, unique tensor of bits (ints)."""
        qbits = sorted({int(str(x).split("q")[-1]) for x in qchoices})
        return torch.tensor(qbits, device=device, dtype=torch.int64)  # [Q]

    def _dynr_signal_from_hidden(h: Tensor) -> Tensor:
        """
        h: [B, D] last-layer hidden at the current step (we use it to set the *next* step bits).
        Signal in [0,1] = 0.5 * normalized dynamic range + 0.5 * tail fraction.
        """
        # Config knobs with safe defaults
        tau = float(getattr(cfg, "dynr_tau", 3.0))        # tail threshold in std units
        rclip = float(getattr(cfg, "dynr_rclip", 12.0))   # clip dynamic-range ratio for normalization

        # Promote to float32 for the computation (safer std/max)
        h32 = h.to(torch.float32)
        std = h32.std(dim=-1, unbiased=False).clamp_min(1e-6)         # [B], fp32
        maxabs = h32.abs().amax(dim=-1)                               # [B], fp32
        r = (maxabs / std)                                            # [B], fp32
        r_norm = (r / rclip).clamp(0.0, 1.0)                          # [B], fp32

        tail = (h32.abs() > (tau * std).unsqueeze(-1)).float().mean(dim=-1)  # [B] in [0,1], fp32

        # signal = 0.5 * (r_norm + tail)
        signal = r_norm
        return signal.clamp(0.0, 1.0)

    def _choose_quant_with_bias_and_guards(
        qr_sorted: Tensor,                 # [Q] sorted QRAT = bits/16 (ascending)
        map_sorted_to_orig: Tensor,        # [Q] map back to original indices
        q_min: float, q_max: float,
        c_req: Tensor,                     # [B_sel] required per-seq quant ratio for target tracking
        signal01: Tensor,                  # [B_sel] dynamic-range signal in [0,1]
        cum_q_steps: Tensor,               # [B_sel]
        cum_q_val: Tensor,                 # [B_sel]
        R: Tensor                          # [B_sel] remaining structured steps including current
    ):
        """
        Bias choice between the two neighbors of c_req using signal01 and keep feasibility.
        Returns (chosen_orig_idx, chosen_qratio)
        """
        # Controller strength (like EMC gamma)
        gamma = float(getattr(cfg, "dynr_gamma", 0.35))

        Q = qr_sorted.numel()
        c = c_req.clamp(qr_sorted[0], qr_sorted[-1])                  # [B_sel]
        hi_sorted = torch.searchsorted(qr_sorted, c, right=False).clamp(0, Q-1)
        lo_sorted = (hi_sorted - 1).clamp(0, Q-1)

        lo_v = qr_sorted[lo_sorted]
        hi_v = qr_sorted[hi_sorted]
        denom = (hi_v - lo_v).clamp_min(1e-8)
        p_hi_base = torch.where((hi_v - lo_v) > 1e-8, (c - lo_v) / denom, torch.ones_like(c))

        # Bias with dynamic-range signal (delta around 0.5)
        delta = gamma * (signal01 - 0.5)
        p_hi_mod = (p_hi_base + delta).clamp(0.0, 1.0)

        # Deterministic pick (>=0.5 -> hi) before feasibility
        choose_hi = (p_hi_mod >= 0.5) & (hi_sorted != lo_sorted)
        chosen_sorted = torch.where(choose_hi, hi_sorted, lo_sorted)  # [B_sel]
        chosen_v = qr_sorted[chosen_sorted]

        # Feasibility guardrails to ensure we can still reach the global target
        R_post = (R - 1.0).clamp_min(0.0)
        target_total = float(target_quant_ratio) * (cum_q_steps + 1.0 + R_post)

        allowed_min = (target_total - cum_q_val - q_max * R_post).clamp(q_min, q_max)
        allowed_max = (target_total - cum_q_val - q_min * R_post).clamp(q_min, q_max)

        lo_feas = torch.searchsorted(qr_sorted, allowed_min, right=False)
        hi_feas = torch.searchsorted(qr_sorted, allowed_max, right=True) - 1
        lo_feas = torch.minimum(lo_feas, hi_feas).clamp(0, Q-1)
        hi_feas = torch.maximum(lo_feas, hi_feas).clamp(0, Q-1)

        chosen_sorted = torch.maximum(chosen_sorted, lo_feas)
        chosen_sorted = torch.minimum(chosen_sorted, hi_feas)
        chosen_v = qr_sorted[chosen_sorted]

        chosen_orig = map_sorted_to_orig[chosen_sorted]
        return chosen_orig, chosen_v

    # -------------------- Setup --------------------

    model.eval()
    m = getattr(model, "module", model)
    try:
        m_dtype = next(m.parameters()).dtype
    except StopIteration:
        m_dtype = torch.float32

    # Quant axis (discrete bits and ratios)
    QBITS = _parse_quant_axis(quant_choices)                     # [Q] ints
    QRAT = (QBITS.to(torch.float32).clamp(min=1.0)) / 16.0       # [Q] in (0,1]
    # Sorted view for searchsorted
    pairs = sorted([(float(v), i) for i, v in enumerate(QRAT.tolist())], key=lambda x: x[0])
    qr_sorted = torch.tensor([v for v, _ in pairs], device=device, dtype=torch.float32)  # [Q]
    q_map_sorted_to_orig = torch.tensor([i for _, i in pairs], device=device, dtype=torch.long)

    Q = int(QBITS.numel())
    q_min, q_max = float(qr_sorted[0].item()), float(qr_sorted[-1].item())
    dense_q_idx = int(torch.argmax(QBITS).item())   # expect 16-bit present

    # Token sparsity is assumed dense (keep=1.0), but we keep the effective-step logic
    thr = Ts + Tw + 1
    KEEP_ONE = torch.ones(1, device=device, dtype=torch.float32)

    # Cumulative stats & trackers
    total_nll = 0.0
    total_tok = 0
    eff_tok   = 0
    total_qratio_eff = 0.0

    # Per-sequence cumulative quant tracking is reset per batch (initialized inside the dataloader loop).
    prev_dynr_signal = None
    cum_q_steps = None
    cum_q_val   = None

    # Histograms
    action_hist_q = torch.zeros(Q, device=device)   # just quant axis here

    # Enable runtime quant controls
    enable_structured_controls(model)
    clear_structured_action(model)

    # If relevancy mask machinery exists in your stack, ensure it's off
    if str(getattr(cfg, "sparsity_criteria", "recency")) == "relevancy":
        clear_relevancy_keep(model)

    # -------------------- Main loop --------------------

    for batch in tqdm(dl, desc="eval dynr quant matched keep"):
        batch = batch.to(device)
        B, _ = batch.shape

        prefill_ids = batch[:, :context_len]
        step_inputs = batch[:, context_len : context_len + rollout_len]
        step_labels = batch[:, context_len + 1 : context_len + rollout_len + 1]

        # Reset per-batch trackers (do NOT persist across unrelated sequences)
        prev_dynr_signal = torch.full((B,), 0.5, device=device, dtype=torch.float32)
        cum_q_steps      = torch.zeros(B, device=device, dtype=torch.float32)
        cum_q_val        = torch.zeros(B, device=device, dtype=torch.float32)

        # Dense prefill to build KV
        out = model(input_ids=prefill_ids, use_cache=True, return_dict=True)
        past_kv = detach_cache_to_tuple(out.past_key_values)
        kv_len = torch.full((B,), context_len + 1, device=device, dtype=torch.long)

        for t in range(rollout_len):
            cur = step_inputs[:, t]           # [B]
            labels_t = step_labels[:, t]      # [B]

            eff_mask = (kv_len > thr)         # [B] bool: effective tokens
            struct_mask = (eff_mask if not struct_on_non_eff else torch.ones_like(eff_mask))
            idx_struct = torch.nonzero(struct_mask, as_tuple=False).squeeze(-1)
            S_struct = int(idx_struct.numel())

            # Default: densest bits for non-structured steps
            a_q = torch.full((B,), dense_q_idx, device=device, dtype=torch.long)

            if S_struct > 0:
                # Remaining structured steps (per sequence)
                if not struct_on_non_eff:
                    T_rem = rollout_len - t
                    neff_rem = (thr - kv_len + 1).clamp_min(0)
                    R_vec_full = (T_rem - neff_rem).clamp_min(0).to(torch.float32)
                else:
                    R_vec_full = torch.full((B,), float(rollout_len - t), device=device, dtype=torch.float32)

                R_vec = R_vec_full.index_select(0, idx_struct)
                # Per-sequence required ratio to stay on target
                # c_req = (target * (cum + R) - sum) / max(R,1)
                c_req_q = (float(target_quant_ratio) * (cum_q_steps.index_select(0, idx_struct) + R_vec)
                           - cum_q_val.index_select(0, idx_struct)) / R_vec.clamp_min(1.0)
                c_req_q = c_req_q.clamp(q_min, q_max)

                # Choose q using previous-step dynamic-range signal
                chosen_orig, chosen_v = _choose_quant_with_bias_and_guards(
                    qr_sorted=qr_sorted,
                    map_sorted_to_orig=q_map_sorted_to_orig,
                    q_min=q_min, q_max=q_max,
                    c_req=c_req_q,
                    signal01=prev_dynr_signal.index_select(0, idx_struct),
                    cum_q_steps=cum_q_steps.index_select(0, idx_struct),
                    cum_q_val=cum_q_val.index_select(0, idx_struct),
                    R=R_vec,
                )
                a_q[idx_struct] = chosen_orig

            # Apply quant per subgroup
            qbits_now = QBITS[a_q]                       # [B]
            qratio_now = (qbits_now.to(torch.float32).clamp(min=1.0) / 16.0)  # [B]

            # Histogram
            action_hist_q.index_add_(0, a_q, torch.ones_like(a_q, dtype=torch.float32))

            # We run attention with keep=1.0 (dense); build mask or pass None
            kappa_now = KEEP_ONE.expand(B)               # [B], all ones
            pq = qbits_now.to(torch.float32).unsqueeze(-1)  # [B,1] only quant groups

            uniq_q, inv = torch.unique(qbits_now, sorted=True, return_inverse=True)
            logits_step = None
            new_cache = None
            pos_ids_all = (kv_len - 1).clamp_min(0).unsqueeze(1)
            dynr_signal_this = torch.empty((B,), device=device, dtype=torch.float32)

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

            for g, q_val in enumerate(uniq_q.tolist()):
                sel = (inv == g).nonzero(as_tuple=False).squeeze(-1)
                if sel.numel() == 0:
                    continue

                set_structured_action(model, prune_keep=1.0, quant_bits=int(q_val))

                cur_g     = cur.index_select(0, sel)
                pos_ids_g = pos_ids_all.index_select(0, sel)
                kappa_g   = kappa_now.index_select(0, sel)

                # Build a dense attention bias for completeness (keep=1.0)
                bias_g = build_sparse_attention_bias(
                    model=model,
                    past_kv_lens=kv_len.index_select(0, sel),
                    keep_fracs=kappa_g,
                    Ts=Ts, Tw=Tw,
                    device=device, dtype=m_dtype,
                    criteria="recency",  # dense anyway; no sparsity
                    tier=getattr(cfg, "relevancy_tier", "per_head"),
                )
                cache_g = select_cache_by_indices(past_kv, sel)
                out_g = model(
                    input_ids=cur_g.unsqueeze(1),
                    use_cache=True,
                    past_key_values=cache_g,
                    position_ids=pos_ids_g,
                    attention_mask=bias_g,
                    output_hidden_states=True,            # <-- get last hidden for dynamic range
                    return_dict=True,
                )
                # Merge logits
                if logits_step is None:
                    logits_step = torch.empty((B, out_g.logits.size(-1)),
                                              device=device, dtype=out_g.logits.dtype)
                logits_step.index_copy_(0, sel, out_g.logits[:, -1, :])

                # Compute per-seq dynamic-range signal for NEXT step (from last hidden)
                h_last = out_g.hidden_states[-1][:, -1, :]      # [sel, D]
                sig_g = _dynr_signal_from_hidden(h_last)        # [sel]
                dynr_signal_this.index_copy_(0, sel, sig_g)

                # Merge KV
                if new_cache is None:
                    new_cache = _init_cache_container_like(out_g.past_key_values, B)
                for li, (k_src, v_src) in enumerate(out_g.past_key_values):
                    k_dst, v_dst = new_cache[li]
                    k_dst.index_copy_(0, sel, k_src)
                    v_dst.index_copy_(0, sel, v_src)

                kv_len.index_add_(0, sel, torch.ones_like(sel, device=device, dtype=kv_len.dtype))

            # Clear for next step
            clear_structured_action(model)
            past_kv = new_cache

            # Loss & stats
            nll_t = F.cross_entropy(logits_step, labels_t, reduction="none")
            total_nll += nll_t.sum().item()
            total_tok += B

            has_old = eff_mask.float()
            eff_tok += int(has_old.sum().item())

            gate = (eff_mask if not struct_on_non_eff else torch.ones_like(eff_mask)).float()
            total_qratio_eff += (qratio_now * gate).sum().item()

            # Update per-sequence cumulative quant tracking
            S_this = int(gate.sum().item())
            if S_this > 0:
                cum_q_steps = cum_q_steps + gate
                cum_q_val   = cum_q_val   + gate * qratio_now

            # Prepare signal for next step
            prev_dynr_signal = dynr_signal_this

    # -------------------- Final metrics --------------------

    ppl = math.exp(total_nll / max(1, total_tok))
    denom_struct = (eff_tok if not struct_on_non_eff else total_tok)
    avg_quant_ratio = (total_qratio_eff / max(1, denom_struct))
    action_probs_q = (action_hist_q / action_hist_q.sum().clamp_min(1)).tolist()

    return {
        "ppl": ppl,
        "avg_quant_ratio": avg_quant_ratio,
        "action_hist_quant": action_hist_q.tolist(),
        "action_probs_quant": action_probs_q,
        "tokens": total_tok,
        "tokens_effective": eff_tok,
        "quant_axis_bits": QBITS.tolist(),
    }

@torch.no_grad()
def evaluate_ecov_prune_matched_keep(
    cfg,
    model,
    dl,
    Ts: int,
    Tw: int,
    keep_fracs: Tuple[float, ...],        # assume len==1 (κ fixed)
    prune_choices: Tuple[str, ...],       # structured pruning axis (the only active axis)
    quant_choices: Tuple[str, ...],       # assume len==1 (bits fixed, e.g., 16)
    target_prune_keep: float,             # target average keep over (effective or all) steps
    context_len: int,
    rollout_len: int,
    device: str,
    coverage_target: float = 0.90,        # ECov threshold on |hidden| energy
    struct_on_non_eff: bool = False,      # prune also on non-effective steps if True
):
    """
    https://chatgpt.com/c/6903a57b-3770-832e-93e8-26c632cd2910
    ECov pruning baseline (causal):

    - For each sequence, choose the current step's prune keep ρ_t from the ECov decision
      computed on the *previous* step's post-gate hidden (silu(gate)*up) at the last decoder
      block, i.e., minimal ρ whose energy coverage ≥ coverage_target.

    - Use a batch-wide residual controller to *raise* some sequences to higher keeps as needed
      to steer the running average toward target_prune_keep, *without lowering below ECov*.

    - No lookahead, no dense teacher. Token-sparsity and quantization are fixed.
    """
    model.eval()
    m = getattr(model, "module", model)

    # ----- Resolve dtypes -----
    try:
        m_dtype = next(m.parameters()).dtype
    except StopIteration:
        m_dtype = torch.float32

    # ----- Build action spec (we only really need PRUNE here) -----
    spec = build_action_spec(
        keep_fracs=keep_fracs,
        prune_choices=prune_choices,
        quant_choices=quant_choices,
    )
    # κ, bits are fixed; pruning axis is the only active axis
    KEEP  = torch.tensor(keep_fracs, device=device, dtype=torch.float32)
    prune_axis = _unique_float_axis(spec.prune_keep)
    quant_axis = _unique_int_axis(spec.q_bits)

    assert KEEP.numel() == 1,  "ECov baseline expects a single keep_frac (no token-sparsity switching)."
    # Hard requirement: ECov baseline assumes dense tokens
    assert abs(float(KEEP.item()) - 1.0) < 1e-6, "ECov baseline expects κ=1.0 (dense tokens)."
    assert len(quant_axis) == 1, "ECov baseline expects a single quant choice (no quant switching)."

    PRUNE = torch.tensor(prune_axis, device=device, dtype=torch.float32)  # [P], not necessarily sorted
    QBITS = torch.tensor(quant_axis, device=device, dtype=torch.int64)    # [1]

    # Sort PRUNE to make ECov selection easy; keep a map back to original indices
    p_vals_sorted, p_map_sorted_to_orig = torch.sort(PRUNE)               # ascending
    P = int(p_vals_sorted.numel())

    # Build inverse map: original -> sorted rank
    inv_sorted_rank = torch.empty_like(p_map_sorted_to_orig)
    inv_sorted_rank[p_map_sorted_to_orig] = torch.arange(P, device=device, dtype=torch.long)

    # Densest index (used for t=0 and for non-struct steps)
    dense_p_idx = int(torch.argmax(PRUNE).item())
    dense_q_idx = int(torch.argmax(QBITS).item())  # unused but passed to set_structured_action
    # Sticky controller: limit rank change per token (default ±1)
    max_rank_step = int(getattr(cfg, "prune_max_rank_delta", 1))
    # ----- Locate the last Llama MLP (we'll hook it to read its input and recompute post-gate hidden) -----
    from transformers.models.llama import modeling_llama as llama_mod
    last_mlp = None
    # Prefer the canonical path if available
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        for lyr in m.model.layers:
            if hasattr(lyr, "mlp"):
                last_mlp = lyr.mlp
        if last_mlp is None:
            # Fallback: scan modules
            for mod in m.modules():
                if isinstance(mod, llama_mod.LlamaMLP):
                    last_mlp = mod
    else:
        for mod in m.modules():
            if isinstance(mod, llama_mod.LlamaMLP):
                last_mlp = mod
    if last_mlp is None:
        raise RuntimeError("Could not locate last LlamaMLP module for ECov baseline.")

    # ----- Running stats -----
    thr = Ts + Tw + 1
    total_nll = 0.0
    total_tok = 0
    eff_tok   = 0
    total_prune_eff = 0.0

    # Composite histogram degenerates to P because K=Q=1, but we keep K*P*Q shape for consistency.
    K, Q = 1, 1
    action_hist = torch.zeros(K * P * Q, device=device)

    enable_structured_controls(model)
    if str(getattr(cfg, "sparsity_criteria", "recency")) == "relevancy":
        clear_relevancy_keep(model)
    clear_structured_action(model)  # teacher/prefill dense

    # Per-step ECov recommendation to apply at *current* step (computed from *previous* step)
    # Initialize with densest keep for the very first decode step.
    rec_idx_for_step = None  # LongTensor[B] filled after t=0 forward
    # We'll fill the recommendations for *next* step inside the hook.
    rec_idx_next = None

    # Shared state for the hook: which global indices are currently in this subgroup forward.
    current_sel = None

    # Hook to compute ECov recommendation for the NEXT step from the current forward:
    def _last_mlp_hook(module, inputs, output):
        nonlocal current_sel, rec_idx_next
        if current_sel is None or current_sel.numel() == 0:
            return
        # inputs[0]: [B_sel, seq_len(=1), hidden] or [B_sel, hidden]
        x = inputs[0]
        if x.dim() == 3:
            x = x[:, -1, :]  # [B_sel, H]
        # Recompute post-gate hidden = silu(gate)*up with *module's* weights
        gate = F.linear(x, module.gate_proj.weight, module.gate_proj.bias)
        up   = F.linear(x,   module.up_proj.weight,  module.up_proj.bias)
        hidden = F.silu(gate) * up                   # [B_sel, D_ff]

        # Energy coverage over |hidden|
        B_sel, D = hidden.shape
        v = hidden.abs().sort(dim=-1, descending=True).values                      # [B_sel, D]
        cumsum = v.cumsum(dim=-1) / v.sum(dim=-1, keepdim=True).clamp_min(1e-12)   # [B_sel, D]
        # coverage at each discrete keep
        idx_counts = (torch.ceil(p_vals_sorted * D).long().clamp(1, D)) - 1        # [P] indices
        cov = cumsum.index_select(-1, idx_counts)                                   # [B_sel, P]
        hit = (cov >= float(coverage_target)).to(torch.int64)                       # [B_sel, P]
        # choose minimal keep that hits coverage; if none hits, use densest (last)
        first_hit = torch.argmax(hit, dim=-1)                                       # [B_sel]
        any_hit = hit.any(dim=-1)
        chosen_sorted = torch.where(any_hit, first_hit, torch.full_like(first_hit, P-1))
        # map to original PRUNE indices
        chosen_orig = p_map_sorted_to_orig.index_select(0, chosen_sorted)           # [B_sel]
        rec_idx_next.index_copy_(0, current_sel, chosen_orig)

    # Register hook once
    hdl = last_mlp.register_forward_hook(_last_mlp_hook)

    try:
        for batch in tqdm(dl, desc="eval ECov prune matched"):
            batch = batch.to(device)
            B, _ = batch.shape

            prefill_ids  = batch[:, :context_len]
            step_inputs  = batch[:, context_len : context_len + rollout_len]
            step_labels  = batch[:, context_len + 1 : context_len + rollout_len + 1]

            # Dense prefill to build KV
            out = model(input_ids=prefill_ids, use_cache=True, return_dict=True)
            past_kv = detach_cache_to_tuple(out.past_key_values)
            kv_len = torch.full((B,), context_len + 1, device=device, dtype=torch.long)

            # initialize recommendations buffers
            rec_idx_for_step = torch.full((B,), dense_p_idx, device=device, dtype=torch.long) if rec_idx_for_step is None else rec_idx_for_step
            rec_idx_next = torch.full((B,), dense_p_idx, device=device, dtype=torch.long)
            # Per-sequence budget trackers (ECov v2: per-seq feasibility)
            cum_pru_steps = torch.zeros(B, device=device, dtype=torch.float32)
            cum_pru_val   = torch.zeros(B, device=device, dtype=torch.float32)
            # Sticky previous selection
            prev_p_idx = torch.full((B,), dense_p_idx, device=device, dtype=torch.long)
            for t in range(rollout_len):
                cur = step_inputs[:, t]             # [B]
                labels_t = step_labels[:, t]        # [B]

                kv_before = kv_len.clone()
                eff_mask = (kv_before > thr)        # [B] bool
                struct_mask = (eff_mask if not struct_on_non_eff else torch.ones_like(eff_mask))
                idx_struct = torch.nonzero(struct_mask, as_tuple=False).squeeze(-1)  # [S_struct]
                S_struct = int(idx_struct.numel())

                # Choose current step prune indices from ECov recs, then *raise* some to match residual target
                a_p = torch.full((B,), dense_p_idx, device=device, dtype=torch.long)  # default for non-struct
                if S_struct > 0:
                    base_idx = rec_idx_for_step.index_select(0, idx_struct)           # [S_struct]
                    # ---- Per-sequence feasibility band (like EMC/DCP) ----
                    if not struct_on_non_eff:
                        T_rem = rollout_len - t
                        neff_rem = (thr - kv_before + 1).clamp_min(0)
                        R_vec_full = (T_rem - neff_rem).clamp_min(0).to(torch.float32)
                    else:
                        R_vec_full = torch.full((B,), float(rollout_len - t), device=device, dtype=torch.float32)
                    Rv = R_vec_full.index_select(0, idx_struct)                       # [S_struct]
                    R_post = (Rv - 1.0).clamp_min(0.0)
                    target_total = float(target_prune_keep) * (cum_pru_steps.index_select(0, idx_struct) + 1.0 + R_post)
                    allowed_min = (target_total - cum_pru_val.index_select(0, idx_struct) - p_vals_sorted[-1] * R_post) \
                                  .clamp(p_vals_sorted[0], p_vals_sorted[-1])
                    allowed_max = (target_total - cum_pru_val.index_select(0, idx_struct) - p_vals_sorted[0]  * R_post) \
                                  .clamp(p_vals_sorted[0], p_vals_sorted[-1])
                    lo_feas = torch.searchsorted(p_vals_sorted, allowed_min, right=False)
                    hi_feas = torch.searchsorted(p_vals_sorted, allowed_max, right=True) - 1
                    lo_feas = torch.minimum(lo_feas, hi_feas).clamp(0, P-1)
                    hi_feas = torch.maximum(lo_feas, hi_feas).clamp(0, P-1)

                    # Base choice from ECov coverage threshold (can be *raised* by feasibility)
                    base_sorted = inv_sorted_rank.index_select(0, base_idx)
                    chosen_sorted = torch.minimum(torch.maximum(base_sorted, lo_feas), hi_feas)

                    # Sticky decision: limit rank change per token
                    prev_sorted = inv_sorted_rank.index_select(0, prev_p_idx.index_select(0, idx_struct))
                    delta = (chosen_sorted - prev_sorted).clamp(min=-max_rank_step, max=max_rank_step)
                    chosen_sorted = (prev_sorted + delta).clamp(0, P-1)

                    chosen_idx = p_map_sorted_to_orig.index_select(0, chosen_sorted)
                    a_p.index_copy_(0, idx_struct, chosen_idx)
                    prev_p_idx.index_copy_(0, idx_struct, chosen_idx)

                # Build subgroup runs (quant bits fixed)
                prune_now  = PRUNE.index_select(0, a_p)               # [B]
                qbits_now  = QBITS.index_select(0, torch.full_like(a_p, dense_q_idx))  # [B], effectively constant
                # hist (K=Q=1 so this is effectively counting prune choices)
                flat_idx = a_p  # since K=Q=1, a_k=0, a_q=0
                action_hist.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))

                # Group by prune value
                pq = prune_now.unsqueeze(-1).to(torch.float32)        # [B,1]
                uniq, inv = torch.unique(pq, dim=0, return_inverse=True)
                logits_step = None
                new_cache = None
                pos_ids_all = (kv_before - 1).clamp_min(0).unsqueeze(1)

                # For each subgroup, set prune keep, run 1 token, and fill ECov rec for *next* step via hook
                for g, (p_val,) in enumerate(uniq.tolist()):
                    sel = (inv == g).nonzero(as_tuple=False).squeeze(-1)
                    if sel.numel() == 0:
                        continue
                    # apply structured control
                    set_structured_action(model, float(p_val), int(QBITS[dense_q_idx].item()))

                    # mark which global indices this subgroup corresponds to for the hook
                    current_sel = sel

                    cur_g     = cur.index_select(0, sel)
                    pos_ids_g = pos_ids_all.index_select(0, sel)

                    # κ is fixed (dense), so we do not pass a custom attention bias
                    cache_g = select_cache_by_indices(past_kv, sel)
                    out_g = model(
                        input_ids=cur_g.unsqueeze(1),
                        use_cache=True,
                        past_key_values=cache_g,
                        position_ids=pos_ids_g,
                        return_dict=True,
                    )

                    # reset subgroup marker for hook
                    current_sel = None

                    if logits_step is None:
                        logits_step = torch.empty((B, out_g.logits.size(-1)),
                                                  device=device, dtype=out_g.logits.dtype)
                    logits_step.index_copy_(0, sel, out_g.logits[:, -1, :])

                    if new_cache is None:
                        # make an empty cache container with batch dim B
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
                        new_cache = _init_cache_container_like(out_g.past_key_values, B)

                    # merge subgroup caches into the full batch cache
                    for li, (k_src, v_src) in enumerate(out_g.past_key_values):
                        k_dst, v_dst = new_cache[li]
                        k_dst.index_copy_(0, sel, k_src)
                        v_dst.index_copy_(0, sel, v_src)

                    kv_len.index_add_(0, sel, torch.ones_like(sel, device=device, dtype=kv_len.dtype))

                # finalize for this step
                past_kv = new_cache
                clear_structured_action(model)

                # Loss and running stats
                nll_t = F.cross_entropy(logits_step, labels_t, reduction="none")
                total_nll += nll_t.sum().item()
                total_tok += B

                has_old = eff_mask.float()
                gate = (has_old if not struct_on_non_eff else torch.ones_like(has_old))
                eff_tok += int(has_old.sum().item())
                total_prune_eff += (prune_now * gate).sum().item()
                s_gate = struct_mask.float()
                cum_pru_steps = cum_pru_steps + s_gate
                cum_pru_val   = cum_pru_val   + s_gate * prune_now

                # Prepare ECov recommendations for the *next* step
                rec_idx_for_step = rec_idx_next.clone()
                rec_idx_next.zero_().fill_(dense_p_idx)

    finally:
        # remove hook
        try:
            hdl.remove()
        except Exception:
            pass
        clear_structured_action(model)

    # Final metrics
    ppl = math.exp(total_nll / max(1, total_tok))
    denom_struct = (eff_tok if not struct_on_non_eff else total_tok)
    avg_prune_keep = (total_prune_eff / max(1, denom_struct)) if denom_struct > 0 else 0.0
    action_probs = (action_hist / action_hist.sum().clamp_min(1)).tolist()

    return {
        "ppl": ppl,
        "avg_prune_keep": avg_prune_keep,
        "tokens": total_tok,
        "tokens_effective": eff_tok,
        "coverage_target": float(coverage_target),
        "action_hist": action_hist.tolist(),     # length = P (since K=Q=1)
        "action_probs": action_probs,
    }


@torch.no_grad()
def evaluate_dcp_prune_matched_keep(
    cfg,
    model,
    dl,
    Ts: int,
    Tw: int,
    keep_fracs: Tuple[float, ...],
    prune_choices: Tuple[str, ...],
    quant_choices: Tuple[str, ...],
    target_prune_keep: float,
    context_len: int,
    rollout_len: int,
    device: str,
    struct_on_non_eff: bool = False,
    cov_target: float = 0.90,       # coverage target for DCP -> minimal keep
    dcp_gamma: float = 0.35,        # bias strength toward hi/lo keep from DCP signal
):
    """
    # Explanation: https://chatgpt.com/c/690779c6-8c30-832b-948d-fec988f8f5bf
    # Discovery: https://chatgpt.com/c/6903a591-1f14-832a-9f23-3c364d5913cc
    Downstream Contribution Proxy (DCP) controller for *pruning only*, with budget matching.

    Assumptions:
      - Token-sparsity (κ) and Quantization (bits) each have a single choice (i.e., fixed).
      - Only pruning keep (ρ) is switched per step & per sequence.
      - Decisions for step t use signals computed at step t-1 (causal, one-step lag).
      - Matching target_prune_keep is enforced with per-sequence feasibility guardrails.

    DCP signal:
      For the last decoder layer's MLP, we capture its *input* x_t and locally compute
      hidden_t = silu(gate_proj(x_t)) * up_proj(x_t) (no pruning/quant here),
      then per-channel saliency s_c = |hidden_t[c]| * ||down_proj[:,c]||_2.
      We convert s into the minimal keep fraction k_dcp achieving energy coverage ≥ cov_target,
      then map to signal01 = (k_dcp - p_min)/(p_max - p_min) ∈ [0,1], which biases the
      budget-matched chooser toward larger/smaller keeps.

    Returns (dict):
      ppl, avg_keep_all, avg_keep_effective, avg_prune_keep, avg_quant_ratio,
      prune_action_hist, prune_action_probs, tokens, tokens_effective,
      and metadata (cov_target, dcp_gamma).
    """
    import math
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from typing import Tuple

    model.eval()
    m = getattr(model, "module", model)
    try:
        m_dtype = next(m.parameters()).dtype
    except StopIteration:
        m_dtype = torch.float32

    # ---- Action spaces (κ fixed, bits fixed, ρ variable) ----
    spec = build_action_spec(
        keep_fracs=keep_fracs,
        prune_choices=prune_choices,
        quant_choices=quant_choices,
    )
    KEEP  = torch.tensor(keep_fracs, device=device, dtype=torch.float32)       # [K] (K==1 expected)
    prune_axis = _unique_float_axis(spec.prune_keep)                           # [P] ρ choices (floats)
    quant_axis = _unique_int_axis(spec.q_bits)                                 # [Q] bits choices (ints)
    PRUNE = torch.tensor(prune_axis, device=device, dtype=torch.float32)       # [P]
    QBITS = torch.tensor(quant_axis, device=device, dtype=torch.int64)         # [Q]
    QRAT  = QBITS.to(torch.float32).clamp(min=1.0) / 16.0                      # [Q] ratio ∈ (0,1]

    K, P, Q = KEEP.numel(), PRUNE.numel(), QBITS.numel()
    assert K == 1, "evaluate_dcp_prune_matched_keep expects a single κ (keep_fracs) choice."
    assert Q == 1, "evaluate_dcp_prune_matched_keep expects a single bits (quant_choices) choice."

    dense_k_idx = int(torch.argmax(KEEP).item())
    dense_q_idx = int(torch.argmax(QBITS).item())

    # Sorted view of PRUNE axis for searchsorted/feasibility guardrails
    p_vals_sorted, p_map_sorted_to_orig = torch.sort(PRUNE)
    p_min, p_max = float(p_vals_sorted[0].item()), float(p_vals_sorted[-1].item())
    inv_sorted_rank = torch.empty_like(p_map_sorted_to_orig)
    inv_sorted_rank[p_map_sorted_to_orig] = torch.arange(P, device=device, dtype=torch.long)
    max_rank_step = int(getattr(cfg, "prune_max_rank_delta", 1))
    _floor_val = getattr(cfg, "prune_keep_floor", None)
    if _floor_val is not None:
        _floor_val = float(_floor_val)
        floor_rank = int(torch.searchsorted(p_vals_sorted, torch.tensor([_floor_val], device=device)).item())
    else:
        floor_rank = None

    # --- Helper: choose prune with DCP bias + feasibility guardrails (per sequence) ---
    def _choose_prune_with_bias_and_guards(
        c_req: torch.Tensor,          # [B_struct] required keep (float)
        signal01: torch.Tensor,       # [B_struct] DCP signal in [0,1]
        cum_steps: torch.Tensor,      # [B_struct] # structured steps so far
        cum_val: torch.Tensor,        # [B_struct] cumulative keep value so far
        R_vec: torch.Tensor,          # [B_struct] structured steps remaining including current
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          chosen_orig_idx: [B_struct] indices into PRUNE (original order)
          chosen_keep:     [B_struct] chosen ρ values
        """
        Bx = c_req.size(0)
        # Clamp target into axis range and find local [lo, hi]
        c = c_req.clamp(p_vals_sorted[0], p_vals_sorted[-1])                   # [B]
        hi_sorted = torch.searchsorted(p_vals_sorted, c, right=False).clamp(0, P-1)
        lo_sorted = (hi_sorted - 1).clamp(0, P-1)

        lo_v = p_vals_sorted[lo_sorted]
        hi_v = p_vals_sorted[hi_sorted]
        denom = (hi_v - lo_v).clamp_min(1e-8)
        p_hi_base = torch.where((hi_v - lo_v) > 1e-8, (c - lo_v) / denom, torch.ones_like(c))

        # Bias toward hi/lo with DCP signal
        delta = dcp_gamma * (signal01 - 0.5)
        p_hi_mod = (p_hi_base + delta).clamp(0.0, 1.0)

        choose_hi = (p_hi_mod >= 0.5) & (hi_sorted != lo_sorted)
        chosen_sorted = torch.where(choose_hi, hi_sorted, lo_sorted)
        chosen_k = p_vals_sorted[chosen_sorted]

        # Feasibility guardrails to keep the per-sequence budget attainable
        R_post = (R_vec - 1.0).clamp_min(0.0)
        target_total = float(target_prune_keep) * (cum_steps + 1.0 + R_post)
        allowed_min = (target_total - cum_val - p_vals_sorted[-1] * R_post).clamp(p_vals_sorted[0], p_vals_sorted[-1])
        allowed_max = (target_total - cum_val - p_vals_sorted[0] * R_post).clamp(p_vals_sorted[0], p_vals_sorted[-1])

        lo_feas = torch.searchsorted(p_vals_sorted, allowed_min, right=False)
        hi_feas = torch.searchsorted(p_vals_sorted, allowed_max, right=True) - 1
        lo_feas = torch.minimum(lo_feas, hi_feas).clamp(0, P-1)
        hi_feas = torch.maximum(lo_feas, hi_feas).clamp(0, P-1)

        chosen_sorted = torch.maximum(chosen_sorted, lo_feas)
        chosen_sorted = torch.minimum(chosen_sorted, hi_feas)
        chosen_k = p_vals_sorted[chosen_sorted]

        chosen_orig = p_map_sorted_to_orig[chosen_sorted]
        return chosen_orig, chosen_k

    # ---- Locate last MLP + precompute ||down_proj[:,c]||_2 ----
    from transformers.models.llama import modeling_llama as llama_mod
    last_mlp = None
    for mod in m.modules():
        if isinstance(mod, llama_mod.LlamaMLP):
            last_mlp = mod
    if last_mlp is None:
        raise RuntimeError("Could not locate LlamaMLP module for DCP.")

    Wdown = last_mlp.down_proj.weight                            # [hidden, inter]
    col_norms = torch.linalg.vector_norm(Wdown, ord=2, dim=0)    # [inter]
    col_norms = col_norms.to(device=device, dtype=torch.float32)  # keep fp32 for stability

    gate_w, gate_b = last_mlp.gate_proj.weight, last_mlp.gate_proj.bias
    up_w,   up_b   = last_mlp.up_proj.weight,   last_mlp.up_proj.bias
    # Create fp32 copies for the probe path to avoid bf16/float matmul mismatches
    gate_w_f = gate_w.to(torch.float32)
    gate_b_f = None if gate_b is None else gate_b.to(torch.float32)
    up_w_f   = up_w.to(torch.float32)
    up_b_f   = None if up_b is None else up_b.to(torch.float32)

    # ---- Forward hook to capture last-MLP *input* x (pre MLP) for each subgroup call ----
    capture = {"sel": None, "store": None}   # will be [B, hidden_size]
    def _mlp_hook(module, inputs, output):
        sel = capture["sel"]
        store = capture["store"]
        if sel is None or store is None:
            return
        x = inputs[0]                        # [B_g, 1, hidden_size] or [B_g, hidden_size]
        x2 = x.reshape(x.size(0), -1)        # [B_g, hidden_size]
        # store as float32 for stable saliency math
        store.index_copy_(0, sel, x2.detach().to(torch.float32))
    hook = last_mlp.register_forward_hook(_mlp_hook)

    # ---- Evaluator accumulators ----
    thr = Ts + Tw + 1
    total_nll = 0.0
    total_tok = 0
    eff_tok   = 0
    total_keep_all = 0.0
    total_keep_eff = 0.0
    total_prune_eff = 0.0

    prune_action_hist = torch.zeros(P, device=device)

    enable_structured_controls(model)
    if str(getattr(cfg, "sparsity_criteria", "recency")) == "relevancy":
        clear_relevancy_keep(model)
    clear_structured_action(model)

    qbits_const  = QBITS[dense_q_idx].item()
    qratio_const = QRAT[dense_q_idx].item()

    try:
        for batch in tqdm(dl, desc="eval DCP prune matched keep"):
            batch = batch.to(device)
            B, _ = batch.shape

            prefill_ids = batch[:, :context_len]
            step_inputs = batch[:, context_len : context_len + rollout_len]
            step_labels = batch[:, context_len + 1 : context_len + rollout_len + 1]

            # Dense prefill to build cache
            out = model(input_ids=prefill_ids, use_cache=True, return_dict=True)
            past_kv = detach_cache_to_tuple(out.past_key_values)
            kv_len = torch.full((B,), context_len + 1, device=device, dtype=torch.long)

            # Per-sequence budget trackers for pruning axis
            cum_pru_steps = torch.zeros(B, device=device, dtype=torch.float32)
            cum_pru_val   = torch.zeros(B, device=device, dtype=torch.float32)
            # Sticky previous selection
            prev_p_idx = torch.full((B,), int(torch.argmax(PRUNE).item()),
                                    dtype=torch.long, device=device)
            # One-step-lag DCP signal (init neutral 0.5)
            prev_dcp_signal = torch.full((B,), 0.5, device=device, dtype=torch.float32)

            # Fixed κ and bits for all steps
            a_k_all = torch.full((B,), dense_k_idx, device=device, dtype=torch.long)
            a_q_all = torch.full((B,), dense_q_idx, device=device, dtype=torch.long)
            kappa_const = KEEP[dense_k_idx].item()

            for t in range(rollout_len):
                cur      = step_inputs[:, t]
                labels_t = step_labels[:, t]

                kv_before = kv_len.clone()
                eff_mask = (kv_before > thr)                   # [B] bool
                has_old = eff_mask.float()

                # --- Choose PRUNE action (current step) using prev DCP signal + budget guards ---
                # default to densest keep for non-structured steps
                a_p = torch.full((B,), int(torch.argmax(PRUNE).item()),
                                 dtype=torch.long, device=device)

                # Structured mask (effective-only, or all)
                struct_mask = eff_mask if not struct_on_non_eff else torch.ones_like(eff_mask)
                idx_struct = torch.nonzero(struct_mask, as_tuple=False).squeeze(-1)
                S_struct = int(idx_struct.numel())

                if S_struct > 0:
                    # Remaining structured steps per sequence (like EMC logic)
                    if not struct_on_non_eff:
                        T_rem = rollout_len - t
                        neff_rem = (thr - kv_before + 1).clamp_min(0)
                        R_vec_full = (T_rem - neff_rem).clamp_min(0).to(torch.float32)
                    else:
                        R_vec_full = torch.full((B,), float(rollout_len - t), device=device, dtype=torch.float32)
                    Rv = R_vec_full[idx_struct]

                    # Required keep to stay on budget (per sequence)
                    c_req_pru = (
                        float(target_prune_keep) * (cum_pru_steps[idx_struct] + Rv) - cum_pru_val[idx_struct]
                    ) / Rv.clamp_min(1.0)
                    c_req_pru = c_req_pru.clamp(p_min, p_max)

                    # Choose with DCP bias + feasibility
                    chosen_orig, chosen_keep = _choose_prune_with_bias_and_guards(
                        c_req=c_req_pru,
                        signal01=prev_dcp_signal[idx_struct],
                        cum_steps=cum_pru_steps[idx_struct],
                        cum_val=cum_pru_val[idx_struct],
                        R_vec=Rv,
                    )
                    chosen_sorted = inv_sorted_rank.index_select(0, chosen_orig)
                    if floor_rank is not None:
                        chosen_sorted = torch.maximum(
                            chosen_sorted,
                            torch.full_like(chosen_sorted, floor_rank)
                        )
                    # Sticky decision: limit rank change per token
                    prev_sorted = inv_sorted_rank.index_select(0, prev_p_idx.index_select(0, idx_struct))
                    delta = (chosen_sorted - prev_sorted).clamp(min=-max_rank_step, max=+max_rank_step)
                    chosen_sorted = (prev_sorted + delta).clamp(0, P-1)
                    a_p.index_copy_(0, idx_struct, p_map_sorted_to_orig.index_select(0, chosen_sorted))
                    prev_p_idx.index_copy_(0, idx_struct, p_map_sorted_to_orig.index_select(0, chosen_sorted))

                # --- Build per-sample κ (fixed) / prune / bits tensors ---
                a_k = a_k_all
                a_q = a_q_all
                kappa_now  = KEEP[a_k]                           # [B] constant
                prune_now  = PRUNE[a_p]                          # [B]
                qbits_now  = QBITS[a_q]                          # [B]

                # Histogram (prune only)
                prune_action_hist.index_add_(0, a_p, torch.ones_like(a_p, dtype=torch.float32))

                # --- Decode step in subgroups of identical (prune, bits) ---
                pq = torch.stack([prune_now, qbits_now.to(torch.float32)], dim=-1)  # [B, 2]
                uniq, inv = torch.unique(pq, dim=0, return_inverse=True)

                logits_step = None
                new_cache = None
                # allocate as float32 for stable downstream saliency computations
                capture["store"] = torch.empty(
                    (B, m.config.hidden_size), device=device, dtype=torch.float32
                )
                capture["store"].fill_(0.0)

                pos_ids_all = (kv_before - 1).clamp_min(0).unsqueeze(1)
 
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
                    sel = (inv == g).nonzero(as_tuple=False).squeeze(-1)
                    if sel.numel() == 0:
                        continue
                    p_scalar = float(p_val)
                    q_scalar = int(q_val)

                    # Apply subgroup structured controls
                    set_structured_action(model, p_scalar, q_scalar)

                    cur_g     = cur.index_select(0, sel)
                    pos_ids_g = pos_ids_all.index_select(0, sel)
                    kappa_g   = kappa_now.index_select(0, sel)

                    # Capture mapping for last-MLP hook
                    capture["sel"] = sel

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
                    )
                    if logits_step is None:
                        logits_step = torch.empty((B, out_g.logits.size(-1)),
                                                  device=device, dtype=out_g.logits.dtype)
                    logits_step.index_copy_(0, sel, out_g.logits[:, -1, :])

                    if new_cache is None:
                        new_cache = _init_cache_container_like(out_g.past_key_values, B)
                    for li, (k_src, v_src) in enumerate(out_g.past_key_values):
                        k_dst, v_dst = new_cache[li]
                        k_dst.index_copy_(0, sel, k_src)
                        v_dst.index_copy_(0, sel, v_src)

                    kv_len.index_add_(0, sel, torch.ones_like(sel, device=device, dtype=kv_len.dtype))

                # Clear structured action for next iteration; clear capture map
                clear_structured_action(model)
                capture["sel"] = None

                # --- Loss & token stats ---
                nll_t = F.cross_entropy(logits_step, labels_t, reduction="none")
                total_nll += nll_t.sum().item()
                total_tok += B

                eff_tok += int(has_old.sum().item())
                total_keep_all += kappa_now.sum().item()
                total_keep_eff += (kappa_now * has_old).sum().item()
                gate_struct = has_old if not struct_on_non_eff else torch.ones_like(has_old)
                total_prune_eff += (prune_now * gate_struct).sum().item()

                # --- Build DCP signal for NEXT step from captured x in last MLP ---
                # x_store: [B, hidden_size]
                x_store = capture["store"]                                  # [B, H]
                # hidden = silu(gate) * up  (use *dense* weights for the probe)+                # Cast probe weights/bias to x_store dtype (fp32) to avoid bf16/float mismatch

                # Use fp32 probe weights/bias to match x_store dtype
                gate = F.linear(x_store, gate_w_f, gate_b_f)                # [B, inter]
                up   = F.linear(x_store,   up_w_f,   up_b_f)                # [B, inter]
                hidden = F.silu(gate) * up                                  # [B, inter]
                hidden = hidden.to(torch.float32)
                # downstream contribution saliency
                # s = |hidden| * ||down[:,c]||_2
                s = hidden.abs() * col_norms.unsqueeze(0)                   # [B, inter]
                s_sum = s.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                s_sorted, _ = torch.sort(s, dim=-1, descending=True)        # [B, inter]
                cumsum = (s_sorted.cumsum(dim=-1) / s_sum)                  # [B, inter]
                # first index where coverage >= cov_target
                mask = (cumsum >= cov_target)
                # torch.argmax on bool gives first True (assumes last is True)
                first_idx = torch.argmax(mask.to(torch.int32), dim=-1)      # [B]
                inter_dim = s.size(-1)
                k_dcp = (first_idx.to(torch.float32) + 1.0) / float(inter_dim)  # [B] ∈ (0,1]

                # Map to [0,1] signal on the PRUNE axis range
                if p_max > p_min + 1e-12:
                    signal01 = ((k_dcp - p_min) / (p_max - p_min)).clamp(0.0, 1.0)
                else:
                    signal01 = torch.full_like(k_dcp, 0.5)
                prev_dcp_signal = signal01.detach().to(torch.float32)

                # --- Advance per-sequence budget trackers for pruning ---
                S_this = int(gate_struct.sum().item())
                if S_this > 0:
                    cum_pru_steps = cum_pru_steps + gate_struct
                    cum_pru_val   = cum_pru_val   + gate_struct * prune_now

        # end loop over batches
    finally:
        # Ensure hook removal even if an error occurs mid-loop
        try:
            hook.remove()
        except Exception:
            pass
        clear_structured_action(model)

    # ---- Final metrics ----
    ppl = math.exp(total_nll / max(1, total_tok))
    avg_keep_all = total_keep_all / max(1, total_tok)
    avg_keep_eff = (total_keep_eff / max(1, eff_tok)) if eff_tok > 0 else 0.0
    denom_struct = (eff_tok if not struct_on_non_eff else total_tok)
    avg_prune_keep  = (total_prune_eff  / max(1, denom_struct)) if denom_struct > 0 else 0.0
    # bits ratio is constant
    avg_quant_ratio = qratio_const

    prune_probs = (prune_action_hist / prune_action_hist.sum().clamp_min(1)).tolist()

    return {
        "ppl": ppl,
        "avg_keep_all": avg_keep_all,
        "avg_keep_effective": avg_keep_eff,
        "avg_prune_keep": avg_prune_keep,
        "avg_quant_ratio": avg_quant_ratio,
        "prune_action_hist": prune_action_hist.tolist(),
        "prune_action_probs": prune_probs,
        "tokens": total_tok,
        "tokens_effective": eff_tok,
        "cov_target": float(cov_target),
        "dcp_gamma": float(dcp_gamma),
    }

@torch.no_grad()
def evaluate_margin_prune_matched_keep(
    cfg,
    model,
    dl,
    Ts: int,
    Tw: int,
    keep_fracs: Tuple[float, ...],    # must be (1.0,)
    prune_choices: Tuple[str, ...],
    quant_choices: Tuple[str, ...],   # must be a single bits choice, e.g., ("w16",)
    target_prune_keep: float,         # desired average ρ over structured steps
    context_len: int,
    rollout_len: int,
    device: str,
    struct_on_non_eff: bool = False,
    margin_smooth: float = 0.05,      # EMA for the margin (stabilize jitter)
    gamma: float = 0.35,              # bias strength toward hi/lo from margin
):
    """
    Per-input, causal prune controller using previous-step logit margin:
      - Compute margin m_{t-1} = p_top1 - p_top2 in [0,1].
      - Convert to signal01 = 1 - m_{t-1}  (low margin => high need).
      - Bias the discrete keep choice around the per-seq budget requirement,
        with feasibility rails so the running average can hit target_prune_keep.
    Assumes κ=1.0 (dense tokens) and a single quantization choice.
    """
    import math
    import torch
    import torch.nn.functional as F
    from typing import Tuple
    from tqdm import tqdm

    model.eval()
    m = getattr(model, "module", model)

    # ----- Hard constraints: token-sparsity OFF, bits fixed -----
    KEEP = torch.tensor(keep_fracs, device=device, dtype=torch.float32)
    assert KEEP.numel() == 1 and abs(float(KEEP.item()) - 1.0) < 1e-6, \
        "Margin controller expects κ=1.0 (dense tokens)."
    spec = build_action_spec(keep_fracs=keep_fracs, prune_choices=prune_choices, quant_choices=quant_choices)
    PRUNE = torch.tensor(_unique_float_axis(spec.prune_keep), device=device, dtype=torch.float32)
    QBITS = torch.tensor(_unique_int_axis(spec.q_bits), device=device, dtype=torch.int64)
    assert QBITS.numel() == 1, "Margin controller expects a single bits choice."

    P = PRUNE.numel()
    p_vals_sorted, p_map_sorted_to_orig = torch.sort(PRUNE)      # ascending keeps
    inv_sorted_rank = torch.empty_like(p_map_sorted_to_orig)
    inv_sorted_rank[p_map_sorted_to_orig] = torch.arange(P, device=device, dtype=torch.long)
    p_min, p_max = float(p_vals_sorted[0]), float(p_vals_sorted[-1])
    max_rank_step = int(getattr(cfg, "prune_max_rank_delta", 1))
    floor_val = getattr(cfg, "prune_keep_floor", None)
    floor_rank = None
    if floor_val is not None:
        floor_val = float(floor_val)
        floor_rank = int(torch.searchsorted(p_vals_sorted, torch.tensor([floor_val], device=device)).item())

    def choose_with_bias_and_feas(c_req, signal01, cum_steps, cum_val, R_vec, prev_idx_sorted):
        # c_req in [p_min, p_max], signal01 in [0,1] (1=needs more keep)
        c = c_req.clamp(p_min, p_max)
        hi_sorted = torch.searchsorted(p_vals_sorted, c, right=False).clamp(0, P-1)
        lo_sorted = (hi_sorted - 1).clamp(0, P-1)
        lo_v, hi_v = p_vals_sorted[lo_sorted], p_vals_sorted[hi_sorted]
        denom = (hi_v - lo_v).clamp_min(1e-8)
        p_hi_base = torch.where(denom > 1e-8, (c - lo_v) / denom, torch.ones_like(c))
        # margin bias (low margin -> push higher keep)
        p_hi_mod = (p_hi_base + gamma * (signal01 - 0.5)).clamp(0.0, 1.0)
        choose_hi = (p_hi_mod >= 0.5) & (hi_sorted != lo_sorted)
        chosen_sorted = torch.where(choose_hi, hi_sorted, lo_sorted)

        # feasibility rails
        R_post = (R_vec - 1.0).clamp_min(0.0)
        target_total = float(target_prune_keep) * (cum_steps + 1.0 + R_post)
        allowed_min = (target_total - cum_val - p_vals_sorted[-1] * R_post).clamp(p_vals_sorted[0], p_vals_sorted[-1])
        allowed_max = (target_total - cum_val - p_vals_sorted[0]  * R_post).clamp(p_vals_sorted[0], p_vals_sorted[-1])
        lo_feas = torch.searchsorted(p_vals_sorted, allowed_min, right=False)
        hi_feas = torch.searchsorted(p_vals_sorted, allowed_max, right=True) - 1
        lo_feas = torch.minimum(lo_feas, hi_feas).clamp(0, P-1)
        hi_feas = torch.maximum(lo_feas, hi_feas).clamp(0, P-1)
        chosen_sorted = torch.maximum(chosen_sorted, lo_feas)
        chosen_sorted = torch.minimum(chosen_sorted, hi_feas)

        # sticky
        delta = (chosen_sorted - prev_idx_sorted).clamp(min=-max_rank_step, max=+max_rank_step)
        chosen_sorted = (prev_idx_sorted + delta).clamp(0, P-1)

        if floor_rank is not None:
            chosen_sorted = torch.maximum(chosen_sorted, torch.full_like(chosen_sorted, floor_rank))
        return chosen_sorted

    thr = Ts + Tw + 1
    total_nll = 0.0
    total_tok = 0
    eff_tok   = 0
    total_prune_eff = 0.0
    hist = torch.zeros(P, device=device)

    enable_structured_controls(model)
    if str(getattr(cfg, "sparsity_criteria", "recency")) == "relevancy":
        clear_relevancy_keep(model)
    clear_structured_action(model)

    try:
        for batch in tqdm(dl, desc="eval margin prune matched"):
            batch = batch.to(device)
            B, _ = batch.shape
            prefill_ids = batch[:, :context_len]
            step_inputs = batch[:, context_len : context_len + rollout_len]
            step_labels = batch[:, context_len + 1 : context_len + rollout_len + 1]

            out = model(input_ids=prefill_ids, use_cache=True, return_dict=True)
            past_kv = detach_cache_to_tuple(out.past_key_values)
            kv_len = torch.full((B,), context_len + 1, device=device, dtype=torch.long)

            # per-seq trackers
            cum_steps = torch.zeros(B, device=device, dtype=torch.float32)
            cum_val   = torch.zeros(B, device=device, dtype=torch.float32)
            prev_idx_sorted = torch.full((B,), int(torch.argmax(PRUNE).item()), device=device, dtype=torch.long)
            prev_idx_sorted = inv_sorted_rank.index_select(0, prev_idx_sorted)

            # margin signal (EMA)
            prev_margin = torch.full((B,), 0.5, device=device, dtype=torch.float32)  # neutral

            # 1 effective token warmup (dense)
            did_warmup = torch.zeros(B, device=device, dtype=torch.bool)

            for t in range(rollout_len):
                cur = step_inputs[:, t]
                labels_t = step_labels[:, t]
                kv_before = kv_len.clone()
                eff_mask = (kv_before > thr)
                struct_mask = eff_mask if not struct_on_non_eff else torch.ones_like(eff_mask)
                idx_struct = torch.nonzero(struct_mask, as_tuple=False).squeeze(-1)
                S = int(idx_struct.numel())

                # choose prune per input
                a_p = torch.full((B,), int(torch.argmax(PRUNE).item()), device=device, dtype=torch.long)

                if S > 0:
                    # warmup: first time each seq becomes effective -> force dense once
                    first_eff = (~did_warmup) & eff_mask
                    a_p[first_eff] = int(torch.argmax(PRUNE).item())
                    did_warmup = did_warmup | first_eff

                    # remaining structured indices
                    idx_use = idx_struct[~first_eff.index_select(0, idx_struct)]

                    if idx_use.numel() > 0:
                        # required per-seq keep to stay on budget
                        if not struct_on_non_eff:
                            T_rem = rollout_len - t
                            neff_rem = (thr - kv_before + 1).clamp_min(0)
                            R_vec_full = (T_rem - neff_rem).clamp_min(0).to(torch.float32)
                        else:
                            R_vec_full = torch.full((B,), float(rollout_len - t), device=device, dtype=torch.float32)
                        Rv = R_vec_full.index_select(0, idx_use)

                        c_req = (
                            float(target_prune_keep) * (cum_steps.index_select(0, idx_use) + Rv) -
                            cum_val.index_select(0, idx_use)
                        ) / Rv.clamp_min(1.0)
                        c_req = c_req.clamp(p_min, p_max)

                        # margin -> need signal
                        signal01 = (1.0 - prev_margin.index_select(0, idx_use)).clamp(0.0, 1.0)

                        chosen_sorted = choose_with_bias_and_feas(
                            c_req=c_req,
                            signal01=signal01,
                            cum_steps=cum_steps.index_select(0, idx_use),
                            cum_val=cum_val.index_select(0, idx_use),
                            R_vec=Rv,
                            prev_idx_sorted=prev_idx_sorted.index_select(0, idx_use),
                        )
                        a_p.index_copy_(0, idx_use, p_map_sorted_to_orig.index_select(0, chosen_sorted))
                        prev_idx_sorted.index_copy_(0, idx_use, chosen_sorted)

                prune_now = PRUNE.index_select(0, a_p)
                hist.index_add_(0, a_p, torch.ones_like(a_p, dtype=torch.float32))

                # run sub-batches (bits fixed; κ=1 so no attention mask)
                uniq, inv = torch.unique(prune_now, sorted=False, return_inverse=True)
                logits_step = None
                new_cache = None
                pos_ids_all = (kv_before - 1).clamp_min(0).unsqueeze(1)

                for g, p_val in enumerate(uniq.tolist()):
                    sel = (inv == g).nonzero(as_tuple=False).squeeze(-1)
                    if sel.numel() == 0: continue
                    set_structured_action(model, float(p_val), int(QBITS[0].item()))
                    cache_g = select_cache_by_indices(past_kv, sel)
                    out_g = model(
                        input_ids=cur.index_select(0, sel).unsqueeze(1),
                        use_cache=True,
                        past_key_values=cache_g,
                        position_ids=pos_ids_all.index_select(0, sel),
                        return_dict=True,
                    )
                    if logits_step is None:
                        logits_step = torch.empty((B, out_g.logits.size(-1)), device=device, dtype=out_g.logits.dtype)
                    logits_step.index_copy_(0, sel, out_g.logits[:, -1, :])

                    if new_cache is None:
                        def _init_like(sub_cache, B_total):
                            c = []
                            for (k_src, v_src) in sub_cache:
                                k_shape, v_shape = list(k_src.shape), list(v_src.shape)
                                k_shape[0] = v_shape[0] = B_total
                                c.append((torch.empty(k_shape, dtype=k_src.dtype, device=k_src.device),
                                          torch.empty(v_shape, dtype=v_src.dtype, device=v_src.device)))
                            return tuple(c)
                        new_cache = _init_like(out_g.past_key_values, B)
                    for li, (k_src, v_src) in enumerate(out_g.past_key_values):
                        k_dst, v_dst = new_cache[li]
                        k_dst.index_copy_(0, sel, k_src)
                        v_dst.index_copy_(0, sel, v_src)
                    kv_len.index_add_(0, sel, torch.ones_like(sel, dtype=kv_len.dtype))

                past_kv = new_cache
                clear_structured_action(model)

                # loss & stats
                nll_t = F.cross_entropy(logits_step, labels_t, reduction="none")
                total_nll += nll_t.sum().item()
                total_tok += B
                gate = (eff_mask.float() if not struct_on_non_eff else torch.ones_like(eff_mask, dtype=torch.float32))
                eff_tok += int(eff_mask.sum().item())
                total_prune_eff += (prune_now * gate).sum().item()

                # update margin for next step (EMA)
                with torch.no_grad():
                    probs = torch.softmax(logits_step, dim=-1)
                    top2 = torch.topk(probs, k=2, dim=-1).values
                    margin = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)
                    prev_margin = (1.0 - margin_smooth) * prev_margin + margin_smooth * margin

                # update budgets
                cum_steps = cum_steps + gate
                cum_val   = cum_val   + gate * prune_now

    finally:
        clear_structured_action(model)

    ppl = math.exp(total_nll / max(1, total_tok))
    avg_prune_keep = total_prune_eff / max(1, (eff_tok if not struct_on_non_eff else total_tok))
    probs = (hist / hist.sum().clamp_min(1)).tolist()
    return {
        "ppl": ppl,
        "avg_prune_keep": avg_prune_keep,
        "tokens": total_tok,
        "tokens_effective": eff_tok,
        "prune_action_hist": hist.tolist(),
        "prune_action_probs": probs,
        "gamma": float(gamma),
        "margin_smooth": float(margin_smooth),
    }
