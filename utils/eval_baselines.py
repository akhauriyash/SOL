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
def evaluate_emc_matched_structured(
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
    target_quant_ratio: float,          # bits/16, e.g. 16->1.0, 8->0.5
    context_len: int,
    rollout_len: int,
    device: str,
    struct_on_non_eff: bool = False,    # if True, apply prune/quant on non-effective steps too
    use_emc_mix: float = None,          # optional override for cfg.emc_mix
    emc_gamma_keep: float = None,       # optional per-axis override
    emc_gamma_prune: float = None,
    emc_gamma_quant: float = None,
    emc_ema: float = None,              # optional EMA smoothing on uncertainty (0 disables)
):
    """
    EMC controller with matched budgets on THREE axes:
      - Token keep (kappa, κ) on effective tokens
      - Channel prune keep (rho, ρ) on struct_mask tokens
      - Quant ratio (q_ratio = bits/16) on struct_mask tokens

    Uses previous-step uncertainty u_{t-1} (entropy/margin mixture) to bias *each axis*
    toward higher-resource actions when u is high, lower-resource actions when u is low.
    Budget steering + feasibility guardrails keep the final averages close to targets.
    """

    import math
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()

    # ---- dtype for attention bias ----
    m = getattr(model, "module", model)
    try:
        m_dtype = next(m.parameters()).dtype
    except StopIteration:
        m_dtype = torch.float32

    # ---- build axes from your spec (same as fixed eval) ----
    spec = build_action_spec(
        keep_fracs=keep_fracs,
        prune_choices=prune_choices,
        quant_choices=quant_choices,
    )

    # κ axis
    KEEP = torch.tensor(keep_fracs, device=device, dtype=torch.float32)  # original order [K]
    K = KEEP.numel()
    k_pairs = sorted([(float(v), i) for i, v in enumerate(keep_fracs)], key=lambda x: x[0])
    k_vals_sorted = torch.tensor([v for v, _ in k_pairs], device=device, dtype=torch.float32)     # sorted values
    k_map_sorted_to_orig = torch.tensor([i for _, i in k_pairs], device=device, dtype=torch.long) # sorted->orig
    k_min = float(k_vals_sorted[0].item())
    k_max = float(k_vals_sorted[-1].item())
    dense_k_idx = keep_fracs.index(1.0) if 1.0 in keep_fracs else int(torch.argmax(KEEP).item())

    # ρ axis
    prune_axis = _unique_float_axis(spec.prune_keep)
    PRUNE = torch.tensor(prune_axis, device=device, dtype=torch.float32)  # original order [P]
    P = PRUNE.numel()
    p_vals_sorted, p_map_sorted_to_orig = torch.sort(PRUNE)
    p_min = float(p_vals_sorted[0].item())
    p_max = float(p_vals_sorted[-1].item())
    dense_p_idx = int(torch.argmax(PRUNE).item())

    # q axis (ratio = bits/16)
    quant_axis = _unique_int_axis(spec.q_bits)
    QBITS = torch.tensor(quant_axis, device=device, dtype=torch.int64)    # original order [Q]
    Q = QBITS.numel()
    QRAT = QBITS.to(torch.float32).clamp(min=1.0) / 16.0
    q_vals_sorted, q_map_sorted_to_orig = torch.sort(QRAT)
    q_min = float(q_vals_sorted[0].item())
    q_max = float(q_vals_sorted[-1].item())
    dense_q_idx = int(torch.argmax(QBITS).item())

    # ---- controller knobs ----
    mix = float(getattr(cfg, "emc_mix", 1.0))
    if use_emc_mix is not None:
        mix = float(use_emc_mix)

    gamma_default = float(getattr(cfg, "emc_gamma", 0.35))
    g_k = float(emc_gamma_keep) if emc_gamma_keep is not None else float(getattr(cfg, "emc_gamma_keep", gamma_default))
    g_p = float(emc_gamma_prune) if emc_gamma_prune is not None else float(getattr(cfg, "emc_gamma_prune", gamma_default))
    g_q = float(emc_gamma_quant) if emc_gamma_quant is not None else float(getattr(cfg, "emc_gamma_quant", gamma_default))

    ema = float(getattr(cfg, "emc_ema", 0.0))
    if emc_ema is not None:
        ema = float(emc_ema)

    thr = Ts + Tw + 1

    # ---- helpers ----
    def _compute_uncert_from_logits(logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, V] -> u in [0,1]
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        H = -(p * logp).sum(dim=-1)  # [B]
        H_norm = H / math.log(logits.size(-1) + 1e-12)
        top2 = torch.topk(p, k=2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)
        u = (mix * H_norm + (1.0 - mix) * (1.0 - margin)).clamp(0.0, 1.0)
        return u

    def _choose_axis_with_bias_and_guards(
        c_req: torch.Tensor,                 # [N]
        signal01: torch.Tensor,              # [N]
        cum_val: torch.Tensor,               # [N]
        cum_steps: torch.Tensor,             # [N]
        R: torch.Tensor,                     # [N] remaining steps including current
        target_avg: float,
        vals_sorted: torch.Tensor,           # [A] sorted discrete values
        map_sorted_to_orig: torch.Tensor,    # [A] sorted->orig indices
        vmin: float,
        vmax: float,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministic hi/lo selection around c_req with uncertainty bias + feasibility band clamp.
        Returns:
          chosen_orig_idx: [N]
          chosen_val:      [N]
        """
        A = vals_sorted.numel()
        c = c_req.clamp(vals_sorted[0], vals_sorted[-1])
        hi = torch.searchsorted(vals_sorted, c, right=False).clamp(0, A - 1)
        lo = (hi - 1).clamp(0, A - 1)

        lo_v = vals_sorted[lo]
        hi_v = vals_sorted[hi]
        denom = (hi_v - lo_v).clamp_min(1e-8)
        p_hi_base = torch.where((hi_v - lo_v) > 1e-8, (c - lo_v) / denom, torch.ones_like(c))

        # bias toward hi when signal > 0.5
        delta = gamma * (signal01 - 0.5)
        p_hi_mod = (p_hi_base + delta).clamp(0.0, 1.0)

        choose_hi = (p_hi_mod >= 0.5) & (hi != lo)
        chosen_sorted = torch.where(choose_hi, hi, lo)
        chosen_val = vals_sorted[chosen_sorted]

        # feasibility guardrails: ensure target remains attainable
        R_post = (R - 1.0).clamp_min(0.0)
        target_total = target_avg * (cum_steps + 1.0 + R_post)  # == target_avg*(cum_steps+R)
        allowed_min = (target_total - cum_val - vmax * R_post).clamp(vmin, vmax)
        allowed_max = (target_total - cum_val - vmin * R_post).clamp(vmin, vmax)

        lo_feas = torch.searchsorted(vals_sorted, allowed_min, right=False)
        hi_feas = torch.searchsorted(vals_sorted, allowed_max, right=True) - 1
        lo_feas = torch.minimum(lo_feas, hi_feas).clamp(0, A - 1)
        hi_feas = torch.maximum(lo_feas, hi_feas).clamp(0, A - 1)

        chosen_sorted = torch.maximum(chosen_sorted, lo_feas)
        chosen_sorted = torch.minimum(chosen_sorted, hi_feas)
        chosen_val = vals_sorted[chosen_sorted]

        chosen_orig = map_sorted_to_orig[chosen_sorted]
        return chosen_orig, chosen_val

    def _mix_report(vals_sorted_1d: torch.Tensor, target: float):
        # just for returning "neighbors of target" like your other evals
        vs = vals_sorted_1d.detach().cpu().tolist()
        if target <= vs[0]:
            return {"lo": vs[0], "hi": vs[0], "p_hi": 0.0}
        if target >= vs[-1]:
            return {"lo": vs[-1], "hi": vs[-1], "p_hi": 1.0}
        lo_i = max(i for i, v in enumerate(vs) if v <= target)
        hi_i = min(i for i, v in enumerate(vs) if v >= target)
        lo_v, hi_v = vs[lo_i], vs[hi_i]
        p_hi = 0.0 if hi_v == lo_v else (target - lo_v) / (hi_v - lo_v)
        return {"lo": lo_v, "hi": hi_v, "p_hi": float(p_hi)}

    # ---- accumulators ----
    total_nll = 0.0
    total_tok = 0
    eff_tok = 0
    total_keep_all = 0.0
    total_keep_eff = 0.0
    total_prune_eff = 0.0
    total_qratio_eff = 0.0
    action_hist = torch.zeros(K * P * Q, device=device)

    # ---- evaluator state ----
    enable_structured_controls(model)

    crit = str(getattr(cfg, "sparsity_criteria", "recency"))
    if crit == "relevancy":
        # if your relevancy implementation caches any state, keep this
        clear_relevancy_keep(model)
    # If you have Quest-specific persistent state, clear it here similarly.
    # e.g., if crit == "quest": clear_quest_keep(model)

    clear_structured_action(model)  # ensure dense for prefill

    for batch in tqdm(dl, desc="eval EMC matched structured"):
        batch = batch.to(device)
        B, _ = batch.shape

        prefill_ids = batch[:, :context_len]
        step_inputs = batch[:, context_len : context_len + rollout_len]
        step_labels = batch[:, context_len + 1 : context_len + rollout_len + 1]

        # Dense prefill
        out = model(input_ids=prefill_ids, use_cache=True, return_dict=True)
        past_kv = detach_cache_to_tuple(out.past_key_values)
        kv_len = torch.full((B,), context_len + 1, device=device, dtype=torch.long)

        # ---- corrected initialization: u_0 from prefill next-token logits ----
        prev_uncert = _compute_uncert_from_logits(out.logits[:, -1, :]).to(torch.float32)  # [B]
        if ema > 0.0:
            prev_uncert_ema = prev_uncert.clone()
        else:
            prev_uncert_ema = None

        # per-sequence trackers for κ
        cum_keep = torch.zeros(B, device=device)
        cum_eff = torch.zeros(B, device=device)

        # per-sequence trackers for ρ and q_ratio (on struct_gate steps)
        cum_pru_steps = torch.zeros(B, device=device)
        cum_q_steps   = torch.zeros(B, device=device)
        cum_pru_val   = torch.zeros(B, device=device)
        cum_q_val     = torch.zeros(B, device=device)

        for t in range(rollout_len):
            cur = step_inputs[:, t]
            labels_t = step_labels[:, t]

            kv_before = kv_len
            eff_mask = (kv_before > thr)
            has_old = eff_mask.to(torch.float32)

            # uncertainty signal for this decision step
            signal01 = prev_uncert
            if prev_uncert_ema is not None:
                signal01 = prev_uncert_ema

            # default actions: dense everywhere
            a_k = torch.full((B,), dense_k_idx, device=device, dtype=torch.long)
            a_p = torch.full((B,), dense_p_idx, device=device, dtype=torch.long)
            a_q = torch.full((B,), dense_q_idx, device=device, dtype=torch.long)

            # Compute remaining effective steps R_eff_all for κ (and for struct when struct_on_non_eff=False)
            T_rem = rollout_len - t
            neff_rem = (thr - kv_before + 1).clamp_min(0)
            R_eff_all = (T_rem - neff_rem).clamp_min(0).to(torch.float32)  # [B]

            # ---- κ decisions on effective tokens ----
            if eff_mask.any():
                idx_eff = torch.nonzero(eff_mask, as_tuple=False).squeeze(-1)
                R_eff = R_eff_all[idx_eff].clamp_min(1.0)

                c_req_k = (target_keep_effective * (cum_eff[idx_eff] + R_eff) - cum_keep[idx_eff]) / R_eff
                c_req_k = c_req_k.clamp(k_min, k_max)

                chosen_k_idx, _ = _choose_axis_with_bias_and_guards(
                    c_req=c_req_k,
                    signal01=signal01[idx_eff],
                    cum_val=cum_keep[idx_eff],
                    cum_steps=cum_eff[idx_eff],
                    R=R_eff,
                    target_avg=float(target_keep_effective),
                    vals_sorted=k_vals_sorted,
                    map_sorted_to_orig=k_map_sorted_to_orig,
                    vmin=k_min,
                    vmax=k_max,
                    gamma=g_k,
                )
                a_k[idx_eff] = chosen_k_idx

            # ---- ρ and q decisions (struct_mask) ----
            struct_mask = eff_mask if not struct_on_non_eff else torch.ones_like(eff_mask)
            if struct_mask.any():
                idx_struct = torch.nonzero(struct_mask, as_tuple=False).squeeze(-1)

                if struct_on_non_eff:
                    R_struct_all = torch.full((B,), float(rollout_len - t), device=device, dtype=torch.float32)
                else:
                    R_struct_all = R_eff_all

                R_vec = R_struct_all[idx_struct].clamp_min(1.0)

                # prune keep (ρ)
                c_req_p = (target_prune_keep * (cum_pru_steps[idx_struct] + R_vec) - cum_pru_val[idx_struct]) / R_vec
                c_req_p = c_req_p.clamp(p_min, p_max)
                chosen_p_idx, _ = _choose_axis_with_bias_and_guards(
                    c_req=c_req_p,
                    signal01=signal01[idx_struct],
                    cum_val=cum_pru_val[idx_struct],
                    cum_steps=cum_pru_steps[idx_struct],
                    R=R_vec,
                    target_avg=float(target_prune_keep),
                    vals_sorted=p_vals_sorted,
                    map_sorted_to_orig=p_map_sorted_to_orig,
                    vmin=p_min,
                    vmax=p_max,
                    gamma=g_p,
                )
                a_p[idx_struct] = chosen_p_idx

                # quant ratio (bits/16)
                c_req_q = (target_quant_ratio * (cum_q_steps[idx_struct] + R_vec) - cum_q_val[idx_struct]) / R_vec
                c_req_q = c_req_q.clamp(q_min, q_max)
                chosen_q_idx, _ = _choose_axis_with_bias_and_guards(
                    c_req=c_req_q,
                    signal01=signal01[idx_struct],
                    cum_val=cum_q_val[idx_struct],
                    cum_steps=cum_q_steps[idx_struct],
                    R=R_vec,
                    target_avg=float(target_quant_ratio),
                    vals_sorted=q_vals_sorted,
                    map_sorted_to_orig=q_map_sorted_to_orig,
                    vmin=q_min,
                    vmax=q_max,
                    gamma=g_q,
                )
                a_q[idx_struct] = chosen_q_idx

            # materialize chosen values
            kappa_now  = KEEP[a_k]      # [B]
            prune_now  = PRUNE[a_p]     # [B]
            qbits_now  = QBITS[a_q]     # [B]
            qratio_now = QRAT[a_q]      # [B]

            # histogram over (k,p,q)
            flat_idx = a_k * (P * Q) + a_p * Q + a_q
            action_hist.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))

            # ---- decode step grouped by (prune, qbits) ----
            pq = torch.stack([prune_now, qbits_now.to(torch.float32)], dim=-1)
            uniq, inv = torch.unique(pq, dim=0, return_inverse=True)

            logits_step = None
            new_cache = None
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
                    logits_step = torch.empty((B, out_g.logits.size(-1)), device=device, dtype=out_g.logits.dtype)
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

            # ---- loss + accounting ----
            nll_t = F.cross_entropy(logits_step, labels_t, reduction="none")
            total_nll += nll_t.sum().item()
            total_tok += B

            eff_tok += int(has_old.sum().item())
            total_keep_all += kappa_now.sum().item()
            total_keep_eff += (kappa_now * has_old).sum().item()

            gate = has_old if not struct_on_non_eff else torch.ones_like(has_old)
            total_prune_eff  += (prune_now  * gate).sum().item()
            total_qratio_eff += (qratio_now * gate).sum().item()

            # per-sequence tracker updates
            cum_eff  = cum_eff  + has_old
            cum_keep = cum_keep + has_old * kappa_now

            cum_pru_steps = cum_pru_steps + gate
            cum_q_steps   = cum_q_steps   + gate
            cum_pru_val   = cum_pru_val   + gate * prune_now
            cum_q_val     = cum_q_val     + gate * qratio_now

            # update uncertainty for next step
            next_u = _compute_uncert_from_logits(logits_step).to(torch.float32)
            if prev_uncert_ema is not None:
                prev_uncert_ema = ema * prev_uncert_ema + (1.0 - ema) * next_u
                prev_uncert = prev_uncert_ema
            else:
                prev_uncert = next_u

    clear_structured_action(model)

    ppl = math.exp(total_nll / max(1, total_tok))
    avg_keep_all = total_keep_all / max(1, total_tok)
    avg_keep_eff = (total_keep_eff / max(1, eff_tok)) if eff_tok > 0 else 0.0
    denom_struct = (eff_tok if not struct_on_non_eff else total_tok)
    avg_prune_keep  = total_prune_eff  / max(1, denom_struct)
    avg_quant_ratio = total_qratio_eff / max(1, denom_struct)

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
        "mix_keep":  _mix_report(k_vals_sorted, float(target_keep_effective)),
        "mix_prune": _mix_report(p_vals_sorted, float(target_prune_keep)),
        "mix_quant": _mix_report(q_vals_sorted, float(target_quant_ratio)),
        "emc": {
            "mix": float(mix),
            "gamma_keep": float(g_k),
            "gamma_prune": float(g_p),
            "gamma_quant": float(g_q),
            "ema": float(ema),
        },
        "criteria": str(getattr(cfg, "sparsity_criteria", "recency")),
    }

@torch.no_grad()
def evaluate_drift_aware_matched_structured(
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
    target_quant_ratio: float,   # bits/16, e.g. 16->1.0, 8->0.5
    context_len: int,
    rollout_len: int,
    device: str,
    struct_on_non_eff: bool = False,
    dac_gamma_keep: float = None,
    dac_gamma_prune: float = None,
    dac_gamma_quant: float = None,
    dac_ema: float = None,
):
    """
    Drift-Aware Controller (DAC) with matched budgets on THREE axes:
      - Token keep kappa (κ): match target_keep_effective on effective tokens.
      - Channel prune keep (ρ): match target_prune_keep on struct_mask tokens.
      - Quant ratio q_ratio (bits/16): match target_quant_ratio on struct_mask tokens.

    Drift signal is causal:
      - t=0: embedding drift between prefill last token and current token.
      - t>=1: cosine drift between last-layer hidden states at (t-1) and (t-2).

    High drift => bias toward larger κ / larger ρ / larger q_ratio.
    Budget steering + feasibility guardrails ensure targets remain attainable.
    """

    import math
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()

    # ---- dtype for attention bias / criteria paths ----
    m = getattr(model, "module", model)
    try:
        m_dtype = next(m.parameters()).dtype
    except StopIteration:
        m_dtype = torch.float32

    thr = Ts + Tw + 1

    # ---- build axes from your spec (same as fixed eval) ----
    spec = build_action_spec(
        keep_fracs=keep_fracs,
        prune_choices=prune_choices,
        quant_choices=quant_choices,
    )

    # κ axis (sorted view + map to original indices)
    KEEP = torch.tensor(keep_fracs, device=device, dtype=torch.float32)  # original order [K]
    K = KEEP.numel()
    k_pairs = sorted([(float(v), i) for i, v in enumerate(keep_fracs)], key=lambda x: x[0])
    k_vals_sorted = torch.tensor([v for v, _ in k_pairs], device=device, dtype=torch.float32)
    k_map_sorted_to_orig = torch.tensor([i for _, i in k_pairs], device=device, dtype=torch.long)
    k_min = float(k_vals_sorted[0].item())
    k_max = float(k_vals_sorted[-1].item())
    dense_k_idx = keep_fracs.index(1.0) if 1.0 in keep_fracs else int(torch.argmax(KEEP).item())

    # ρ axis
    prune_axis = _unique_float_axis(spec.prune_keep)
    PRUNE = torch.tensor(prune_axis, device=device, dtype=torch.float32)  # original order [P]
    P = PRUNE.numel()
    p_vals_sorted, p_map_sorted_to_orig = torch.sort(PRUNE)
    p_min = float(p_vals_sorted[0].item())
    p_max = float(p_vals_sorted[-1].item())
    dense_p_idx = int(torch.argmax(PRUNE).item())

    # q axis: q_ratio = bits/16
    quant_axis = _unique_int_axis(spec.q_bits)
    QBITS = torch.tensor(quant_axis, device=device, dtype=torch.int64)    # original order [Q]
    Q = QBITS.numel()
    QRAT = QBITS.to(torch.float32).clamp(min=1.0) / 16.0
    q_vals_sorted, q_map_sorted_to_orig = torch.sort(QRAT)
    q_min = float(q_vals_sorted[0].item())
    q_max = float(q_vals_sorted[-1].item())
    dense_q_idx = int(torch.argmax(QBITS).item())

    # ---- controller knobs ----
    gamma_default = float(getattr(cfg, "dac_gamma", 0.35))
    g_k = float(dac_gamma_keep)  if dac_gamma_keep  is not None else float(getattr(cfg, "dac_gamma_keep",  gamma_default))
    g_p = float(dac_gamma_prune) if dac_gamma_prune is not None else float(getattr(cfg, "dac_gamma_prune", gamma_default))
    g_q = float(dac_gamma_quant) if dac_gamma_quant is not None else float(getattr(cfg, "dac_gamma_quant", gamma_default))

    ema = float(getattr(cfg, "dac_ema", 0.0))
    if dac_ema is not None:
        ema = float(dac_ema)

    emb_layer = model.get_input_embeddings()

    def _cosine_drift(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # returns in [0,1]: 0=no change, 1=opposite direction
        return 0.5 * (1.0 - F.cosine_similarity(a, b, dim=-1, eps=eps)).clamp(0.0, 1.0)

    def _choose_axis_with_bias_and_guards(
        c_req: torch.Tensor,                 # [N]
        signal01: torch.Tensor,              # [N] drift in [0,1]
        cum_val: torch.Tensor,               # [N] sum of chosen values so far
        cum_steps: torch.Tensor,             # [N] count of steps so far
        R: torch.Tensor,                     # [N] remaining steps incl current
        target_avg: float,
        vals_sorted: torch.Tensor,           # [A] sorted discrete values
        map_sorted_to_orig: torch.Tensor,    # [A] sorted->orig indices
        vmin: float,
        vmax: float,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministic hi/lo selection around c_req with drift bias + feasibility band clamp.
        Returns:
          chosen_orig_idx: [N]
          chosen_val:      [N]
        """
        A = vals_sorted.numel()
        c = c_req.clamp(vals_sorted[0], vals_sorted[-1])
        hi = torch.searchsorted(vals_sorted, c, right=False).clamp(0, A - 1)
        lo = (hi - 1).clamp(0, A - 1)

        lo_v = vals_sorted[lo]
        hi_v = vals_sorted[hi]
        denom = (hi_v - lo_v).clamp_min(1e-8)
        p_hi_base = torch.where((hi_v - lo_v) > 1e-8, (c - lo_v) / denom, torch.ones_like(c))

        # drift bias: high drift -> push toward hi
        delta = gamma * (signal01 - 0.5)
        p_hi_mod = (p_hi_base + delta).clamp(0.0, 1.0)

        choose_hi = (p_hi_mod >= 0.5) & (hi != lo)
        chosen_sorted = torch.where(choose_hi, hi, lo)
        chosen_val = vals_sorted[chosen_sorted]

        # feasibility guardrails
        R_post = (R - 1.0).clamp_min(0.0)
        target_total = target_avg * (cum_steps + 1.0 + R_post)
        allowed_min = (target_total - cum_val - vmax * R_post).clamp(vmin, vmax)
        allowed_max = (target_total - cum_val - vmin * R_post).clamp(vmin, vmax)

        lo_feas = torch.searchsorted(vals_sorted, allowed_min, right=False)
        hi_feas = torch.searchsorted(vals_sorted, allowed_max, right=True) - 1
        lo_feas = torch.minimum(lo_feas, hi_feas).clamp(0, A - 1)
        hi_feas = torch.maximum(lo_feas, hi_feas).clamp(0, A - 1)

        chosen_sorted = torch.maximum(chosen_sorted, lo_feas)
        chosen_sorted = torch.minimum(chosen_sorted, hi_feas)
        chosen_val = vals_sorted[chosen_sorted]

        chosen_orig = map_sorted_to_orig[chosen_sorted]
        return chosen_orig, chosen_val

    def _mix_report(vals_sorted_1d: torch.Tensor, target: float):
        vs = vals_sorted_1d.detach().cpu().tolist()
        if target <= vs[0]:
            return {"lo": vs[0], "hi": vs[0], "p_hi": 0.0}
        if target >= vs[-1]:
            return {"lo": vs[-1], "hi": vs[-1], "p_hi": 1.0}
        lo_i = max(i for i, v in enumerate(vs) if v <= target)
        hi_i = min(i for i, v in enumerate(vs) if v >= target)
        lo_v, hi_v = vs[lo_i], vs[hi_i]
        p_hi = 0.0 if hi_v == lo_v else (target - lo_v) / (hi_v - lo_v)
        return {"lo": lo_v, "hi": hi_v, "p_hi": float(p_hi)}

    # ---- accumulators ----
    total_nll = 0.0
    total_tok = 0
    eff_tok = 0

    total_keep_all = 0.0
    total_keep_eff = 0.0
    total_prune_eff = 0.0
    total_qratio_eff = 0.0

    action_hist = torch.zeros(K * P * Q, device=device)

    # ---- global model state hygiene ----
    enable_structured_controls(model)
    crit = str(getattr(cfg, "sparsity_criteria", "recency"))
    if crit == "relevancy":
        clear_relevancy_keep(model)
    # If you have Quest-specific persistent state, clear it here similarly:
    # if crit == "quest": clear_quest_keep(model)

    clear_structured_action(model)

    for batch in tqdm(dl, desc="eval DAC matched structured"):
        batch = batch.to(device)
        B, _ = batch.shape

        prefill_ids = batch[:, :context_len]
        step_inputs = batch[:, context_len : context_len + rollout_len]
        step_labels = batch[:, context_len + 1 : context_len + rollout_len + 1]

        # Dense prefill: need hidden state for drift reference
        out = model(
            input_ids=prefill_ids,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
        past_kv = detach_cache_to_tuple(out.past_key_values)
        kv_len = torch.full((B,), context_len + 1, device=device, dtype=torch.long)

        last_h_prev = out.hidden_states[-1][:, -1, :].detach().to(torch.float32)  # h_{prefill_last}
        last_h_prevprev = last_h_prev.clone()

        # embedding drift reference for t=0
        prev_tok = prefill_ids[:, -1]
        prev_emb = emb_layer(prev_tok).detach().to(torch.float32)

        # optional EMA over drift
        drift_ema = None
        if ema > 0.0:
            drift_ema = torch.full((B,), 0.5, device=device, dtype=torch.float32)

        # per-sequence trackers (κ)
        cum_keep = torch.zeros(B, device=device)
        cum_eff  = torch.zeros(B, device=device)

        # per-sequence trackers (ρ and q_ratio)
        cum_pru_steps = torch.zeros(B, device=device)
        cum_q_steps   = torch.zeros(B, device=device)
        cum_pru_val   = torch.zeros(B, device=device)
        cum_q_val     = torch.zeros(B, device=device)

        for t in range(rollout_len):
            cur = step_inputs[:, t]
            labels_t = step_labels[:, t]

            kv_before = kv_len
            eff_mask = (kv_before > thr)
            has_old = eff_mask.to(torch.float32)

            # ---- drift signal in [0,1] (causal) ----
            if t == 0:
                cur_emb = emb_layer(cur).detach().to(torch.float32)
                drift_now = _cosine_drift(prev_emb, cur_emb)
            else:
                drift_now = _cosine_drift(last_h_prev, last_h_prevprev)

            if drift_ema is not None:
                drift_ema = ema * drift_ema + (1.0 - ema) * drift_now
                signal01 = drift_ema
            else:
                signal01 = drift_now

            # default actions: dense
            a_k = torch.full((B,), dense_k_idx, device=device, dtype=torch.long)
            a_p = torch.full((B,), dense_p_idx, device=device, dtype=torch.long)
            a_q = torch.full((B,), dense_q_idx, device=device, dtype=torch.long)

            # remaining effective steps per sequence (for κ, and for struct when struct_on_non_eff=False)
            T_rem = rollout_len - t
            neff_rem = (thr - kv_before + 1).clamp_min(0)
            R_eff_all = (T_rem - neff_rem).clamp_min(0).to(torch.float32)  # [B]

            # ---- κ decision on effective tokens ----
            if eff_mask.any():
                idx_eff = torch.nonzero(eff_mask, as_tuple=False).squeeze(-1)
                R_eff = R_eff_all[idx_eff].clamp_min(1.0)

                c_req_k = (target_keep_effective * (cum_eff[idx_eff] + R_eff) - cum_keep[idx_eff]) / R_eff
                c_req_k = c_req_k.clamp(k_min, k_max)

                chosen_k_idx, _ = _choose_axis_with_bias_and_guards(
                    c_req=c_req_k,
                    signal01=signal01[idx_eff],
                    cum_val=cum_keep[idx_eff],
                    cum_steps=cum_eff[idx_eff],
                    R=R_eff,
                    target_avg=float(target_keep_effective),
                    vals_sorted=k_vals_sorted,
                    map_sorted_to_orig=k_map_sorted_to_orig,
                    vmin=k_min,
                    vmax=k_max,
                    gamma=g_k,
                )
                a_k[idx_eff] = chosen_k_idx

            # ---- structured axes (ρ, q_ratio) decision ----
            struct_mask = eff_mask if not struct_on_non_eff else torch.ones_like(eff_mask)
            if struct_mask.any():
                idx_struct = torch.nonzero(struct_mask, as_tuple=False).squeeze(-1)

                if struct_on_non_eff:
                    R_struct_all = torch.full((B,), float(rollout_len - t), device=device, dtype=torch.float32)
                else:
                    R_struct_all = R_eff_all

                R_vec = R_struct_all[idx_struct].clamp_min(1.0)

                # prune keep (ρ)
                c_req_p = (target_prune_keep * (cum_pru_steps[idx_struct] + R_vec) - cum_pru_val[idx_struct]) / R_vec
                c_req_p = c_req_p.clamp(p_min, p_max)

                chosen_p_idx, _ = _choose_axis_with_bias_and_guards(
                    c_req=c_req_p,
                    signal01=signal01[idx_struct],
                    cum_val=cum_pru_val[idx_struct],
                    cum_steps=cum_pru_steps[idx_struct],
                    R=R_vec,
                    target_avg=float(target_prune_keep),
                    vals_sorted=p_vals_sorted,
                    map_sorted_to_orig=p_map_sorted_to_orig,
                    vmin=p_min,
                    vmax=p_max,
                    gamma=g_p,
                )
                a_p[idx_struct] = chosen_p_idx

                # quant ratio (q_ratio)
                c_req_q = (target_quant_ratio * (cum_q_steps[idx_struct] + R_vec) - cum_q_val[idx_struct]) / R_vec
                c_req_q = c_req_q.clamp(q_min, q_max)

                chosen_q_idx, _ = _choose_axis_with_bias_and_guards(
                    c_req=c_req_q,
                    signal01=signal01[idx_struct],
                    cum_val=cum_q_val[idx_struct],
                    cum_steps=cum_q_steps[idx_struct],
                    R=R_vec,
                    target_avg=float(target_quant_ratio),
                    vals_sorted=q_vals_sorted,
                    map_sorted_to_orig=q_map_sorted_to_orig,
                    vmin=q_min,
                    vmax=q_max,
                    gamma=g_q,
                )
                a_q[idx_struct] = chosen_q_idx

            # materialize chosen values
            kappa_now  = KEEP[a_k]          # [B]
            prune_now  = PRUNE[a_p]         # [B]
            qbits_now  = QBITS[a_q]         # [B]
            qratio_now = QRAT[a_q]          # [B]

            # histogram over (k,p,q)
            flat_idx = a_k * (P * Q) + a_p * Q + a_q
            action_hist.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))

            # ---- decode step grouped by (prune, qbits) ----
            pq = torch.stack([prune_now, qbits_now.to(torch.float32)], dim=-1)
            uniq, inv = torch.unique(pq, dim=0, return_inverse=True)

            logits_step = None
            new_cache = None
            new_state = torch.empty_like(last_h_prev)  # [B, D]
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
                    logits_step = torch.empty((B, out_g.logits.size(-1)), device=device, dtype=out_g.logits.dtype)
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

            # ---- loss + stats ----
            nll_t = F.cross_entropy(logits_step, labels_t, reduction="none")
            total_nll += nll_t.sum().item()
            total_tok += B

            eff_tok += int(has_old.sum().item())
            total_keep_all += kappa_now.sum().item()
            total_keep_eff += (kappa_now * has_old).sum().item()

            gate = has_old if not struct_on_non_eff else torch.ones_like(has_old)
            total_prune_eff  += (prune_now  * gate).sum().item()
            total_qratio_eff += (qratio_now * gate).sum().item()

            # ---- update trackers ----
            cum_eff  = cum_eff  + has_old
            cum_keep = cum_keep + has_old * kappa_now

            cum_pru_steps = cum_pru_steps + gate
            cum_q_steps   = cum_q_steps   + gate
            cum_pru_val   = cum_pru_val   + gate * prune_now
            cum_q_val     = cum_q_val     + gate * qratio_now

            # ---- advance drift state ----
            last_h_prevprev = last_h_prev
            last_h_prev = last_h

    clear_structured_action(model)

    ppl = math.exp(total_nll / max(1, total_tok))
    avg_keep_all = total_keep_all / max(1, total_tok)
    avg_keep_eff = (total_keep_eff / max(1, eff_tok)) if eff_tok > 0 else 0.0
    denom_struct = (eff_tok if not struct_on_non_eff else total_tok)
    avg_prune_keep  = total_prune_eff / max(1, denom_struct)
    avg_quant_ratio = total_qratio_eff / max(1, denom_struct)

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
        "mix_keep":  _mix_report(k_vals_sorted, float(target_keep_effective)),
        "mix_prune": _mix_report(p_vals_sorted, float(target_prune_keep)),
        "mix_quant": _mix_report(q_vals_sorted, float(target_quant_ratio)),
        "dac": {
            "gamma_keep": float(g_k),
            "gamma_prune": float(g_p),
            "gamma_quant": float(g_q),
            "ema": float(ema),
        },
        "criteria": str(getattr(cfg, "sparsity_criteria", "recency")),
    }
