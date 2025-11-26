import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

import numpy as np
from tqdm import tqdm
from itertools import islice
from .cache import detach_cache_to_tuple
from .masks import build_sparse_attention_bias

@torch.no_grad()
def probe_losses_with_lookahead(
    model,
    past_kv_live,           # cache *before* step t
    kv_len,                 # [B], includes current position (past+1)
    cur,                    # [B] token at step t
    step_inputs,            # [B, rollout_len] (we'll index future tokens from here)
    step_labels,            # [B, rollout_len] (aligned labels)
    t,                      # current step index in rollout
    keep_fracs,             # tuple/list of κ values
    Ts, Tw,
    device,
    dtype,
    horizon=2,              # lookahead H (1 = your current behavior)
    future_mask_rule="dense",  # "dense" or "repeat" (use same κ for future steps)
    metric: str = "lookahead_ce",  # "lookahead_ce" (label CE) or "dense_kl" (KL to dense logits)
    policy_for_future: Optional[nn.Module] = None,  # frozen Δ-policy, returns signed_rel_delta_hat
    emb_layer: Optional[nn.Module] = None,          # LM embedding layer
    eps_rel: float = 0.02,                          # feasibility threshold on Δ̂_rel
    sparsity_criteria: str = "recency",
    relevancy_tier: str = "per_head",
):
    """
    Returns: losses_a [B, A] where each entry aggregates per-step losses over the horizon:
      - metric="lookahead_ce": sum of label CE over the next H steps under the action dynamics.
      - metric="dense_kl":    sum over h of KL( p_dense_{t+h} || p_action_{t+h} ), comparing
                              the dense rollout logits to those under the action dynamics.
    """
    B = cur.size(0)
    A = len(keep_fracs)
    # Precompute κ tensors once
    kap_list = [torch.full((B,), float(k), device=device, dtype=torch.float32) for k in keep_fracs]
    KEEP = torch.tensor(keep_fracs, device=device, dtype=torch.float32) if A > 0 else None
    thr = Ts + Tw + 1
    losses = []
    # tokens/labels for future steps within horizon
    max_h = min(horizon, step_inputs.size(1) - t)  # how many steps we still have (including current)
    # For convenience, grab a slice [t : t+max_h]
    fut_tokens = step_inputs[:, t : t + max_h]      # first column is cur for h==0 if you prefer; we'll handle explicitly
    fut_labels = step_labels[:, t : t + max_h]      # labels for t..t+max_h-1

    # Optionally precompute the dense (κ=1) rollout logprobs once for KL supervision.
    dense_logprobs = None
    if metric == "dense_kl":
        cache_d = detach_cache_to_tuple(past_kv_live)
        kv_len_d = kv_len.clone()
        ones = torch.ones_like(kv_len_d, dtype=torch.float32, device=device)
        dense_logprobs = []
        # h = 0
        pos_ids_d0 = (kv_len_d - 1).clamp_min(0).unsqueeze(1)
        bias_d0 = build_sparse_attention_bias(
            model=model,
            past_kv_lens=kv_len_d,
            keep_fracs=ones,
            Ts=Ts,
            Tw=Tw,
            device=device,
            dtype=dtype,
            criteria=sparsity_criteria,
            tier=relevancy_tier,
        )
        out_d0 = model(
            input_ids=cur.unsqueeze(1),
            use_cache=True,
            past_key_values=cache_d,
            position_ids=pos_ids_d0,
            attention_mask=bias_d0,
            return_dict=True,
        )
        dense_logprobs.append(F.log_softmax(out_d0.logits[:, -1, :], dim=-1))
        cache_d = out_d0.past_key_values
        kv_len_d = kv_len_d + 1
        for h in range(1, max_h):
            tok_h = fut_tokens[:, h]
            pos_ids_dh = (kv_len_d - 1).clamp_min(0).unsqueeze(1)
            bias_dh = build_sparse_attention_bias(
                model=model,
                past_kv_lens=kv_len_d,
                keep_fracs=ones,
                Ts=Ts,
                Tw=Tw,
                device=device,
                dtype=dtype,
                criteria=sparsity_criteria,
                tier=relevancy_tier,
            )
            out_dh = model(input_ids=tok_h.unsqueeze(1), use_cache=True, past_key_values=cache_d,
                           position_ids=pos_ids_dh, attention_mask=bias_dh, return_dict=True)
            dense_logprobs.append(F.log_softmax(out_dh.logits[:, -1, :], dim=-1))
            cache_d = out_dh.past_key_values
            kv_len_d = kv_len_d + 1

    for a_idx, kap in enumerate(kap_list):
        cache_a = detach_cache_to_tuple(past_kv_live)
        kv_len_a = kv_len.clone()
        loss_sum = torch.zeros((B,), device=device)

        pos_ids = (kv_len_a - 1).clamp_min(0).unsqueeze(1)
        bias0 = build_sparse_attention_bias(
            model=model,
            past_kv_lens=kv_len_a,
            keep_fracs=kap,
            Ts=Ts,
            Tw=Tw,
            device=device,
            dtype=dtype,
            criteria=sparsity_criteria,
            tier=relevancy_tier,
        )
        out0 = model(
            input_ids=cur.unsqueeze(1),
            use_cache=True,
            past_key_values=cache_a,
            position_ids=pos_ids,
            attention_mask=bias0,
            return_dict=True,
            output_hidden_states=(future_mask_rule == "policy"),
        )
        logits0 = out0.logits[:, -1, :]
        state_a = out0.hidden_states[-1][:, -1, :] if future_mask_rule == "policy" else None
        if metric == "dense_kl":
            # KL( p_dense || p_action )
            logprobs_a0 = F.log_softmax(logits0, dim=-1)
            kl0 = torch.sum(torch.exp(dense_logprobs[0]) * (dense_logprobs[0] - logprobs_a0), dim=-1)
            loss0 = kl0
        else:  # "lookahead_ce"
            loss0 = F.cross_entropy(logits0, fut_labels[:, 0], reduction="none")

        # accumulate
        loss_sum = loss_sum + loss0
        cache_a = out0.past_key_values
        kv_len_a = kv_len_a + 1

        # ---- h = 1..max_h-1 : evaluate short lookahead (dense by default) ----
        for h in range(1, max_h):
            tok_h = fut_tokens[:, h]       # next input
            lab_h = fut_labels[:, h]

            if future_mask_rule == "repeat":
                kap_h = kap
            elif future_mask_rule == "policy" and (policy_for_future is not None) and (emb_layer is not None) and (KEEP is not None):
                # --- Policy-aware: pick κ using frozen Δ-policy with feasibility-then-cost rule ---
                # features: [state, tok_embed(next), frac_old]
                tok_embed = emb_layer(tok_h)                                # [B, H_emb]
                num_old = (kv_len_a - thr).clamp_min(0).to(torch.float32)   # [B]
                frac_old = (num_old / (num_old + 1e-6)).unsqueeze(1)        # [B,1]
                feats = torch.cat([state_a.to(torch.float32),
                                   tok_embed.to(torch.float32),
                                   frac_old], dim=-1)                       # [B, 2H+1]
                with torch.inference_mode():
                    signed_rel_delta_hat = policy_for_future(feats).to(torch.float32) # [B, A]
                # compute per-action cost over "old" region
                kappas_row = KEEP.view(1, -1)                                         # [1, A]
                Ke_a = torch.ceil(num_old.unsqueeze(-1) * kappas_row)                 # [B, A]
                costs_a = torch.where(
                    num_old.unsqueeze(-1) > 0,
                    Ke_a / (num_old.unsqueeze(-1) + 1e-6),
                    torch.zeros_like(Ke_a),
                )

                feasible = (signed_rel_delta_hat <= eps_rel)                           # [B, A]
                feasible_any = feasible.any(dim=-1)                                    # [B]
                BIG = 1e9
                a_feasible = costs_a.masked_fill(~feasible, BIG).argmin(dim=-1)        # [B]
                a_best = signed_rel_delta_hat.argmin(dim=-1)                            # [B]
                a_next = torch.where(feasible_any, a_feasible, a_best)                 # [B]
                kap_h = KEEP[a_next]                                                   # [B]
            else:
                # default: "dense"
                kap_h = torch.ones_like(kv_len_a, dtype=torch.float32, device=device)

            pos_ids = (kv_len_a - 1).clamp_min(0).unsqueeze(1)
            bias_h = build_sparse_attention_bias(
                model=model,
                past_kv_lens=kv_len_a,
                keep_fracs=kap_h,
                Ts=Ts,
                Tw=Tw,
                device=device,
                dtype=dtype,
                criteria=sparsity_criteria,
                tier=relevancy_tier,
            )

            out_h = model(
                input_ids=tok_h.unsqueeze(1),
                use_cache=True,
                past_key_values=cache_a,
                position_ids=pos_ids,
                attention_mask=bias_h,
                return_dict=True,
                output_hidden_states=(future_mask_rule == "policy"),
            )
            logits_h = out_h.logits[:, -1, :]
            if future_mask_rule == "policy":
                state_a = out_h.hidden_states[-1][:, -1, :]
            if metric == "dense_kl":
                logprobs_ah = F.log_softmax(logits_h, dim=-1)
                # KL( p_dense_h || p_action_h )
                kl_h = torch.sum(torch.exp(dense_logprobs[h]) * (dense_logprobs[h] - logprobs_ah), dim=-1)
                loss_h = kl_h
            else:
                loss_h = F.cross_entropy(logits_h, lab_h, reduction="none")
            loss_sum = loss_sum + loss_h

            cache_a = out_h.past_key_values
            kv_len_a = kv_len_a + 1
        losses.append(loss_sum)

    return torch.stack(losses, dim=-1) / float(max_h)