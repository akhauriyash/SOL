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


def _expand_to_batch(values: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Repeat or trim `values` so that it has length `batch_size`."""
    flat = values.reshape(-1)
    if flat.numel() == batch_size:
        return flat
    if flat.numel() == 1:
        return flat.expand(batch_size)
    if flat.numel() > batch_size:
        return flat[:batch_size]
    reps = (batch_size + flat.numel() - 1) // flat.numel()
    return flat.repeat(reps)[:batch_size]


def build_recency_mask_2d(past_kv_lens: torch.Tensor,
                          keep_fracs: torch.Tensor,
                          Ts: int, Tw: int, device, dtype) -> torch.Tensor:
    """
    Return key mask of shape [B, K], with 1=keep, 0=mask.
    K = kv_len = (past length + current).
    """
    B = past_kv_lens.size(0)
    max_K = int(past_kv_lens.max().item())
    mask2d = torch.zeros((B, max_K), device=device, dtype=torch.int64)

    for b in range(B):
        K = int(past_kv_lens[b].item())
        if K <= Ts + Tw + 1:
            mask2d[b, :K] = 1
            continue
        num_old = K - (Ts + Tw + 1)
        Ke = math.ceil(float(keep_fracs[b].item()) * num_old)
        start = max(Ts, K - (Tw + 1 + Ke))
        mask2d[b, 0:Ts] = 1
        mask2d[b, start:K] = 1

    return mask2d.to(dtype=torch.float32)


def build_recency_bias_4d(past_kv_lens: torch.Tensor,
                          keep_fracs: torch.Tensor,
                          Ts: int, Tw: int, device, dtype) -> torch.Tensor:
    # 1) Build your existing 2-D keep mask [B, K] with 1=keep, 0=mask
    keep2d = build_recency_mask_2d(past_kv_lens, keep_fracs, Ts, Tw, device, dtype=torch.float32)
    # 2) Convert to additive bias of shape [B, 1, 1, K]
    neg_inf = torch.finfo(dtype).min
    bias4d = (1.0 - keep2d).unsqueeze(1).unsqueeze(1) * neg_inf
    return bias4d.to(dtype)



def nanmax(tensor, dim=None, keepdim=False):
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    return output

def nanmin(tensor, dim=None, keepdim=False):
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)
    return output


def _build_quest_mask(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    token_budgets: torch.Tensor,
    page_size: int,
):
    # HF LLaMA repeat_kv gives [B, H_kv * groups == H, K, Dh]
    from transformers.models.llama import modeling_llama as llama_mod

    key_states = llama_mod.repeat_kv(key, module.num_key_value_groups)  # [B, H, K, Dh]

    B, H, Q, Dh = query.shape
    K = key_states.shape[-2]
    device = query.device
    q_dtype = query.dtype

    budgets = torch.as_tensor(token_budgets, device=device, dtype=torch.int64)
    budgets = _expand_to_batch(budgets, B)             # [B]
    budgets = budgets.clamp(min=0, max=K)              # out-of-place clamp while it's still dense
    budgets = budgets.view(B, 1, 1).expand(B, H, Q)    # broadcast to [B,H,Q] (no more in-place ops on this!)

    if attention_mask is not None:
        allowed = attention_mask[:, :, :, :K].eq(0)  # [B, ?, Q, K]
        if allowed.size(1) != 1:                     # normalize to [B,1,Q,K] so we control expansion
            allowed = allowed[:, :1]
    else:
        allowed = torch.ones((B, 1, Q, K), dtype=torch.bool, device=device)

    available_tokens = allowed.sum(dim=-1)  # [B,1,Q]
    if torch.all(available_tokens == 0):
        return None

    budgets = torch.minimum(budgets, available_tokens.to(budgets.dtype))
    budgets = torch.where(available_tokens > 0, budgets.clamp(min=1), torch.zeros_like(budgets))

    # If we're effectively keeping everything allowed, let eager attn run dense
    if torch.all(budgets == available_tokens.to(budgets.dtype)):
        return None

    page_size = max(int(page_size), 1)
    K_eff = ((K + page_size - 1) // page_size) * page_size
    if K_eff != K:
        pad_k = K_eff - K
        pad_keys = torch.full((B, H, pad_k, Dh), float("nan"), device=device, dtype=key_states.dtype)
        keys_padded = torch.cat([key_states, pad_keys], dim=-2)  # [B,H,K_eff,Dh]

        pad_allowed = torch.zeros((B, 1, Q, pad_k), dtype=torch.bool, device=device)
        allowed_padded = torch.cat([allowed, pad_allowed], dim=-1)  # [B,1,Q,K_eff]
    else:
        keys_padded = key_states
        allowed_padded = allowed

    P = K_eff // page_size
    keys_pages = keys_padded.view(B, H, P, page_size, Dh)  # [B,H,P,S,Dh]

    page_max_pos = nanmax(keys_pages, dim=-2).values  # [B,H,P,Dh]
    page_min     = nanmin(keys_pages, dim=-2).values  # [B,H,P,Dh]
    page_max_neg = -page_min                                # [B,H,P,Dh]

    q_abs   = query.abs().to(q_dtype)                       # [B,H,Q,Dh]
    pos_sel = (query >= 0).to(q_abs.dtype)
    q_pos   = q_abs * pos_sel
    q_neg   = q_abs - q_pos

    # [B,H,Q,Dh] @ [B,H,Dh,P] -> [B,H,Q,P]
    page_scores = (
        torch.matmul(q_pos.to(page_max_pos.dtype), page_max_pos.transpose(-1, -2)) +
        torch.matmul(q_neg.to(page_max_neg.dtype), page_max_neg.transpose(-1, -2))
    ).to(q_dtype)

    # Allowed pages (do not expand to H yet)
    allowed_pages = allowed_padded.view(B, 1, Q, P, page_size).any(dim=-1)  # [B,1,Q,P]
    neg_inf = torch.finfo(page_scores.dtype).min
    page_scores = torch.where(allowed_pages, page_scores, torch.full_like(page_scores, neg_inf))

    # Page budgets
    page_budgets = torch.div(budgets + page_size - 1, page_size, rounding_mode="floor")  # [B,H,Q]
    available_pages = allowed_pages.view(B, 1, Q, P).sum(dim=-1)                          # [B,1,Q]
    page_budgets = torch.minimum(page_budgets, available_pages.to(page_budgets.dtype))    # [B,H,Q]

    if torch.all(page_budgets == available_pages.to(page_budgets.dtype)):
        return None

    # Top-k over pages (flatten (B,H,Q))
    page_scores_flat   = page_scores.reshape(-1, P)    # [(B*H*Q), P]
    page_budgets_flat  = page_budgets.reshape(-1)      # [(B*H*Q)]
    max_pages = int(page_budgets_flat.max().item())

    if max_pages == 0:
        keep_pages_flat = torch.zeros_like(page_scores_flat, dtype=torch.bool)
    else:
        topk_vals, _ = torch.topk(page_scores_flat, k=max_pages, dim=-1)
        row_idx = torch.arange(page_scores_flat.size(0), device=device)
        gather_idx = torch.clamp(page_budgets_flat - 1, min=0)
        thresholds = torch.where(
            page_budgets_flat > 0,
            topk_vals[row_idx, gather_idx],
            torch.full_like(gather_idx, float("inf"), dtype=page_scores_flat.dtype),
        )
        keep_pages_flat = page_scores_flat >= thresholds.unsqueeze(-1)

    allowed_pages_flat = allowed_pages.expand(B, H, Q, P).reshape(-1, P)
    keep_pages_flat &= allowed_pages_flat

    keep_pages = keep_pages_flat.view(B, H, Q, P)  # [B,H,Q,P]

    token_keep = keep_pages.unsqueeze(-1).expand(-1, -1, -1, -1, page_size)
    token_keep = token_keep.reshape(B, H, Q, -1)[..., :K]           # [B,H,Q,K]

    allowed_K = allowed[..., :K]                                    # [B,1,Q,K]

    selected = (token_keep & allowed_K).sum(dim=-1)                 # [B,H,Q]
    fallback = (selected == 0) & (available_tokens > 0)             # [B,H,Q] via broadcast

    if fallback.any():
        rev_allowed  = torch.flip(allowed_K, dims=[-1]).to(torch.int64)  # [B,1,Q,K]
        last_offsets = torch.argmax(rev_allowed, dim=-1)                  # [B,1,Q]
        last_indices = (K - 1) - last_offsets                             # [B,1,Q]
        last_one_hot = F.one_hot(last_indices, num_classes=K).to(torch.bool)  # [B,1,Q,K]
        scatter = last_one_hot.expand(B, H, Q, K) & allowed_K.expand(B, H, Q, K)
        token_keep = torch.where(fallback.unsqueeze(-1), token_keep | scatter, token_keep)

    mask = torch.zeros((B, H, Q, K), dtype=q_dtype, device=device)
    mask = mask.masked_fill(~(token_keep & allowed_K), torch.finfo(q_dtype).min)
    return mask

def enable_quest_attention(model, page_size: int = 16) -> None:
    from transformers.models.llama import modeling_llama as llama_mod

    if not hasattr(llama_mod, "_quest_prev_eager_attention_forward"):
        orig_forward = llama_mod.eager_attention_forward

        def eager_attention_forward_with_quest(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout: float = 0.0,
            **kwargs,
        ):
            enabled = bool(getattr(module, "_quest_enabled", False))
            budgets = getattr(module, "_quest_token_budgets", None)
            local_page = getattr(module, "_quest_page_size", page_size)
            if (not enabled) or (budgets is None):
                return orig_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask,
                    scaling,
                    dropout=dropout,
                    **kwargs,
                )

            quest_mask = _build_quest_mask(
                module,
                query,
                key,
                attention_mask,
                budgets,
                local_page,
            )

            if quest_mask is not None:
                if attention_mask is not None:
                    quest_mask = quest_mask.to(attention_mask.dtype)
                if attention_mask is None:
                    new_attention_mask = quest_mask
                else:
                    tgt_len = attention_mask.size(-1)
                    if quest_mask.size(-1) != tgt_len:
                        pad = torch.zeros(
                            (*quest_mask.shape[:-1], tgt_len),
                            dtype=quest_mask.dtype,
                            device=quest_mask.device,
                        )
                        pad[..., : quest_mask.size(-1)] = quest_mask
                        quest_mask = pad
                    new_attention_mask = attention_mask + quest_mask
            else:
                new_attention_mask = attention_mask

            return orig_forward(
                module,
                query,
                key,
                value,
                new_attention_mask,
                scaling,
                dropout=dropout,
                **kwargs,
            )

        llama_mod._quest_prev_eager_attention_forward = orig_forward
        llama_mod.eager_attention_forward = eager_attention_forward_with_quest

    LlamaAttention = llama_mod.LlamaAttention

    modules = []
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            module._quest_enabled = True
            module._quest_page_size = page_size
            module._quest_token_budgets = None
            modules.append(module)

    model._quest_attention_modules = modules
    if not modules:
        raise ValueError("Quest attention is only supported for LLaMA attention modules.")


def set_quest_token_budgets(
    model, keep_fracs: torch.Tensor, past_kv_lens: torch.Tensor
) -> None:
    if not hasattr(model, "_quest_attention_modules"):
        raise ValueError("Quest attention has not been enabled for this model")

    keep_vals = torch.as_tensor(keep_fracs, dtype=torch.float32)
    kv_vals = torch.as_tensor(past_kv_lens, dtype=torch.float32)
    batch = int(kv_vals.numel())

    keep_vals = _expand_to_batch(keep_vals, batch)
    kv_vals = kv_vals.reshape(-1)

    if torch.all((keep_vals >= 0.0) & (keep_vals <= 1.0 + 1e-6)):
        budgets = torch.ceil(keep_vals * kv_vals).to(torch.int64)
    else:
        budgets = keep_vals.round().to(torch.int64)

    max_tokens = kv_vals.to(torch.int64)
    budgets = torch.clamp(budgets, min=0)
    budgets = torch.minimum(budgets, max_tokens)

    for module in model._quest_attention_modules:
        module._quest_token_budgets = budgets.clone()


def clear_quest_token_budgets(model) -> None:
    if not hasattr(model, "_quest_attention_modules"):
        return
    for module in model._quest_attention_modules:
        module._quest_token_budgets = None

def enable_relevancy_attention(model, tier: str = "per_head", cfg=None) -> None:
    """
    Monkey-patch LLaMA eager attention to support relevancy-driven masking while
    always delegating the actual attention computation to the original eager path.

    Behavior:
      * If relevancy is disabled or no keep fractions are set, we call the original eager attention unchanged.
      * Otherwise, we:
          1) compute attn logits (once) to decide which keys to keep,
          2) turn that decision into an additive mask (0 where kept, -inf where dropped),
          3) add it to the caller's attention_mask, and
          4) call the original eager attention with the modified attention_mask.

    This is compatible with llama-3.2-1b (HF `transformers`) and avoids re-implementing attention math.
    """
    import torch
    from transformers.models.llama import modeling_llama as llama_mod

    # Only patch once
    if not hasattr(llama_mod, "_sparsity_original_eager_attention_forward"):
        orig_forward = llama_mod.eager_attention_forward

        def eager_attention_forward_with_relevancy(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout: float = 0.0,
            **kwargs,
        ):
            # Gate: only apply when enabled and keep fractions are provided
            keep_fracs = getattr(module, "_sparsity_keep", None)
            enabled = bool(getattr(module, "_sparsity_relevancy_enabled", False))
            if (not enabled) or (keep_fracs is None):
                return orig_forward(
                    module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
                )

            # Config for "always keep" buckets
            tier_now = getattr(module, "_sparsity_tier", "per_head")
            Ts = int(getattr(module, "_sparsity_Ts", 0))
            Tw = int(getattr(module, "_sparsity_Tw", 0))

            # --- Build logits ONLY to decide the mask; real attention stays in orig_forward ---
            # Repeat KV so logits match the query head dimension
            key_states = llama_mod.repeat_kv(key, module.num_key_value_groups)   # [B, H, K, Dh]
            attn_logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling  # [B, H, Q, K]

            # Respect caller's mask while *deciding* relevance: disallowed spots never get selected
            if attention_mask is not None:
                attn_logits = attn_logits + attention_mask[:, :, :, : key_states.shape[-2]]

            B, H, Q, K = attn_logits.shape
            device, dtype = attn_logits.device, attn_logits.dtype

            # Robustly convert keep_fracs to tensor
            keep_fracs = torch.as_tensor(keep_fracs, device=device, dtype=dtype).view(-1)

            # Derive "valid" indices: where the caller allows attention; else fall back to strict causal
            if attention_mask is not None:
                valid = attention_mask[:, :, :, :K].eq(0)  # additive mask: 0 = allowed; -inf = blocked
            else:
                valid = torch.ones((B, 1, Q, K), dtype=torch.bool, device=device)

            # Always-keep regions: prefix Ts and last Tw valid positions
            always_keep = torch.zeros((B, 1, Q, K), dtype=torch.bool, device=device)
            if Ts > 0:
                always_keep[..., : min(Ts, K)] = True
            if Tw >= 0:
                valid_cum = valid.cumsum(dim=-1)
                last_valid_count = valid_cum.amax(dim=-1, keepdim=True)   # [B,1,Q,1]
                keep_recent = valid & (valid_cum >= (last_valid_count - (Tw + 1)).clamp_min(1))
                always_keep |= keep_recent

            eligible = valid & (~always_keep)
            neg_inf = torch.finfo(dtype).min
            masked_logits = torch.where(eligible, attn_logits, neg_inf)
            # after masked_logits is computed
            B, H, Q, K = masked_logits.shape
            neg_inf = torch.finfo(masked_logits.dtype).min

            eligible_row = (masked_logits > neg_inf/2).sum(dim=-1)                      # [B,H,Q]
            kf = torch.as_tensor(keep_fracs, device=device, dtype=dtype).view(-1)
            if kf.numel() == 1:
                kf = kf.expand(B)
            elif kf.numel() != B:
                # best-effort match to this call's batch size
                if kf.numel() > B:
                    kf = kf[:B]
                else:
                    reps = (B + kf.numel() - 1) // kf.numel()
                    kf = kf.repeat(reps)[:B]
            kf = kf.view(B, 1, 1)
            keep_counts = torch.ceil(kf * eligible_row.to(dtype)).to(torch.int64)
            keep_mask = _topk_mask(masked_logits, keep_counts, B, H, Q, K)  # [B,H,Q,K] bool

            # Merge with always-keep; result = True where we KEEP attention
            full_keep = keep_mask | always_keep.expand(B, H, Q, K)

            # Convert to an additive mask: 0 where kept, -inf where dropped. Allow per-head masking.
            extra_mask = torch.zeros((B, H, Q, K), dtype=dtype, device=device)
            extra_mask = extra_mask.masked_fill(~full_keep, neg_inf)

            # Combine with caller's attention_mask. If their mask is longer than K, pad ours to match.
            if attention_mask is not None:
                tgt_len = attention_mask.size(-1)
                if tgt_len != K:
                    pad_extra = torch.zeros((B, H, Q, tgt_len), dtype=dtype, device=device)
                    pad_extra[..., :K] = extra_mask
                    new_attention_mask = attention_mask + pad_extra  # broadcasts 1->H on head dim
                else:
                    new_attention_mask = attention_mask + extra_mask
            else:
                new_attention_mask = extra_mask  # [B, H, Q, K]

            # Delegate to the original implementation (it will recompute logits and do all the work)
            return orig_forward(
                module, query, key, value, new_attention_mask, scaling, dropout=dropout, **kwargs
            )

        # Install the patch
        llama_mod._sparsity_original_eager_attention_forward = orig_forward
        llama_mod.eager_attention_forward = eager_attention_forward_with_relevancy

    # --- Attach control attrs to each LlamaAttention module ---
    from transformers.models.llama import modeling_llama as llama_mod  # ensure same module object
    LlamaAttention = llama_mod.LlamaAttention
    Ts = int(getattr(cfg, "Ts", 0)) if cfg is not None else 0
    Tw = int(getattr(cfg, "Tw", 0)) if cfg is not None else 0

    modules = []
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            module._sparsity_relevancy_enabled = True
            module._sparsity_keep = None  # set at runtime to a scalar/tensor (per-batch) of keep fractions
            module._sparsity_tier = tier
            module._sparsity_Ts = Ts
            module._sparsity_Tw = Tw
            modules.append(module)

    model._relevancy_attention_modules = modules
    if not modules:
        raise ValueError("Relevancy sparsity is only supported for LLaMA attention modules.")




def set_relevancy_keep(model, keep_fracs: torch.Tensor, tier: Optional[str] = None) -> None:
    if not hasattr(model, "_relevancy_attention_modules"):
        raise ValueError("Relevancy attention has not been enabled for this model")
    keep_flat = keep_fracs.detach().view(-1)
    if tier is not None:
        for module in model._relevancy_attention_modules:
            module._sparsity_tier = tier
    for module in model._relevancy_attention_modules:
        module._sparsity_keep = keep_flat


def clear_relevancy_keep(model) -> None:
    if not hasattr(model, "_relevancy_attention_modules"):
        return
    for module in model._relevancy_attention_modules:
        module._sparsity_keep = None


def build_sparse_attention_bias(
    *,
    model,
    past_kv_lens: torch.Tensor,
    keep_fracs: torch.Tensor,
    Ts: int,
    Tw: int,
    device,
    dtype,
    criteria: str,
    tier: str,
):
    if criteria == "recency":
        clear_relevancy_keep(model)
        clear_quest_token_budgets(model)
        return build_recency_bias_4d(past_kv_lens, keep_fracs, Ts, Tw, device, dtype)
    if criteria == "quest":
        clear_relevancy_keep(model)
        set_quest_token_budgets(model, keep_fracs, past_kv_lens)
        return None
    if criteria != "relevancy":
        clear_quest_token_budgets(model)
        raise ValueError(f"Unknown sparsity criteria: {criteria}")
    clear_quest_token_budgets(model)
    set_relevancy_keep(model, keep_fracs, tier=tier)
    return None


def _expand_to_batch(values: torch.Tensor, batch_size: int) -> torch.Tensor:
    flat = values.reshape(-1)
    if flat.numel() == batch_size:
        return flat
    if flat.numel() == 1:
        return flat.expand(batch_size)
    if flat.numel() > batch_size:
        return flat[:batch_size]
    reps = (batch_size + flat.numel() - 1) // flat.numel()
    return flat.repeat(reps)[:batch_size]

def _fake_quantize_bits(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16:
        return x

    x_f = x.to(torch.float32)
    qmax = (1 << (bits - 1)) - 1

    if x_f.dim() == 3:
        ax = x_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        ax = x_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)

    scale = ax / float(qmax)
    x_int = torch.round(x_f / scale).clamp_(-qmax, qmax)
    x_q = x_int * scale

    return x_q.to(x.dtype)


def _fake_quantize_batched(x: torch.Tensor, bits_vec: torch.Tensor) -> torch.Tensor:
    """Apply _fake_quantize_bits per-example for possibly mixed bitwidths."""
    if bits_vec.numel() == 1:
        return _fake_quantize_bits(x, int(bits_vec.item()))
    # x is [B, T, D] or [B, D]; select by batch row
    B = x.size(0)
    bits = _expand_to_batch(bits_vec.reshape(-1), B).to(x.device)
    out = torch.empty_like(x)
    uniq = torch.unique(bits)
    for b in uniq.tolist():
        sel = (bits == int(b))
        if x.dim() == 3:
            out[sel, :, :] = _fake_quantize_bits(x[sel, :, :], int(b))
        else:
            out[sel, :] = _fake_quantize_bits(x[sel, :], int(b))
    return out


def enable_structured_controls(model, *, apply_to_attention_input: bool = True) -> None:
    """
    Monkey-patch LLaMA decoder layer & MLP to support:
      - channel pruning at the layer *input* (so KV reflects it)
      - activation fake-quant inside MLP
    Works with meta-llama/Llama-3.2-1B (HF).
    """
    from transformers.models.llama import modeling_llama as llama_mod

    # Patch decoder layer once
    if not hasattr(llama_mod, "_struct_prev_decoderlayer_forward"):
        orig_layer_forward = llama_mod.LlamaDecoderLayer.forward

        def _apply_channel_prune(hidden_states: torch.Tensor, keep_fracs_b: torch.Tensor) -> torch.Tensor:
            if keep_fracs_b is None:
                return hidden_states
            B, T, D = hidden_states.shape
            keep = _expand_to_batch(keep_fracs_b.reshape(-1), B).to(hidden_states)
            if torch.allclose(keep, torch.ones_like(keep)):
                return hidden_states
            k_counts = (keep * D).ceil().clamp_(1, D).to(torch.int64)  # [B]
            # score per channel = max |activation| across tokens in this call
            scores = hidden_states.abs().amax(dim=1)  # [B, D]
            # scores = hidden_states.pow(2).mean(dim=1).sqrt() 
            max_k = int(k_counts.max().clamp(1, D).item())
            topk_vals, _ = torch.topk(scores, k=max_k, dim=-1)
            row = torch.arange(B, device=hidden_states.device)
            thr = topk_vals[row, (k_counts - 1).clamp_min(0)]    # [B]
            mask = (scores >= thr.unsqueeze(-1)).to(hidden_states.dtype).unsqueeze(1)  # [B,1,D]
            return hidden_states * mask  # broadcast across T

        def decoder_forward_with_struct(module, *args, **kwargs):
            # No pruning at decoder input: Q/K/V stay dense
            return orig_layer_forward(module, *args, **kwargs)

        llama_mod._struct_prev_decoderlayer_forward = orig_layer_forward
        llama_mod.LlamaDecoderLayer.forward = decoder_forward_with_struct

    # Patch MLP once
    if not hasattr(llama_mod, "_struct_prev_mlp_forward"):
        orig_mlp_forward = llama_mod.LlamaMLP.forward
        
        def mlp_forward_with_struct(module, hidden_states, *args, **kwargs):
            # 1) Optional pruning on MLP input ONLY (doesn't touch attention)
            keep = getattr(module, "_struct_prune_keep", None)
            if keep is not None:
                hidden_states = _apply_channel_prune(hidden_states, keep)

            # 2) Optional quantization: now applied ONLY to the MLP output
            bits = getattr(module, "_struct_quant_bits", None)
            if bits is None:
                # No quantization -> just run the original MLP
                return orig_mlp_forward(module, hidden_states, *args, **kwargs)

            # Run the original MLP in full precision
            out = orig_mlp_forward(module, hidden_states, *args, **kwargs)

            # Fake-quantize the MLP *output* (residual stream)
            out_q = _fake_quantize_batched(out, bits)
            return out_q

        llama_mod._struct_prev_mlp_forward = orig_mlp_forward
        llama_mod.LlamaMLP.forward = mlp_forward_with_struct

    # Attach control attrs to each relevant module
    from transformers.models.llama import modeling_llama as llama_mod  # same module object
    for module in model.modules():
        if isinstance(module, llama_mod.LlamaDecoderLayer):
            module._struct_prune_keep = None  # torch.Tensor per-batch or None
        if isinstance(module, llama_mod.LlamaMLP):
            module._struct_quant_bits = None  # torch.Tensor[int] per-batch or None
            module._struct_prune_keep = None  # NEW: MLP-level prune control
    model._structured_controls_enabled = True


def set_structured_action(model, prune_keep, quant_bits) -> None:
    """Set per-batch controls for the next forward call."""
    if not getattr(model, "_structured_controls_enabled", False):
        enable_structured_controls(model)
    # Accept Python scalars, 0-D, or 1-D tensors; normalize to tensors on model device
    dev = next(model.parameters()).device
    prune_keep = torch.as_tensor(prune_keep, device=dev, dtype=torch.float32)
    quant_bits = torch.as_tensor(quant_bits, device=dev, dtype=torch.int64)
    # Save tensors; decoder/MLP wrappers will expand per forward call
    for module in model.modules():
        cls = module.__class__.__name__
        if hasattr(module, "_struct_prune_keep"):
            module._struct_prune_keep = prune_keep
        if hasattr(module, "_struct_quant_bits"):
            module._struct_quant_bits = quant_bits


def clear_structured_action(model) -> None:
    """Reset to dense/no-quant behavior."""
    if not getattr(model, "_structured_controls_enabled", False):
        return
    for module in model.modules():
        if hasattr(module, "_struct_prune_keep"):
            module._struct_prune_keep = None
        if hasattr(module, "_struct_quant_bits"):
            module._struct_quant_bits = None


def _build_relevancy_mask(attn_probs: torch.Tensor, keep_fracs: torch.Tensor, tier: str) -> torch.Tensor:
    B, H, Q, K = attn_probs.shape
    keep_vals = keep_fracs.clamp(0.0, 1.0)
    keep_counts = torch.clamp((keep_vals * K).ceil().to(dtype=torch.int64), min=1, max=K)
    if tier == "per_layer":
        base = attn_probs.mean(dim=1, keepdim=True)
        mask = _topk_mask(base, keep_counts, B, 1, Q, K)
        return mask.expand(B, H, Q, K).to(attn_probs.dtype)

    # Default: per-head relevancy
    mask = _topk_mask(attn_probs, keep_counts, B, H, Q, K)
    return mask.to(attn_probs.dtype)

def _topk_mask(values, keep_counts, B, H, Q, K, valid_mask=None):
    flat_vals = values.reshape(B*H*Q, K)
    if valid_mask is not None:
        vm = valid_mask.reshape(B*H*Q, K)
        neg_inf = torch.finfo(values.dtype).min
        flat_vals = torch.where(vm, flat_vals, torch.tensor(neg_inf, dtype=values.dtype, device=values.device))

    # keep_counts: either [B] or [B,H,Q]
    if keep_counts.dim() == 1:
        expanded = keep_counts.view(B,1,1).expand(B,H,Q).reshape(-1)
    else:
        expanded = keep_counts.reshape(-1)

    max_k = int(expanded.max().clamp(0, K).item())
    if max_k == 0:
        return torch.zeros_like(values, dtype=torch.bool)

    topk_vals, _ = torch.topk(flat_vals, k=max_k, dim=-1)
    row_idx = torch.arange(flat_vals.size(0), device=flat_vals.device)
    gather_idx = (expanded - 1).clamp(min=0)
    thresholds = topk_vals[row_idx, gather_idx]
    return (flat_vals >= thresholds.unsqueeze(-1)).reshape(B, H, Q, K)