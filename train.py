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
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from transformers.cache_utils import DynamicCache
import datetime
import numpy as np
import wandb
from tqdm import tqdm
from itertools import islice
from argparse import ArgumentParser
import json
import sys
from utils.seeds import set_seed, snapshot_code
from utils.model import load_lm_and_tokenizer, unwrap
from utils.data import make_dataloader, limited_dl
from utils.masks import (
    build_sparse_attention_bias,
    clear_relevancy_keep,
    enable_structured_controls,
    set_structured_action,
    clear_structured_action,
)
from utils.actions import build_action_spec
from utils.cache import detach_cache_to_tuple
from utils.probe import probe_losses_with_lookahead
from utils.eval import (
    evaluate_stateful_policy_rollout,
)
from utils.eval_baselines import (
    evaluate_dense_full,
    evaluate_fixed_matched_keep,
)
from utils.masks import enable_quest_attention, enable_relevancy_attention

from utils.config import Config, apply_cfg_overrides_from_file
from predictor import RecurrentActorCriticPolicy
from utils.actions import build_action_spec

import torch.backends.cuda as sdp
sdp.enable_flash_sdp(False)
sdp.enable_math_sdp(False)
sdp.enable_mem_efficient_sdp(True)  # SDPA only
from transformers.cache_utils import DynamicCache


def ensure_dynamic_cache(past_kv):
    """Convert legacy tuple cache -> DynamicCache if needed."""
    if isinstance(past_kv, tuple):
        return DynamicCache.from_legacy_cache(past_kv)
    return past_kv  # already a Cache

def repeat_legacy_cache_k(past_kv_tuple, K: int):
    """Repeat a tuple-of-(k,v) cache Kx along batch dim, return DynamicCache."""
    tup_rep = tuple(
        (k.repeat_interleave(K, dim=0), v.repeat_interleave(K, dim=0))
        for (k, v) in past_kv_tuple
    )
    return DynamicCache.from_legacy_cache(tup_rep)

def _dense_idx(fracs):
    """Robustly find the 'dense' action index even if 1.0 isn't present."""
    try:
        return fracs.index(1.0)
    except ValueError:
        return int(np.argmax(np.array(fracs)))

def _repeat_cache_k(past_kv, K: int):
    """Repeat a tuple-of-(k,v) cache K times along batch dim."""
    return tuple((k.repeat_interleave(K, dim=0), v.repeat_interleave(K, dim=0)) for (k, v) in past_kv)

def train_one_epoch_grpo(tok,
                         model,
                         policy,
                         cfg,
                         dl: DataLoader,
                         epoch=0,
                         run=None,
                         val_dl=None,
                         eval_every=100,
                         global_step_state=None,
                         optimizer=None,
                         ckpt_dir: Optional[str] = None,
                         best_state: Optional[dict] = None,
                         meta: Optional[dict] = None):
    device = next(policy.parameters()).device
    is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    policy_mod = unwrap(policy)
    if optimizer is None:
        optimizer = torch.optim.AdamW(unwrap(policy).parameters(), lr=cfg.lr, fused=True)
    policy.train()

    K = int(getattr(cfg, "grpo_rollouts_per_input", 16))
    grpo_level = str(getattr(cfg, "grpo_level", "process")).lower()
    grpo_norm  = str(getattr(cfg, "grpo_norm", "center")).lower()
    adv_whiten_global = bool(getattr(cfg, "adv_whiten_global", True))
    ppo_clip = float(getattr(cfg, "ppo_clip", 0.2))
    mb_size = int(getattr(cfg, "ppo_minibatch_size", 512))
    target_N = int(getattr(cfg, "ppo_target_batch_size", 2048))
    entropy_coef = float(getattr(cfg, "entropy_coef", 1e-3))
    pi_temperature = float(getattr(cfg, "pi_temperature", 0.7))
    grad_accum_steps = int(getattr(cfg, "grad_accum_steps", 1))

    kl_pi_ref_coef = float(getattr(cfg, "kl_pi_ref_coef", 0.0))
    if kl_pi_ref_coef > 0.0 and "_policy_ref" not in (global_step_state or {}):
        policy_ref = copy.deepcopy(unwrap(policy)).eval()
        for p in policy_ref.parameters():
            p.requires_grad_(False)
        if global_step_state is None:
            global_step_state = {}
        global_step_state["_policy_ref"] = policy_ref
        
    policy_ref = (global_step_state or {}).get("_policy_ref", None)


    # === Multi-budget targets (token keep, prune keep, quant bits) ===
    # These are *default* targets; per‑rollout targets are sampled around them below.
    C_tok_default   = float(getattr(cfg, "C_target", getattr(cfg, "C_target_token", getattr(cfg, "keep_target", 1.0))))
    C_pru_default   = float(getattr(cfg, "C_target_prune", 0.70))
    C_qbits_default = float(getattr(cfg, "C_target_quant_bits", 8.0))
    C_q_default     = C_qbits_default / 16.0

    # Fixed trade‑off weights between accuracy (delta CE) and compute costs.
    alpha_tok   = float(getattr(cfg, "alpha_tok", getattr(cfg, "cost_tradeoff_alpha", 1.0)))
    alpha_pru   = float(getattr(cfg, "alpha_prune", alpha_tok))
    alpha_quant = float(getattr(cfg, "alpha_quant", alpha_pru))

    if global_step_state is None:
        global_step_state = {"micro": 0, "update": 0}
    else:
        global_step_state.setdefault("micro", 0)
        global_step_state.setdefault("update", 0)
    action_spec = build_action_spec(
        keep_fracs=cfg.keep_fracs,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
    )
    A = action_spec.n_actions
    P_MAX = float(max(action_spec.prune_keep)) if len(action_spec.prune_keep) > 0 else 1.0
    
    has_keep_dof  = len(set(action_spec.token_keep)) > 1
    has_prune_dof = len(set(action_spec.prune_keep)) > 1
    has_quant_dof = len(set(action_spec.q_bits)) > 1

    # If a dimension has no DOF, its cost weight is effectively zero.
    if not has_keep_dof:
        alpha_tok = 0.0
    if not has_prune_dof:
        alpha_pru = 0.0
    if not has_quant_dof:
        alpha_quant = 0.0
    KEEP_TOKEN = torch.tensor(action_spec.token_keep, device=device, dtype=torch.float32)
    KEEP_PRUNE = torch.tensor(action_spec.prune_keep, device=device, dtype=torch.float32)
    Q_BITS     = torch.tensor(action_spec.q_bits,     device=device, dtype=torch.int64)

    pol_n = unwrap(policy).n_actions if hasattr(unwrap(policy), "n_actions") else unwrap(policy).pi.out_features
    if pol_n != A:
        raise ValueError(
            f"Policy action dim ({pol_n}) != composite action size ({A}). "
            f"Recreate policy with n_actions={A}."
        )

    enable_structured_controls(model)
    crit = str(getattr(cfg, "sparsity_criteria", "recency"))
    if crit == "quest":
        enable_quest_attention(model, page_size=getattr(cfg, "quest_page_size", 16))
    elif crit == "relevancy":
        enable_relevancy_attention(model, tier=getattr(cfg, "relevancy_tier", "per_head"), cfg=cfg)
    thr = cfg.Ts + cfg.Tw + 1
    emb_layer = unwrap(model).get_input_embeddings()

    logs = {
        "avg_reward": 0.0,
        "avg_r_task": 0.0,
        "avg_penalty": 0.0,
        "avg_abs_kl": 0.0,
        "avg_abs_penalty": 0.0,
        "avg_penalty_over_task_abs": 0.0,
        "avg_cost_eff": 0.0,
        "avg_ppl_approx": 0.0,
        "avg_keep_chosen": 0.0,
    }
    steps_done = 0

    agg_h_seq, agg_e_seq, agg_scalars_seq, agg_prev_actions_seq = [], [], [], []
    agg_actions_seq, agg_logp_old_seq, agg_adv_seq = [], [], []
    agg_count = 0
    agg_cost_eff_sum = 0.0
    agg_eff_tok      = 0.0
    agg_prune_sum    = 0.0
    agg_qratio_sum   = 0.0
    agg_tok_steps    = 0.0
 

    for batch in tqdm(dl, desc="Training (GRPO)...", disable=not is_main):
        batch = batch.to(device)
        B, total_len = batch.shape
        assert total_len == cfg.context_len + cfg.rollout_len + 1

        # --- Sample per‑sequence target budgets for this batch (normalized to [0,1]) ---
        def _sample_budget_1d(name: str, default: float) -> torch.Tensor:
            """
            Sample a target budget for each sequence i in the batch.
            Supports:
              - cfg.<name>_list : discrete set of values to sample from
              - cfg.<name>_min / cfg.<name>_max : uniform range
            Falls back to a constant `default`.
            """
            choices = getattr(cfg, f"{name}_list", None)
            if choices is not None:
                vals = torch.as_tensor(choices, dtype=torch.float32, device=device)
                idx = torch.randint(low=0, high=vals.numel(), size=(B,), device=device)
                return vals[idx]
            lo = getattr(cfg, f"{name}_min", None)
            hi = getattr(cfg, f"{name}_max", None)
            if (lo is not None) and (hi is not None):
                return torch.empty(B, device=device).uniform_(float(lo), float(hi))
            return torch.full((B,), float(default), device=device)

        # Token‑level keep target is already in [0,1].
        C_tok_target_B = _sample_budget_1d("budget_tok", C_tok_default)          # [B]
        # Prune budget is expressed in normalized prune_keep ρ in [0,1].
        C_pru_target_B = _sample_budget_1d("budget_prune", C_pru_default)        # [B]
        # Quantization budget uses qratio = bits/16 in [0,1].
        C_qratio_target_B = _sample_budget_1d("budget_q_ratio", C_q_default)     # [B]

        if global_step_state.get("save_stride") in (None, 0):
            try:
                num_batches = len(dl)
            except Exception:
                num_batches = 1
            T = int(cfg.rollout_len)
            adv_per_batch = T * B * K
            target_N = int(getattr(cfg, "ppo_target_batch_size", 2048))
            mb_size = int(getattr(cfg, "ppo_minibatch_size", 512))
            grad_acc = max(1, int(getattr(cfg, "grad_accum_steps", 1)))
            cycles = max(1, math.ceil((num_batches * adv_per_batch) / max(1, target_N)))
            updates_per_cycle = max(1, math.ceil(math.ceil(max(1, target_N) / max(1, mb_size)) / grad_acc))
            total_updates_est = cycles * updates_per_cycle
            global_step_state["save_stride"] = max(1, total_updates_est // 10)
        prefill_ids = batch[:, :cfg.context_len]
        step_inputs = batch[:, cfg.context_len : cfg.context_len + cfg.rollout_len]
        step_labels = batch[:, cfg.context_len + 1 : cfg.context_len + cfg.rollout_len + 1]

        if getattr(cfg, "sparsity_criteria", "recency") == "relevancy":
            clear_relevancy_keep(model)
        clear_structured_action(model)
        with torch.inference_mode():
            outputs = model(
                input_ids=prefill_ids,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
        past_kv_ref = detach_cache_to_tuple(outputs.past_key_values)
        past_kv_pol = detach_cache_to_tuple(outputs.past_key_values)
        kv_len_ref = torch.full((B,), cfg.context_len + 1, device=device, dtype=torch.long)
        kv_len_pol = torch.full((B,), cfg.context_len + 1, device=device, dtype=torch.long)
        state_pol = outputs.hidden_states[-1][:, -1, :].detach()

        dense_logprobs = []
        with torch.inference_mode():
            ones = torch.ones_like(kv_len_ref, dtype=torch.float32, device=device)
            for t in range(cfg.rollout_len):
                cur = step_inputs[:, t]
                labels_t = step_labels[:, t]
                pos_ids = (kv_len_ref - 1).clamp_min(0).unsqueeze(1)
                bias_ref = build_sparse_attention_bias(
                    model=model,
                    past_kv_lens=kv_len_ref,
                    keep_fracs=ones,
                    Ts=cfg.Ts,
                    Tw=cfg.Tw,
                    device=device,
                    dtype=model.dtype,
                    criteria=getattr(cfg, "sparsity_criteria", "recency"),
                    tier=getattr(cfg, "relevancy_tier", "per_head"),
                )
                out_ref = model(
                    input_ids=cur.unsqueeze(1),
                    use_cache=True,
                    past_key_values=past_kv_ref,
                    position_ids=pos_ids,
                    attention_mask=bias_ref,
                    return_dict=True,
                )
                logits_ref = out_ref.logits[:, -1, :]
                dense_logprobs.append(F.log_softmax(logits_ref, dim=-1))  # [B, V]
                past_kv_ref = out_ref.past_key_values
                kv_len_ref = kv_len_ref + 1
        dense_logprobs = torch.stack(dense_logprobs, dim=0)  # [W, B, V]

        if isinstance(past_kv_pol, tuple):
            past_kv_pol = repeat_legacy_cache_k(past_kv_pol, K)
        else:
            past_kv_pol = repeat_legacy_cache_k(past_kv_pol.to_legacy_cache(), K)
        past_kv_ref = ensure_dynamic_cache(past_kv_ref)

        kv_len_pol = kv_len_pol.repeat_interleave(K, dim=0)      # [B*K]
        state_pol = state_pol.repeat_interleave(K, dim=0)        # [B*K]
        Bk = B * K
        # Expand per‑sequence targets across K rollouts per input.
        C_tok_target_BK    = C_tok_target_B.repeat_interleave(K, dim=0)      # [BK]
        C_pru_target_BK    = C_pru_target_B.repeat_interleave(K, dim=0)      # [BK]
        C_qratio_target_BK = C_qratio_target_B.repeat_interleave(K, dim=0)   # [BK]

        # Buffers (time-major)
        h_seq_buf, e_seq_buf, scalars_seq_buf = [], [], []
        prev_actions_seq_buf, actions_seq_buf = [], []
        logp_old_seq_buf = []
        rewards_buf, r_task_buf = [], []
        keep_buf, eff_mask_buf = [], []
        prune_keep_buf, qratio_buf = [], []

        # Logging accumulators
        nll_sum = torch.tensor(0.0, device=device)
        tok_count = 0
        action_counts = torch.zeros(A, device=device)
        keep_chosen_sum = torch.tensor(0.0, device=device)
        cost_eff_sum = torch.tensor(0.0, device=device)
        eff_tok = torch.tensor(0.0, device=device)
        # Recurrent policy state (BK) & per-episode budget trackers
        # pi_state = policy.init_state(Bk, device=device)
        pi_state = policy_mod.init_state(Bk, device=device)
        prev_action_ids = pi_state.last_action.clone()
        cum_keep   = torch.zeros(Bk, device=device)  # sum_t eff_t * kappa_t
        cum_eff    = torch.zeros(Bk, device=device)  # sum_t eff_t
        cum_prune  = torch.zeros(Bk, device=device)  # sum_t eff_t * (prune_t / P_MAX)
        cum_qratio = torch.zeros(Bk, device=device)  # sum_t eff_t * q_ratio_t

        for t in range(cfg.rollout_len):
            # Tile input token & label across K
            cur = step_inputs[:, t].repeat_interleave(K, dim=0)      # [BK]
            labels_t = step_labels[:, t].repeat_interleave(K, dim=0) # [BK]

            eff_mask = (kv_len_pol > thr)                             # [BK]
            tok_embed = emb_layer(cur).detach()

            # --- Structured scalar features (8D) ---
            # 0: t_frac            \in [0, 1]               (progress through rollout)
            # 1: eff_flag          \in {0, 1}               (controllable vs warmup)
            # 2: C_tok_target      \in [0, 1]               (token keep budget)
            # 3: C_pru_target      \in [0, 1]               (normalized prune budget)
            # 4: C_qratio_target   \in [0, 1]               (quant bits / 16)
            # 5: dev_keep          = mean_keep_prev   - C_tok_target
            # 6: dev_prune         = mean_prune_prev  - C_pru_target
            # 7: dev_qratio        = mean_qratio_prev - C_qratio_target
            t_frac = torch.full(
                (Bk, 1),
                (t + 1) / float(cfg.rollout_len),
                device=device,
                dtype=torch.float32,
            )
            eff_flag = eff_mask.float().unsqueeze(1)

            # Running means up to *previous* step
            mean_keep_prev = torch.where(
                cum_eff > 0,
                cum_keep / cum_eff,
                C_tok_target_BK,
            )
            mean_prune_prev = torch.where(
                cum_eff > 0,
                cum_prune / cum_eff,
                C_pru_target_BK,            
            )
            mean_qratio_prev = torch.where(
                cum_eff > 0,
                cum_qratio / cum_eff,
                C_qratio_target_BK,
            )

            dev_keep   = mean_keep_prev   - C_tok_target_BK
            dev_prune  = mean_prune_prev  - C_pru_target_BK
            dev_qratio = mean_qratio_prev - C_qratio_target_BK

            # [BK, 8] : [t_frac, eff_flag, C_tok, C_pru, C_q, dev_keep, dev_prune, dev_q]
            scalars = torch.cat(
                [
                    t_frac,
                    eff_flag,
                    C_tok_target_BK.unsqueeze(1),
                    C_pru_target_BK.unsqueeze(1),
                    C_qratio_target_BK.unsqueeze(1),
                    dev_keep.unsqueeze(1),
                    dev_prune.unsqueeze(1),
                    dev_qratio.unsqueeze(1),
                ],
                dim=-1,
            )
            h_prev_for_policy = state_pol.to(torch.float32)
            # logits, _value_unused, pi_state = policy.step(
            #     # h_lm=state_pol.to(torch.float32),
            #     h_lm=h_prev_for_policy,
            #     e_tok=tok_embed.to(torch.float32),
            #     scalars=scalars,
            #     state=pi_state,
            #     temperature=pi_temperature,
            # )

            # Rollout collection does not need gradients through the policy
            with torch.no_grad():
                logits, _value_unused, pi_state = policy_mod.step(
                    h_lm=h_prev_for_policy,
                    e_tok=tok_embed.to(torch.float32),
                    scalars=scalars,
                    state=pi_state,
                    temperature=pi_temperature,
                )
            dist_pi = torch.distributions.Categorical(logits=logits)
            action = dist_pi.sample()                      # [BK]
            logp_old = dist_pi.log_prob(action)            # [BK]
            pi_state.last_action = action.detach()

            kappa_now = KEEP_TOKEN[action]                 # [BK]
            prune_now = KEEP_PRUNE[action]                 # [BK]
            qbits_now = Q_BITS[action]                     # [BK]

            eff = eff_mask.float()
            p_dense = float(max(action_spec.prune_keep))
            q_dense = int(max(action_spec.q_bits))

            kappa_now = torch.where(eff_mask, kappa_now, torch.ones_like(kappa_now))
            prune_now = torch.where(eff_mask, prune_now, torch.full_like(prune_now, p_dense))
            qbits_now = torch.where(eff_mask, qbits_now, torch.full_like(qbits_now, q_dense))
            q_ratio   = qbits_now.to(torch.float32) / 16.0
            set_structured_action(model, prune_now, qbits_now)
            q_ratio   = qbits_now.to(torch.float32) / 16.0          # [BK]
            pos_ids = (kv_len_pol - 1).clamp_min(0).unsqueeze(1)
            attn_bias = build_sparse_attention_bias(
                model=model,
                past_kv_lens=kv_len_pol,
                keep_fracs=kappa_now,
                Ts=cfg.Ts,
                Tw=cfg.Tw,
                device=device,
                dtype=model.dtype,
                criteria=getattr(cfg, "sparsity_criteria", "recency"),
                tier=getattr(cfg, "relevancy_tier", "per_head"),
            )

            with torch.inference_mode():
                out_step = model(
                    input_ids=cur.unsqueeze(1),
                    use_cache=True,
                    past_key_values=past_kv_pol,
                    position_ids=pos_ids,
                    attention_mask=attn_bias,
                    return_dict=True,
                    output_hidden_states=True,
                )
                logits_sparse = out_step.logits[:, -1, :]
                past_kv_pol = out_step.past_key_values
                kv_len_pol = kv_len_pol + 1
                state_pol = out_step.hidden_states[-1][:, -1, :].detach()

            clear_structured_action(model)
            logp_dense = dense_logprobs[t].float().repeat_interleave(K, dim=0)  # [BK, V]
            logp_sparse = F.log_softmax(logits_sparse.float(), dim=-1)          # [BK, V]

            kl_t = F.kl_div(logp_sparse, logp_dense, log_target=True, reduction='none').sum(dim=-1)  # [BK]
            ce_dense  = F.nll_loss(logp_dense,  labels_t, reduction='none')
            ce_sparse = F.nll_loss(logp_sparse, labels_t, reduction='none')
            delta_ce  = -ce_sparse
            w_kl = float(getattr(cfg, "task_w_kl", 0.0))
            r_task_t = (1.0 - w_kl) * delta_ce - w_kl * kl_t
            eff = eff_mask.float()                                     # [BK]
            cost_t_eff   = eff * kappa_now                             # [BK]

            # Episode-wise running sums (used for scalars on next step)
            cum_eff    = cum_eff    + eff
            cum_keep   = cum_keep   + cost_t_eff
            cum_prune  = cum_prune  + eff * (prune_now / P_MAX)
            cum_qratio = cum_qratio + eff * q_ratio

            mean_keep_so_far = torch.where(
                cum_eff > 0, cum_keep / cum_eff, torch.zeros_like(cum_eff)
            )
            # LM state for policy is the *previous* LM state (before taking this action)
            h_seq_buf.append(h_prev_for_policy)
            e_seq_buf.append(tok_embed.to(torch.float32))
            scalars_seq_buf.append(scalars)
            prev_actions_seq_buf.append(prev_action_ids)
            actions_seq_buf.append(action)
            logp_old_seq_buf.append(logp_old.detach())

            rewards_buf.append(r_task_t)
            r_task_buf.append(r_task_t)
            keep_buf.append(kappa_now)
            eff_mask_buf.append(eff_mask)
            prune_keep_buf.append(prune_now)
            qratio_buf.append(q_ratio)
            prev_action_ids = action.detach()

            nll_sum += F.cross_entropy(logits_sparse, labels_t, reduction="sum")
            tok_count += Bk
            action_counts.index_add_(0, action, torch.ones_like(action, dtype=torch.float32))
            keep_chosen_sum += kappa_now[eff_mask].sum()
            cost_eff_sum += cost_t_eff.sum()
            eff_tok += eff.sum()
            # Accumulate normalized prune ratio (0..1) for budget tracking
            agg_prune_sum  += float(((prune_now / P_MAX) * eff).sum().item())
            agg_qratio_sum += float((q_ratio   * eff).sum().item())
            agg_tok_steps  += float(eff.sum().item())

        h_seq = torch.stack(h_seq_buf, dim=0)                     # [W, BK, H]
        e_seq = torch.stack(e_seq_buf, dim=0)                     # [W, BK, E]
        scalars_seq = torch.stack(scalars_seq_buf, dim=0)         # [W, BK, 10]
        prev_actions_seq = torch.stack(prev_actions_seq_buf, dim=0)  # [W, BK]
        actions_seq = torch.stack(actions_seq_buf, dim=0)         # [W, BK]
        logp_old_seq = torch.stack(logp_old_seq_buf, dim=0)       # [W, BK]
        rewards = torch.stack(rewards_buf, dim=0)                 # [T, BK]
        r_task_all = torch.stack(r_task_buf, dim=0)               # [T, BK]
        keep_all = torch.stack(keep_buf, dim=0)                   # [W, BK]
        eff_all  = torch.stack(eff_mask_buf, dim=0).float()       # [W, BK]
        prune_all  = torch.stack(prune_keep_buf, dim=0)           # [W, BK]
        qratio_all = torch.stack(qratio_buf, dim=0)               # [W, BK]

        T, BK = rewards.shape
        assert BK == Bk

        def _grpo_adv(x_flat, level):
            if level == "process":
                x3 = x_flat.view(T, B, K)
                if grpo_norm == "zscore":
                    mu = x3.mean(dim=2, keepdim=True)
                    sd = x3.std(dim=2, unbiased=False, keepdim=True).clamp_min(1e-6)
                    A = (x3 - mu) / sd
                else:
                    A = x3 - x3.mean(dim=2, keepdim=True)
                return A.view(T, BK)
            else:
                X = x_flat.sum(dim=0).view(B, K)
                if grpo_norm == "zscore":
                    mu = X.mean(dim=1, keepdim=True)
                    sd = X.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
                    A_grp = (X - mu) / sd
                else:
                    A_grp = X - X.mean(dim=1, keepdim=True)
                return A_grp.view(1, BK).expand(T, BK)


        rag   = getattr(cfg, "reward_agg", None)
        gamma = float(getattr(cfg, "reward_gamma", 0.82))

        x_for_adv = r_task_all
        if rag in ("sum", "max") and 0.0 <= gamma < 1.0:
            T, BK = r_task_all.shape
            returns = torch.zeros_like(r_task_all)
            if rag == "sum":
                running = torch.zeros(BK, device=r_task_all.device, dtype=r_task_all.dtype)
                for t in range(T - 1, -1, -1):
                    running = r_task_all[t] + gamma * running
                    returns[t] = running
            else:  # rag == "max"
                running_best = torch.full((BK,), float("-inf"),
                                          device=r_task_all.device, dtype=r_task_all.dtype)
                for t in range(T - 1, -1, -1):
                    running_best = torch.maximum(r_task_all[t], gamma * running_best)
                    returns[t] = running_best

            if grpo_level == "outcome":
                x_for_adv = returns[0].unsqueeze(0).expand_as(r_task_all)  # shape [T, BK]
            else:
                x_for_adv = returns
                
        sum_eff_seq    = eff_all.sum(dim=0).clamp_min(1.0)               # [BK]
        mean_keep_seq  = (eff_all * keep_all).sum(dim=0) / sum_eff_seq   # [BK]
        prune_all_ratio = prune_all / P_MAX
        mean_prune_seq = (eff_all * prune_all_ratio).sum(dim=0) / sum_eff_seq   # [BK]
        mean_qratio_seq= (eff_all * qratio_all).sum(dim=0) / sum_eff_seq  # [BK]

        # --- Multi‑budget compute costs (all roughly in [0,1]) ---
        keep_gap   = mean_keep_seq  - C_tok_target_BK           # [BK]
        prune_gap  = mean_prune_seq - C_pru_target_BK           # [BK]
        qratio_gap = mean_qratio_seq - C_qratio_target_BK       # [BK]

        # For logging: batch-mean requested budgets and adherence
        mean_C_tok_target_batch    = float(C_tok_target_BK.mean().item())
        mean_C_pru_target_batch    = float(C_pru_target_BK.mean().item())
        mean_C_qratio_target_batch = float(C_qratio_target_BK.mean().item())

        avg_abs_keep_gap   = float(keep_gap.abs().mean().item())
        avg_abs_prune_gap  = float(prune_gap.abs().mean().item())
        avg_abs_qratio_gap = float(qratio_gap.abs().mean().item())
            
        tolerance = 0.02

        def huber_sq(gap, tol):
            dev = gap.abs() - tol
            dev = dev.clamp_min(0.0)
            return dev * dev

        cost_tok_seq    = huber_sq(keep_gap,   tolerance)
        cost_pru_seq    = huber_sq(prune_gap,  tolerance)
        cost_qratio_seq = huber_sq(qratio_gap, tolerance)

        # Broadcast per‑trajectory costs across rollout time to match x_for_adv shape.
        cost_tok = cost_tok_seq.view(1, -1).expand_as(x_for_adv)         # [T, BK]
        cost_pru = cost_pru_seq.view(1, -1).expand_as(x_for_adv)         # [T, BK]
        cost_q   = cost_qratio_seq.view(1, -1).expand_as(x_for_adv)      # [T, BK]

        # Total reward for GRPO: accuracy (delta CE / KL‑mix) minus compute costs.
        computational_component = (
            alpha_tok   * cost_tok
            + alpha_pru * cost_pru
            + alpha_quant * cost_q
        )
        r_total = x_for_adv - computational_component

        # Compute advantage of the combined objective and (optionally) whiten it.
        adv = _grpo_adv(r_total, level=grpo_level)
        if adv_whiten_global:
            adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
        agg_h_seq.append(h_seq)
        agg_e_seq.append(e_seq)
        agg_scalars_seq.append(scalars_seq)
        agg_prev_actions_seq.append(prev_actions_seq)
        agg_actions_seq.append(actions_seq)
        agg_logp_old_seq.append(logp_old_seq)
        agg_adv_seq.append(adv.detach())
        agg_count += adv.numel()

        agg_cost_eff_sum += float(cost_eff_sum.item())
        agg_eff_tok  += float(eff_tok.item())
        if agg_count >= target_N:
            # if (run is not None) and (val_dl is not None) and (global_step_state["update"] % eval_every == 0):
            if (eval_every is not None) and (eval_every > 0) and (run is not None) and (val_dl is not None) and (global_step_state["update"] % eval_every == 0):
                try:
                    # Allow eval_* to be missing or None; fall back to the training defaults.
                    _eval_C_tok   = getattr(cfg, "eval_C_tok", None)
                    _eval_C_pru   = getattr(cfg, "eval_C_pru", None)
                    _eval_C_qbits = getattr(cfg, "eval_C_qbits", None)

                    eval_C_tok   = C_tok_default   if _eval_C_tok   is None else float(_eval_C_tok)
                    eval_C_pru   = C_pru_default   if _eval_C_pru   is None else float(_eval_C_pru)
                    eval_C_qbits = C_qbits_default if _eval_C_qbits is None else float(_eval_C_qbits)
                    sparse_stats = evaluate_stateful_policy_rollout(
                        cfg, model, policy, val_dl,
                        Ts=cfg.Ts, Tw=cfg.Tw, keep_fracs=cfg.keep_fracs,
                        context_len=cfg.context_len, rollout_len=cfg.rollout_len,
                        device=cfg.device, greedy=True, temperature=1.0,
                        target_C_tok=eval_C_tok,
                        target_C_pru=eval_C_pru,
                        target_C_qbits=eval_C_qbits,
                    )
                    avg_prune_keep  = float(sparse_stats.get("avg_prune_keep", 0.0))
                    avg_quant_ratio = float(sparse_stats.get("avg_quant_ratio", 0.0))

                    # If there are no structural/pruning/quant DOFs, treat this as the
                    # single-budget regime and compare to a fixed matched baseline
                    # evaluated at the *actual* policy budgets.
                    if (not has_prune_dof) and (not has_quant_dof):
                        policy_keep_eff_actual   = float(sparse_stats["avg_keep_effective"])
                        policy_prune_keep_actual = avg_prune_keep
                        policy_quant_ratio_actual = avg_quant_ratio

                        fixed_matched = evaluate_fixed_matched_keep(
                            cfg,
                            model,
                            val_dl,
                            Ts=cfg.Ts,
                            Tw=cfg.Tw,
                            keep_fracs=tuple(cfg.keep_fracs),
                            prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
                            quant_choices=getattr(cfg, "quant_choices", ("q16",)),
                            target_keep_effective=policy_keep_eff_actual,
                            target_prune_keep=policy_prune_keep_actual,
                            target_quant_ratio=policy_quant_ratio_actual,
                            context_len=cfg.context_len,
                            rollout_len=cfg.rollout_len,
                            device=cfg.device,
                            struct_on_non_eff=False,
                        )
                        gap_nats = math.log(sparse_stats["ppl"]) - math.log(fixed_matched["ppl"])
                        gap_ratio = sparse_stats["ppl"] / fixed_matched["ppl"]
                        if run is not None:
                            run.log({
                                "special/gap_to_fixed_ln_ppl": gap_nats,
                                "special/avg_keep_effective": sparse_stats["avg_keep_effective"],
                                "special/gap_ratio_to_fixed": gap_ratio,
                                "special/fixed_ppl": fixed_matched["ppl"],
                                "special/sparse_ppl": sparse_stats["ppl"],
                                "special/avg_prune_keep": avg_prune_keep,
                                "special/avg_quant_ratio": avg_quant_ratio,
                                "update_step": global_step_state["update"],
                            })
                    else:
                        if run is not None:
                            run.log({
                                "eval/avg_ppl": sparse_stats["ppl"],
                                "eval/avg_keep_effective": sparse_stats["avg_keep_effective"],
                                "eval/avg_prune_keep": sparse_stats["avg_prune_keep"],
                                "eval/avg_quant_ratio": sparse_stats["avg_quant_ratio"],
                                "update_step": global_step_state["update"],
                            })
                except Exception as _e:
                    if is_main:
                        import traceback
                        traceback.print_exc()
                        print(f"[warn] eval (GRPO) failed with recurrent policy: {_e}")

                if is_main and ckpt_dir is not None:
                    print(f"Saving checkpoint to {ckpt_dir} ... at update {global_step_state['update']}")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "update_step": global_step_state["update"],
                            "policy_state_dict": unwrap(policy).state_dict(),
                            "cfg": asdict(cfg),
                            "meta": meta,
                            "global_step_state": copy.deepcopy(global_step_state),
                            "best_metric": best_state.get("best_metric", float("inf")) if best_state is not None else None,
                        },
                        os.path.join(ckpt_dir, "policy_latest.pt"),
                    )

            h_total       = torch.cat(agg_h_seq, dim=1)            # [T, sumB, H]
            e_total       = torch.cat(agg_e_seq, dim=1)            # [T, sumB, E]
            scalars_total = torch.cat(agg_scalars_seq, dim=1)      # [T, sumB, 10]
            prev_actions_total = torch.cat(agg_prev_actions_seq, dim=1)  # [T, sumB]
            actions_total  = torch.cat(agg_actions_seq, dim=1)     # [T, sumB]
            logp_old_total = torch.cat(agg_logp_old_seq, dim=1)    # [T, sumB]
            adv_total      = torch.cat(agg_adv_seq, dim=1)         # [T, sumB]

            T, Btot = actions_total.shape[0], actions_total.shape[1]
            mb_seqs = max(1, mb_size // max(1, T))
            idx_seq = torch.randperm(Btot, device=device)

            clip_fracs, approx_kls = [], []
            tbptt_k = int(getattr(cfg, "policy_tbptt_k", 0))

            grad_accum_steps = max(1, int(grad_accum_steps))
            if "_accum_i" not in global_step_state:
                global_step_state["_accum_i"] = 0
            if global_step_state["_accum_i"] % grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            num_mb = (Btot + mb_seqs - 1) // mb_seqs  # ceil
            for start in range(0, Btot, mb_seqs):
                mb_idx = idx_seq[start:start+mb_seqs]

                logits_seq, values_seq = policy(
                    h_total[:, mb_idx, :],
                    e_total[:, mb_idx, :],
                    scalars_total[:, mb_idx, :],
                    prev_actions_total[:, mb_idx],
                    tbptt_k=tbptt_k,
                    temperature=pi_temperature,
                )  # [T,Bmb,A], [T,Bmb]
                # logits_seq, _values_unused = policy.forward_sequence(
                #     h_seq=h_total[:, mb_idx, :],
                #     e_seq=e_total[:, mb_idx, :],
                #     scalars_seq=scalars_total[:, mb_idx, :],
                #     prev_actions_seq=prev_actions_total[:, mb_idx],
                #     tbptt_k=tbptt_k,
                #     temperature=pi_temperature,
                # )  # [T,Bmb,A], [T,Bmb]
                dist_new = torch.distributions.Categorical(
                    logits=logits_seq.reshape(T*mb_idx.numel(), -1)
                )
                actions_flat = actions_total[:, mb_idx].reshape(-1)
                logp_new = dist_new.log_prob(actions_flat)
                logp_old = logp_old_total[:, mb_idx].reshape(-1)
                ratio = torch.exp(logp_new - logp_old)

                adv_flat = adv_total[:, mb_idx].reshape(-1)
                clipped = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
                pg = torch.min(ratio * adv_flat, clipped * adv_flat)
                policy_loss = -pg.mean()

                entropy = dist_new.entropy().mean()

                loss = policy_loss - entropy_coef * entropy
                # Ensure critic params are "used" so grads are tensors (not None).
                # 0.0 => no training signal, but avoids DDP+wandb grad=None issues.
                loss = loss + 0.0 * values_seq.mean()
                # loss = policy_loss - entropy_coef * entropy
                ent_ratio = (entropy_coef * entropy).abs() / policy_loss.abs().clamp_min(1e-8)
                if kl_pi_ref_coef > 0.0 and policy_ref is not None:
                    with torch.no_grad():
                        logits_ref_seq, _ = policy_ref.forward_sequence(
                            h_seq=h_total[:, mb_idx, :],
                            e_seq=e_total[:, mb_idx, :],
                            scalars_seq=scalars_total[:, mb_idx, :],
                            prev_actions_seq=prev_actions_total[:, mb_idx],
                            tbptt_k=tbptt_k,
                            temperature=pi_temperature,
                        )
                    dist_ref = torch.distributions.Categorical(
                        logits=logits_ref_seq.reshape(T*mb_idx.numel(), -1)
                    )
                    kl_pi = torch.distributions.kl_divergence(dist_new, dist_ref).mean()
                    loss = loss + kl_pi_ref_coef * kl_pi

                (loss / grad_accum_steps).backward()
                global_step_state["_accum_i"] += 1

                if global_step_state["_accum_i"] % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(unwrap(policy).parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step_state["update"] += 1
                    stride = global_step_state.get("save_stride")
                    if is_main and ckpt_dir and stride:
                        upd = int(global_step_state["update"])
                        if upd % int(stride) == 0:
                            path = os.path.join(ckpt_dir, f"policy_{upd}.pt")
                            torch.save(
                                {
                                    "epoch": epoch + 1,
                                    "update_step": upd,
                                    "policy_state_dict": unwrap(policy).state_dict(),
                                    "cfg": asdict(cfg),
                                    "meta": meta,
                                    "global_step_state": copy.deepcopy(global_step_state),
                                },
                                path,
                            )
                            torch.save(
                                {
                                    "epoch": epoch + 1,
                                    "update_step": upd,
                                    "policy_state_dict": unwrap(policy).state_dict(),
                                    "cfg": asdict(cfg),
                                    "meta": meta,
                                    "global_step_state": copy.deepcopy(global_step_state),
                                },
                                os.path.join(ckpt_dir, "policy_latest.pt"),
                            )
                clip_fracs.append(((ratio > (1+ppo_clip)) | (ratio < (1-ppo_clip))).float().mean())
                approx_kls.append((logp_old - logp_new).mean())

            if run is not None:
                run.log({
                    "train/clip_frac": torch.stack(clip_fracs).mean().item(),
                    "train/approx_kl": torch.stack(approx_kls).mean().item(),
                    "train/ent_ratio": ent_ratio.item(),
                    "update_step": global_step_state["update"],
                })

            agg_h_seq, agg_e_seq, agg_scalars_seq, agg_prev_actions_seq = [], [], [], []
            agg_actions_seq, agg_logp_old_seq, agg_adv_seq = [], [], []
            agg_count = 0
            agg_cost_eff_sum = 0.0
            agg_eff_tok = 0.0
            agg_prune_sum    = 0.0
            agg_qratio_sum   = 0.0
            agg_tok_steps    = 0.0

        ppl_approx = math.exp((nll_sum / max(1, tok_count)).item()) if tok_count > 0 else 0.0
        keep_mean = (keep_chosen_sum / eff_tok.clamp_min(1.0)).item() if eff_tok.item() > 0 else 0.0
        mean_r = rewards.mean().item()  # same as r_task mean now
        mean_r_task = r_task_all.mean().item()
        abs_task = r_task_all.abs().mean().item()
        log_cost_eff = (cost_eff_sum / eff_tok.clamp_min(1.0)).item() if eff_tok.item() > 0 else float("nan")
        mean_r_total = r_total.mean().item()
        mean_x_for_adv = x_for_adv.mean().item()
        mean_comp = computational_component.mean().item()

        mean_abs_x_for_adv = x_for_adv.abs().mean().item()
        mean_abs_comp = computational_component.abs().mean().item()
        comp_over_task_abs = (
            mean_abs_comp / max(mean_abs_x_for_adv, 1e-8)
        ) if mean_abs_x_for_adv > 0 else float("inf")

        # For backward‑compatible logging names: "penalty" now means compute penalty.
        mean_penalty = mean_comp
        abs_penalty = mean_abs_comp
        ratio_abs = (abs_penalty / max(abs_task, 1e-8)) if abs_task > 0 else float("inf")

        mean_cost_tok = cost_tok.mean().item()
        mean_cost_pru = cost_pru.mean().item()
        mean_cost_q   = cost_q.mean().item()

        action_frac = {}
        denom = action_counts.sum().clamp_min(1.0)
        for i in range(A):
            tag = action_spec.tags[i]
            action_frac[f"action_fracs/action_frac/{tag}"] = float((action_counts[i] / denom).item())

        if run is not None:
            mean_C_tok_target = float(C_tok_target_B.mean().item())
            budget_gap_token = (
                keep_mean - mean_C_tok_target
            ) if eff_tok.item() > 0 else float("nan")
            metrics = {
                "train/avg_reward": mean_r,
                "train/avg_r_task": mean_r_task,
                "train/penalty_mean": mean_penalty,
                "train/abs_kl_mean": abs_task,
                "train/abs_penalty_mean": abs_penalty,
                "train/penalty_over_task_abs": ratio_abs,
                "train/avg_keep_effective": keep_mean,
                "train/mean_cost_eff": log_cost_eff,
                # In the multi‑budget regime, "budget_gap_token" is keep_mean − E[C_tok_target].
                "train/budget_gap_token": budget_gap_token,
                "train/ppl_approx": ppl_approx,
                # Budget sampling (what we asked for, on average this batch)
                "budgets/target_tok_mean":    mean_C_tok_target_batch,
                "budgets/target_prune_mean":  mean_C_pru_target_batch,
                "budgets/target_qratio_mean": mean_C_qratio_target_batch,
                # Budget adherence (how far we are from what we asked for)
                "budgets/abs_keep_gap_mean":   avg_abs_keep_gap,
                "budgets/abs_prune_gap_mean":  avg_abs_prune_gap,
                "budgets/abs_qratio_gap_mean": avg_abs_qratio_gap,
                "reward_comps/r_total_mean": mean_r_total,
                "reward_comps/x_for_adv_mean": mean_x_for_adv,
                "reward_comps/computational_component_mean": mean_comp,
                "reward_comps/x_for_adv_abs_mean": mean_abs_x_for_adv,
                "reward_comps/computational_component_abs_mean": mean_abs_comp,
                "reward_comps/comp_vs_task_abs_ratio": comp_over_task_abs,
                "reward_comps/cost_tok_mean": mean_cost_tok,
                "reward_comps/cost_prune_mean": mean_cost_pru,
                "reward_comps/cost_quant_mean": mean_cost_q,

                "update_step": global_step_state["update"],
                "micro_step": global_step_state["micro"],
            }
            metrics.update(action_frac)
            run.log(metrics)

        logs["avg_reward"] += mean_r
        logs["avg_r_task"] += mean_r_task
        logs["avg_penalty"] += mean_penalty
        logs["avg_abs_kl"] += abs_task
        logs["avg_abs_penalty"] += abs_penalty
        logs["avg_penalty_over_task_abs"] += ratio_abs
        logs["avg_cost_eff"] += log_cost_eff
        logs["avg_ppl_approx"] += ppl_approx
        logs["avg_keep_chosen"] += keep_mean
        steps_done += 1
        global_step_state["micro"] += 1

    for k in logs:
        logs[k] /= max(1, steps_done)
    if global_step_state.get("_accum_i", 0) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(unwrap(policy).parameters(), cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step_state["update"] += 1
        stride = global_step_state.get("save_stride")
        if is_main and ckpt_dir and stride:
            upd = int(global_step_state["update"])
            if upd % int(stride) == 0:
                path = os.path.join(ckpt_dir, f"policy_{upd}.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "update_step": upd,
                        "policy_state_dict": unwrap(policy).state_dict(),
                        "cfg": asdict(cfg),
                        "meta": meta,
                        "global_step_state": copy.deepcopy(global_step_state),
                    },
                    path,
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "update_step": upd,
                        "policy_state_dict": unwrap(policy).state_dict(),
                        "cfg": asdict(cfg),
                        "meta": meta,
                        "global_step_state": copy.deepcopy(global_step_state),
                    },
                    os.path.join(ckpt_dir, "policy_latest.pt"),
                )
    return logs

def train_one_epoch_sft(
    tok,
    model,
    policy,
    cfg,
    dl,
    epoch: int = 0,
    run=None,
    val_dl=None,
    eval_every: int = 100,
    global_step_state: Optional[dict] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ckpt_dir: Optional[str] = None,
    best_state: Optional[dict] = None,
    meta: Optional[dict] = None,
):
    """
    Supervised-Finetuning of the policy against the matched-keep teacher.

    Key difference vs the old version:
      * All teacher logic lives in `evaluate_sft_teacher_matched_keep`.
      * For each batch, we call the teacher once (on that batch) with
        `collect_policy_tensors=True`, then do a single CE step on the policy.

    The LM stays frozen/inference-mode the whole time.
    """

    device = next(policy.parameters()).device
    is_dist = dist.is_available() and dist.is_initialized()
    is_main = (not is_dist) or dist.get_rank() == 0
    grad_accum = int(getattr(cfg, "grad_accum_steps", 1))

    unwrap_policy = unwrap(policy) if "unwrap" in globals() else policy
    if optimizer is None:
        optimizer = torch.optim.AdamW(unwrap_policy.parameters(), lr=getattr(cfg, "lr", 2e-4), fused=True)

    if global_step_state is None:
        global_step_state = {"micro": 0, "update": 0}
    if "last_eval_update" not in global_step_state:
        global_step_state["last_eval_update"] = -1

    total_updates = int(getattr(cfg, "_sft_total_updates", 0))
    if total_updates <= 0:
        total_updates = max(1, math.ceil((len(dl) if hasattr(dl, "__len__") else 1) / max(1, int(getattr(cfg, "grad_accum_steps", 1)))))
    warmup_updates = max(1, int(0.05 * total_updates))
    peak_lr = float(getattr(cfg, "lr", 2e-4))
    min_lr  = float(getattr(cfg, "sft_min_lr", 5e-5))
    for g in optimizer.param_groups:
        g["lr"] = peak_lr
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates,
        num_cycles=0.5,
    )
    if global_step_state.get("save_stride") in (None, 0):
        global_step_state["save_stride"] = max(1, total_updates // 10)
    C_target = float(getattr(cfg, "C_target", getattr(cfg, "keep_target", 1.0)))
    pi_temperature = float(getattr(cfg, "pi_temperature", 0.7))
    tbptt_k = int(getattr(cfg, "policy_tbptt_k", 0))
    A = len(cfg.keep_fracs)

    if global_step_state is None:
        global_step_state = {"micro": 0, "update": 0}
    if "lambda_keep" not in global_step_state:
        global_step_state["lambda_keep"] = float(getattr(cfg, "lambda_init", 0.0))

    try:
        init_state_probe = policy.init_state(1, device=device)
        initial_prev_action_idx = int(init_state_probe.last_action[0].item())
    except Exception:
        if 1.0 in cfg.keep_fracs:
            initial_prev_action_idx = cfg.keep_fracs.index(1.0)
        else:
            initial_prev_action_idx = int(torch.tensor(cfg.keep_fracs).argmax().item())

    model.eval()
    policy.train()

    logs = {
        "avg_policy_ce": 0.0,
        "avg_policy_kl": 0.0,
        "avg_policy_acc": 0.0,
        "avg_keep_chosen": 0.0, 
        "avg_cost_eff": 0.0,
        "avg_ppl_approx": 0.0,
        "avg_abs_kl": 0.0,
        "avg_penalty": 0.0,
        "avg_abs_penalty": 0.0,
        "avg_penalty_over_task_abs": 0.0,
    }
    steps_done = 0
    action_hist_epoch = torch.zeros(A, device=device)

    for batch in tqdm(dl, desc="Training (SFT, teacher-driven)...", disable=not is_main):
        batch = batch.to(device)
        B, total_len = batch.shape
        assert total_len == cfg.context_len + cfg.rollout_len + 1, \
            f"got {total_len}, expected {cfg.context_len + cfg.rollout_len + 1}"

        teach = evaluate_sft_teacher_matched_keep(
            cfg=cfg,
            model=model,
            dl=[batch],
            Ts=cfg.Ts,
            Tw=cfg.Tw,
            keep_fracs=tuple(cfg.keep_fracs),
            target_keep_effective=C_target,
            context_len=cfg.context_len,
            rollout_len=cfg.rollout_len,
            device=cfg.device,
            return_assignments=False,
            collect_policy_tensors=True,
            lambda_keep_value=float(global_step_state.get("lambda_keep", getattr(cfg, "lambda_init", 0.0))),
            initial_prev_action=initial_prev_action_idx,
        )
        pb = teach["policy_batches"][0]
        h_seq = pb["h_seq"]                 # [T, B, H],
        e_seq = pb["e_seq"]                 # [T, B, E],
        scalars_seq = pb["scalars_seq"]     # [T, B, 10],
        prev_actions_seq = pb["prev_actions_seq"]         # [T, B]
        teacher_actions_seq = pb["teacher_actions_seq"]   # [T, B]
        same_prev = (prev_actions_seq[1:] == teacher_actions_seq[:-1]).float().mean()
        logits_seq, _values_unused = policy.forward_sequence(
            h_seq=h_seq,
            e_seq=e_seq,
            scalars_seq=scalars_seq,
            prev_actions_seq=prev_actions_seq,
            tbptt_k=tbptt_k,
            temperature=1.0,
        )  # [T, B, A]

        T, Bmb, _ = logits_seq.shape
        # eff_flag is scalar feature index 1: {0,1} for "controllable" steps
        eff_mask = (scalars_seq[..., 1] > 0.5).float()    # [T,B]

        ce = F.cross_entropy(
            logits_seq.reshape(T * Bmb, A),
            teacher_actions_seq.reshape(T * Bmb),
            reduction="mean",
        )
        acc_exact = (logits_seq.argmax(-1) == teacher_actions_seq).float().mean()
        p_true = F.softmax(logits_seq, -1).gather(-1, teacher_actions_seq.unsqueeze(-1)).mean()
        ce_again = -p_true.log()
        print(f"\nCE={ce.item():.4f}, acc={acc_exact.item():.4f}, CE(approx)={ce_again.item():.4f}\n")
        loss = ce
        with torch.no_grad():
            acc = (logits_seq.argmax(dim=-1) == teacher_actions_seq).float().mean()
        assert torch.all(prev_actions_seq[1:] == teacher_actions_seq[:-1])

        did_step = False
        if (global_step_state["micro"] % grad_accum) == 0:
            optimizer.zero_grad(set_to_none=True)
        (loss / grad_accum).backward()
        if ((global_step_state["micro"] + 1) % grad_accum) == 0:
            torch.nn.utils.clip_grad_norm_(unwrap_policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            for g in optimizer.param_groups:
                if g["lr"] < min_lr:
                    g["lr"] = min_lr
            global_step_state["update"] += 1
            did_step = True
            stride = global_step_state.get("save_stride")
            if is_main and ckpt_dir and stride:
                upd = int(global_step_state["update"])
                if upd % int(stride) == 0:
                    path = os.path.join(ckpt_dir, f"policy_{upd}.pt")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "update_step": upd,
                            "policy_state_dict": unwrap(policy).state_dict(),
                            "cfg": asdict(cfg),
                            "meta": meta,
                            "global_step_state": copy.deepcopy(global_step_state),
                        },
                        path,
                    )
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "update_step": upd,
                            "policy_state_dict": unwrap(policy).state_dict(),
                            "cfg": asdict(cfg),
                            "meta": meta,
                            "global_step_state": copy.deepcopy(global_step_state),
                        },
                        os.path.join(ckpt_dir, "policy_latest.pt"),
                    )

        ppl_batch = float(teach["ppl"])
        keep_eff_mean = float(teach["avg_keep_effective"])
        action_counts = torch.tensor(teach["action_hist"], device=device, dtype=torch.float32)
        if run is not None:
            total_actions = action_counts.sum().clamp_min(1.0)
            action_frac = {
                f"action_fracs/action_frac/k={float(cfg.keep_fracs[i]):.2f}": float((action_counts[i] / total_actions).item())
                for i in range(A)
            }
            metrics = {
                "train/prev_equals_next": float(same_prev),
                "train/policy_kl": float(loss.item()),
                "train/policy_ce": float(loss.item()),
                "train/policy_acc": float(acc.item()),
                "train/avg_keep_effective": keep_eff_mean,
                "train/mean_cost_eff": keep_eff_mean,
                "train/ppl_approx": ppl_batch,
                "train/lambda_keep": float(global_step_state.get("lambda_keep", 0.0)),
                "train/avg_reward": 0.0,
                "train/penalty_mean": 0.0,
                "train/abs_kl_mean": 0.0,
                "train/abs_penalty_mean": 0.0,
                "train/penalty_over_task_abs": 0.0,
                "update_step": global_step_state["update"],
                "micro_step": global_step_state["micro"],
            }
            metrics.update(action_frac)
            run.log(metrics)

        logs["avg_policy_kl"] += float(loss.item())
        logs["avg_policy_acc"] += float(acc.item())
        logs["avg_keep_chosen"] += keep_eff_mean
        logs["avg_cost_eff"] += keep_eff_mean
        logs["avg_ppl_approx"] += ppl_batch
        steps_done += 1
        global_step_state["micro"] += 1
        action_hist_epoch += action_counts

        if did_step and (run is not None) and (val_dl is not None):
            upd = int(global_step_state["update"])
            if (upd > 0) and (upd % eval_every == 0) and (upd != global_step_state.get("last_eval_update", -1)):
                try:
                    # sparse_stats = evaluate_stateful_policy_rollout(
                    #     cfg, model, policy, val_dl,

                    sparse_stats = evaluate_stateful_policy_rollout(
                        cfg, model, unwrap(policy), val_dl,
                        Ts=cfg.Ts, Tw=cfg.Tw, keep_fracs=cfg.keep_fracs,
                        context_len=cfg.context_len, rollout_len=cfg.rollout_len,
                        device=cfg.device, greedy=True, temperature=1.0,
                        lambda_keep=float(global_step_state.get("lambda_keep", 0.0)),
                        lambda_prune=float(global_step_state.get("lambda_prune", 0.0)),
                        lambda_quant=float(global_step_state.get("lambda_quant", 0.0)),
                    )
                    
                    # Match a fixed baseline to the *actual* policy budgets.
                    policy_keep_eff_actual = float(sparse_stats["avg_keep_effective"])
                    policy_prune_keep_actual = float(sparse_stats.get("avg_prune_keep", 0.0))
                    policy_quant_ratio_actual = float(sparse_stats.get("avg_quant_ratio", 0.0))

                    fixed_matched = evaluate_fixed_matched_keep(
                        cfg,
                        model,
                        val_dl,
                        Ts=cfg.Ts,
                        Tw=cfg.Tw,
                        keep_fracs=tuple(cfg.keep_fracs),
                        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
                        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
                        target_keep_effective=policy_keep_eff_actual,
                        target_prune_keep=policy_prune_keep_actual,
                        target_quant_ratio=policy_quant_ratio_actual,
                        context_len=cfg.context_len,
                        rollout_len=cfg.rollout_len,
                        device=cfg.device,
                        struct_on_non_eff=False,
                    )

                    gap_nats = math.log(sparse_stats["ppl"]) - math.log(fixed_matched["ppl"])
                    gap_ratio = sparse_stats["ppl"] / fixed_matched["ppl"]
                    run.log({
                        "special/gap_to_fixed_ln_ppl": float(gap_nats),
                        "special/avg_keep_effective": float(sparse_stats["avg_keep_effective"]),
                        "special/gap_ratio_to_fixed": float(gap_ratio),
                        "special/fixed_ppl": float(fixed_matched["ppl"]),
                        "special/sparse_ppl": float(sparse_stats["ppl"]),
                        "update_step": upd,
                    })
                except Exception as _e:
                    import pdb; pdb.set_trace()
                    if is_main:
                        print(f"[warn] eval (SFT) failed: {_e}")

                if is_main and ckpt_dir is not None:
                    print(f"Saving checkpoint to {ckpt_dir} ... at update {upd}")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "update_step": upd,
                            "policy_state_dict": unwrap_policy.state_dict(),
                            "cfg": asdict(cfg),
                            "meta": meta,
                            "global_step_state": copy.deepcopy(global_step_state),
                            "best_metric": best_state.get("best_metric", float("inf")) if best_state is not None else None,
                        },
                        os.path.join(ckpt_dir, "policy_latest.pt"),
                    )
                global_step_state["last_eval_update"] = upd

    if (global_step_state["micro"] % grad_accum) != 0:
        torch.nn.utils.clip_grad_norm_(unwrap_policy.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        for g in optimizer.param_groups:
            if g["lr"] < min_lr:
                g["lr"] = min_lr
        global_step_state["update"] += 1
        stride = global_step_state.get("save_stride")
        if is_main and ckpt_dir and stride:
            upd = int(global_step_state["update"])
            if upd % int(stride) == 0:
                path = os.path.join(ckpt_dir, f"policy_{upd}.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "update_step": upd,
                        "policy_state_dict": unwrap(policy).state_dict(),
                        "cfg": asdict(cfg),
                        "meta": meta,
                        "global_step_state": copy.deepcopy(global_step_state),
                    },
                    path,
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "update_step": upd,
                        "policy_state_dict": unwrap(policy).state_dict(),
                        "cfg": asdict(cfg),
                        "meta": meta,
                        "global_step_state": copy.deepcopy(global_step_state),
                    },
                    os.path.join(ckpt_dir, "policy_latest.pt"),
                )

    for k in logs:
        logs[k] /= max(1, steps_done)

    logs["action_hist_epoch"] = action_hist_epoch.tolist()
    logs["avg_reward"] = 0.0
    logs["avg_abs_kl"] = 0.0
    logs["avg_penalty"] = 0.0
    logs["avg_abs_penalty"] = 0.0
    logs["avg_penalty_over_task_abs"] = 0.0

    return logs


def main():
    parser = ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default="RL4E")
    parser.add_argument("--wandb_entity", type=str, default="akhauriyash")
    parser.add_argument("--wandb_run_name", type=str, default="SFT")
    parser.add_argument("--config", type=str, default="/home/ya255/rl4e/configs/config_template.yml", help="Path to JSON config with overrides")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a policy checkpoint .pt to warm start from")
    parser.add_argument("--total_updates", type=int, default=None,
                        help="Expected total update steps; if set, save policy every 10% of these steps")
    args = parser.parse_args()
    repo_root = os.getcwd()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.cuda.set_device(local_rank if torch.cuda.is_available() else 0)
        dist.init_process_group(backend=backend, init_method="env://", timeout=datetime.timedelta(hours=2),)
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    is_main = (rank == 0)
    cfg = Config()
    cfg.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    cfg.dtype = torch.float32
    if args.config:
        apply_cfg_overrides_from_file(cfg, args.config, is_main=is_main)

    base_cfg_rel = os.path.relpath(args.config, repo_root) if args.config else None
    meta = {
        "kind": "SFT",
        "command": " ".join(sys.argv),
        "config_paths": {
            "base": base_cfg_rel,
            "rl": None,
        },
    }

    set_seed(cfg.seed + rank)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    run = None
    if is_main:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.wandb_run_name}-{timestamp}",
            job_type="train",
            config=asdict(cfg),
            tags=[
                "sparse-attn", "SFT", "critic-free",
                f"model={cfg.model_name}",
                f"dataset={cfg.dataset_name}/{cfg.dataset_config}",
            ],
        )
        wandb.config.update(
            {
                "meta/base_config_relpath": base_cfg_rel, "meta/command": meta["command"], "meta/kind": meta["kind"]
            },
            allow_val_change=True,
        )
    if is_main:
        wandb.define_metric("micro_step")
        wandb.define_metric("update_step")
        wandb.define_metric("train/*", step_metric="micro_step")
        wandb.define_metric("eval/*", step_metric="update_step")
        wandb.define_metric("epoch/*", step_metric="update_step")
        wandb.define_metric("baseline/*", step_metric="update_step")
        wandb.define_metric("observe/*", step_metric="update_step")


    ckpt_dir = None
    if is_main:
        ckpt_dir = os.path.join("checkpoints", (run.name if run is not None else f"RL4E-SFT-{timestamp}"))
        os.makedirs(ckpt_dir, exist_ok=True)
        # snapshot_code(ckpt_dir, root_dir=os.getcwd(), skip_dirs = [
        #         ".venv", ".git", "__pycache__", "wandb", "checkpoints", "block_cache",
        #         "official_configs", "official_results", "newckpt", "old_ch", "sol",
        #         "dec1_checkpoints", "dec6_checkpoints", "dec6_backup", "current_valid"
        #     ])
        snapshot_code(
            ckpt_dir,
            root_dir=os.getcwd(),
            include_top_level=["official_configs", "utils"],
        )
        try:
            with open(os.path.join(ckpt_dir, "train_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[train_sft] Could not write train_meta.json: {e}")
    global_step_state = {"micro": 0, "update": 0}
    # Milestone stride (every 10% of the run)
    global_step_state["save_stride"] = (args.total_updates // 10) if args.total_updates else None
    # Track the best validation metric across the whole run
    best_ckpt = {"best_metric": float("inf")}

    tok, model = load_lm_and_tokenizer(cfg)
    model.to(cfg.device, dtype=cfg.dtype)

    base_model = unwrap(model)
    hidden_size = getattr(base_model.config, "hidden_size", getattr(base_model.config, "n_embd", None))
    if hidden_size is None:
        raise ValueError("Could not infer hidden size from model.config")

    emb_layer = unwrap(model).get_input_embeddings()
    embed_dim = getattr(emb_layer, "embedding_dim", emb_layer.weight.shape[1])

    # === Recurrent policy hyperparams (defaults; can be overridden in cfg) ===
    pol_d_model  = int(getattr(cfg, "policy_d_model", 768))
    pol_heads    = int(getattr(cfg, "policy_n_heads", 8))
    pol_layers   = int(getattr(cfg, "policy_n_layers", 2))
    pol_mlp_mult = float(getattr(cfg, "policy_mlp_ratio", 4.0))
    pol_act_dim  = int(getattr(cfg, "policy_action_dim", 32))
    pol_max_len  = int(getattr(cfg, "policy_max_len", max(1024, cfg.rollout_len + 8)))
    # Number of scalar features provided to the policy per step.
    # Must match the construction in train_one_epoch_grpo / teacher code.
    SCALAR_D = int(getattr(cfg, "policy_scalar_dim", 8))
    spec = build_action_spec(
        keep_fracs=cfg.keep_fracs,
        prune_choices=cfg.struct_prune_choices,
        quant_choices=cfg.quant_choices,
    )
    policy = RecurrentActorCriticPolicy(
        h_dim=int(hidden_size),
        e_dim=int(embed_dim),
        n_actions=spec.n_actions,
        d_model=pol_d_model,
        n_heads=pol_heads,
        n_layers=pol_layers,
        mlp_ratio=pol_mlp_mult,
        action_dim=pol_act_dim,
        max_len=pol_max_len,
        dropout=float(getattr(cfg, "policy_dropout", 0.0)),
        scalar_dim=SCALAR_D,
    ).to(cfg.device, dtype=torch.float32)

    if args.checkpoint_path:
        if not os.path.isfile(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location=cfg.device)
        state_key = "policy_state_dict" if "policy_state_dict" in ckpt else (
            "state_dict" if "state_dict" in ckpt else None
        )
        if state_key is None:
            raise KeyError(f"'{args.checkpoint_path}' missing 'policy_state_dict' or 'state_dict'")
        unwrap(policy).load_state_dict(ckpt[state_key], strict=True)
        if is_main:
            print(f"[load] Loaded policy weights from {args.checkpoint_path}")

    if distributed:
        policy = DDP(
            policy,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    # if distributed:
    #     policy = DDP(policy, device_ids=[local_rank] if torch.cuda.is_available() else None,
    #                  output_device=local_rank if torch.cuda.is_available() else None)
    if is_main:
        _watch_target = unwrap(policy)
        # wandb.watch(_watch_target, log="gradients", log_freq=500)
        wandb.config.update(
            {
                "policy_num_params": sum(p.numel() for p in _watch_target.parameters()),
                "train_rollout_len": cfg.rollout_len,
                "sinks_Ts": cfg.Ts,
                "trailing_Tw": cfg.Tw,
                "policy_d_model": pol_d_model,
                "policy_n_heads": pol_heads,
                "policy_n_layers": pol_layers,
                "policy_tbptt_k": int(getattr(cfg, "policy_tbptt_k", 0)),
                "policy_scalar_dim": SCALAR_D,
            },
            allow_val_change=True,
        )

    dl = make_dataloader(cfg, tok, split="train", shuffle=True, distributed=distributed)
    valcfg = copy.deepcopy(cfg)
    valcfg.dataset_name = "wikitext"
    valcfg.dataset_config = "wikitext-2-raw-v1"
    val_dl = make_dataloader(valcfg, tok, split="validation", shuffle=False, distributed=False) if is_main else None
    baseline_batches = 100

    optimizer = torch.optim.AdamW(unwrap(policy).parameters(), lr=cfg.lr, fused=True)


    algo = str(getattr(cfg, "algo", "grpo")).lower()
    assert cfg.epochs == 1, "Only single-epoch training is supported in this script."
    for epoch in range(cfg.epochs):
        print(f"\n=== Epoch {epoch+1} ({algo.upper()}) ===")

        if distributed and isinstance(dl.sampler, DistributedSampler):
            dl.sampler.set_epoch(epoch)

        TRAIN_FRACTION = 0.2
        max_batches = max(1, int(len(dl) * TRAIN_FRACTION))
        dl_epoch = limited_dl(dl, max_batches)

        if algo == "grpo":
            stats = train_one_epoch_grpo(
                tok, model, policy, cfg, dl_epoch, epoch=epoch,
                run=run, val_dl=val_dl, eval_every=-1,
                global_step_state=global_step_state, optimizer=optimizer,
                ckpt_dir=ckpt_dir, best_state=best_ckpt, meta=meta,
            )
        elif algo == "sft":
            cfg._sft_total_updates = math.ceil(max_batches / max(1, int(getattr(cfg, "grad_accum_steps", 1))))
            stats = train_one_epoch_sft(
                tok, model, policy, cfg, dl_epoch, epoch=epoch,
                run=run, val_dl=val_dl, eval_every=cfg.eval_every_updates,
                global_step_state=global_step_state, optimizer=optimizer,
                ckpt_dir=ckpt_dir, best_state=best_ckpt, meta=meta,
            )
        else:
            raise NotImplementedError(f"Unknown algo '{algo}'. Expected 'grpo' or 'sft'.")

        if is_main:
            print(f"epoch={epoch+1} "
                f"ppl~{stats['avg_ppl_approx']:.2f} "
                f"keep={stats['avg_keep_chosen']:.3f} "
                f"cost={stats['avg_cost_eff']:.3f}")
 
        if is_main:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "epoch/avg_reward": stats["avg_reward"],
                    "epoch/ppl_approx": stats["avg_ppl_approx"],
                    "epoch/keep": stats["avg_keep_chosen"],
                    "epoch/cost_eff": stats["avg_cost_eff"],
                    "epoch/lambda_keep": float(global_step_state.get("lambda_keep", 0.0)),
                    "epoch/kl_abs": stats["avg_abs_kl"],
                    "epoch/penalty_abs": stats["avg_abs_penalty"],
                    "epoch/penalty_over_task_abs": stats["avg_penalty_over_task_abs"],
                    "epoch/penalty_mean": stats["avg_penalty"],
                    "update_step": global_step_state["update"],
                }
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "policy_state_dict": unwrap(policy).state_dict(),
                    "meta": meta,
                    "cfg": asdict(cfg),
                    "global_step_state": copy.deepcopy(global_step_state),
                },
                os.path.join(ckpt_dir, f"policy_epoch{epoch+1:03d}.pt"),
            )
            latest_path = os.path.join(ckpt_dir, "policy_latest.pt")
            if os.path.islink(latest_path) or os.path.exists(latest_path):
                try:
                    os.remove(latest_path)
                except OSError:
                    pass
            try:
                os.symlink(f"policy_epoch{epoch+1:03d}.pt", latest_path)
            except OSError:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "meta": meta,
                        "policy_state_dict": unwrap(policy).state_dict(),
                        "cfg": asdict(cfg),
                        "global_step_state": copy.deepcopy(global_step_state),
                    },
                    latest_path,
                )

if __name__ == "__main__":
    try:
        main()
    finally:
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            wandb.finish()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
