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
    evaluate_sft_teacher_matched_keep,
)
from utils.eval_baselines import (
    evaluate_dense_full,
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

# Goal conditioned policy discussions
# https://chatgpt.com/c/68f1c284-7a04-8330-80d9-57bf4c420377
# https://chatgpt.com/c/68f1c2bf-5330-8326-82af-88c6164883ec
# https://chatgpt.com/codex/tasks/task_e_68f1c2cd3a148331b96ee873de421412
# Be careful, GPT made many bugs, go line by line.
# Multi action spaces
# https://chatgpt.com/c/68fbf5ad-102c-8326-a141-00406d76b291


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
    """
    GRPO training (critic-free) with grouped rollouts per input.
    - K rollouts per (context, rollout window)
    - Group-relative advantages (process-level or outcome-level)
    - Lagrangian constrained RL: separate reward (quality) and cost (sparsity) advantages.
      The actor uses A = A_r - λ * A_c, and λ is updated from aggregated cost stats.
    """
    device = next(policy.parameters()).device
    is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    if optimizer is None:
        optimizer = torch.optim.AdamW(unwrap(policy).parameters(), lr=cfg.lr, fused=True)
    policy.train()

    # === Core knobs (defaults try to match your PPO code) ===
    K = int(getattr(cfg, "grpo_rollouts_per_input", 16))
    grpo_level = str(getattr(cfg, "grpo_level", "process")).lower()     # "process" or "outcome"
    grpo_norm  = str(getattr(cfg, "grpo_norm", "center")).lower()       # "center" or "zscore"
    adv_whiten_global = bool(getattr(cfg, "adv_whiten_global", True))   # final global whitening
    ppo_clip = float(getattr(cfg, "ppo_clip", 0.2))
    mb_size = int(getattr(cfg, "ppo_minibatch_size", 512))
    target_N = int(getattr(cfg, "ppo_target_batch_size", 2048))
    entropy_coef = float(getattr(cfg, "entropy_coef", 1e-3))
    pi_temperature = float(getattr(cfg, "pi_temperature", 0.7))
    grad_accum_steps = int(getattr(cfg, "grad_accum_steps", 1))

    # Optional stability: KL(π || π_ref) to a frozen snapshot of the policy (default off)
    kl_pi_ref_coef = float(getattr(cfg, "kl_pi_ref_coef", 0.0))
    if kl_pi_ref_coef > 0.0 and "_policy_ref" not in (global_step_state or {}):
        # store once in global state so it's shared across ranks/epochs
        policy_ref = copy.deepcopy(unwrap(policy)).eval()
        for p in policy_ref.parameters():
            p.requires_grad_(False)
        if global_step_state is None:
            global_step_state = {}
        global_step_state["_policy_ref"] = policy_ref
    policy_ref = (global_step_state or {}).get("_policy_ref", None)


    # === Multi-constraint budgets (token keep, prune keep, quant bits) ===
    # Token keep (legacy path compatibility)
    C_tok   = float(getattr(cfg, "C_target_token", getattr(cfg, "C_target", getattr(cfg, "keep_target", 1.0))))
    tol_tok = float(getattr(cfg, "tol_token", getattr(cfg, "budget_tolerance", getattr(cfg, "keep_tolerance", 0.01))))
    lr_tok  = float(getattr(cfg, "lambda_lr_token", getattr(cfg, "lambda_lr", 0.5)))
    init_tok= float(getattr(cfg, "lambda_init_token", getattr(cfg, "lambda_init", 25)))
    # Prune keep (MLP)
    C_pru   = float(getattr(cfg, "C_target_prune", 0.70))
    tol_pru = float(getattr(cfg, "tol_prune", 0.05))
    lr_pru  = float(getattr(cfg, "lambda_lr_prune", lr_tok))
    init_pru= float(getattr(cfg, "lambda_init_prune", init_tok))
    # Quant bits (normalize to ratio of 16)
    C_qbits = float(getattr(cfg, "C_target_quant_bits", 8.0))
    C_q     = C_qbits / 16.0
    tol_q   = float(getattr(cfg, "tol_quant_bits", 1.0)) / 16.0
    lr_q    = float(getattr(cfg, "lambda_lr_quant", lr_tok))
    init_q  = float(getattr(cfg, "lambda_init_quant", init_tok))
    lambda_max = float(getattr(cfg, "lambda_max", 20000.0))

    if global_step_state is None:
        global_step_state = {"micro": 0, "update": 0}
    # Initialize three dual variables
    global_step_state.setdefault("lambda_keep",  init_tok)  # token keep
    global_step_state.setdefault("lambda_prune", init_pru)  # prune keep
    global_step_state.setdefault("lambda_quant", init_q)    # quant ratio

    # Build composite action space (token_keep × prune_keep × q_bits)
    action_spec = build_action_spec(
        keep_fracs=cfg.keep_fracs,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
    )
    A = action_spec.n_actions
    KEEP_TOKEN = torch.tensor(action_spec.token_keep, device=device, dtype=torch.float32)
    KEEP_PRUNE = torch.tensor(action_spec.prune_keep, device=device, dtype=torch.float32)
    Q_BITS     = torch.tensor(action_spec.q_bits,     device=device, dtype=torch.int64)
    # Sanity: policy head size must match
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

    # Running logs
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

    # Aggregation buffers (to hit PPO target_N before an update)
    agg_h_seq, agg_e_seq, agg_scalars_seq, agg_prev_actions_seq = [], [], [], []
    agg_actions_seq, agg_logp_old_seq, agg_adv_seq = [], [], []
    agg_count = 0
    agg_cost_eff_sum = 0.0     # ∑ eff * kappa
    agg_eff_tok      = 0.0     # ∑ eff
    agg_prune_sum    = 0.0     # ∑ s_now over all samples/steps
    agg_qratio_sum   = 0.0     # ∑ q_ratio over all samples/steps
    agg_tok_steps    = 0.0     # total steps counted (denominator for prune/quant)
 

    for batch in tqdm(dl, desc="Training (GRPO)...", disable=not is_main):
        batch = batch.to(device)
        B, total_len = batch.shape
        assert total_len == cfg.context_len + cfg.rollout_len + 1

        # Auto-set milestone stride on first batch (assumes 1 epoch; estimates GRPO updates)
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
            # Aggregation rounds across the run
            cycles = max(1, math.ceil((num_batches * adv_per_batch) / max(1, target_N)))
            # Each round ≈ ceil(target_N/mb_size) microbatches; one optimizer step per grad_acc
            updates_per_cycle = max(1, math.ceil(math.ceil(max(1, target_N) / max(1, mb_size)) / grad_acc))
            total_updates_est = cycles * updates_per_cycle
            global_step_state["save_stride"] = max(1, total_updates_est // 10)
        prefill_ids = batch[:, :cfg.context_len]
        step_inputs = batch[:, cfg.context_len : cfg.context_len + cfg.rollout_len]
        step_labels = batch[:, cfg.context_len + 1 : cfg.context_len + cfg.rollout_len + 1]

        # ----- Dense reference rollout (teacher-forced, κ=1) -----
        if getattr(cfg, "sparsity_criteria", "recency") == "relevancy":
            clear_relevancy_keep(model)  # <<< add this
        clear_structured_action(model)   # ensure teacher is dense & unquantized
        with torch.inference_mode():
            outputs = model(
                input_ids=prefill_ids,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
        past_kv_ref = detach_cache_to_tuple(outputs.past_key_values)
        past_kv_pol = detach_cache_to_tuple(outputs.past_key_values)  # will become K-tiled
        kv_len_ref = torch.full((B,), cfg.context_len + 1, device=device, dtype=torch.long)
        kv_len_pol = torch.full((B,), cfg.context_len + 1, device=device, dtype=torch.long)
        state_pol = outputs.hidden_states[-1][:, -1, :].detach()     # [B, H]

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

        # Buffers (time-major)
        h_seq_buf, e_seq_buf, scalars_seq_buf = [], [], []
        prev_actions_seq_buf, actions_seq_buf = [], []
        logp_old_seq_buf = []
        rewards_buf, r_task_buf, penalty_buf = [], [], []  # rewards_buf now stores r_task only
        cost_delta_buf = []  # (eff * kappa_now - C_target * eff), used for cost advantages
        keep_buf, eff_mask_buf = [], []  # <-- NEW: to compute per-example mean keep
        prune_keep_buf, qratio_buf = [], []  # <-- NEW: for new constraints

        # Logging accumulators
        nll_sum = torch.tensor(0.0, device=device)
        tok_count = 0
        action_counts = torch.zeros(A, device=device)
        keep_chosen_sum = torch.tensor(0.0, device=device)
        cost_eff_sum = torch.tensor(0.0, device=device)
        eff_tok = torch.tensor(0.0, device=device)

        # Recurrent policy state (BK) & budget trackers
        pi_state = policy.init_state(Bk, device=device)
        prev_action_ids = pi_state.last_action.clone()
        cum_keep = torch.zeros(Bk, device=device)
        cum_eff  = torch.zeros(Bk, device=device)
        phi_prev = torch.zeros(Bk, device=device)

        for t in range(cfg.rollout_len):
            # Tile input token & label across K
            cur = step_inputs[:, t].repeat_interleave(K, dim=0)      # [BK]
            labels_t = step_labels[:, t].repeat_interleave(K, dim=0) # [BK]

            eff_mask = (kv_len_pol > thr)                             # [BK] bool
            tok_embed = emb_layer(cur).detach()

            # Scalars to the policy (same schema as your PPO loop)
            win = float(cfg.Ts + cfg.Tw + 1)
            fill_frac = min(t + 1, win) / win
            frac_old = torch.full((Bk, 1), fill_frac, device=device, dtype=torch.float32)
            t_frac = torch.full_like(frac_old, 2.0 * ((t + 1) / float(cfg.rollout_len)) - 1.0)
            lambda_keep_now  = torch.full_like(frac_old, float(global_step_state["lambda_keep"]))
            lambda_prune_now = torch.full_like(frac_old, float(global_step_state["lambda_prune"]))
            lambda_quant_now = torch.full_like(frac_old, float(global_step_state["lambda_quant"]))

            mean_keep_prev = torch.where(cum_eff > 0, cum_keep / cum_eff, torch.zeros_like(cum_keep))

            dev_prev = mean_keep_prev - C_tok
            if tol_tok > 0:
                tol_t = torch.as_tensor(tol_tok, device=device, dtype=dev_prev.dtype)
                dev_norm_prev = dev_prev / tol_t
                gap_prev = (dev_prev.abs() - tol_t).clamp_min(0.0) / tol_t
            else:
                dev_norm_prev = dev_prev
                gap_prev = (dev_prev.abs()).clamp_min(0.0)

            eff_flag = eff_mask.float()
            eff_count_norm = torch.where(cum_eff > 0, cum_eff / float(cfg.rollout_len), torch.zeros_like(cum_eff))
            steps_rem_frac = torch.full_like(frac_old, (cfg.rollout_len - t) / float(cfg.rollout_len))
            phi_prev_feat = phi_prev

            scalars = torch.cat([
                frac_old, t_frac, lambda_keep_now, lambda_prune_now, lambda_quant_now,
                eff_flag.unsqueeze(1),
                mean_keep_prev.unsqueeze(1),
                dev_norm_prev.unsqueeze(1),
                gap_prev.unsqueeze(1),
                steps_rem_frac,
                eff_count_norm.unsqueeze(1),
                phi_prev_feat.unsqueeze(1),
            ], dim=-1)  # [BK, 12]

            # Policy step (critic output ignored; value head stays unused)
            logits, _value_unused, pi_state = policy.step(
                h_lm=state_pol.to(torch.float32),
                e_tok=tok_embed.to(torch.float32),
                scalars=scalars,
                state=pi_state,
                temperature=pi_temperature,
            )
            dist_pi = torch.distributions.Categorical(logits=logits)
            action = dist_pi.sample()                      # [BK]
            logp_old = dist_pi.log_prob(action)            # [BK]
            pi_state.last_action = action.detach()

            # Sparse step through LM
            # Decode composite action and program the model
            kappa_now = KEEP_TOKEN[action]                 # [BK]
            prune_now = KEEP_PRUNE[action]                 # [BK]
            qbits_now = Q_BITS[action]                     # [BK]

            eff = eff_mask.float()
            p_dense = float(max(action_spec.prune_keep))
            q_dense = int(max(action_spec.q_bits))

            # mirror evaluator behavior
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
            # Teacher logprobs for this t (expand B -> BK)
            logp_dense = dense_logprobs[t].float().repeat_interleave(K, dim=0)  # [BK, V]
            logp_sparse = F.log_softmax(logits_sparse.float(), dim=-1)          # [BK, V]

            # Reward components (same as your PPO loop)
            kl_t = F.kl_div(logp_sparse, logp_dense, log_target=True, reduction='none').sum(dim=-1)  # [BK]
            ce_dense  = F.nll_loss(logp_dense,  labels_t, reduction='none') # It is not used, as its cancelled out in whitening anyway
            ce_sparse = F.nll_loss(logp_sparse, labels_t, reduction='none')
            delta_ce  = -ce_sparse
            w_kl = float(getattr(cfg, "task_w_kl", 0.0))
            # Here, if we want to 'optimize' for next-token CE, w_kl = 0.0
            # However, if we want to match dense baseline, not 'exceed it', w_kl = 1
            r_task_t = (1.0 - w_kl) * delta_ce - w_kl * kl_t
            # Immediate cost (no shaping mixed into reward)
            eff = eff_mask.float()                                     # [BK]
            cost_t_eff   = eff * kappa_now                             # [BK]
            cost_delta_t = cost_t_eff - eff * C_tok                    # deviation from token target
            # logging only; keep token-penalty for continuity (structural penalties enter via adv)
            penalty_t    = float(global_step_state["lambda_keep"]) * cost_delta_t
            # Advance trackers for aggregated λ update
            cum_eff  = cum_eff  + eff
            cum_keep = cum_keep + cost_t_eff

            # --- NEW: update phi_prev to match evaluator logic (potential over keep deviation) ---
            mean_keep_so_far = torch.where(cum_eff > 0, cum_keep / cum_eff, torch.zeros_like(cum_eff))
            dev_abs = (mean_keep_so_far - C_tok).abs()
            phi_now = (dev_abs - tol_tok).clamp_min(0.0)
            phi_prev = phi_now.detach()
            # Record time-major buffers
            h_seq_buf.append(state_pol.to(torch.float32))
            e_seq_buf.append(tok_embed.to(torch.float32))
            scalars_seq_buf.append(scalars)
            prev_actions_seq_buf.append(prev_action_ids)
            actions_seq_buf.append(action)
            logp_old_seq_buf.append(logp_old.detach())

            rewards_buf.append(r_task_t)         # keep "rewards" as the pure task reward for logging
            r_task_buf.append(r_task_t)
            penalty_buf.append(penalty_t)        # λ * (cost - C_target * eff) for visibility
            cost_delta_buf.append(cost_delta_t)  # (cost - C_target * eff), used to form A_c
            keep_buf.append(kappa_now)               # <-- NEW
            eff_mask_buf.append(eff_mask)            # <-- NEW
            prune_keep_buf.append(prune_now)         # <-- NEW
            qratio_buf.append(q_ratio)               # <-- NEW
            prev_action_ids = action.detach()

            # Logging aggregates
            nll_sum += F.cross_entropy(logits_sparse, labels_t, reduction="sum")
            tok_count += Bk
            action_counts.index_add_(0, action, torch.ones_like(action, dtype=torch.float32))
            keep_chosen_sum += kappa_now[eff_mask].sum()
            cost_eff_sum += cost_t_eff.sum()
            eff_tok += eff.sum()
            # Aggregate prune/quant observed values over *all* steps/samples
            agg_prune_sum  += float((prune_now * eff).sum().item())
            agg_qratio_sum += float((q_ratio   * eff).sum().item())
            agg_tok_steps  += float(eff.sum().item())
        # Stack time-major
        h_seq = torch.stack(h_seq_buf, dim=0)                     # [W, BK, H]
        e_seq = torch.stack(e_seq_buf, dim=0)                     # [W, BK, E]
        scalars_seq = torch.stack(scalars_seq_buf, dim=0)         # [W, BK, 10]
        prev_actions_seq = torch.stack(prev_actions_seq_buf, dim=0)  # [W, BK]
        actions_seq = torch.stack(actions_seq_buf, dim=0)         # [W, BK]
        logp_old_seq = torch.stack(logp_old_seq_buf, dim=0)       # [W, BK]
        rewards = torch.stack(rewards_buf, dim=0)                 # [W, BK]
        r_task_all = torch.stack(r_task_buf, dim=0)               # [W, BK]
        penalty_all = torch.stack(penalty_buf, dim=0)             # [W, BK]
        cost_delta_all = torch.stack(cost_delta_buf, dim=0)       # [W, BK]
        keep_all = torch.stack(keep_buf, dim=0)                   # [W, BK]
        eff_all  = torch.stack(eff_mask_buf, dim=0).float()       # [W, BK]
        prune_all  = torch.stack(prune_keep_buf, dim=0)           # [W, BK]
        qratio_all = torch.stack(qratio_buf, dim=0)               # [W, BK]
        # ----- GRPO advantages (critic-free, Lagrangian) -----
        T, BK = rewards.shape
        assert BK == Bk

        def _grpo_adv(x_flat, level):  # x_flat: [T, BK]
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
                X = x_flat.sum(dim=0).view(B, K)  # sum over time per rollout
                if grpo_norm == "zscore":
                    mu = X.mean(dim=1, keepdim=True)
                    sd = X.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
                    A_grp = (X - mu) / sd
                else:
                    A_grp = X - X.mean(dim=1, keepdim=True)
                return A_grp.view(1, BK).expand(T, BK)

        # # === Reward advantage (quality) ===
        # adv_r = _grpo_adv(r_task_all, level=grpo_level)
        # if adv_whiten_global:
        #     adv_r = (adv_r - adv_r.mean()) / adv_r.std(unbiased=False).clamp_min(1e-6)

        # === Reward advantage (quality) with optional discounted future aggregation ===
        # Config (optional):
        #   cfg.reward_agg   in {None, "sum", "max"}
        #   cfg.reward_gamma in [0, 1)
        rag   = getattr(cfg, "reward_agg", None)
        gamma = float(getattr(cfg, "reward_gamma", 0.82))

        x_for_adv = r_task_all  # default: no aggregation
        if rag in ("sum", "max") and 0.0 <= gamma < 1.0:
            T, BK = r_task_all.shape
            returns = torch.zeros_like(r_task_all)
            if rag == "sum":
                # R_t = sum_{i=t}^{T-1} gamma^{i-t} r_i
                running = torch.zeros(BK, device=r_task_all.device, dtype=r_task_all.dtype)
                for t in range(T - 1, -1, -1):
                    running = r_task_all[t] + gamma * running
                    returns[t] = running
            else:  # rag == "max"
                # R_t = max_{i>=t} gamma^{i-t} r_i  (reverse scan with discounted running max)
                running_best = torch.full((BK,), float("-inf"),
                                          device=r_task_all.device, dtype=r_task_all.dtype)
                for t in range(T - 1, -1, -1):
                    running_best = torch.maximum(r_task_all[t], gamma * running_best)
                    returns[t] = running_best

            # Process: use per-step R_t. Outcome: give each step the rollout's total discounted return R_0.
            if grpo_level == "outcome":
                x_for_adv = returns[0].unsqueeze(0).expand_as(r_task_all)  # shape [T, BK]
            else:
                x_for_adv = returns

        adv_r = _grpo_adv(x_for_adv, level=grpo_level)
        if adv_whiten_global:
            adv_r = (adv_r - adv_r.mean()) / adv_r.std(unbiased=False).clamp_min(1e-6)
        # === Sparsity stats reused by both modes ===
        sum_eff_seq    = eff_all.sum(dim=0).clamp_min(1.0)               # [BK]
        mean_keep_seq  = (eff_all * keep_all).sum(dim=0) / sum_eff_seq   # [BK]
        mean_prune_seq = (eff_all * prune_all).sum(dim=0) / sum_eff_seq   # [BK]
        mean_qratio_seq= (eff_all * qratio_all).sum(dim=0) / sum_eff_seq  # [BK]
        alpha_c = float(getattr(cfg, "cost_tradeoff_alpha", 1.0))

        # ---- Multi-constraint mixing (token, prune, quant) ----
        # Token keep gates (one-sided)
        d_tok = (mean_keep_seq - (C_tok + tol_tok)).clamp_min(0.0)         # [BK]
        s_tok = torch.where(mean_keep_seq > C_tok + tol_tok,
                            torch.tensor(1.0, device=device),
                            torch.tensor(0.0, device=device))
        # Prune keep gates
        d_pru = (mean_prune_seq - (C_pru + tol_pru)).clamp_min(0.0)        # [BK]
        s_pru = torch.where(mean_prune_seq > C_pru + tol_pru,
                            torch.tensor(1.0, device=device),
                            torch.tensor(0.0, device=device))
        # Quant ratio gates
        d_q   = (mean_qratio_seq - (C_q + tol_q)).clamp_min(0.0)           # [BK]
        s_q   = torch.where(mean_qratio_seq > C_q + tol_q,
                            torch.tensor(1.0, device=device),
                            torch.tensor(0.0, device=device))
        # Per-step deviations
        dev_tok = eff_all * (keep_all - C_tok)                              # [T, BK]
        dev_pru = eff_all * (prune_all  - C_pru)                            # [T, BK]
        dev_q   = eff_all * (qratio_all - C_q)                              # [T, BK]
        lam_tok = float(global_step_state.get("lambda_keep",  0.0))
        lam_pru = float(global_step_state.get("lambda_prune", 0.0))
        lam_q   = float(global_step_state.get("lambda_quant", 0.0))
        adv = (adv_r
               - alpha_c * lam_tok * d_tok.view(1, -1) * s_tok.view(1, -1) * dev_tok
               - alpha_c * lam_pru * d_pru.view(1, -1) * s_pru.view(1, -1) * dev_pru
               - alpha_c * lam_q   * d_q.view(1, -1)   * s_q.view(1, -1)   * dev_q)
        # Aggregate whole sequences (no values/returns)
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
        # When enough samples collected, do clipped PG update
        if agg_count >= target_N:
            # Periodic eval + checkpoint (unchanged)
            if (run is not None) and (val_dl is not None) and (global_step_state["update"] % eval_every == 0):
                try:
                    sparse_stats = evaluate_stateful_policy_rollout(
                        cfg, model, policy, val_dl,
                        Ts=cfg.Ts, Tw=cfg.Tw, keep_fracs=cfg.keep_fracs,
                        context_len=cfg.context_len, rollout_len=cfg.rollout_len,
                        device=cfg.device, greedy=True, temperature=1.0,
                        lambda_keep=float(global_step_state.get("lambda_keep", 0.0)),
                        lambda_prune=float(global_step_state.get("lambda_prune", 0.0)),
                        lambda_quant=float(global_step_state.get("lambda_quant", 0.0)),
                    )
                    if "lambda_prune" not in global_step_state and "lambda_quant" not in global_step_state:
                        teach = evaluate_sft_teacher_matched_keep(
                            cfg, model, val_dl, Ts=cfg.Ts, Tw=cfg.Tw, keep_fracs=tuple(cfg.keep_fracs),
                            target_keep_effective=float(sparse_stats["avg_keep_effective"]),
                            context_len=cfg.context_len, rollout_len=cfg.rollout_len, device=cfg.device,
                        )
                        gap_nats = math.log(sparse_stats["ppl"]) - math.log(teach["ppl"])
                        gap_ratio = sparse_stats["ppl"] / teach["ppl"]
                        if run is not None:
                            run.log({
                                "special/gap_to_teacher_ln_ppl": gap_nats,
                                "special/avg_keep_effective": sparse_stats["avg_keep_effective"],
                                "special/gap_ratio_to_teacher": gap_ratio,
                                "special/teacher_ppl": teach["ppl"],
                                "special/sparse_ppl": sparse_stats["ppl"],
                                "update_step": global_step_state["update"],
                            })
                    else:
                        # log just the sparse policy stats
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

            # Concatenate and update
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

            # ---- Accumulate across micro-batches AND across updates ----
            grad_accum_steps = max(1, int(grad_accum_steps))
            if "_accum_i" not in global_step_state:
                global_step_state["_accum_i"] = 0
            # Zero ONLY at the start of a new accumulation cycle
            if global_step_state["_accum_i"] % grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            num_mb = (Btot + mb_seqs - 1) // mb_seqs  # ceil
            for start in range(0, Btot, mb_seqs):
                mb_idx = idx_seq[start:start+mb_seqs]
                logits_seq, _values_unused = policy.forward_sequence(
                    h_seq=h_total[:, mb_idx, :],
                    e_seq=e_total[:, mb_idx, :],
                    scalars_seq=scalars_total[:, mb_idx, :],
                    prev_actions_seq=prev_actions_total[:, mb_idx],
                    tbptt_k=tbptt_k,
                    temperature=pi_temperature,
                )  # [T,Bmb,A], [T,Bmb]
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
                ent_ratio = (entropy_coef * entropy).abs() / policy_loss.abs().clamp_min(1e-8)
                # Optional π vs π_ref KL (stability), off by default
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

                # ---- Gradient accumulation across the whole run ----
                (loss / grad_accum_steps).backward()
                global_step_state["_accum_i"] += 1

                if global_step_state["_accum_i"] % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(unwrap(policy).parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step_state["update"] += 1
                    # --- 10% milestone save ---
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

            if dist.is_available() and dist.is_initialized():
                totals = torch.tensor([agg_cost_eff_sum, agg_eff_tok, agg_prune_sum, agg_qratio_sum, agg_tok_steps], device=device)
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                agg_cost_eff_sum, agg_eff_tok, agg_prune_sum, agg_qratio_sum, agg_tok_steps = [t.item() for t in totals]

            ema_beta = float(getattr(cfg, "lambda_ema_beta", 0.9))
            # --- Token keep controller (legacy) ---
            if agg_eff_tok > 0:
                mean_keep_eff = agg_cost_eff_sum / agg_eff_tok
                if "ema_cost_tok" not in global_step_state:
                    global_step_state["ema_cost_tok"] = mean_keep_eff
                else:
                    global_step_state["ema_cost_tok"] = ema_beta * global_step_state["ema_cost_tok"] + (1 - ema_beta) * mean_keep_eff
                gap_tok = float(global_step_state["ema_cost_tok"]) - (C_tok + tol_tok)
                new_lam_tok = float(global_step_state["lambda_keep"]) + lr_tok * gap_tok
                global_step_state["lambda_keep"] = float(max(0.0, min(lambda_max, new_lam_tok)))
                if run is not None:
                    run.log({
                        "observe/mean_keep_eff": mean_keep_eff,
                        "observe/keep_target": C_tok,
                        "observe/budget_violation_end_token": max(gap_tok, 0.0),
                        "update_step": global_step_state["update"],
                    })
            # --- Prune keep controller ---
            if agg_tok_steps > 0:
                mean_prune_obs = agg_prune_sum / agg_tok_steps
                if "ema_cost_pru" not in global_step_state:
                    global_step_state["ema_cost_pru"] = mean_prune_obs
                else:
                    global_step_state["ema_cost_pru"] = ema_beta * global_step_state["ema_cost_pru"] + (1 - ema_beta) * mean_prune_obs
                gap_pru = float(global_step_state["ema_cost_pru"]) - (C_pru + tol_pru)
                new_lam_pru = float(global_step_state["lambda_prune"]) + lr_pru * gap_pru
                global_step_state["lambda_prune"] = float(max(0.0, min(lambda_max, new_lam_pru)))
                if run is not None:
                    run.log({
                        "observe/mean_prune_keep": mean_prune_obs,
                        "observe/prune_target": C_pru,
                        "observe/prune_gap": mean_prune_obs - (C_pru + tol_pru),
                        "observe/budget_violation_end_prune": max(gap_pru, 0.0),
                        "update_step": global_step_state["update"],
                    })
            # --- Quant controller ---
            if agg_tok_steps > 0:
                mean_qratio_obs = agg_qratio_sum / agg_tok_steps
                if "ema_cost_q" not in global_step_state:
                    global_step_state["ema_cost_q"] = mean_qratio_obs
                else:
                    global_step_state["ema_cost_q"] = ema_beta * global_step_state["ema_cost_q"] + (1 - ema_beta) * mean_qratio_obs
                gap_q = float(global_step_state["ema_cost_q"]) - (C_q + tol_q)
                new_lam_q = float(global_step_state["lambda_quant"]) + lr_q * gap_q
                global_step_state["lambda_quant"] = float(max(0.0, min(lambda_max, new_lam_q)))
                if run is not None:
                    run.log({
                        "observe/mean_quant_ratio": mean_qratio_obs,
                        "observe/quant_target_ratio": C_q,
                        "observe/quant_gap": mean_qratio_obs - (C_q + tol_q),
                        "observe/budget_violation_end_quant": max(gap_q, 0.0),
                        "update_step": global_step_state["update"],
                    })
            # reset aggregation buffers
            agg_h_seq, agg_e_seq, agg_scalars_seq, agg_prev_actions_seq = [], [], [], []
            agg_actions_seq, agg_logp_old_seq, agg_adv_seq = [], [], []
            agg_count = 0
            agg_cost_eff_sum = 0.0
            agg_eff_tok = 0.0
            agg_prune_sum    = 0.0
            agg_qratio_sum   = 0.0
            agg_tok_steps    = 0.0

        # ----- Per-batch logging (matches your PPO stats) -----
        ppl_approx = math.exp((nll_sum / max(1, tok_count)).item()) if tok_count > 0 else 0.0
        keep_mean = (keep_chosen_sum / eff_tok.clamp_min(1.0)).item() if eff_tok.item() > 0 else 0.0
        mean_r = rewards.mean().item()  # same as r_task mean now
        mean_r_task = r_task_all.mean().item()
        mean_penalty = penalty_all.mean().item()
        abs_task = r_task_all.abs().mean().item()
        abs_penalty = penalty_all.abs().mean().item()
        ratio_abs = (abs_penalty / max(abs_task, 1e-8)) if abs_task > 0 else float('inf')
        log_cost_eff = (cost_eff_sum / eff_tok.clamp_min(1.0)).item() if eff_tok.item() > 0 else float("nan")

        # action_frac = {f"train/action_frac/k={cfg.keep_fracs[i]:.2f}": float((action_counts[i] / action_counts.sum().clamp_min(1.0)).item())
        #                for i in range(A)}
        # Log per-action fractions with tags (kXX-sYY-qZZ)
        action_frac = {}
        denom = action_counts.sum().clamp_min(1.0)
        for i in range(A):
            tag = action_spec.tags[i]
            action_frac[f"train/action_frac/{tag}"] = float((action_counts[i] / denom).item())
        if run is not None:
            metrics = {
                "train/avg_reward": mean_r,
                "train/avg_r_task": mean_r_task,
                "train/penalty_mean": mean_penalty,
                "train/abs_kl_mean": abs_task,
                "train/abs_penalty_mean": abs_penalty,
                "train/penalty_over_task_abs": ratio_abs,
                "train/avg_keep_effective": keep_mean,
                "train/mean_cost_eff": log_cost_eff,
                "train/lambda_keep":  global_step_state["lambda_keep"],
                "train/lambda_prune": global_step_state["lambda_prune"],
                "train/lambda_quant": global_step_state["lambda_quant"],
                "train/budget_gap_token": (log_cost_eff - C_tok) if not math.isnan(log_cost_eff) else float("nan"),
                "train/ppl_approx": ppl_approx,
                "update_step": global_step_state["update"],
                "micro_step": global_step_state["micro"],
            }
            metrics.update(action_frac)
            run.log(metrics)

        # Epoch running averages
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
        # --- 10% milestone save (flush case) ---
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
    tok,  # unused here, kept for signature compatibility
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
    # if optimizer is None:
    #     optimizer = torch.optim.AdamW(unwrap_policy.parameters(), lr=cfg.lr, fused=True)
    if optimizer is None:
        optimizer = torch.optim.AdamW(unwrap_policy.parameters(), lr=getattr(cfg, "lr", 2e-4), fused=True)

    # Ensure step state exists before any .get() usage
    if global_step_state is None:
        global_step_state = {"micro": 0, "update": 0}
    if "last_eval_update" not in global_step_state:
        global_step_state["last_eval_update"] = -1

    # --- LR scheduler (per-epoch): 5% warmup -> linear decay to 5e-5 ---
    total_updates = int(getattr(cfg, "_sft_total_updates", 0))
    if total_updates <= 0:
        # Fallback if __len__ exists; otherwise default to 1 update to avoid div-by-zero
        total_updates = max(1, math.ceil((len(dl) if hasattr(dl, "__len__") else 1) / max(1, int(getattr(cfg, "grad_accum_steps", 1)))))
    warmup_updates = max(1, int(0.05 * total_updates))
    peak_lr = float(getattr(cfg, "lr", 2e-4))
    min_lr  = float(getattr(cfg, "sft_min_lr", 5e-5))
    for g in optimizer.param_groups:
        g["lr"] = peak_lr  # warm up *to* this LR
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates,
        num_cycles=0.5,  # half-cycle: peak -> floor
    )
    # Auto-set milestone stride (every 10%) if not provided via CLI
    if global_step_state.get("save_stride") in (None, 0):
        global_step_state["save_stride"] = max(1, total_updates // 10)
    # knobs
    C_target = float(getattr(cfg, "C_target", getattr(cfg, "keep_target", 1.0)))
    pi_temperature = float(getattr(cfg, "pi_temperature", 0.7))
    tbptt_k = int(getattr(cfg, "policy_tbptt_k", 0))
    A = len(cfg.keep_fracs)

    # initialize step state (kept for parity; no-op if already set above)
    if global_step_state is None:
        global_step_state = {"micro": 0, "update": 0}
    if "lambda_keep" not in global_step_state:
        global_step_state["lambda_keep"] = float(getattr(cfg, "lambda_init", 0.0))

    # figure out initial prev action id to keep feature parity with your policy state
    try:
        init_state_probe = policy.init_state(1, device=device)
        initial_prev_action_idx = int(init_state_probe.last_action[0].item())
    except Exception:
        # fall back to dense index (k=1.0) or argmax KEEP
        if 1.0 in cfg.keep_fracs:
            initial_prev_action_idx = cfg.keep_fracs.index(1.0)
        else:
            # safe default
            initial_prev_action_idx = int(torch.tensor(cfg.keep_fracs).argmax().item())

    # model/policy modes
    model.eval()
    policy.train()

    # logging accumulators
    logs = {
        "avg_policy_ce": 0.0,
        "avg_policy_kl": 0.0,
        "avg_policy_acc": 0.0,
        "avg_keep_chosen": 0.0,  # effective-token keep (from teacher)
        "avg_cost_eff": 0.0,     # same as avg_keep_chosen in this matched-keep framing
        "avg_ppl_approx": 0.0,   # teacher ppl on the batch
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

        # 1) Ask the teacher (ON THIS BATCH ONLY) to produce the per-step assignments
        #    and the policy tensors we need (no heavy logic here).
        teach = evaluate_sft_teacher_matched_keep(
            cfg=cfg,
            model=model,
            dl=[batch],  # one-batch iterable
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
        h_seq = pb["h_seq"]                 # [T, B, H], detached
        e_seq = pb["e_seq"]                 # [T, B, E], detached
        scalars_seq = pb["scalars_seq"]     # [T, B, 10], detached (float32)
        prev_actions_seq = pb["prev_actions_seq"]         # [T, B] (long)
        teacher_actions_seq = pb["teacher_actions_seq"]   # [T, B] (long)
        same_prev = (prev_actions_seq[1:] == teacher_actions_seq[:-1]).float().mean()
        # 2) Policy forward + CE against teacher actions
        logits_seq, _values_unused = policy.forward_sequence(
            h_seq=h_seq,
            e_seq=e_seq,
            scalars_seq=scalars_seq,
            prev_actions_seq=prev_actions_seq,
            tbptt_k=tbptt_k,
            temperature=1.0,
        )  # [T, B, A]

        T, Bmb, _ = logits_seq.shape
        # effective-step mask weighting (index 3 in your scalar features)
        eff_mask = (scalars_seq[..., 3] > 0.5).float()    # [T,B]

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

        # ---- Gradient accumulation ----
        did_step = False
        if (global_step_state["micro"] % grad_accum) == 0:
            optimizer.zero_grad(set_to_none=True)
        (loss / grad_accum).backward()
        # Only step/clip at accumulation boundary
        if ((global_step_state["micro"] + 1) % grad_accum) == 0:
            torch.nn.utils.clip_grad_norm_(unwrap_policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            # clamp to floor
            for g in optimizer.param_groups:
                if g["lr"] < min_lr:
                    g["lr"] = min_lr
            global_step_state["update"] += 1
            did_step = True
            # --- 10% milestone save ---
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

        # 3) Logging (batch)
        ppl_batch = float(teach["ppl"])
        keep_eff_mean = float(teach["avg_keep_effective"])
        action_counts = torch.tensor(teach["action_hist"], device=device, dtype=torch.float32)
        if run is not None:
            total_actions = action_counts.sum().clamp_min(1.0)
            action_frac = {
                f"train/action_frac/k={float(cfg.keep_fracs[i]):.2f}": float((action_counts[i] / total_actions).item())
                for i in range(A)
            }
            metrics = {
                "train/prev_equals_next": float(same_prev),
                # "train/policy_ce": float(ce.item()),
                "train/policy_kl": float(loss.item()),
                "train/policy_ce": float(loss.item()),  # compat alias (same scalar)
                "train/policy_acc": float(acc.item()),
                "train/avg_keep_effective": keep_eff_mean,
                "train/mean_cost_eff": keep_eff_mean,  # same notion here
                "train/ppl_approx": ppl_batch,
                "train/lambda_keep": float(global_step_state.get("lambda_keep", 0.0)),
                # compatibility placeholders
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

        # 4) Epoch accumulators
        # logs["avg_policy_ce"] += float(ce.item())
        logs["avg_policy_kl"] += float(loss.item())
        logs["avg_policy_acc"] += float(acc.item())
        logs["avg_keep_chosen"] += keep_eff_mean
        logs["avg_cost_eff"] += keep_eff_mean
        logs["avg_ppl_approx"] += ppl_batch
        steps_done += 1
        global_step_state["micro"] += 1
        action_hist_epoch += action_counts

        # 5) Optional periodic eval + checkpoint (once per optimizer update)
        if did_step and (run is not None) and (val_dl is not None):
            upd = int(global_step_state["update"])
            if (upd > 0) and (upd % eval_every == 0) and (upd != global_step_state.get("last_eval_update", -1)):
                try:
                    sparse_stats = evaluate_stateful_policy_rollout(
                        cfg, model, policy, val_dl,
                        Ts=cfg.Ts, Tw=cfg.Tw, keep_fracs=cfg.keep_fracs,
                        context_len=cfg.context_len, rollout_len=cfg.rollout_len,
                        device=cfg.device, greedy=True, temperature=1.0,
                        lambda_keep=float(global_step_state.get("lambda_keep", 0.0)),
                        lambda_prune=float(global_step_state.get("lambda_prune", 0.0)),
                        lambda_quant=float(global_step_state.get("lambda_quant", 0.0)),
                    )
                    teach_val = evaluate_sft_teacher_matched_keep(
                        cfg, model, val_dl, Ts=cfg.Ts, Tw=cfg.Tw,
                        keep_fracs=tuple(cfg.keep_fracs),
                        target_keep_effective=C_target,
                        context_len=cfg.context_len, rollout_len=cfg.rollout_len,
                        device=cfg.device,
                    )
                    gap_nats = math.log(sparse_stats["ppl"]) - math.log(teach_val["ppl"])
                    gap_ratio = sparse_stats["ppl"] / teach_val["ppl"]
                    run.log({
                        "special/gap_to_teacher_ln_ppl": float(gap_nats),
                        "special/avg_keep_effective": float(sparse_stats["avg_keep_effective"]),
                        "special/gap_ratio_to_teacher": float(gap_ratio),
                        "special/teacher_ppl": float(teach_val["ppl"]),
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

    # Flush remaining grads if dataloader length isn't divisible by grad_accum
    if (global_step_state["micro"] % grad_accum) != 0:
        torch.nn.utils.clip_grad_norm_(unwrap_policy.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        # clamp to floor
        for g in optimizer.param_groups:
            if g["lr"] < min_lr:
                g["lr"] = min_lr
        global_step_state["update"] += 1
        # --- 10% milestone save (flush case) ---
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

    # normalize epoch logs
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
        snapshot_code(ckpt_dir, root_dir=os.getcwd(), skip_dirs=[".venv", ".git", "backup", "checkpoints", "__pycache__", "wandb", "block_cache"])
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
    SCALAR_D = int(getattr(cfg, "policy_scalar_dim", 12))

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
        # map_location to the current training device so we can resume on GPU/CPU seamlessly
        ckpt = torch.load(args.checkpoint_path, map_location=cfg.device)
        # Accept either 'policy_state_dict' (our save format) or generic 'state_dict'
        state_key = "policy_state_dict" if "policy_state_dict" in ckpt else (
            "state_dict" if "state_dict" in ckpt else None
        )
        if state_key is None:
            raise KeyError(f"'{args.checkpoint_path}' missing 'policy_state_dict' or 'state_dict'")
        unwrap(policy).load_state_dict(ckpt[state_key], strict=True)
        if is_main:
            print(f"[load] Loaded policy weights from {args.checkpoint_path}")

    if distributed:
        policy = DDP(policy, device_ids=[local_rank] if torch.cuda.is_available() else None,
                     output_device=local_rank if torch.cuda.is_available() else None)
    if is_main:
        _watch_target = unwrap(policy)
        wandb.watch(_watch_target, log="gradients", log_freq=500)
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
    # dataset_name: Optional[str] = "wikitext"
    # dataset_config: Optional[str] = "wikitext-2-raw-v1"
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

        TRAIN_FRACTION = 0.6
        max_batches = max(1, int(len(dl) * TRAIN_FRACTION))
        dl_epoch = limited_dl(dl, max_batches)

        if algo == "grpo":
            stats = train_one_epoch_grpo(
                tok, model, policy, cfg, dl_epoch, epoch=epoch,
                run=run, val_dl=val_dl, eval_every=cfg.eval_every_updates,
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
 

        # epoch-level summary
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
