import math
import os
import random
import time
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional

import copy
from pathlib import Path
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
from argparse import ArgumentParser
import json
import yaml

@dataclass
class Config:
    # --- Core model/runtime ---
    model_name: str = "meta-llama/Llama-3.2-1B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    epochs: int = 5
    seed: int = 1234
    eval_sparsity_bias: float = 0.0
    eval_quant_bias: float = 0.0
    eval_prune_bias: float = 0.0
    quest_page_size: int = 16

    algo: str = "grpo"
    reward_agg: str = None  # "mean" or "sum"
    grpo_quality_topq: Optional[float] = None  # if set, use quality-first GRPO with top-q fraction, otherwise lambda controller.
    grpo_rollouts_per_input: int = 16
    grpo_level: str = "process"  # "process" or "outcome"
    task_w_kl: float = 0.0       # LCE Style ground truth access
    grpo_norm: str = "center"    # "center" or "zscore"
    adv_whiten_global: bool = True
    kl_pi_ref_coef: float = 0.0  # turn on if you want π vs π_ref regularization
    reward_gamma: float = 0.82
    tol_token: float = 0.01
    tol_prune: float = 0.05
    tol_quant_bits: float = 1.0
    keep_tolerance: float = 0.01
    budget_tolerance: float = 0.1
    budget_penalty: str = "linear"
    budget_steer_beta: float = 0.5
    steer_beta: float = 0.5
    margin_soft_tau: float = 0.5
    score_soft_tau: float = 0.5

    lambda_lr_token: float = 0.5
    lambda_lr_prune: float = 0.5
    lambda_lr_quant: float = 0.5

    lambda_init_token: float = 25.0
    lambda_init_prune: float = 25.0
    lambda_init_quant: float = 25.0
    # --- Sparse attention controls used by train_one_epoch_ppo ---
    Ts: int = 4                      # number of "sinks"
    Tw: int = 1                      # trailing dense window
    keep_fracs: Tuple[float, ...] = (0.05, 0.3, 0.6, 0.85, 1.0)  # κ actions

    # Pruning: s40 -> keep 60% of channels, s70 -> keep 30%, s100 -> keep 100% (no pruning)
    struct_prune_choices: Tuple[str, ...] = ("s100",)
    # Quantization: q4/q8/q16 (16 -> identity)
    quant_choices: Tuple[str, ...] = ("q16",)
    keep_target: float = 0.3         # budget target (a.k.a. C_target)
    C_target_token: float = 0.3
    # Optional explicit alias; if None, code falls back to keep_target
    C_target: Optional[float] = None
    sparsity_criteria: str = "recency"   # "recency" | "relevancy" | "quest"
    relevancy_tier: str = "per_head"     # "per_head" | "per_layer"

    enable_prune_quant: bool = False
    C_target_prune: float = 0.70      # overrides keep_target for pruning budget
    C_target_quant_bits: float = 8.0   # e.g., 16, 8, 4
    # --- Sequence lengths / batching ---
    context_len: int = 512           # dense prefill length
    rollout_len: int = 16            # on-policy window length
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    lr: float = 3e-4
    sft_min_lr: float = 5e-5
    _sft_total_updates: int = 0

    # --- PPO / RL hyperparams (all read via getattr in training loop) ---
    ppo_clip: float = 0.2
    ppo_epochs: int = 3
    ppo_minibatch_size: int = 512
    ppo_target_batch_size: int = 2048  # total W*B before each PPO update
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    gamma: float = 0.95
    gae_lambda: float = 0.9
    pi_temperature: float = 0.7
    lambda_lr: float = 1e-2
    lambda_init: float = 0.0
    lambda_max: float = 20000.0
    lambda_ema_beta: float = 0.9
    cost_tradeoff_alpha: float = 1.0

    # --- Recurrent policy network hyperparams ---
    policy_d_model: int = 768
    policy_n_heads: int = 8
    policy_n_layers: int = 2
    policy_mlp_ratio: float = 4.0
    policy_action_dim: int = 32
    policy_max_len: int = 1024
    policy_scalar_dim: int = 8
    policy_dropout: float = 0.0
    policy_tbptt_k: int = 0

    # --- Probe / teacher settings (used by probing/eval helpers) ---
    # "lookahead_ce" | "dense_kl"
    probe_metric: str = "dense_kl"
    eps_acc: float = 0.04
    eps_rel: float = 0.02
    future_mask_rule: str = "policy"  # "dense" | "repeat" | "policy"
    horizon: int = 4
    tr_kl_coef: float = 0.0

    # --- Data ---
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    text_field: str = "text"
    dataset_pct: Optional[int] = 100   # e.g., 1 -> use split 'train[:1%]'
    max_blocks: Optional[int] = None

    # --- Eval cadence ---
    eval_every_updates: int = 10

def apply_cfg_overrides_from_file(cfg, path, is_main=True):
    """
    Load a JSON file and overwrite matching fields in `cfg` in-place.
    Special handling:
      - dtype: accepts "float32", "float16", "bfloat16", and aliases "fp32", "fp16", "bf16"
      - tuple fields (e.g., keep_fracs) can be provided as lists
      - null maps to None
    Unknown keys are ignored with a one-line notice on rank-0.
    """
    p = Path(path)
    with open(p, "r") as f:
        text = f.read()
    # Accept JSON or YAML (YAML only if pyyaml is installed)
    overrides = None
    if (p.suffix in {".yml", ".yaml"}) and yaml is not None:
        overrides = yaml.safe_load(text)
    if overrides is None:
        overrides = json.loads(text)

    def coerce_value(cur, v):
        if v is None:
            return None
        # torch.dtype from string
        if isinstance(cur, torch.dtype) and isinstance(v, str):
            m = {
                "float32": torch.float32, "fp32": torch.float32,
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            }
            return m.get(v.lower(), cur)
        # tuples from lists
        if isinstance(cur, tuple) and isinstance(v, (list, tuple)):
            return tuple(v)
        # basic scalar cast (int/float/bool/str)
        if isinstance(cur, (int, float, bool, str)):
            try:
                return type(cur)(v)
            except Exception:
                return v
        return v

    applied = 0
    for k, v in overrides.items():
        if hasattr(cfg, k):
            cur = getattr(cfg, k)
            setattr(cfg, k, coerce_value(cur, v))
            applied += 1
        else:
            if is_main:
                print(f"[config] Ignoring unknown key: {k}")
    if is_main:
        print(f"[config] Applied {applied} override(s) from {path}")
