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
import wandb
from tqdm import tqdm
from itertools import islice



def unwrap(module):
    """Return underlying nn.Module if wrapped in DDP/FSDP-like wrappers."""
    return module.module if hasattr(module, "module") else module

# ----------------------------
# Load model/tokenizer and freeze LM
# ----------------------------
def load_lm_and_tokenizer(cfg, dense_only=False):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, trust_remote_code=True)
    # Ensure model sees pad token if needed
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # In DDP, keep one full model copy per process on its local device.
    # Avoid model-parallel 'device_map="auto"'.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=cfg.dtype,
    )
    model.to(cfg.device)
    # Force eager attention so our 4-D mask is honored; disable dropout
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"
    model.eval()
    sparsity_mode = getattr(cfg, "sparsity_criteria", "recency")
    if not dense_only:
        if sparsity_mode == "relevancy":
            from .masks import enable_relevancy_attention

            enable_relevancy_attention(model, tier=getattr(cfg, "relevancy_tier", "per_head"), cfg=cfg)
        elif sparsity_mode == "quest":
            from .masks import enable_quest_attention

            enable_quest_attention(model, page_size=getattr(cfg, "quest_page_size", 16))
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model

