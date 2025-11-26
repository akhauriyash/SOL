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


def detach_cache_to_tuple(cache):
    """
    Normalize any HF cache object (DynamicCache, list/tuple of kv tuples, etc.)
    into a pure tuple of (k, v, ...)-tensors and clone each tensor so later
    appends cannot mutate the original buffers.
    """
    if cache is None:
        return None
    # Normalize to legacy format
    legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
    # Deep-clone all tensors
    cloned_legacy = tuple(tuple(t.clone() for t in layer) for layer in legacy)
    # Convert back to a DynamicCache (what Qwen expects)
    return DynamicCache.from_legacy_cache(cloned_legacy)

# ---- cache helpers used for grouped forwards ----
@torch.no_grad()
def _cache_select(past_kv: DynamicCache, idx: torch.LongTensor) -> DynamicCache:
    legacy = past_kv.to_legacy_cache() if hasattr(past_kv, "to_legacy_cache") else past_kv
    sel = tuple((k.index_select(0, idx), v.index_select(0, idx)) for (k, v) in legacy)
    return DynamicCache.from_legacy_cache(sel)

@torch.no_grad()
def _cache_merge(full_cache: DynamicCache, idx: torch.LongTensor, sub_cache: DynamicCache) -> DynamicCache:
    full = full_cache.to_legacy_cache() if hasattr(full_cache, "to_legacy_cache") else full_cache
    sub  = sub_cache.to_legacy_cache()  if hasattr(sub_cache,  "to_legacy_cache")  else sub_cache
    out = []
    for (kf, vf), (ks, vs) in zip(full, sub):
        kf2 = kf.clone(); vf2 = vf.clone()
        kf2.index_copy_(0, idx, ks)
        vf2.index_copy_(0, idx, vs)
        out.append((kf2, vf2))
    return DynamicCache.from_legacy_cache(tuple(out))


select_cache_by_indices = _cache_select
merge_cache_by_indices = _cache_merge