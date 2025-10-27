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
import json, hashlib
from pathlib import Path
import tempfile


def _cache_key(cfg, tokenizer, split):
    meta = {
        "dataset": f"{cfg.dataset_name}:{getattr(cfg, 'dataset_config', None)}",
        "split": split,
        "text_field": cfg.text_field,
        "tokenizer": getattr(tokenizer, "name_or_path", str(tokenizer)),
        "block_size": cfg.context_len + cfg.rollout_len + 1,
        "max_blocks": cfg.max_blocks,  # affects early stop/total
        "version": 1,  # bump if you change chunking logic
    }
    payload = json.dumps(meta, sort_keys=True).encode()
    key = hashlib.sha1(payload).hexdigest()
    return key, meta

def _cache_paths(cfg, tokenizer, split):
    key, meta = _cache_key(cfg, tokenizer, split)
    cache_dir = Path(getattr(cfg, "cache_dir", "./block_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = cache_dir / f"{split}-{key}"
    return base.with_suffix(".npy"), base.with_suffix(".json"), meta


class TokenBlockDataset(torch.utils.data.Dataset):
    """
    Emits blocks of length (context_len + rollout_len + 1) so we can:
      - prefill on [:context_len]
      - roll W steps over [context_len : context_len+W]
      - compute labels shifted by +1
    """
    def __init__(self, cfg, tokenizer, block_size: int, split="train"):
        main = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
        t0 = time.perf_counter()

        cache_path, meta_path, meta = _cache_paths(cfg, tokenizer, split)
        rebuild = getattr(cfg, "rebuild_cache", False)

        # Try to load cache
        if (not rebuild) and cache_path.exists():
            if main:
                print(f"[data] Loading cached blocks from {cache_path}")
            arr = np.load(cache_path, mmap_mode="r")  # shape: (N, block_size)
            self.blocks = arr
            if main:
                print(f"[data] Loaded {len(self.blocks)} cached blocks in {time.perf_counter()-t0:.1f}s")
            return

        # If distributed: non‑zero ranks wait for rank 0 to build & save
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            if main:
                print("[data] Waiting for rank 0 to build cache...")
            dist.barrier()
            # After barrier, cache should exist
            if not cache_path.exists():
                raise FileNotFoundError(f"Expected cache at {cache_path} after barrier, but not found.")
            arr = np.load(cache_path, mmap_mode="r")
            self.blocks = arr
            return

        # Rank 0 builds
        if main:
            print(f"[data] Loading split='{split}' (streaming=True); block_size={block_size}")
        raw = load_dataset(cfg.dataset_name, cfg.dataset_config, split=split, streaming=True)
        if split != "train":
            raw = islice(raw, 2_000)
        else:
            raw = islice(raw, 1_000_000)
        texts = (x[cfg.text_field] for x in raw if x.get(cfg.text_field) and x[cfg.text_field].strip())

        blocks = chunk_tokens(cfg, tokenizer, texts, block_size)
        if cfg.max_blocks is not None:
            blocks = blocks[:cfg.max_blocks]

        if main:
            dt = time.perf_counter() - t0
            print(f"[data] Built {len(blocks)} blocks in {dt:.1f}s; saving cache -> {cache_path}")

        # Save cache atomically (avoid np.save adding '.npy' to string paths)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(blocks, dtype=np.int32)  # compress in 32bit
        with tempfile.NamedTemporaryFile(
            dir=str(cache_path.parent),
            prefix=cache_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as f:
            np.save(f, arr)           # pass FILE HANDLE -> no auto '.npy' suffix
            f.flush()
            os.fsync(f.fileno())      # ensure durability before rename
            tmp_name = f.name
        os.replace(tmp_name, cache_path)  # atomic within same filesystem
        with open(meta_path, "w") as f:
            json.dump({**meta, "num_blocks": int(arr.shape[0])}, f)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Load memmap view
        arr = np.load(cache_path, mmap_mode="r")
        self.blocks = arr

    def __len__(self):
        # works for list or numpy array/memmap
        return len(self.blocks)

    def __getitem__(self, idx: int):
        return torch.tensor(self.blocks[idx], dtype=torch.long)

def chunk_tokens(cfg, tokenizer, texts, block_size: int, batch_size: int = 512) -> List[List[int]]:
    """Tokenize a text stream and emit fixed-size blocks"""
    blocks, buf, batch = [], [], []
    show_bar = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
    pbar = tqdm(
        total=cfg.max_blocks if cfg.max_blocks is not None else None,
        desc="building blocks",
        mininterval=1.0,
        disable=not show_bar,
    )

    def _flush_batch(batch_list):
        nonlocal blocks, buf
        if not batch_list:
            return
        enc = tokenizer(batch_list, add_special_tokens=False)["input_ids"]
        for ids in enc:
            buf.extend(ids)
            while len(buf) >= block_size:
                blocks.append(buf[:block_size])
                buf = buf[block_size:]
                pbar.update(1)
                if cfg.max_blocks is not None and len(blocks) >= cfg.max_blocks:
                    return True  # early stop
        batch_list.clear()
        return False

    for t in texts:
        batch.append(t)
        if len(batch) >= batch_size:
            if _flush_batch(batch):
                break

    # flush tail
    _flush_batch(batch)
    pbar.close()
    return blocks

def make_dataloader(cfg, tokenizer, split="train", shuffle=None, distributed=False):
    block_size = cfg.context_len + cfg.rollout_len + 1
    ds = TokenBlockDataset(cfg, tokenizer, block_size=block_size, split=split)
    if shuffle is None:
        shuffle = (split == "train")
    sampler = None
    gen = None
    seed = int(getattr(cfg, "seed", 0))
    if distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed)
        # when using a sampler, DataLoader must have shuffle=False
        shuffle = False
    elif shuffle:
        # Deterministic shuffling when not using DistributedSampler
        gen = torch.Generator()
        gen.manual_seed(seed)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        generator=gen,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[data] DataLoader('{split}'): {len(ds)} blocks • batch_size={cfg.batch_size} • shuffle={shuffle} • distributed={distributed}")
    return dl

def limited_dl(dl, max_batches=None):
    """
    Yield at most max_batches from a dataloader (or all if None),
    but keep a known __len__ so tqdm shows a proper progress bar.
    """
    if max_batches is None:
        return dl

    class _SizedSlice:
        __slots__ = ("_dl", "_n")
        def __init__(self, _dl, _n):
            self._dl = _dl
            self._n = int(_n)
        def __iter__(self):
            # fresh iterator each time (per-epoch safe)
            return islice(iter(self._dl), self._n)
        def __len__(self):
            return self._n

    return _SizedSlice(dl, max_batches)