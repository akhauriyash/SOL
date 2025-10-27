# utilities.py
import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
import torch
import torch.nn.functional as F
from utils.config import Config
from tqdm import tqdm

import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
import numpy as np


def find_latest_ckpt(ckpt_dir: str, mode: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    latest = os.path.join(ckpt_dir, f"policy_{mode}.pt")
    if os.path.exists(latest):
        return latest
    cands = [f for f in os.listdir(ckpt_dir) if f.startswith("policy_epoch") and f.endswith(".pt")]
    if not cands:
        return None
    cands.sort()
    return os.path.join(ckpt_dir, cands[-1])

def load_cfg_from_checkpoint_or_yaml(
    ckpt_dir: str,
    ckpt_path: str,
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
) -> Config:
    cfg = Config()
    sd_cpu = torch.load(ckpt_path, map_location="cpu")
    sd_cfg = sd_cpu.get("cfg")
    if sd_cfg:
        for k, v in sd_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    else:
        meta = sd_cpu.get("meta", None)
        if meta is None:
            meta_path = os.path.join(ckpt_dir, "train_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
        base_rel = None
        if meta is not None:
            base_rel = meta.get("config_paths", {}).get("base")
        if base_rel:
            yaml_path = os.path.join(ckpt_dir, "code", base_rel)
            if os.path.exists(yaml_path):
                from utils.config import apply_cfg_overrides_from_file
                apply_cfg_overrides_from_file(cfg, yaml_path, is_main=True)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        cfg.dtype = torch.bfloat16
    else:
        cfg.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if dataset_name is not None:
        cfg.dataset_name = dataset_name
    if dataset_config is not None:
        cfg.dataset_config = dataset_config
    return cfg