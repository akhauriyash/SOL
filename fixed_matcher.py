#!/usr/bin/env python
# fixed_only_profile.py
#
# Same CLI args/defaults as policy_action_variability.py,
# but evaluates ONLY the fixed matched baseline at the requested targets.
#
# Usage stays the same:
#   python fixed_only_profile.py --ckpt_dir ... --tgt_keep ... --tgt_prune_keep ... --tgt_quant_bits ...

import os
import json
import time
import csv
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.backends.cuda as sdp

from utils.seeds import set_seed
from utils.model import load_lm_and_tokenizer
from utils.data import make_dataloader, limited_dl
from utils.eval_baselines import evaluate_fixed_matched_keep
from utils.config import Config

# SDPA: mem-efficient only
sdp.enable_flash_sdp(False)
sdp.enable_math_sdp(False)
sdp.enable_mem_efficient_sdp(True)


@dataclass
class EvalCfg:
    ckpt_dir: str = ""
    eval_batches: Optional[int] = None   # None = run full loader
    split: str = "validation"
    mode: str = "latest"                # "latest" or "best"
    dataset_name: Optional[str] = "wikitext"
    dataset_config: Optional[str] = "en"
    text_field: Optional[str] = "text"
    batch_size: Optional[int] = 16
    seed: int = 1234


def find_latest_ckpt(ckpt_dir: str, mode: str) -> Optional[str]:
    """
    Prefer policy_{mode}.pt, otherwise fall back to the latest policy_epoch*.pt.
    """
    if not os.path.isdir(ckpt_dir):
        return None

    preferred = os.path.join(ckpt_dir, f"policy_{mode}.pt")
    if os.path.exists(preferred):
        return preferred

    cands = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith("policy_epoch") and f.endswith(".pt")
    ]
    if not cands:
        return None
    cands.sort()
    return os.path.join(ckpt_dir, cands[-1])


def load_cfg_from_checkpoint_or_yaml(
    ckpt_dir: str,
    ckpt_path: str,
    dataset_name: Optional[str],
    dataset_config: Optional[str],
    text_field: Optional[str],
    batch_size: Optional[int],
) -> Config:
    """
    Load the *training* config for evaluation.
    Priority:
      1) 'cfg' dict embedded in the checkpoint
      2) YAML recorded in 'meta.config_paths.base' -> ckpt_dir/code/<relpath>
         (or train_meta.json)
      3) Default Config()

    Then apply only dataset-related overrides.
    """
    cfg = Config()

    sd_cpu = torch.load(ckpt_path, map_location="cpu")
    sd_cfg = sd_cpu.get("cfg")

    if sd_cfg:
        print("[eval] Using training config embedded in checkpoint.")
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
        if meta is None:
            print("[eval] No 'meta' found in checkpoint or train_meta.json.")
        base_rel = None
        if meta is not None:
            base_rel = meta.get("config_paths", {}).get("base")
            kind = meta.get("kind", "unknown")
            print(f"[eval] meta.kind={kind}")
            print(f"[eval] meta.config_paths={meta.get('config_paths', {})}")
        if base_rel:
            yaml_path = os.path.join(ckpt_dir, "code", base_rel)
            if os.path.exists(yaml_path):
                from utils.config import apply_cfg_overrides_from_file
                apply_cfg_overrides_from_file(cfg, yaml_path, is_main=True)
                print(f"[eval] Applied base YAML overrides from snapshot: {yaml_path}")
            else:
                print(f"[eval] Saved base config not found at {yaml_path}; using defaults.")
        else:
            print("[eval] No training config in checkpoint or meta; using defaults.")

    # Device / dtype
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Dataset overrides (optional)
    if dataset_name is not None:
        cfg.dataset_name = dataset_name
    if dataset_config is not None:
        cfg.dataset_config = dataset_config
    if text_field is not None:
        cfg.text_field = text_field
    if batch_size is not None:
        cfg.batch_size = batch_size

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="FIXED-ONLY: Evaluate fixed matched baseline at requested budgets and log results."
    )

    # Checkpoint location (same flags)
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Directory containing policy checkpoints (policy_latest.pt, etc.).")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Optional explicit path to a checkpoint .pt file. "
                             "If set, overrides --ckpt_dir/--mode.")
    parser.add_argument("--mode", type=str, default="latest", choices=["latest", "best"])

    # Data options (same flags)
    parser.add_argument("--dataset_name", type=str,
                        choices=["wikitext", "allenai/c4"], default=None)
    parser.add_argument("--eval_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--seed", type=int, default=1234)

    # Targets (budgets) (same flags)
    parser.add_argument("--tgt_keep", type=float, default=None,
                        help="Target effective keep ratio (tokens_effective / tokens) in [0,1].")
    parser.add_argument("--tgt_prune_keep", type=float, default=None,
                        help="Target structured prune keep ratio in [0,1].")
    parser.add_argument("--tgt_quant_ratio", type=float, default=None,
                        help="Target quantization ratio in [0,1], where bits = 16 * ratio. "
                             "Alternative to --tgt_quant_bits.")
    parser.add_argument("--tgt_quant_bits", type=float, default=None,
                        help="Target quantization *bit* budget. E.g. 7 -> ratio 7/16.")

    # Optional eval biases (same flags; we just pass through to cfg like before)
    parser.add_argument("--sparsity_bias", type=float, default=None,
                        help="Optional eval sparsity bias.")
    parser.add_argument("--quant_bias", type=float, default=None,
                        help="Optional eval quantization bias.")
    parser.add_argument("--prune_bias", type=float, default=None,
                        help="Optional eval pruning bias.")

    # CSV logging (same flags)
    parser.add_argument("--csv_path", type=str, default="policy_eval_results.csv",
                        help="CSV file to append results to.")

    # Kept for CLI compatibility; ignored
    parser.add_argument("--num_trials", type=int, default=0,
                        help="(ignored in fixed-only) kept for CLI compatibility.")
    parser.add_argument("--do_emc_and_driftaware", action="store_true",
                        help="(ignored in fixed-only) kept for CLI compatibility.")

    args = parser.parse_args()

    if args.ckpt_path is None and args.ckpt_dir is None:
        parser.error("You must specify either --ckpt_dir or --ckpt_path.")

    # -------------------- Build EvalCfg -------------------- #
    E = EvalCfg()
    E.split = args.split
    E.mode = args.mode
    E.eval_batches = args.eval_batches
    E.seed = args.seed

    if args.dataset_name is not None:
        E.dataset_name = args.dataset_name
        E.dataset_config = ("wikitext-2-raw-v1" if args.dataset_name == "wikitext" else "en")

    if args.batch_size is not None:
        E.batch_size = args.batch_size

    # Resolve checkpoint path (same behavior)
    if args.ckpt_path is not None:
        ckpt_path = args.ckpt_path
        if args.ckpt_dir is not None:
            E.ckpt_dir = args.ckpt_dir
        else:
            E.ckpt_dir = os.path.dirname(ckpt_path)
    else:
        E.ckpt_dir = args.ckpt_dir
        ckpt_path = find_latest_ckpt(E.ckpt_dir, E.mode)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in {E.ckpt_dir}")

    print(f"[fixed-only] Using checkpoint: {ckpt_path}")

    # -------------------- Config / model / data -------------------- #
    set_seed(E.seed)

    cfg = load_cfg_from_checkpoint_or_yaml(
        ckpt_dir=E.ckpt_dir,
        ckpt_path=ckpt_path,
        dataset_name=E.dataset_name,
        dataset_config=E.dataset_config,
        text_field=E.text_field,
        batch_size=E.batch_size,
    )

    # Biases (same pattern)
    cfg.eval_sparsity_bias = float(getattr(cfg, "eval_sparsity_bias", 0.0))
    cfg.eval_quant_bias = float(getattr(cfg, "eval_quant_bias", 0.0))
    cfg.eval_prune_bias = float(getattr(cfg, "eval_prune_bias", 0.0))

    if args.sparsity_bias is not None:
        cfg.eval_sparsity_bias = float(args.sparsity_bias)
    if args.quant_bias is not None:
        cfg.eval_quant_bias = float(args.quant_bias)
    if args.prune_bias is not None:
        cfg.eval_prune_bias = float(args.prune_bias)

    # Load LM + tokenizer
    tok, model = load_lm_and_tokenizer(cfg)

    # Data loader
    dl = make_dataloader(
        cfg,
        tok,
        split=E.split,
        shuffle=False,
        distributed=False,
    )

    # -------------------- Targets / budgets -------------------- #
    # Defaults if user doesn't pass them (dense)
    target_keep_effective = float(args.tgt_keep) if args.tgt_keep is not None else 1.0
    target_prune_keep = float(args.tgt_prune_keep) if args.tgt_prune_keep is not None else 1.0

    if args.tgt_quant_bits is not None:
        target_qbits = float(args.tgt_quant_bits)
    elif args.tgt_quant_ratio is not None:
        target_qbits = 16.0 * float(args.tgt_quant_ratio)
    else:
        target_qbits = 16.0

    target_quant_ratio = target_qbits / 16.0

    print(
        f"[fixed-only] targets: keep_eff={target_keep_effective:.3f}, "
        f"prune_keep={target_prune_keep:.3f}, quant_bits={target_qbits:.3f}"
    )

    # -------------------- FIXED baseline ONLY -------------------- #
    start_time = time.time()
    fixed = evaluate_fixed_matched_keep(
        cfg,
        model,
        limited_dl(dl, E.eval_batches),
        Ts=cfg.Ts,
        Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_keep_effective=target_keep_effective,
        target_prune_keep=target_prune_keep,
        target_quant_ratio=target_quant_ratio,
        context_len=cfg.context_len,
        rollout_len=cfg.rollout_len,
        device=cfg.device,
        struct_on_non_eff=False,
    )
    total_time = time.time() - start_time

    print(
        f"\nFixed (forced targets)\t\t: "
        f"ppl={fixed['ppl']:.3f}  "
        f"keep_all={fixed['avg_keep_all']:.3f}  "
        f"keep_eff={fixed['avg_keep_effective']:.3f}  "
        f"prune_keep={fixed['avg_prune_keep']:.3f}  "
        f"quant_bits={16*fixed['avg_quant_ratio']:.3f}\t"
        f"tokens={fixed['tokens_effective']}/{fixed['tokens']}  "
        f"(time={total_time:.2f}s)\n"
    )

    # -------------------- CSV logging (fixed-only row) -------------------- #
    csv_path = args.csv_path
    ckpt_dir_last = os.path.basename(os.path.normpath(E.ckpt_dir))

    row = {
        "ckpt_dir": ckpt_dir_last,
        "ckpt_path": ckpt_path,
        "mode": E.mode,
        "dataset_name": cfg.dataset_name,
        "dataset_config": getattr(cfg, "dataset_config", None),
        "split": E.split,
        "eval_batches": E.eval_batches,

        # Targets requested
        "target_keep_effective": float(target_keep_effective),
        "target_prune_keep": float(target_prune_keep),
        "target_quant_ratio": float(target_quant_ratio),
        "target_quant_bits": float(target_qbits),

        # Fixed metrics
        "fixed_ppl": float(fixed["ppl"]),
        "fixed_keep_all": float(fixed["avg_keep_all"]),
        "fixed_keep_effective": float(fixed["avg_keep_effective"]),
        "fixed_prune_keep": float(fixed["avg_prune_keep"]),
        "fixed_quant_ratio": float(fixed["avg_quant_ratio"]),
        "fixed_tokens_effective": int(fixed["tokens_effective"]),
        "fixed_tokens": int(fixed["tokens"]),

        # Biases used
        "sparsity_bias": float(cfg.eval_sparsity_bias),
        "prune_bias": float(cfg.eval_prune_bias),
        "quant_bias": float(cfg.eval_quant_bias),

        "elapsed_s": float(total_time),
    }

    fieldnames = list(row.keys())
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[csv] Appended fixed-only results to {csv_path}")


if __name__ == "__main__":
    main()
