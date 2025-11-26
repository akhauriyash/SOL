import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.backends.cuda as sdp

from utils.seeds import set_seed
from utils.model import load_lm_and_tokenizer
from utils.data import make_dataloader, limited_dl
from utils.eval_baselines import (
    evaluate_randomized_matched_sparsity,
    evaluate_dense_full,
)
from utils.config import Config

# SDPA settings (same as your existing script)
sdp.enable_flash_sdp(False)
sdp.enable_math_sdp(False)
sdp.enable_mem_efficient_sdp(True)  # SDPA only


# python eval_random_alloc.py \
#   --ckpt_dir /mnt/home/ya255/projects/SOL/checkpoints/RL_LCE_Quest4-20251028-213126 \
#   --mode latest \
#   --dataset_name wikitext \
#   --tgt_keep 1.0 \
#   --tgt_prune 0.65 \
#   --tgt_quant 1.0 \
#   --quant_choices q16 \
#   --prune_choices s100,s80,s60,s40 \

# -------------------------
# Basic eval configuration
# -------------------------
@dataclass
class EvalRandomCfg:
    CKPT_DIR: str = "/home/ya255/rl4e/checkpoints/GRPO_DKL_Relv-20251017-002523/"
    eval_batches: Optional[int] = None   # None = full loader
    split: str = "validation"
    mode: str = "latest"                 # "latest" or "best"
    dataset_name: Optional[str] = "allenai/c4"
    dataset_config: Optional[str] = "en"
    text_field: Optional[str] = "text"
    batch_size: Optional[int] = 16
    seed: int = 1234


# -------------------------
# Helpers
# -------------------------
def find_latest_ckpt(ckpt_dir: str, mode: str) -> Optional[str]:
    """
    Same helper as in your original script:
    - If policy_{mode}.pt exists, use it.
    - Otherwise, fall back to latest policy_epoch*.pt
    """
    if not os.path.isdir(ckpt_dir):
        return None
    latest = os.path.join(ckpt_dir, f"policy_{mode}.pt")
    if os.path.exists(latest):
        return latest
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

    # Dataset overrides
    if dataset_name is not None:
        cfg.dataset_name = dataset_name
    if dataset_config is not None:
        cfg.dataset_config = dataset_config
    if text_field is not None:
        cfg.text_field = text_field
    if batch_size is not None:
        cfg.batch_size = batch_size

    return cfg


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Random quantization/pruning/sparsity evaluation (no policy)."
    )
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None, choices=["latest", "best"])
    parser.add_argument(
        "--dataset_name", type=str, choices=["wikitext", "allenai/c4"], default=None
    )
    parser.add_argument("--eval_batches", type=int, default=None)

    # Target knobs (your main controls)
    parser.add_argument(
        "--tgt_keep",
        type=float,
        required=True,
        help="Target *effective* keep ratio for token sparsity (0-1).",
    )
    parser.add_argument(
        "--tgt_prune",
        type=float,
        default=1.0,
        help="Target structural keep ratio (1.0 means no pruning).",
    )
    parser.add_argument(
        "--tgt_quant",
        type=float,
        default=1.0,
        help=(
            "Target quantization ratio (semantics: 16 * tgt_quant "
            "≈ effective bit-width; e.g. 1.0 -> 16-bit, 0.5 -> 8-bit)."
        ),
    )

    # Optional overrides for choices (comma-separated)
    parser.add_argument(
        "--quant_choices",
        type=str,
        default=None,
        help="Comma-separated quantization choices, e.g. 'q16,q8,q4'. "
             "If not set, uses cfg.quant_choices.",
    )
    parser.add_argument(
        "--prune_choices",
        type=str,
        default=None,
        help="Comma-separated pruning choices, e.g. 's100,s75,s50'. "
             "If not set, uses cfg.struct_prune_choices.",
    )

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    # --- Build simple eval config ---
    E = EvalRandomCfg()
    if args.ckpt_dir is not None:
        E.CKPT_DIR = args.ckpt_dir
    if args.mode is not None:
        E.mode = args.mode
    if args.eval_batches is not None:
        E.eval_batches = args.eval_batches
    if args.dataset_name is not None:
        E.dataset_name = args.dataset_name
        E.dataset_config = (
            "wikitext-2-raw-v1" if args.dataset_name == "wikitext" else "en"
        )
    if args.batch_size is not None:
        E.batch_size = args.batch_size
    if args.seed is not None:
        E.seed = args.seed

    # --- Seed everything (for reproducible randomness) ---
    set_seed(E.seed)

    # --- Find and load checkpoint (just for cfg/meta; not using policy) ---
    ckpt_path = find_latest_ckpt(E.CKPT_DIR, E.mode)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {E.CKPT_DIR}")

    cfg = load_cfg_from_checkpoint_or_yaml(
        ckpt_dir=E.CKPT_DIR,
        ckpt_path=ckpt_path,
        dataset_name=E.dataset_name,
        dataset_config=E.dataset_config,
        text_field=E.text_field,
        batch_size=E.batch_size,
    )

    # Override eval batch size if needed
    if E.batch_size is not None:
        cfg.batch_size = E.batch_size

    # Optional: override quant/prune choices from CLI
    if args.quant_choices is not None:
        cfg.quant_choices = tuple(
            [q.strip() for q in args.quant_choices.split(",") if q.strip()]
        )
    if args.prune_choices is not None:
        cfg.struct_prune_choices = tuple(
            [p.strip() for p in args.prune_choices.split(",") if p.strip()]
        )

    # Load checkpoint once more just to possibly pick up updated keep_fracs
    sd = torch.load(ckpt_path, map_location="cpu")
    sd_cfg = sd.get("cfg", None)
    if sd_cfg is not None and "keep_fracs" in sd_cfg:
        cfg.keep_fracs = tuple(sd_cfg["keep_fracs"])

    meta = sd.get("meta")
    if meta:
        kind = meta.get("kind", "unknown")
        cmd = meta.get("command", "")
        paths = meta.get("config_paths", {})
        print(f"[ckpt] kind: {kind}")
        print(f"[ckpt] command: {cmd}")
        print(
            f"[ckpt] config_paths: base={paths.get('base')}, rl={paths.get('rl')}"
        )
    else:
        print("[ckpt] No meta found in checkpoint.")

    # In your current setup you use "quest" sparsity criteria; keep the same assert
    assert cfg.sparsity_criteria == "quest", (
        "This eval script currently assumes 'quest' sparsity_criteria. "
        "Adjust if you're using another scheme."
    )

    # --- Load LM and tokenizer ---
    tok, model = load_lm_and_tokenizer(cfg)

    # --- DataLoader ---
    dl = make_dataloader(
        cfg,
        tok,
        split=E.split,
        shuffle=False,
        distributed=False,
    )

    print(f"\nLoaded checkpoint (for config/meta): {ckpt_path}")
    print(f"Device={cfg.device}, dtype={cfg.dtype}, model={cfg.model_name}")
    print(
        f"Eval split='{E.split}', batches={E.eval_batches}, "
        f"batch_size={cfg.batch_size}"
    )
    print(f"Structure: Ts={cfg.Ts}  Tw={cfg.Tw}  keep_fracs={cfg.keep_fracs}")
    print(
        f"context_len={cfg.context_len}  rollout_len={cfg.rollout_len}\n"
    )

    # -------------------------
    # Dense baseline (optional but useful for comparison)
    # -------------------------
    print("Running dense baseline (no sparsity / no quant / no pruning)...")
    start_time = time.time()
    dense_res = evaluate_dense_full(
        model,
        limited_dl(dl, E.eval_batches),
        cfg.context_len,
        cfg.rollout_len,
        cfg.device,
    )
    dense_time = time.time() - start_time

    print(
        f"\nDense baseline\t\t: ppl={dense_res['ppl']:.3f}  "
        f"tokens={dense_res['tokens']}\t(time={dense_time:.2f}s)\n"
    )

    # -------------------------
    # Randomized matched evaluation
    # -------------------------
    print(
        "Running RANDOM allocation with user-specified targets:\n"
        f"  tgt_keep_eff={args.tgt_keep:.4f}, "
        f"tgt_prune_keep={args.tgt_prune:.4f}, "
        f"tgt_quant_ratio={args.tgt_quant:.4f} "
        f"(~{16 * args.tgt_quant:.2f} effective bits)\n"
    )

    start_time = time.time()
    rand_res = evaluate_randomized_matched_sparsity(
        cfg,
        model,
        limited_dl(dl, E.eval_batches),
        Ts=cfg.Ts,
        Tw=cfg.Tw,
        keep_fracs=cfg.keep_fracs,
        prune_choices=getattr(cfg, "struct_prune_choices", ("s100",)),
        quant_choices=getattr(cfg, "quant_choices", ("q16",)),
        target_keep_effective=args.tgt_keep,
        target_prune_keep=args.tgt_prune,
        target_quant_ratio=args.tgt_quant,
        context_len=cfg.context_len,
        rollout_len=cfg.rollout_len,
        device=cfg.device,
        struct_on_non_eff=False,
    )
    rand_time = time.time() - start_time

    # -------------------------
    # Print summary
    # -------------------------
    print(
        f"\nRandom alloc\t\t: ppl={rand_res['ppl']:.3f}  "
        f"keep_all={rand_res['avg_keep_all']:.3f}  "
        f"keep_eff={rand_res['avg_keep_effective']:.3f}  "
        f"prune_keep={rand_res['avg_prune_keep']:.3f}  "
        f"quant_ratio={16 * rand_res['avg_quant_ratio']:.3f}  "
        f"tokens={rand_res['tokens_effective']}/{rand_res['tokens']}\t"
        f"(time={rand_time:.2f}s)\n"
    )

    if "action_probs" in rand_res:
        probs = ", ".join(f"{p:.3f}" for p in rand_res["action_probs"])
        levels = ", ".join(f"{k:.3f}" for k in cfg.keep_fracs)
        print(f"Sparsity levels (κ order)      : [{levels}]")
        print(f"Random action probs (κ order)  : [{probs}]")

    print("\n=== Comparison (validation) ===\n")
    print(
        f"Dense baseline\t\t: ppl={dense_res['ppl']:.3f}  "
        f"tokens={dense_res['tokens']}"
    )
    print(
        f"Random alloc\t\t: ppl={rand_res['ppl']:.3f}  "
        f"keep_all={rand_res['avg_keep_all']:.3f}  "
        f"keep_eff={rand_res['avg_keep_effective']:.3f}  "
        f"prune_keep={rand_res['avg_prune_keep']:.3f}  "
        f"quant_ratio={16 * rand_res['avg_quant_ratio']:.3f}  "
        f"tokens={rand_res['tokens_effective']}/{rand_res['tokens']}"
    )


if __name__ == "__main__":
    main()
