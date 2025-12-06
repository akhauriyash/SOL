import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.backends.cuda as sdp

from utils.seeds import set_seed
from utils.model import load_lm_and_tokenizer
from utils.data import make_dataloader, limited_dl
from utils.eval_baselines import evaluate_randomized_matched_sparsity
from utils.config import Config

import csv
from datetime import datetime

# SDPA settings (same as your existing script)
sdp.enable_flash_sdp(False)
sdp.enable_math_sdp(False)
sdp.enable_mem_efficient_sdp(True)  # SDPA only

# python action_variability.py \
#   --tgt_keep 1.0 \
#   --tgt_prune 1.0 \
#   --tgt_quant 0.4 \
#   --quant_choices q5,q8,q16 \
#   --prune_choices s100 \
#   --eval_batches 100 \
#   --seed 5612 \
#   --num_trials 10 \
#   --csv_path action_variability.csv


# -------------------------
# Basic eval configuration
# -------------------------
@dataclass
class EvalRandomCfg:
    CKPT_DIR: str = "/mnt/home/ya255/projects/SOL/current_valid/nLRL_LCE_TokSparse-20251205-233007"
    eval_batches: Optional[int] = None   # None = full loader
    split: str = "validation"
    mode: str = "latest"                 # "latest" or "best"
    dataset_name: Optional[str] = "wikitext"
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


def sanitize_csv_field(value) -> str:
    """
    Convert anything to a string and replace commas with '|' so that
    CSV delimiter logic (',') is never confused.
    """
    if value is None:
        return ""
    s = str(value)
    return s.replace(",", "|")

def join_list(values, float_fmt: Optional[str] = None) -> str:
    """
    Join a list/tuple (or single value) into a '|' separated string.
    If float_fmt is provided, apply it to each element.
    Robust against accidentally passing scalars or strings.
    """
    # If we get a scalar or string, treat it as a single element
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        values = [values]

    out = []
    for v in values:
        if float_fmt is not None:
            out.append(float_fmt.format(v))
        else:
            out.append(str(v))
    return "|".join(out)
# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Repeat random quant/prune/sparsity evaluation for variability, log to CSV."
    )
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None, choices=["latest", "best"])
    parser.add_argument(
        "--dataset_name", type=str, choices=["wikitext", "allenai/c4"], default="wikitext"
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
    parser.add_argument(
        "--keep_fracs",
        type=str,
        default=None,
        help="Comma-separated keep_fracs, e.g. '1.0,0.5,0.25'. "
             "If not set, uses cfg.keep_fracs (from training/checkpoint).",
    )

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234,
                        help="Base seed; each trial uses seed+i.")
    parser.add_argument("--num_trials", type=int, default=5,
                        help="Number of random allocation trials.")
    parser.add_argument("--csv_path", type=str, default="random_alloc_results.csv",
                        help="Path to CSV file to append results to.")

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

    if args.keep_fracs is not None:
        cfg.keep_fracs = tuple(
            float(x.strip()) for x in args.keep_fracs.split(",") if x.strip()
        )
        print(f"[cfg] Overriding keep_fracs from CLI: {cfg.keep_fracs}")
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

    print(
        "Running RANDOM allocation with user-specified targets:\n"
        f"  tgt_keep_eff={args.tgt_keep:.4f}, "
        f"tgt_prune_keep={args.tgt_prune:.4f}, "
        f"tgt_quant_ratio={args.tgt_quant:.4f} "
        f"(~{16 * args.tgt_quant:.2f} effective bits)\n"
    )

    # -------------------------
    # Repeat random allocation multiple times
    # -------------------------
    ppl_trials: List[float] = []
    keep_eff_trials: List[float] = []
    prune_keep_trials: List[float] = []
    quant_bits_trials: List[float] = []
    tokens_eff_trials: List[int] = []
    tokens_trials: List[int] = []
    seeds_used: List[int] = []

    for i in range(args.num_trials):
        trial_seed = args.seed + i
        seeds_used.append(trial_seed)
        set_seed(trial_seed)

        print(f"\n[trial {i+1}/{args.num_trials}] seed={trial_seed}")
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
        trial_time = time.time() - start_time

        ppl = rand_res["ppl"]
        keep_eff = rand_res["avg_keep_effective"]
        prune_keep = rand_res["avg_prune_keep"]
        quant_bits = 16.0 * rand_res["avg_quant_ratio"]
        tokens_eff = rand_res["tokens_effective"]
        tokens_total = rand_res["tokens"]

        ppl_trials.append(ppl)
        keep_eff_trials.append(keep_eff)
        prune_keep_trials.append(prune_keep)
        quant_bits_trials.append(quant_bits)
        tokens_eff_trials.append(tokens_eff)
        tokens_trials.append(tokens_total)

        print(
            f"[trial {i+1}] ppl={ppl:.3f}  "
            f"keep_eff={keep_eff:.3f}  "
            f"prune_keep={prune_keep:.3f}  "
            f"quant_bits={quant_bits:.3f}  "
            f"tokens={tokens_eff}/{tokens_total}  "
            f"(time={trial_time:.2f}s)"
        )
        print(f"Frequency: ", rand_res["action_probs"])

    # -------------------------
    # Write CSV row
    # -------------------------
    csv_path = args.csv_path
    csv_exists = os.path.exists(csv_path)

    fieldnames = [
        "timestamp",
        "ckpt_dir",
        "ckpt_path",
        "mode",
        "dataset_name",
        "dataset_config",
        "split",
        "eval_batches",
        "batch_size",
        "num_trials",
        "seeds",
        "tgt_keep",
        "tgt_prune",
        "tgt_quant",
        "quant_choices",
        "prune_choices",
        "Ts",
        "Tw",
        "keep_fracs",
        "context_len",
        "rollout_len",
        "ppl_trials",
        "keep_eff_trials",
        "prune_keep_trials",
        "quant_bits_trials",
        "tokens_eff_trials",
        "tokens_trials",
    ]

    # Build row dict, sanitizing anything that might have commas
    row = {
        "timestamp": datetime.now().isoformat(),
        "ckpt_dir": sanitize_csv_field(E.CKPT_DIR),
        "ckpt_path": sanitize_csv_field(ckpt_path),
        "mode": sanitize_csv_field(E.mode),
        "dataset_name": sanitize_csv_field(E.dataset_name),
        "dataset_config": sanitize_csv_field(E.dataset_config),
        "split": sanitize_csv_field(E.split),
        "eval_batches": sanitize_csv_field(E.eval_batches),
        "batch_size": sanitize_csv_field(cfg.batch_size),
        "num_trials": sanitize_csv_field(args.num_trials),
        "seeds": sanitize_csv_field(join_list(seeds_used)),
        "tgt_keep": sanitize_csv_field(args.tgt_keep),
        "tgt_prune": sanitize_csv_field(args.tgt_prune),
        "tgt_quant": sanitize_csv_field(args.tgt_quant),
        "quant_choices": sanitize_csv_field(
            join_list(getattr(cfg, "quant_choices", ("q16",)))
        ),
        "prune_choices": sanitize_csv_field(
            join_list(getattr(cfg, "struct_prune_choices", ("s100",)))
        ),
        # Ts/Tw can be plain ints; don't assume they're iterable
        "Ts": sanitize_csv_field(cfg.Ts),
        "Tw": sanitize_csv_field(cfg.Tw),
        "keep_fracs": sanitize_csv_field(
            join_list(getattr(cfg, "keep_fracs", []), float_fmt="{:.4f}")
        ),
        "context_len": sanitize_csv_field(cfg.context_len),
        "rollout_len": sanitize_csv_field(cfg.rollout_len),
        "ppl_trials": sanitize_csv_field(join_list(ppl_trials, float_fmt="{:.6f}")),
        "keep_eff_trials": sanitize_csv_field(join_list(keep_eff_trials, float_fmt="{:.6f}")),
        "prune_keep_trials": sanitize_csv_field(join_list(prune_keep_trials, float_fmt="{:.6f}")),
        "quant_bits_trials": sanitize_csv_field(join_list(quant_bits_trials, float_fmt="{:.6f}")),
        "tokens_eff_trials": sanitize_csv_field(join_list(tokens_eff_trials)),
        "tokens_trials": sanitize_csv_field(join_list(tokens_trials)),
    }

    # Append to CSV
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n[done] Appended results to CSV: {csv_path}\n")
    print("Trials ppl:", join_list(ppl_trials, float_fmt="{:.3f}"))


if __name__ == "__main__":
    main()
