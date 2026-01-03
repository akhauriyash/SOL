# eval_policy_lmeval.py
import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
import json
import argparse
from os.path import dirname, basename, join, splitext
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
import torch
import re
import math
import torch.nn.functional as F
from tqdm import tqdm
from lm_eval.api.model import LM
from lm_eval import evaluator
import os
import numpy as np
import sys

from policy_runtime import PolicyLMRunner
from policy_harness import PolicyHarnessLM, FixedHarnessLM

def _select_accuracy_metric(metrics: dict) -> Optional[float]:
    """
    Pick a standard accuracy-like metric from an lm-eval results block.
    Falls back to the first numeric entry if needed.
    """
    for k in ("acc,none", "acc", "exact_match,none", "exact_match"):
        if k in metrics and isinstance(metrics[k], (int, float)):
            return float(metrics[k])
    for _, v in metrics.items():
        if isinstance(v, (int, float)):
            return float(v)
    return None

def _extract_accuracy(res: dict) -> dict:
    """
    Returns: {"macro": float|None, "per_task": {task: float}}
    (No per-question metrics.)
    """
    per_task = {}
    for task, metrics in (res.get("results") or {}).items():
        v = _select_accuracy_metric(metrics)
        if v is not None:
            per_task[task] = v

    macro = None
    agg = res.get("aggregated") or res.get("groups") or {}
    for key in ("macro_avg", "overall", "total"):
        block = agg.get(key)
        if isinstance(block, dict):
            mv = _select_accuracy_metric(block)
            if mv is not None:
                macro = mv
                break
    if macro is None and per_task:
        macro = sum(per_task.values()) / len(per_task)
    return {"macro": macro, "per_task": per_task}

def _extract_sparsity_means(stats: dict) -> dict:
    """
    Pulls averaged keep/prune/quant numbers from export_sparsity_stats().
    Converts to intuitive 'rates': token_sparsity = 1 - keep, prune_rate = 1 - prune_keep.
    Also includes avg_bits = 16 * quant_ratio.
    """
    g = stats.get("global", stats)
    eff_steps = int(g.get("effective_steps", 0))
    keep = float(g["keep_avg_eff"] if eff_steps > 0 else g.get("keep_avg_all", 1.0))
    prune_keep = float(g["prune_avg_eff"] if eff_steps > 0 else g.get("prune_avg_all", 1.0))
    qratio = float(g["quant_ratio_avg_eff"] if eff_steps > 0 else g.get("quant_ratio_avg_all", 1.0))
    return {
        "token_keep_avg": keep,
        "token_sparsity_avg": 1.0 - keep,
        "prune_keep_avg": prune_keep,
        "prune_rate_avg": 1.0 - prune_keep,
        "quant_ratio_avg": qratio,
        "avg_bits": 16.0 * qratio,
    }
    
def _write_key_metrics(sidecar_path: str,
                       stats_policy: Optional[dict] = None, res_policy: Optional[dict] = None,
                       stats_dense: Optional[dict] = None,  res_dense: Optional[dict] = None,
                       stats_fixed: Optional[dict] = None,  res_fixed: Optional[dict] = None):
    out_dir, base = dirname(sidecar_path), basename(sidecar_path)
    key_path = join(out_dir, "key_metrics_" + base)
    payload = {}
    if stats_policy is not None and res_policy is not None:
        payload["policy"] = {**_extract_sparsity_means(stats_policy), "accuracy": _extract_accuracy(res_policy)}
    if stats_dense is not None and res_dense is not None:
        payload["dense_baseline"] = {**_extract_sparsity_means(stats_dense), "accuracy": _extract_accuracy(res_dense)}
    if stats_fixed is not None and res_fixed is not None:
        payload["fixed_baseline"] = {**_extract_sparsity_means(stats_fixed), "accuracy": _extract_accuracy(res_fixed)}
    with open(key_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"[saved] key metrics → {key_path}")

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        if o.ndim == 0:
            return o.item()
        return o.detach().cpu().tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)

def print_compact_summary(res):
    results = res.get("results", {})
    accs = []
    print("\n=== Per-task accuracy ===")
    for task, metrics in results.items():
        # lm-eval tasks often expose acc_norm (ARC, etc.) in addition to acc.
        for k in ("acc_norm,none", "acc_norm", "acc,none", "acc", "exact_match,none", "exact_match"):
            if k in metrics:
                v = metrics[k]
                print(f"{task}: {v:.4f}")
                accs.append(v)
                break
        else:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"{task} ({k}): {v:.4f}")
                    break

    agg = res.get("aggregated") or res.get("groups") or {}
    macro = None
    for key in ("macro_avg", "overall", "total"):
        if isinstance(agg, dict) and key in agg:
            block = agg[key]
            for k in ("acc,none", "acc", "exact_match,none", "exact_match"):
                if k in block:
                    macro = block[k]
                    break
            if macro is not None:
                break

    print("\n=== Overall ===")
    if macro is not None:
        print(f"Macro average accuracy: {macro:.4f}")
    elif accs:
        print(f"Macro average accuracy (computed): {sum(accs)/len(accs):.4f}")
    else:
        print("No accuracy-like metric found in results.")


_BIAS_KINDS = ("quant", "sparsity", "prune")
_KIND_TO_METRIC = {
    "quant": "quant_ratio_avg",
    "sparsity": "token_keep_avg",
    "prune": "prune_keep_avg",
}

def _parse_bias_from_filename(path: str) -> Tuple[str, float]:
    """
    Extract (kind, value) from filenames like '..._quant_-4.json' or '..._sparsity_1.5.json'.
    """
    name = basename(path)
    m = re.search(r'_(quant|sparsity|prune)_([+-]?\d+(?:\.\d+)?)', name)
    if not m:
        raise ValueError(f"Could not parse bias kind/value from filename: {name}")
    kind = m.group(1)
    val = float(m.group(2))
    return kind, val

def _resolve_key_metrics_path(ref_path: str) -> str:
    """
    Accept either a key-metrics file itself or the base eval file and return the key-metrics path.
    """
    d, b = dirname(ref_path), basename(ref_path)
    if b.startswith("key_metrics_"):
        return ref_path
    candidate = join(d, "key_metrics_" + b)
    return candidate if os.path.exists(candidate) else ref_path

def _load_reference_metrics(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference key-metrics file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def _pick_metric_from_reference(ref_payload: dict, metric_name: str) -> Optional[float]:
    """
    Try common blocks in priority order: policy → fixed_baseline → dense_baseline.
    """
    for block in ("policy", "fixed_baseline", "dense_baseline"):
        if block in ref_payload and isinstance(ref_payload[block], dict):
            blk = ref_payload[block]
            if metric_name in blk:
                return float(blk[metric_name])
    # Flat payload fallback (unlikely)
    return float(ref_payload.get(metric_name)) if metric_name in ref_payload else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True, help="Directory with policy_*.pt")
    p.add_argument("--mode", type=str, default="latest")
    p.add_argument("--tasks", type=str, default="piqa,arc_easy")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="Limit samples per task (LM-Eval)")
    p.add_argument("--only_dense", action="store_true", help="Run dense-only LM-Eval and exit.")
    p.add_argument("--episode_len", type=int, default=None, help="Override episode length (default cfg.rollout_len)")
    p.add_argument("--dense_refresh_tail", type=int, default=None, help="Tail tokens to dense-prefill between episodes (default Ts+Tw+1)")
    p.add_argument("--policy_temperature", type=float, default=0.6)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--greedy_policy", dest="greedy_policy", action="store_true",
                   help="Use argmax over policy actions (default).")
    g.add_argument("--stochastic_policy", dest="greedy_policy", action="store_false",
                   help="Sample policy actions (stochastic).")
    p.set_defaults(greedy_policy=True)
    p.add_argument("--dense_baseline", action="store_true", help="Also run dense baseline (no policy, no sparse masks)")
    p.add_argument("--export_sparsity_json", type=str, default=None,
                   help="If set, write per-request sparsity stats to this JSON file")
    p.add_argument("--sparsity_bias", type=float, default=0.0, help=">0 favors sparser keep_fracs; <0 favors denser")
    p.add_argument("--prune_bias", type=float, default=0.0, help=">0 favors more structural pruning (lower keep after pruning)")
    p.add_argument("--quant_bias", type=float, default=0.0, help=">0 favors lower-bit quantization")
    p.add_argument("--tgt_keep", type=float, default=None,
                   help="Override target token keep C_tok in [0,1] for policy eval.")
    p.add_argument("--tgt_prune_keep", type=float, default=None,
                   help="Override target prune keep C_pru in [0,1] for policy eval.")
    p.add_argument("--tgt_quant_bits", type=float, default=None,
                   help="Override target quant bits (e.g. 8, 16) for policy eval.")

    p.add_argument("--fixed_baseline_reference", type=str, default=None,
                   help="Path to (or sibling of) a key_metrics_*.json file from a previous run. "
                        "Will parse bias kind/value from the filename and use the matching "
                        "metric (quant_ratio_avg/token_keep_avg/prune_keep_avg) as the fixed target. "
                        "Skips policy and dense runs.")
    p.add_argument("--fixed_from_policy", action="store_true",
                   help="After running the policy, run a FixedHarnessLM that targets the policy's observed averages (token_keep/prune_keep/quant_ratio).")
    args = p.parse_args()

    # Validate explicit targets if provided
    def _check01(name: str, v: Optional[float]):
        if v is None:
            return
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {v}")

    _check01("--tgt_keep", args.tgt_keep)
    _check01("--tgt_prune_keep", args.tgt_prune_keep)
    if args.tgt_quant_bits is not None and float(args.tgt_quant_bits) <= 0:
        raise ValueError(f"--tgt_quant_bits must be > 0, got {args.tgt_quant_bits}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # --- NEW: dense-only mode (run once, useful for scans) ---
    if args.only_dense:
        dense_model = PolicyHarnessLM(
            ckpt_dir=args.ckpt_dir,
            mode=args.mode,
            greedy_policy=True,
            policy_temperature=args.policy_temperature,
            episode_len=args.episode_len,
            dense_refresh_tail=args.dense_refresh_tail,
            sparsity_bias=float(args.sparsity_bias),
            prune_bias=float(args.prune_bias),
            quant_bias=float(args.quant_bias),
            target_C_tok=args.tgt_keep,
            target_C_pru=args.tgt_prune_keep,
            target_C_qbits=args.tgt_quant_bits,
            dense_only=True,
            max_batch=args.batch_size,
        )

        res_dense = evaluator.simple_evaluate(
            model=dense_model,
            tasks=tasks,
            batch_size=args.batch_size,
            limit=args.limit,
            num_fewshot=0,
        )
        dense_stats = dense_model.export_sparsity_stats()
        print("\n=== Observed sparsity (dense-only) ===")
        print(json.dumps(dense_stats["global"], indent=2, default=_json_default))
        print("\n\n## Dense-only result")
        print_compact_summary(res_dense)

        if args.export_sparsity_json:
            with open(args.export_sparsity_json, "w") as f:
                json.dump(dense_stats, f, indent=2, default=_json_default)
            print(f"[saved] per-request sparsity (dense-only) → {args.export_sparsity_json}")

            _write_key_metrics(
                sidecar_path=args.export_sparsity_json,
                stats_dense=dense_stats, res_dense=res_dense,
            )

        # Optionally dump full results JSON to stdout (already printed above in compact form)
        # print(json.dumps(res_dense, indent=2, default=_json_default))
        return

    # If a reference for fixed baseline is provided, run fixed-only path and exit.
    if args.fixed_baseline_reference:
        # Validate exactly one non-zero bias is provided
        bias_map = {
            "quant": float(args.quant_bias),
            "sparsity": float(args.sparsity_bias),
            "prune": float(args.prune_bias),
        }
        nonzero = [(k, v) for k, v in bias_map.items() if abs(v) > 1e-12]
        if len(nonzero) != 1:
            raise ValueError("Exactly one of --quant_bias/--sparsity_bias/--prune_bias must be non-zero "
                             "when using --fixed_baseline_reference.")
        (bias_kind_cli, bias_val_cli) = nonzero[0]

        ref_path = _resolve_key_metrics_path(args.fixed_baseline_reference)
        bias_kind_file, bias_val_file = _parse_bias_from_filename(ref_path)
        if bias_kind_file != bias_kind_cli:
            raise AssertionError(f"Bias kind mismatch: CLI={bias_kind_cli} vs file={bias_kind_file} ({basename(ref_path)})")
        # Allow integer vs float nuances (e.g., -4 vs -4.0)
        if not math.isclose(bias_val_file, bias_val_cli, rel_tol=0, abs_tol=1e-6):
            raise AssertionError(f"Bias value mismatch: CLI={bias_val_cli} vs file={bias_val_file} ({basename(ref_path)})")

        ref_payload = _load_reference_metrics(ref_path)

        # --- NEW: read ALL axes from reference so fixed mirrors policy’s observed budgets ---
        tk = _pick_metric_from_reference(ref_payload, "token_keep_avg")
        pr = _pick_metric_from_reference(ref_payload, "prune_keep_avg")
        qr = _pick_metric_from_reference(ref_payload, "quant_ratio_avg")
        if tk is None and pr is None and qr is None:
            raise KeyError(f"No usable targets found in {ref_path}")

        # Fallbacks: keep defaults from FixedHarnessLM if any are missing
        # (tk -> cfg.C_target/keep_target inside harness; pr/qr default to 1.0 there)
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        fixed_kwargs_all = {
            "ckpt_dir": args.ckpt_dir,
            "mode": args.mode,
            "episode_len": args.episode_len,
            "dense_refresh_tail": args.dense_refresh_tail,
            "max_batch": args.batch_size,
        }
        if tk is not None:
            fixed_kwargs_all["target_keep_effective"] = float(tk)
        if pr is not None:
            fixed_kwargs_all["target_prune_keep"] = float(pr)
        if qr is not None:
            fixed_kwargs_all["target_quant_ratio"] = float(qr)

        fixed_model = FixedHarnessLM(**fixed_kwargs_all)
        res_fixed = evaluator.simple_evaluate(
            model=fixed_model, tasks=tasks, batch_size=args.batch_size, limit=args.limit, num_fewshot=0
        )
        fixed_stats = fixed_model.export_sparsity_stats()
        print("\n=== Observed sparsity (fixed from reference) ===")
        print(json.dumps(fixed_stats["global"], indent=2, default=_json_default))

        if args.export_sparsity_json:
            with open(args.export_sparsity_json, "w") as f:
                json.dump(fixed_stats, f, indent=2, default=_json_default)
            print(f"[saved] per-request sparsity (fixed) → {args.export_sparsity_json}")
            _write_key_metrics(
                sidecar_path=args.export_sparsity_json,
                stats_fixed=fixed_stats, res_fixed=res_fixed,
            )

        print("\n\n## Fixed baseline (from reference)")
        print_compact_summary(res_fixed)
        return

    # --- default flow (policy, optional fixed-from-policy, optional dense baseline) ---
    model = PolicyHarnessLM(
        ckpt_dir=args.ckpt_dir,
        mode=args.mode,
        greedy_policy = args.greedy_policy,
        policy_temperature=args.policy_temperature,
        episode_len=args.episode_len,
        dense_refresh_tail=args.dense_refresh_tail,
        sparsity_bias=args.sparsity_bias,
        prune_bias=args.prune_bias,
        quant_bias=args.quant_bias,
        target_C_tok=args.tgt_keep,
        target_C_pru=args.tgt_prune_keep,
        target_C_qbits=args.tgt_quant_bits,
        dense_only=False,
        max_batch=args.batch_size,
    )

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    res_policy = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=0,
    )
    stats_all = model.export_sparsity_stats()
    print("\n=== Observed sparsity (policy run) ===")
    print(json.dumps(stats_all["global"], indent=2))

    if args.export_sparsity_json:
        with open(args.export_sparsity_json, "w") as f:
            json.dump(stats_all, f, indent=2, default=_json_default)
        print(f"[saved] per-request sparsity → {args.export_sparsity_json}")

    # --- NEW: optionally mirror policy budgets with FixedHarnessLM (no reference file) ---
    res_fixed = None
    fixed_stats = None
    if args.fixed_from_policy:
        means = _extract_sparsity_means(stats_all)
        fixed_kwargs_all = {
            "ckpt_dir": args.ckpt_dir,
            "mode": args.mode,
            "episode_len": args.episode_len,
            "dense_refresh_tail": args.dense_refresh_tail,
            "max_batch": args.batch_size,
        }
        # Map our means into FixedHarnessLM targets.
        if "token_keep_avg" in means and means["token_keep_avg"] is not None:
            fixed_kwargs_all["target_keep_effective"] = float(means["token_keep_avg"])
        if "prune_keep_avg" in means and means["prune_keep_avg"] is not None:
            fixed_kwargs_all["target_prune_keep"] = float(means["prune_keep_avg"])
        if "quant_ratio_avg" in means and means["quant_ratio_avg"] is not None:
            fixed_kwargs_all["target_quant_ratio"] = float(means["quant_ratio_avg"])

        fixed_model = FixedHarnessLM(**fixed_kwargs_all)
        res_fixed = evaluator.simple_evaluate(
            model=fixed_model, tasks=tasks, batch_size=args.batch_size, limit=args.limit, num_fewshot=0
        )
        fixed_stats = fixed_model.export_sparsity_stats()
        print("\n=== Observed sparsity (fixed-from-policy) ===")
        print(json.dumps(fixed_stats["global"], indent=2, default=_json_default))

        if args.export_sparsity_json:
            root, ext = os.path.splitext(args.export_sparsity_json)
            fixed_path = root + "_fixed" + ext
            with open(fixed_path, "w") as f:
                json.dump(fixed_stats, f, indent=2, default=_json_default)
            print(f"[saved] per-request sparsity (fixed) → {fixed_path}")

    # If we saved JSON, (re)write compact key-metrics including fixed if present.
    if args.export_sparsity_json:
        _write_key_metrics(
            sidecar_path=args.export_sparsity_json,
            stats_policy=stats_all, res_policy=res_policy,
            stats_fixed=fixed_stats,  res_fixed=res_fixed,
        )

    if not args.dense_baseline:
        if res_fixed is None:
            print(json.dumps(res_policy, indent=2, default=_json_default))
        else:
            print(json.dumps({"policy_sparse": res_policy, "fixed_from_policy": res_fixed},
                             indent=2, default=_json_default))
        return

    print_compact_summary(res_policy)
    dense_model = PolicyHarnessLM(
        ckpt_dir=args.ckpt_dir,
        mode=args.mode,
        greedy_policy=True,
        policy_temperature=args.policy_temperature,
        episode_len=args.episode_len,
        dense_refresh_tail=args.dense_refresh_tail,
        sparsity_bias=args.sparsity_bias,
        prune_bias=args.prune_bias,
        quant_bias=args.quant_bias,
        target_C_tok=args.tgt_keep,
        target_C_pru=args.tgt_prune_keep,
        target_C_qbits=args.tgt_quant_bits,
        dense_only=True,
        max_batch=args.batch_size,
    )
    res_dense = evaluator.simple_evaluate(
        model=dense_model, tasks=tasks, batch_size=args.batch_size, limit=args.limit, num_fewshot=0
    )

    print(json.dumps({"policy_sparse": res_policy, "dense_baseline": res_dense},
                    indent=2, default=_json_default))
    dense_stats = dense_model.export_sparsity_stats()
    print("\n=== Observed sparsity (dense baseline) ===")
    print(json.dumps(dense_stats["global"], indent=2))
    if args.export_sparsity_json:
        root, ext = os.path.splitext(args.export_sparsity_json)
        dense_path = root + "_dense" + ext
        with open(dense_path, "w") as f:
            json.dump(dense_stats, f, indent=2)
        print(f"[saved] per-request sparsity (dense) → {dense_path}")

    print("\n\n## Policy Result")
    print_compact_summary(res_policy)
    print("\n\n## Dense baseline")
    print_compact_summary(res_dense)

    # Also dump a compact joint key-metrics file when --dense_baseline is used.
    if args.export_sparsity_json:
        _write_key_metrics(
            sidecar_path=args.export_sparsity_json,
            stats_policy=stats_all, res_policy=res_policy,
            stats_dense=dense_stats, res_dense=res_dense,
        )


if __name__ == "__main__":
    main()
