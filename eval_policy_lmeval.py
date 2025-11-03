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

from policy_runtime import PolicyLMRunner
from policy_harness import PolicyHarnessLM

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

def _write_key_metrics(sidecar_path: str, stats_policy: dict, res_policy: dict,
                       stats_dense: Optional[dict] = None, res_dense: Optional[dict] = None):
    out_dir, base = dirname(sidecar_path), basename(sidecar_path)
    key_path = join(out_dir, "key_metrics_" + base)
    payload = {
        "policy": {**_extract_sparsity_means(stats_policy), "accuracy": _extract_accuracy(res_policy)}
    }
    if stats_dense is not None and res_dense is not None:
        payload["dense_baseline"] = {**_extract_sparsity_means(stats_dense), "accuracy": _extract_accuracy(res_dense)}
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
        for k in ("acc,none", "acc", "exact_match,none", "exact_match"):
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True, help="Directory with policy_*.pt")
    p.add_argument("--mode", type=str, default="latest")
    p.add_argument("--tasks", type=str, default="piqa,arc_easy")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="Limit samples per task (LM-Eval)")
    p.add_argument("--episode_len", type=int, default=None, help="Override episode length (default cfg.rollout_len)")
    p.add_argument("--dense_refresh_tail", type=int, default=None, help="Tail tokens to dense-prefill between episodes (default Ts+Tw+1)")
    p.add_argument("--policy_temperature", type=float, default=0.6)
    p.add_argument("--greedy_policy", action="store_true", help="Use argmax over κ actions (default True)")
    p.add_argument("--dense_baseline", action="store_true", help="Also run dense baseline (no policy, no sparse masks)")
    p.add_argument("--export_sparsity_json", type=str, default=None,
                   help="If set, write per-request sparsity stats to this JSON file")
    p.add_argument("--sparsity_bias", type=float, default=0.0, help=">0 favors sparser keep_fracs; <0 favors denser")
    p.add_argument("--prune_bias", type=float, default=0.0, help=">0 favors more structural pruning (lower keep after pruning)")
    p.add_argument("--quant_bias", type=float, default=0.0, help=">0 favors lower-bit quantization")
    args = p.parse_args()

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

        # Write compact sidecar with just key metrics for easy plotting.
        _write_key_metrics(
            sidecar_path=args.export_sparsity_json,
            stats_policy=stats_all,
            res_policy=res_policy,
        )
    if not args.dense_baseline:
        print(json.dumps(res_policy, indent=2, default=_json_default))
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
