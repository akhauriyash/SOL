# eval_policy_lmeval.py
import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
import torch
import torch.nn.functional as F
from tqdm import tqdm
from lm_eval.api.model import LM
from lm_eval import evaluator
import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
import numpy as np

from policy_runtime import PolicyLMRunner
from policy_harness import PolicyHarnessLM

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
    p.add_argument("--mode", type=str, default="latest", choices=["latest", "best"])
    p.add_argument("--tasks", type=str, default="piqa,arc_easy")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Limit samples per task (LM-Eval)")
    p.add_argument("--episode_len", type=int, default=None, help="Override episode length (default cfg.rollout_len)")
    p.add_argument("--dense_refresh_tail", type=int, default=None, help="Tail tokens to dense-prefill between episodes (default Ts+Tw+1)")
    p.add_argument("--policy_temperature", type=float, default=0.6)
    p.add_argument("--greedy_policy", action="store_true", help="Use argmax over κ actions (default True)")
    p.add_argument("--dense_baseline", action="store_true", help="Also run dense baseline (no policy, no sparse masks)")
    p.add_argument("--export_sparsity_json", type=str, default=None,
                   help="If set, write per-request sparsity stats to this JSON file")
    args = p.parse_args()

    model = PolicyHarnessLM(
        ckpt_dir=args.ckpt_dir,
        mode=args.mode,
        greedy_policy = args.greedy_policy,
        policy_temperature=args.policy_temperature,
        episode_len=args.episode_len,
        dense_refresh_tail=args.dense_refresh_tail,
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

if __name__ == "__main__":
    main()
