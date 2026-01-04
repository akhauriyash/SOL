#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import copy
import random


REQUIRED_COLS = [
    "ckpt_path",
    "policy_ppl",
    "policy_keep_effective_actual",
    "policy_prune_keep_actual",
    "policy_quant_ratio_actual",
    "tgt_keep_cli",
    "tgt_prune_keep_cli",
    "tgt_quant_bits_cli",
    "run_timestamp",
    "pareto_tuples",
]


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def normalize_path(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def format_hms(seconds: float) -> str:
    if seconds < 0 or not (seconds == seconds):  # NaN check
        return "?"
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def row_matches_ckpt_dir(row: Dict[str, str], ckpt_dir: str) -> bool:
    """
    Match if ckpt_path lives under ckpt_dir.
    """
    ckpt_dir_n = normalize_path(ckpt_dir)
    ckpt_path = (row.get("ckpt_path") or "").strip()
    if not ckpt_path:
        return False
    ckpt_path_n = normalize_path(ckpt_path)

    prefix = ckpt_dir_n + os.sep
    return ckpt_path_n == ckpt_dir_n or ckpt_path_n.startswith(prefix)


def load_latest_matching_row(pareto_db: str, ckpt_dir: str) -> Dict[str, str]:
    if not os.path.exists(pareto_db):
        raise FileNotFoundError(f"pareto db not found: {pareto_db}")

    best_row: Optional[Dict[str, str]] = None
    best_ts: Optional[datetime] = None

    with open(pareto_db, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row_matches_ckpt_dir(row, ckpt_dir):
                continue

            ts_str = (row.get("run_timestamp") or "").strip()
            if not ts_str:
                continue

            try:
                ts = parse_iso(ts_str)
            except Exception:
                continue

            if best_ts is None or ts > best_ts:
                best_ts = ts
                best_row = row

    if best_row is None:
        raise RuntimeError(
            f"No matching entries found in {pareto_db} for ckpt_dir={ckpt_dir}"
        )
    return best_row


def coerce_bits(v: Any) -> Any:
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return v
    iv = int(round(fv))
    return iv if abs(fv - iv) < 1e-9 else fv


def parse_pareto_tuples(row: Dict[str, str]) -> List[Tuple[float, float, Any]]:
    s = (row.get("pareto_tuples") or "").strip()
    if not s:
        return []
    data = json.loads(s)

    tuples: List[Tuple[float, float, Any]] = []
    for t in data:
        if not (isinstance(t, (list, tuple)) and len(t) == 3):
            continue
        keep, prune, bits = t
        tuples.append((float(keep), float(prune), coerce_bits(bits)))
    return tuples


def run_one(
    python_exe: str,
    policy_script: str,
    ckpt_dir: str,
    ckpt_path: Optional[str],
    dataset_name: str,
    eval_batches: Optional[int],
    split: str,
    mode: str,
    batch_size: Optional[int],
    seed: int,
    tgt_keep: float,
    tgt_prune_keep: float,
    tgt_quant_bits: Any,
    num_trials: int,
    out_csv: str,
    do_emc_and_driftaware: bool,
    extra_args: List[str],
    cwd: Optional[str],
) -> float:
    cmd = [python_exe, policy_script]

    if ckpt_path:
        cmd += ["--ckpt_path", ckpt_path, "--ckpt_dir", ckpt_dir]
    else:
        cmd += ["--ckpt_dir", ckpt_dir, "--mode", mode]

    cmd += ["--dataset_name", dataset_name]
    cmd += ["--split", split]
    cmd += ["--seed", str(seed)]
    cmd += ["--num_trials", str(num_trials)]
    cmd += ["--csv_path", out_csv]

    if eval_batches is not None:
        cmd += ["--eval_batches", str(eval_batches)]
    if batch_size is not None:
        cmd += ["--batch_size", str(batch_size)]

    cmd += ["--tgt_keep", f"{tgt_keep:.6f}"]
    cmd += ["--tgt_prune_keep", f"{tgt_prune_keep:.6f}"]
    cmd += ["--tgt_quant_bits", str(tgt_quant_bits)]
    if do_emc_and_driftaware:
        cmd += ["--do_emc_and_driftaware"]

    if extra_args:
        cmd += extra_args

    print("[cmd]", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=cwd)  # streams output normally
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser(
        description="Find latest Pareto tuple list for a ckpt_dir and rerun policy_action_variability.py sequentially with progress+ETA."
    )
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--pareto_db", default="pareto_feasibility.csv")
    ap.add_argument("--policy_script", default="policy_action_variability.py")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--num_trials", type=int, default=30)
    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--do_emc_and_driftaware", action="store_true",
                    help="Also run EMC + Drift-Aware baselines inside policy_action_variability.py "
                         "for each Pareto point.")

    ap.add_argument("--dataset_name", default="wikitext", choices=["wikitext", "allenai/c4"])
    ap.add_argument("--eval_batches", type=int, default=7)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--mode", default="latest", choices=["latest", "best"])
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--use_ckpt_path", action="store_true",
                    help="Use ckpt_path stored in pareto_feasibility.csv (recommended).")

    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use for subprocess (default: this interpreter / current venv).")

    ap.add_argument("--cwd", default=None,
                    help="Working directory to run policy_script from (useful if relative imports need repo root).")

    ap.add_argument("--dedupe", action="store_true")

    ap.add_argument("--reverse", action="store_true",
                    help="Reverse the Pareto tuple order before running.")
    ap.add_argument("--subsample", type=int, default=None,
                    help="If set, randomly sample this many tuples from the Pareto list (no replacement).")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Extra args passed to policy_action_variability.py (after '--extra').")

    args = ap.parse_args()

    ckpt_dir = normalize_path(args.ckpt_dir)
    row = load_latest_matching_row(args.pareto_db, ckpt_dir)
    tuples = parse_pareto_tuples(row)
    if args.dedupe:
        seen = set()
        uniq = []
        for t in tuples:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        tuples = uniq

    if args.subsample is not None:
        m = int(args.subsample)
        if m <= 0:
            raise RuntimeError("--subsample must be a positive int.")
        if m < len(tuples):
            rng = random.Random(args.seed)

            # Net keep for tuple (keep, prune_keep, quant_bits):
            # x = (keep + prune_keep + (quant_bits/16)) / 3
            rates = []
            for (k, p, b) in tuples:
                q = float(b) / 16.0
                rates.append((float(k) + float(p) + q) / 3.0)

            rmin, rmax = float(min(rates)), float(max(rates))
            if rmax <= rmin + 1e-12:
                # Degenerate: all same net keep -> fall back to uniform sample
                idxs = sorted(rng.sample(range(len(tuples)), m))
            else:
                width = (rmax - rmin) / m
                buckets = [[] for _ in range(m)]
                for idx, r in enumerate(rates):
                    bi = int((r - rmin) / width)
                    if bi >= m:
                        bi = m - 1  # handle r == rmax
                    buckets[bi].append(idx)

                chosen = []
                chosen_set = set()

                # One per bucket (low->high net keep)
                for bucket in buckets:
                    if bucket:
                        pick = rng.choice(bucket)
                        chosen.append(pick)
                        chosen_set.add(pick)
                    else:
                        chosen.append(None)

                # Fill empty buckets (if any) from remaining tuples
                if any(c is None for c in chosen):
                    remaining = [i for i in range(len(tuples)) if i not in chosen_set]
                    rng.shuffle(remaining)
                    for j in range(len(chosen)):
                        if chosen[j] is None:
                            if not remaining:
                                break
                            pick = remaining.pop()
                            chosen[j] = pick
                            chosen_set.add(pick)

                idxs = [i for i in chosen if i is not None][:m]

            tuples = [tuples[i] for i in idxs]
            sel_rates = [rates[i] for i in idxs]
            print(
                f"[subsample] stratified {len(tuples)} tuples across net_keep "
                f"in [{rmin:.4f}, {rmax:.4f}] (seed={args.seed}) -> "
                f"{', '.join(f'{r:.4f}' for r in sel_rates)}"
            )
            # idxs = sorted(rng.sample(range(len(tuples)), m))
            # tuples = [tuples[i] for i in idxs]
            # print(f"[subsample] selected {len(tuples)} tuples (seed={args.seed})")
        else:
            print(f"[subsample] requested {m} >= {len(tuples)}; using all tuples")


    if args.reverse:
        tuples = list(reversed(tuples))
    if not tuples:
        raise RuntimeError("No pareto tuples found in selected row.")

    selected_ts = row.get("run_timestamp")
    selected_ckpt_path = row.get("ckpt_path") if args.use_ckpt_path else None

    print("\n[selected]")
    print("  now:", datetime.now(ZoneInfo(args.tz)).isoformat(timespec="seconds"))
    print("  pareto_row_timestamp:", selected_ts)
    print("  ckpt_dir:", ckpt_dir)
    print("  ckpt_path_used:", selected_ckpt_path or "(not used)")
    print("  source_csv_path:", row.get("source_csv_path"))
    print("  pareto_count:", len(tuples))
    print("  out_csv:", os.path.abspath(args.out_csv))
    print("  python:", args.python)
    if args.cwd:
        print("  cwd:", args.cwd)

    overall_t0 = time.time()
    ema: Optional[float] = None
    alpha = 0.3  # EMA smoothing

    n = len(tuples)
    for i, (k, p, b) in enumerate(tuples, 1):
        done = i - 1
        elapsed = time.time() - overall_t0
        avg = (elapsed / done) if done > 0 else None
        eta = (avg * (n - done)) if avg is not None else None
        if ema is not None:
            eta = ema * (n - done)

        print("\n" + "=" * 80)
        print(f"[progress] {i}/{n}  elapsed={format_hms(elapsed)}  eta={format_hms(eta or float('nan'))}")
        print(f"[pareto] tgt_keep={k:.6f}  tgt_prune_keep={p:.6f}  tgt_quant_bits={b}")
        print("=" * 80)

        dt = run_one(
            python_exe=args.python,
            policy_script=args.policy_script,
            ckpt_dir=ckpt_dir,
            ckpt_path=selected_ckpt_path,
            dataset_name=args.dataset_name,
            eval_batches=args.eval_batches,
            split=args.split,
            mode=args.mode,
            batch_size=args.batch_size,
            seed=args.seed,
            tgt_keep=k,
            tgt_prune_keep=p,
            tgt_quant_bits=b,
            num_trials=args.num_trials,
            out_csv=args.out_csv,
            do_emc_and_driftaware=bool(args.do_emc_and_driftaware),
            extra_args=args.extra,
            cwd=args.cwd,
        )

        ema = dt if ema is None else (alpha * dt + (1 - alpha) * ema)
        elapsed2 = time.time() - overall_t0
        remaining = n - i
        eta2 = (ema * remaining) if ema is not None else None
        print(f"[done] point {i}/{n}  last={format_hms(dt)}  elapsed={format_hms(elapsed2)}  eta={format_hms(eta2 or float('nan'))}")

    print("\n[all done] total elapsed:", format_hms(time.time() - overall_t0))


if __name__ == "__main__":
    main()
