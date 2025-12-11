import os
import re
import csv
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Global font size
plt.rcParams.update({"font.size": 16})


# ---------- Helpers copied / adapted from your existing script ----------

def normalize_numeric_key(key: str) -> str:
    """
    If key looks like comma-separated floats, sort them numerically
    and format consistently, e.g.:

        '1.0000,0.1000' -> '0.1000,1.0000'

    Otherwise, return key unchanged.
    """
    parts = [p.strip() for p in key.split(",") if p.strip()]
    if not parts:
        return key
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        # Not purely numeric; leave as-is (for q5,q8,q16 etc.)
        return key
    vals_sorted = sorted(vals)
    return ",".join(f"{v:.4f}" for v in vals_sorted)


def parse_action_space_from_ckpt_dir(ckpt_dir: str) -> Optional[str]:
    """
    Parse ckpt_dir strings like:
        'Prune_S30S100-20251206-161746' -> 's30,s100'
        'Quant_Q5Q8Q16-2025...'         -> 'q5,q8,q16'
        'TokSparse_K10K100-2025...'     -> '0.1000,1.0000'   (for keep_fracs)
    """
    base = os.path.basename(ckpt_dir)
    if "_" not in base or "-" not in base:
        return None

    try:
        after_prefix = base.split("_", 1)[1]          # e.g. 'S30S100-2025...' or 'K10K100-...'
        action_part = after_prefix.split("-", 1)[0]   # e.g. 'S30S100' or 'K10K100'
    except IndexError:
        return None

    # Find letter+digits chunks: 'S30', 'S100', 'Q5', 'K10', 'K100', ...
    tokens = re.findall(r"[A-Za-z]\d+", action_part)
    if not tokens:
        return None

    parts = []
    for t in tokens:
        letter = t[0].upper()
        num_str = t[1:]
        try:
            num = int(num_str)
        except ValueError:
            return None

        if letter == "K":
            # TokSparse_K10K100 -> keep_fracs 0.10, 1.00
            parts.append(f"{num / 100.0:.4f}")
        else:
            # Prune / Quant: keep old behavior (lowercase letter + number)
            parts.append(letter.lower() + num_str)

    key = ",".join(parts)
    key = normalize_numeric_key(key)
    return key


def intersect_with_policy(
    base_data: Dict[str, List[float]],
    policy_data: Optional[Dict[str, float]],
) -> Tuple[Dict[str, List[float]], Optional[Dict[str, float]]]:
    """
    Restrict both base_data and policy_data to keys that appear in BOTH,
    as you did: drop entries that don't have a policy counterpart.
    """
    if not policy_data:
        # Nothing to intersect with; keep base_data as-is.
        return base_data, None

    common_keys = sorted(set(base_data.keys()) & set(policy_data.keys()))
    if not common_keys:
        # No overlap: empty both.
        return {}, {}

    filtered_base = {k: base_data[k] for k in common_keys}
    filtered_policy = {k: policy_data[k] for k in common_keys}
    return filtered_base, filtered_policy


def load_random_and_policy_from_single_csv(
    csv_path: str,
    ckpt_dir_col: str = "ckpt_dir",
    policy_ppl_col: str = "policy_ppl",
    fixed_ppl_col: str = "fixed_ppl",
    rand_ppl_col: str = "rand_ppl_trials",
) -> Tuple[Dict[str, List[float]], Dict[str, float], Dict[str, float]]:
    """
    Load a *single* CSV that contains:
      - policy_ppl (one per row),
      - fixed_ppl (one per row),
      - rand_ppl_trials (pipe-separated PPLs),

    and return:
        (action_to_rand_ppls, mean_policy_ppl, mean_fixed_ppl)

    All keyed by a normalized action-space string, parsed from ckpt_dir.
    """
    action_to_rand: Dict[str, List[float]] = defaultdict(list)
    per_action_policy: Dict[str, List[float]] = defaultdict(list)
    per_action_fixed: Dict[str, List[float]] = defaultdict(list)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ckpt_dir = row.get(ckpt_dir_col, "")
            if not ckpt_dir:
                continue

            action_key = parse_action_space_from_ckpt_dir(ckpt_dir)
            if not action_key:
                continue

            # Policy ppl
            ppl_str = row.get(policy_ppl_col)
            if ppl_str:
                try:
                    per_action_policy[action_key].append(float(ppl_str))
                except ValueError:
                    pass

            # Fixed ppl
            fixed_str = row.get(fixed_ppl_col)
            if fixed_str:
                try:
                    per_action_fixed[action_key].append(float(fixed_str))
                except ValueError:
                    pass

            # Random trials ppl (pipe-separated)
            rand_raw = row.get(rand_ppl_col) or ""
            if rand_raw:
                for s in rand_raw.split("|"):
                    s = s.strip()
                    if not s:
                        continue
                    try:
                        val = float(s)
                    except ValueError:
                        continue
                    action_to_rand[action_key].append(val)

    # Collapse policy / fixed to means
    policy_mean: Dict[str, float] = {}
    fixed_mean: Dict[str, float] = {}

    for k, vals in per_action_policy.items():
        if vals:
            policy_mean[k] = sum(vals) / float(len(vals))

    for k, vals in per_action_fixed.items():
        if vals:
            fixed_mean[k] = sum(vals) / float(len(vals))

    # Restrict to keys that have policy entries (drop pure-random-only actions)
    action_to_rand, policy_mean = intersect_with_policy(action_to_rand, policy_mean)
    if fixed_mean:
        fixed_mean = {k: v for k, v in fixed_mean.items() if k in action_to_rand}

    return action_to_rand, policy_mean, fixed_mean


# ---------- New: statistics + plots ----------

def pretty_label(action_key: str) -> str:
    """
    Nicify display labels:
        's30,s100' -> 'S30|S100'
        'q5,q8,q16' -> 'Q5|Q8|Q16'
        '0.1000,1.0000' -> '0.1|1.0'
    """
    label = (
        action_key.replace("q", "Q")
        .replace("s", "S")
        .replace(",", "|")
        .replace("0000", "0")
        .replace("000", "0")
    )
    # Optional: shorten long floats like '0.1000' -> '0.1'
    label = re.sub(r"(\d+\.\d)0+", r"\1", label)
    return label


def compute_stats_for_axis(
    action_to_rand: Dict[str, List[float]],
    policy_mean: Optional[Dict[str, float]],
    fixed_mean: Optional[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    For each action space key, compute:
      - N_random
      - rand_mean, rand_std
      - policy_ppl
      - fixed_ppl (if available)
      - delta = policy - rand_mean
      - z = delta / rand_std
      - p_one = (1 + #rand <= policy) / (N + 1)  [one-sided, lower ppl is better]
      - p_two = two-sided p from that (symmetric around 0.5)
    """
    stats: Dict[str, Dict[str, float]] = {}

    if not policy_mean:
        return stats

    fixed_mean = fixed_mean or {}

    for key, rand_vals in action_to_rand.items():
        if key not in policy_mean:
            continue
        if not rand_vals:
            continue

        rand_arr = np.asarray(rand_vals, dtype=float)
        N = rand_arr.size
        rand_mean = float(rand_arr.mean())
        if N > 1:
            rand_std = float(rand_arr.std(ddof=1))
        else:
            rand_std = 0.0

        policy_ppl = float(policy_mean[key])
        fixed_ppl = float(fixed_mean[key]) if key in fixed_mean else math.nan

        delta = policy_ppl - rand_mean
        z = delta / rand_std if rand_std > 0 else math.nan

        # One-sided: P(null gives <= policy_ppl)
        # lower ppl is better
        p_one = (1.0 + float((rand_arr <= policy_ppl).sum())) / float(N + 1)
        # Symmetric two-sided around 0.5
        p_two = 2.0 * min(p_one, 1.0 - p_one)

        stats[key] = {
            "N": float(N),
            "rand_mean": rand_mean,
            "rand_std": rand_std,
            "policy_ppl": policy_ppl,
            "fixed_ppl": fixed_ppl,
            "delta": delta,
            "z": z,
            "p_one": p_one,
            "p_two": p_two,
        }

    return stats


def plot_z_scores(ax, stats: Dict[str, Dict[str, float]], title: str):
    if not stats:
        ax.set_title(f"{title}\n(no data)")
        ax.axhline(0.0, linestyle="--", linewidth=1)
        return

    keys = sorted(stats.keys())
    zs = [stats[k]["z"] for k in keys]
    labels = [pretty_label(k) for k in keys]
    positions = np.arange(1, len(keys) + 1)

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.bar(positions, zs, width=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Policy z-score vs random")


def plot_p_values(ax, stats: Dict[str, Dict[str, float]], title: str):
    if not stats:
        ax.set_title(f"{title}\n(no data)")
        return

    keys = sorted(stats.keys())
    ps = [stats[k]["p_one"] for k in keys]
    labels = [pretty_label(k) for k in keys]
    positions = np.arange(1, len(keys) + 1)

    ax.scatter(positions, ps)
    ax.axhline(0.05, linestyle="--", linewidth=1, label="0.05 threshold")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.0)
    ax.set_title(title)
    ax.set_ylabel("One-sided randomization p-value")
    ax.legend(loc="best")


def plot_effect_sizes_by_axis(
    ax,
    quant_stats: Dict[str, Dict[str, float]],
    prune_stats: Dict[str, Dict[str, float]],
    tok_stats: Dict[str, Dict[str, float]],
):
    data = []
    labels = []

    def collect_z(stats_dict: Dict[str, Dict[str, float]]) -> List[float]:
        zs = []
        for v in stats_dict.values():
            z = v["z"]
            if not math.isnan(z):
                zs.append(z)
        return zs

    q_z = collect_z(quant_stats)
    p_z = collect_z(prune_stats)
    t_z = collect_z(tok_stats)

    if q_z:
        data.append(q_z)
        labels.append("Quantization")
    if p_z:
        data.append(p_z)
        labels.append("Pruning")
    if t_z:
        data.append(t_z)
        labels.append("Token sparsity")

    if not data:
        ax.set_title("Effect sizes by axis\n(no data)")
        return

    positions = np.arange(1, len(data) + 1)
    vp = ax.violinplot(data, positions=positions, showmeans=True, showextrema=True, showmedians=True)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Policy z-score vs random")
    ax.set_title("Policy effect sizes across axes")


def plot_rand_vs_policy_fixed(
    ax,
    stats: Dict[str, Dict[str, float]],
    title: str,
):
    if not stats:
        ax.set_title(f"{title}\n(no data)")
        return

    keys = sorted(stats.keys())
    rand_means = [stats[k]["rand_mean"] for k in keys]
    policy_vals = [stats[k]["policy_ppl"] for k in keys]
    fixed_vals = [stats[k]["fixed_ppl"] for k in keys]

    # Diagonal range
    all_vals = rand_means + policy_vals + [v for v in fixed_vals if not math.isnan(v)]
    v_min = min(all_vals)
    v_max = max(all_vals)

    ax.plot([v_min, v_max], [v_min, v_max], linestyle="--", linewidth=1, label="y=x (random mean)")

    ax.scatter(rand_means, policy_vals, marker="o", s=60, label="Policy")
    # Only plot fixed where available
    fx_x = [rm for rm, fv in zip(rand_means, fixed_vals) if not math.isnan(fv)]
    fx_y = [fv for fv in fixed_vals if not math.isnan(fv)]
    if fx_x:
        ax.scatter(fx_x, fx_y, marker="x", s=60, label="Fixed baseline")

    ax.set_xlabel("Mean random baseline PPL")
    ax.set_ylabel("PPL")
    ax.set_title(title)
    ax.legend(loc="best")


# ---------- Main ----------

def main():
    quant_csv = "quant_variability.csv"
    prune_csv = "prune_variability.csv"
    toksparse_csv = "toksparse_variability.csv"

    quant_data = {}
    prune_data = {}
    toksparse_data = {}
    quant_policy = quant_fixed = None
    prune_policy = prune_fixed = None
    toksparse_policy = toksparse_fixed = None

    if os.path.exists(quant_csv):
        quant_data, quant_policy, quant_fixed = load_random_and_policy_from_single_csv(quant_csv)
    if os.path.exists(prune_csv):
        prune_data, prune_policy, prune_fixed = load_random_and_policy_from_single_csv(prune_csv)
    if os.path.exists(toksparse_csv):
        toksparse_data, toksparse_policy, toksparse_fixed = load_random_and_policy_from_single_csv(toksparse_csv)

    # Compute stats for each axis
    quant_stats = compute_stats_for_axis(quant_data, quant_policy, quant_fixed)
    prune_stats = compute_stats_for_axis(prune_data, prune_policy, prune_fixed)
    tok_stats = compute_stats_for_axis(toksparse_data, toksparse_policy, toksparse_fixed)

    # Console summary
    def print_summary(axis_name: str, stats: Dict[str, Dict[str, float]]):
        print(f"\n=== {axis_name} ===")
        if not stats:
            print("  (no data)")
            return
        for key in sorted(stats.keys()):
            s = stats[key]
            print(
                f"  {pretty_label(key)}: "
                f"N={int(s['N'])}, rand_mean={s['rand_mean']:.3f}, rand_std={s['rand_std']:.3f}, "
                f"policy={s['policy_ppl']:.3f}, Δ={s['delta']:.3f}, "
                f"z={s['z']:.2f}, p_one={s['p_one']:.3g}, p_two={s['p_two']:.3g}"
            )

    print_summary("Quantization", quant_stats)
    print_summary("Pruning", prune_stats)
    print_summary("Token sparsity", tok_stats)

    # ---------- Figure 1: z-scores per action space ----------
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    plot_z_scores(axes1[0], quant_stats, "Quantization (policy vs random)")
    plot_z_scores(axes1[1], prune_stats, "Pruning (policy vs random)")
    plot_z_scores(axes1[2], tok_stats, "Token sparsity (policy vs random)")
    fig1.tight_layout()
    fig1.savefig("action_z_scores_by_space.pdf")
    plt.close(fig1)
    print("Saved action_z_scores_by_space.pdf")

    # ---------- Figure 2: p-values per action space ----------
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    plot_p_values(axes2[0], quant_stats, "Quantization randomization p-values")
    plot_p_values(axes2[1], prune_stats, "Pruning randomization p-values")
    plot_p_values(axes2[2], tok_stats, "Token sparsity randomization p-values")
    fig2.tight_layout()
    fig2.savefig("action_pvalues_by_space.pdf")
    plt.close(fig2)
    print("Saved action_pvalues_by_space.pdf")

    # ---------- Figure 3: effect-size distribution by axis ----------
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
    plot_effect_sizes_by_axis(ax3, quant_stats, prune_stats, tok_stats)
    fig3.tight_layout()
    fig3.savefig("action_effect_sizes_by_axis.pdf")
    plt.close(fig3)
    print("Saved action_effect_sizes_by_axis.pdf")

    # ---------- Figure 4: mean random vs policy/fixed ----------
    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    plot_rand_vs_policy_fixed(axes4[0], quant_stats, "Quantization: policy vs random")
    plot_rand_vs_policy_fixed(axes4[1], prune_stats, "Pruning: policy vs random")
    plot_rand_vs_policy_fixed(axes4[2], tok_stats, "Token sparsity: policy vs random")
    fig4.tight_layout()
    fig4.savefig("policy_vs_random_scatter.pdf")
    plt.close(fig4)
    print("Saved policy_vs_random_scatter.pdf")


if __name__ == "__main__":
    main()
