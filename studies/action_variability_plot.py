import os
import re
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Global font size
plt.rcParams.update({'font.size': 16})


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
        after_prefix = base.split("_", 1)[1]       # e.g. 'S30S100-20251206-161746' or 'K10K100-...'
        action_part = after_prefix.split("-", 1)[0]  # e.g. 'S30S100' or 'K10K100'
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
            # TokSparse_K10K100 -> keep_fracs 0.10, 1.00 (formatted like the CSV)
            parts.append(f"{num / 100.0:.4f}")
        else:
            # Prune / Quant: keep old behavior (lowercase letter + number)
            parts.append(letter.lower() + num_str)

    key = ",".join(parts)
    key = normalize_numeric_key(key)
    return key

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
    We also intersect so we only keep action spaces that *have* a policy value.
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



def intersect_with_policy(
    base_data: Dict[str, List[float]],
    policy_data: Optional[Dict[str, float]],
) -> Tuple[Dict[str, List[float]], Optional[Dict[str, float]]]:
    """
    Restrict both base_data and policy_data to keys that appear in BOTH,
    as requested: drop violin entries that don't have a policy counterpart.
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

def make_violin_subplot(
    ax,
    action_to_ppl: Dict[str, List[float]],
    title: str,
    ylabel: str = "Perplexity",
    rotate_xticks: bool = True,
    policy_values: Optional[Dict[str, float]] = None,
    fixed_values: Optional[Dict[str, float]] = None,
    show_legend: bool = True,
):
    """
    Draw a violin plot on a given Axes, optionally overlaying policy points.

    action_to_ppl: dict {action_space_str: [ppl_values]}
    policy_values: dict {action_space_str: mean_policy_ppl}
                    (keys must match action_to_ppl keys, e.g. 's30,s100')
    """
    if not action_to_ppl:
        ax.set_title(f"{title}\n(no data)")
        ax.set_xticks([])
        return

    # Sort x-axis labels for consistent order
    orig_labels = sorted(action_to_ppl.keys())                # internal keys, e.g. 's30,s100'
    data = [action_to_ppl[label] for label in orig_labels]

    # Nicify display labels: 's30,s100' -> 'S30|S100', with some cleanup
    display_labels = [
        x.replace("q", "Q")
         .replace("s", "S")
         .replace(",", "|")
         .replace("0000", "0")
         .replace("000", "0")
        for x in orig_labels
    ]

    vp = ax.violinplot(data, showmeans=True, showextrema=True, showmedians=True)

    positions = list(range(1, len(display_labels) + 1))
    ax.set_xticks(positions)
    if rotate_xticks:
        ax.set_xticklabels(display_labels, rotation=25, ha="right")
    else:
        ax.set_xticklabels(display_labels)

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    # Overlay policy points (red dots)
    if policy_values:
        xs, ys = [], []
        for idx, key in enumerate(orig_labels):
            if key not in policy_values:
                continue
            xs.append(idx + 1)
            ys.append(policy_values[key])

        if xs:
            ax.scatter(
                xs,
                ys,
                color="red",
                marker="o",
                s=60,
                zorder=3,
                label="Policy",
            )

    # Overlay fixed baseline points (green dots)
    if fixed_values:
        xs2, ys2 = [], []
        for idx, key in enumerate(orig_labels):
            if key not in fixed_values:
                continue
            xs2.append(idx + 1)
            ys2.append(fixed_values[key])

        if xs2:
            ax.scatter(
                xs2,
                ys2,
                color="green",
                marker="x",
                s=60,
                zorder=3,
                label="Fixed baseline",
            )

    if show_legend and (policy_values or fixed_values):
        ax.legend(loc="best")
    # # Overlay policy points (red dots) at the mean policy perplexity
    # if policy_values:
    #     xs, ys = [], []
    #     for idx, key in enumerate(orig_labels):
    #         if key not in policy_values:
    #             continue
    #         xs.append(idx + 1)          # positions are 1-based
    #         ys.append(policy_values[key])

    #     if xs:
    #         ax.scatter(
    #             xs,
    #             ys,
    #             color="red",
    #             marker="o",
    #             s=60,
    #             zorder=3,
    #             label="Policy",
    #         )
    #         if show_legend:
    #             ax.legend(loc="best")

def main():
    # Unified CSVs: each contains policy + fixed + random trials
    quant_csv = "quant_variability.csv"
    prune_csv = "prune_variability.csv"
    toksparse_csv = "toksparse_variability.csv"
    # quant_csv = "quant_policy_variability_v2.csv"
    # prune_csv = "prune_policy_variability_v2.csv"
    # toksparse_csv = "toksparse_policy_variability_v2.csv"

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
    # import pdb; pdb.set_trace()
    # ---------- Create 3 side-by-side subplots ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    make_violin_subplot(
        axes[0],
        quant_data,
        title="Quantization Action-Space",
        ylabel="Perplexity",
        policy_values=quant_policy,
        fixed_values=quant_fixed,
        show_legend=True,
    )

    make_violin_subplot(
        axes[1],
        prune_data,
        title="Pruning Action-Space",
        ylabel="Perplexity",
        policy_values=prune_policy,
        fixed_values=prune_fixed,
        show_legend=False,
    )

    make_violin_subplot(
        axes[2],
        toksparse_data,
        title="Token-Sparsity Action-Space",
        ylabel="Perplexity",
        policy_values=toksparse_policy,
        fixed_values=toksparse_fixed,
        show_legend=False,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = "action_variability_with_policy.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved combined plot to {output_path}")


if __name__ == "__main__":
    main()

