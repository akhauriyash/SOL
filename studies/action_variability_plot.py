import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List

# Global font size
plt.rcParams.update({'font.size': 16})


def load_ppl_by_action_space(
    csv_path: str,
    action_key: str,
    ppl_key: str = "ppl_trials",
) -> Dict[str, List[float]]:
    """
    Load a CSV produced by action_variability.py and return a dict:
        { action_space_str: [ppl1, ppl2, ...] }

    action_key: column name for the action space
                e.g. "quant_choices", "prune_choices", or "keep_fracs"
    ppl_key:    column name containing pipe-separated perplexities, e.g. "ppl_trials"
    """
    action_to_ppl: Dict[str, List[float]] = defaultdict(list)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if action_key not in row or ppl_key not in row:
                continue

            action_space_raw = row[action_key]
            ppl_raw = row[ppl_key]

            if not action_space_raw or not ppl_raw:
                continue

            # In the CSV we encoded lists as '|' separated, and replaced commas with '|'.
            # For display on x-axis, we convert '|' back to ',' for nicer labels.
            action_space_display = action_space_raw.replace("|", ",")

            ppl_strs = ppl_raw.split("|")
            for s in ppl_strs:
                s = s.strip()
                if not s:
                    continue
                try:
                    val = float(s)
                except ValueError:
                    continue
                action_to_ppl[action_space_display].append(val)

    return action_to_ppl


def make_violin_subplot(
    ax,
    action_to_ppl: Dict[str, List[float]],
    title: str,
    ylabel: str = "Perplexity",
    rotate_xticks: bool = True,
):
    """
    Draw a violin plot on a given Axes.

    action_to_ppl: dict {action_space_str: [ppl_values]}
    """
    if not action_to_ppl:
        ax.set_title(f"{title}\n(no data)")
        ax.set_xticks([])
        return

    # Sort x-axis labels for consistent order
    labels = sorted(action_to_ppl.keys())
    data = [action_to_ppl[label] for label in labels]
    labels = [x.replace("q", "Q").replace("s", "S").replace(",", "|").replace("0000", "0").replace("000", "0") for x in labels]

    vp = ax.violinplot(data, showmeans=True, showextrema=True, showmedians=True)

    positions = range(1, len(labels) + 1)
    ax.set_xticks(list(positions))
    if rotate_xticks:
        ax.set_xticklabels(labels, rotation=25, ha="right")
    else:
        ax.set_xticklabels(labels)

    ax.set_title(title)
    ax.set_ylabel(ylabel)


def main():
    # Paths (adjust if needed)
    quant_csv = "quant_action_variability.csv"
    prune_csv = "prune_action_variability.csv"
    toksparse_csv = "toksparse_action_variability.csv"

    # Load data
    quant_data = load_ppl_by_action_space(
        quant_csv,
        action_key="quant_choices",
        ppl_key="ppl_trials",
    )
    prune_data = load_ppl_by_action_space(
        prune_csv,
        action_key="prune_choices",
        ppl_key="ppl_trials",
    )
    # For sparsity, we treat the action space as keep_fracs
    toksparse_data = load_ppl_by_action_space(
        toksparse_csv,
        action_key="keep_fracs",
        ppl_key="ppl_trials",
    )

    # Create 3 side-by-side subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    # Left: Quantization
    make_violin_subplot(
        axes[0],
        quant_data,
        title="Quantization Action-Space",
        ylabel="Perplexity",
    )

    # Middle: Pruning
    make_violin_subplot(
        axes[1],
        prune_data,
        title="Pruning Action-Space",
        ylabel="Perplexity",
    )

    # Right: Token Sparsity
    make_violin_subplot(
        axes[2],
        toksparse_data,
        title="Token-Sparsity Action-Space",
        ylabel="Perplexity",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = "action_variability.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved combined plot to {output_path}")


if __name__ == "__main__":
    main()
