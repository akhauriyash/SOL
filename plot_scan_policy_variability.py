import pandas as pd
import matplotlib.pyplot as plt
import math


def parse_rand_ppl_column(series: pd.Series) -> pd.Series:
    """
    Parse the rand_ppl_trials column (pipe-separated floats) and return the
    first value as a float. For num_trials=1 this is the only value.
    """
    def parse_cell(cell):
        # Handle NaNs / non-strings
        if isinstance(cell, str):
            cell = cell.strip()
            if not cell:
                return math.nan
            first = cell.split("|")[0]
            try:
                return float(first)
            except ValueError:
                return math.nan
        try:
            return float(cell)
        except (TypeError, ValueError):
            return math.nan

    return series.apply(parse_cell)


def load_and_prepare(path: str) -> pd.DataFrame:
    """
    Load a scan CSV and add a 'rand_ppl' column extracted from rand_ppl_trials.
    """
    df = pd.read_csv(path)
    if "rand_ppl_trials" in df.columns:
        df["rand_ppl"] = parse_rand_ppl_column(df["rand_ppl_trials"])
    else:
        df["rand_ppl"] = math.nan
    return df


def main():
    # Input CSVs
    files = {
        "prune": "s40s100_prune.csv",
        "quant": "q6q16_quant.csv",
        "toksparse": "k10k100_toksparse.csv",
    }

    # X-axis columns for each row
    # Row 1: actual budgets used by the policy / matched baselines
    row1_x = {
        "prune": "target_prune_keep",
        "quant": "target_quant_ratio",
        "toksparse": "target_keep_effective",
    }

    # Row 2: CLI target budgets
    row2_x = {
        "prune": "tgt_prune_keep_cli",
        "quant": "tgt_quant_ratio_cli",
        "toksparse": "tgt_keep_cli",
    }

    # Human-readable titles
    titles = {
        "prune": "Structured pruning",
        "quant": "Quantization",
        "toksparse": "Token sparsity",
    }

    # Axis label mappings
    achieved_labels = {
        "target_prune_keep": "Achieved Prune Rate",
        "target_quant_ratio": "Achieved Quantization Ratio",
        "target_keep_effective": "Achieved Token Sparsity",
    }

    cli_labels = {
        "tgt_prune_keep_cli": "Target Prune Rate",
        "tgt_quant_ratio_cli": "Target Quantization Ratio",
        "tgt_keep_cli": "Target Token Sparsity",
    }

    # Row 3: CLI target vs achieved token sparsity for each kind
    row3_x = {
        "prune": "tgt_prune_keep_cli",
        "quant": "tgt_quant_ratio_cli",
        "toksparse": "tgt_keep_cli",
    }
    # 3 rows x 3 columns:
    #   Row 1: achieved budgets vs ppl
    #   Row 2: CLI targets vs ppl
    #   Row 3: target_keep_cli vs target_keep_effective
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))


    for col_idx, (kind, path) in enumerate(files.items()):
        df = load_and_prepare(path)

        # ---------------- Row 1: actual budgets ---------------- #
        x_col1 = row1_x[kind]
        df1 = df.dropna(subset=[x_col1]).sort_values(x_col1)

        ax1 = axes[0, col_idx]
        ax1.plot(df1[x_col1], df1["policy_ppl"], marker="o", label="Policy")
        ax1.plot(df1[x_col1], df1["fixed_ppl"], marker="s", label="Fixed matched")
        ax1.plot(df1[x_col1], df1["rand_ppl"], marker="^", label="Random matched")

        ax1.set_title(titles[kind])
        ax1.set_xlabel(achieved_labels.get(x_col1, x_col1))
        if col_idx == 0:
            ax1.set_ylabel("Perplexity")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # ---------------- Row 2: CLI targets ---------------- #
        x_col2 = row2_x[kind]
        df2 = df.dropna(subset=[x_col2]).sort_values(x_col2)

        ax2 = axes[1, col_idx]
        ax2.plot(df2[x_col2], df2["policy_ppl"], marker="o", label="Policy")
        ax2.plot(df2[x_col2], df2["fixed_ppl"], marker="s", label="Fixed matched")
        ax2.plot(df2[x_col2], df2["rand_ppl"], marker="^", label="Random matched")

        ax2.set_xlabel(cli_labels.get(x_col2, x_col2))
        if col_idx == 0:
            ax2.set_ylabel("Perplexity")
        ax2.grid(True, alpha=0.3)

        ax2.legend()

        # ---------------- Row 3: target_keep_cli vs target_keep_effective ---------------- #
        x_col3 = row3_x[kind]
        y_col3 = "target_keep_effective"
        if x_col3 in df.columns and y_col3 in df.columns:
            df3 = df.dropna(subset=[x_col3, y_col3]).sort_values(x_col3)
        else:
            df3 = pd.DataFrame()

        ax3 = axes[2, col_idx]
        if not df3.empty:
            ax3.plot(
                df3[x_col3],
                df3[y_col3],
                marker="o",
                label="Achieved Token Sparsity",
            )
        else:
            ax3.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )

        x_label3 = cli_labels.get(x_col3, x_col3)
        y_label3 = achieved_labels.get(y_col3, y_col3)
        ax3.set_xlabel(x_label3)
        ax3.set_ylabel(y_label3)
        ax3.set_title(f"{titles[kind]}: {x_label3} vs {y_label3}")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    fig.tight_layout()
    fig.savefig("scan_policy_variability.pdf")
    print("Saved figure to scan_policy_variability.pdf")


if __name__ == "__main__":
    main()
