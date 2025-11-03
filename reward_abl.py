import pandas as pd
import matplotlib.pyplot as plt

# ---- configurable baseline: choose from {"fixed", "random", "drift_aware", "emc"} ----
BASELINE_NAME = "fixed"  # e.g., "random", "fixed", "drift_aware", or "emc"
# --------------------------------------------------------------------------------------

def detect_run_type(name: str) -> str | None:
    s = str(name)
    if "RL_LCE_Rec_Outc-" in s:
        return "Outcome"
    if "RL_LCE_Rec_Hybrid-" in s:
        return "Hybrid"
    if "RL_LCE_Rec-" in s:
        return "Process"
    return None

def main():
    df = pd.read_csv("ckpt_perplexities.csv")

    # Tag run types from ckpt_dir
    df["run_type"] = df["ckpt_dir"].map(detect_run_type)
    df = df.dropna(subset=["run_type"])

    # Ensure numeric types
    numeric_cols = [
        "policy_keep_all", "policy_ppl", "sparsity_bias",
        "random_ppl", "fixed_ppl", "drift_aware_ppl", "emc_ppl"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Validate baseline column
    baseline_col = f"{BASELINE_NAME}_ppl"
    if baseline_col not in df.columns:
        raise ValueError(
            f"Baseline '{BASELINE_NAME}' not available. "
            f"Choose one of: 'random', 'fixed', 'drift_aware', 'emc'."
        )

    plt.figure(figsize=(6, 4), dpi=150)

    # Plot main curves and dashed baselines (same color, no legend)
    for label in ["Process", "Outcome", "Hybrid"]:
        sub = df[df["run_type"] == label][["policy_keep_all", "policy_ppl", baseline_col]].dropna()
        if sub.empty:
            continue

        # Aggregate by x for smooth lines (mean across same x)
        main_line = (
            sub.groupby("policy_keep_all", as_index=False)["policy_ppl"]
               .mean()
               .sort_values("policy_keep_all")
        )
        baseline_line = (
            sub.groupby("policy_keep_all", as_index=False)[baseline_col]
               .mean()
               .sort_values("policy_keep_all")
        )

        # Plot main solid line with label
        (line,) = plt.plot(
            main_line["policy_keep_all"],
            main_line["policy_ppl"],
            marker="o",
            label=label
        )

        # # Plot baseline dashed line in the SAME color, no legend entry
        # plt.plot(
        #     baseline_line["policy_keep_all"],
        #     baseline_line[baseline_col],
        #     linestyle="--",
        #     linewidth=1.0,
        #     color=line.get_color(),
        #     label="_nolegend_"
        # )

    plt.xlabel("policy_keep_all")
    plt.ylabel("policy_ppl")
    plt.title("Reward Ablation")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Run Type")
    plt.tight_layout()
    plt.savefig("reward_ablation.pdf")

    # Print simple table: columns=sparsity_bias, rows=run_type, values=policy_keep_all
    if "sparsity_bias" in df.columns:
        table = (
            df.dropna(subset=["sparsity_bias", "policy_keep_all"])
              .pivot_table(
                  index="run_type",
                  columns="sparsity_bias",
                  values="policy_keep_all",
                  aggfunc="mean"
              )
              .sort_index(axis=1)
        )
        print("\npolicy_keep_all by run_type (rows) and sparsity_bias (columns):")
        print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    else:
        print("Column 'sparsity_bias' not found; skipping table.")

if __name__ == "__main__":
    main()
