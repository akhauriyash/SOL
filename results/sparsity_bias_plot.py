import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("lce_more_baselines.csv")

num_cols = [
    "sparsity_bias",
    "dense_ppl", "sft_teacher_ppl", "policy_ppl", "random_ppl", "fixed_ppl", "drift_aware_ppl", "emc_ppl",
    "dense_keep_all", "policy_keep_all", "sft_teacher_keep_all", "random_keep_all", "fixed_keep_all", "drift_aware_keep_all", "emc_keep_all",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
})

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4))

bias_df = (
    df[["sparsity_bias", "policy_keep_all"]]
    .dropna()
    .groupby("sparsity_bias", as_index=False)
    .mean()
    .sort_values("sparsity_bias")
)
ax_left.plot(bias_df["sparsity_bias"], bias_df["policy_keep_all"], marker="o", linewidth=3, markersize=8)
ax_left.set_xlabel("Sparsity Bias")
ax_left.set_ylabel("Keep-Rate")
ax_left.grid(True, linewidth=0.5, alpha=0.5)

series_specs = {
    "Policy": ("policy_keep_all", "policy_ppl"),
    "Random": ("random_keep_all", "random_ppl"),
    "Fixed": ("fixed_keep_all", "fixed_ppl"),
    "Greedy Oracle": ("sft_teacher_keep_all", "sft_teacher_ppl"),
    "DAC": ("drift_aware_keep_all", "drift_aware_ppl"),
    "EMC": ("emc_keep_all", "emc_ppl"),
}

colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
for label, (xcol, ycol) in series_specs.items():
    if xcol in df.columns and ycol in df.columns:
        d = (
            df[[xcol, ycol]]
            .dropna()
            .groupby(xcol, as_index=False)
            .mean()
            .sort_values(xcol)
        )
        if not d.empty:
            ax_right.plot(d[xcol], d[ycol], marker="o", linewidth=2, label=label, markersize=6,
                          color=colors.pop(0))

if "dense_ppl" in df.columns:
    dense_level = float(df["dense_ppl"].dropna().mean())
    ax_right.axhline(dense_level, linestyle="--", linewidth=2, color="black")

ax_right.set_xlabel("Keep-Rate")
ax_right.set_ylabel("Perplexity")
ax_right.set_xlim(0.4, 0.82)
ax_right.set_ylim(top=12.0)
ax_right.grid(True, linewidth=0.5, alpha=0.5)
ax_right.legend(ncol=2)

plt.tight_layout()
plt.savefig("lce_more_sparsity_bias_study.pdf", bbox_inches="tight")
