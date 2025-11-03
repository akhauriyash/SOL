import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("early_multiact_ppl.csv")

# Keep only rows where exactly one of the three biases is non-zero
bias_cols = ["sparsity_bias", "prune_bias", "quant_bias"]
only_one_nonzero = (df[bias_cols].ne(0)).sum(axis=1) == 1
sub = df.loc[only_one_nonzero].copy()

# Prepare grouped tables (averaged over duplicate x values) for each bias/metric pair
tables = {
    "Pruning (policy_prune_keep vs prune_bias)":
        sub.loc[sub["prune_bias"].ne(0), ["prune_bias", "policy_prune_keep"]]
           .dropna()
           .sort_values("prune_bias")
           .groupby("prune_bias", as_index=False)["policy_prune_keep"].mean(),

    "Quantization (policy_quant_ratio vs quant_bias)":
        sub.loc[sub["quant_bias"].ne(0), ["quant_bias", "policy_quant_ratio"]]
           .dropna()
           .sort_values("quant_bias")
           .groupby("quant_bias", as_index=False)["policy_quant_ratio"].mean(),

    "Sparsity (policy_keep_all vs sparsity_bias)":
        sub.loc[sub["sparsity_bias"].ne(0), ["sparsity_bias", "policy_keep_all"]]
           .dropna()
           .sort_values("sparsity_bias")
           .groupby("sparsity_bias", as_index=False)["policy_keep_all"].mean(),
}

# --- Print each table individually ---
for title, t in tables.items():
    print("\n" + title)
    print(t.to_string(index=False))

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

plots = [
    ("prune_bias", "policy_prune_keep", "Pruning"),
    ("quant_bias", "policy_quant_ratio", "Quantization"),
    ("sparsity_bias", "policy_keep_all", "Sparsity"),
]

for ax, (xcol, ycol, title) in zip(axes, plots):
    d = tables[[k for k in tables if k.startswith(title)][0]]  # reuse the grouped table
    ax.plot(d[xcol], d[ycol], marker="o")
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.3)

fig.suptitle("Bias scan (only-one-bias-nonzero subset)", y=1.03, fontsize=12)
plt.savefig("bias_scan_x.pdf")
plt.close(fig)
