import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

df_lce = pd.read_csv("dkl_vs_lce___lce.csv")
df_dkl = pd.read_csv("dkl_vs_lce___dkl.csv")

for df in (df_lce, df_dkl):
    for c in ["policy_keep_all", "policy_ppl"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

lce = (
    df_lce[["policy_keep_all", "policy_ppl"]]
    .dropna()
    .groupby("policy_keep_all", as_index=False)
    .mean()
    .sort_values("policy_keep_all")
)

dkl = (
    df_dkl[["policy_keep_all", "policy_ppl"]]
    .dropna()
    .groupby("policy_keep_all", as_index=False)
    .mean()
    .sort_values("policy_keep_all")
)

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(lce["policy_keep_all"], lce["policy_ppl"], marker="o", linewidth=2, markersize=6, label="LCE Policy")
ax.plot(dkl["policy_keep_all"], dkl["policy_ppl"], marker="o", linewidth=2, markersize=6, label="DKL Policy")

ax.set_xlabel("Keep-Rate")
ax.set_ylabel("Perplexity")
ax.set_xlim(0.4, 0.85)
ax.set_ylim(10, 11.5)
ax.grid(True, linewidth=0.5, alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig("dkl_vs_lce.pdf", bbox_inches="tight")
