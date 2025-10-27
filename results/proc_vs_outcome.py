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

df_process = pd.read_csv("dkl_sparsity_bias_ckpt_perplexities.csv")
df_outcome = pd.read_csv("dkl_sparsity_bias_outcome_ckpt_perplexities.csv")

for df in (df_process, df_outcome):
    for c in ["policy_keep_all", "policy_ppl"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

process = (
    df_process[["policy_keep_all", "policy_ppl"]]
    .dropna()
    .groupby("policy_keep_all", as_index=False)
    .mean()
    .sort_values("policy_keep_all")
)

outcome = (
    df_outcome[["policy_keep_all", "policy_ppl"]]
    .dropna()
    .groupby("policy_keep_all", as_index=False)
    .mean()
    .sort_values("policy_keep_all")
)
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(process["policy_keep_all"], process["policy_ppl"], marker="o", linewidth=2, markersize=6, label="Process Policy")
ax.plot(outcome["policy_keep_all"], outcome["policy_ppl"], marker="o", linewidth=2, markersize=6, label="Outcome Policy")

ax.set_xlabel("Keep-Rate")
ax.set_ylabel("Perplexity")
ax.set_xlim(0.25, 0.85)
ax.grid(True, linewidth=0.5, alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig("process_vs_outcome.pdf", bbox_inches="tight")
