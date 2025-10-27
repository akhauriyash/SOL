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


path_tuples = [("lce_sparsity_questp16_bias_ckpt_perplexities.csv", "quest_scaling_p16.pdf"),
               ("lce_sparsity_questp4_bias_ckpt_perplexities.csv", "quest_scaling_p4.pdf")]

for path, imgname in path_tuples:
    df_quest = pd.read_csv(path)
    df_quest = df_quest.sort_values("policy_keep_all")
    df_quest.to_csv(path, index=False)


    for c in ["policy_keep_all", "policy_ppl", "fixed_ppl", "random_ppl", "sft_teacher_ppl", "drift_aware_ppl", "emc_ppl", "fixed_keep_all", "random_keep_all", "sft_teacher_keep_all", "drift_aware_ppl", "emc_keep_all"]:
        if c in df_quest.columns:
            df_quest[c] = pd.to_numeric(df_quest[c], errors="coerce")

    quest = (
        df_quest[["policy_keep_all", "policy_ppl"]]
        .dropna()
        .groupby("policy_keep_all", as_index=False)
        .mean()
        .sort_values("policy_keep_all")
    )

    fixed = (
        df_quest[["fixed_keep_all", "fixed_ppl"]]
        .dropna()
        .groupby("fixed_keep_all", as_index=False)
        .mean()
        .sort_values("fixed_keep_all")
    )

    random = (
        df_quest[["random_keep_all", "random_ppl"]]
        .dropna()
        .groupby("random_keep_all", as_index=False)
        .mean()
        .sort_values("random_keep_all")
    )

    teacher = (
        df_quest[["sft_teacher_keep_all", "sft_teacher_ppl"]]
        .dropna()
        .groupby("sft_teacher_keep_all", as_index=False)
        .mean()
        .sort_values("sft_teacher_keep_all")
    )

    drift_aware = (
        df_quest[["drift_aware_keep_all", "drift_aware_ppl"]]
        .dropna()
        .groupby("drift_aware_keep_all", as_index=False)
        .mean()
        .sort_values("drift_aware_keep_all")
    )

    emc = (
        df_quest[["emc_keep_all", "emc_ppl"]]
        .dropna()
        .groupby("emc_keep_all", as_index=False)
        .mean()
        .sort_values("emc_keep_all")
    )


    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(quest["policy_keep_all"], quest["policy_ppl"], marker="o", linewidth=2, markersize=6, label="Quest + Policy")
    ax.plot(fixed["fixed_keep_all"], fixed["fixed_ppl"], marker="o", linewidth=2, markersize=6, label="Fixed")
    ax.plot(random["random_keep_all"], random["random_ppl"], marker="o", linewidth=2, markersize=6, label="Random")
    ax.plot(teacher["sft_teacher_keep_all"], teacher["sft_teacher_ppl"], marker="o", linewidth=2, markersize=6, label="Greedy Oracle")

    ax.axhline(y=9.810783548925864, linestyle="--", linewidth=2, color="black", label="_nolegend_")
    ax.set_xlabel("Keep-Rate")
    ax.set_ylabel("Perplexity")
    if "p16" in imgname:
        ax.set_xlim(left=0.35)
        ax.set_ylim(top=20)
    else:
        ax.set_xlim(left=0.4)
        ax.set_ylim(top=10.2)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(imgname, bbox_inches="tight")