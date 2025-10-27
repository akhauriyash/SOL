import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
})

def load_curves(
    csv_path,
    x_col="policy_keep_all",
    y_main="policy_ppl",
    y_teacher="sft_teacher_ppl",
    y_fixed=None
):
    df = pd.read_csv(csv_path)

    for c in (x_col, y_main, y_teacher, y_fixed):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    main = (
        df[[x_col, y_main]]
        .dropna()
        .groupby(x_col, as_index=False)
        .mean()
        .sort_values(x_col)
        .rename(columns={y_main: "y"})
    )

    teacher = None
    if y_teacher in df.columns:
        t = (
            df[[x_col, y_teacher]]
            .dropna()
            .groupby(x_col, as_index=False)
            .mean()
            .sort_values(x_col)
            .rename(columns={y_teacher: "y"})
        )
        if not t.empty:
            teacher = t

    fixed = None
    if y_fixed in df.columns:
        f = (
            df[[x_col, y_fixed]]
            .dropna()
            .groupby(x_col, as_index=False)
            .mean()
            .sort_values(x_col)
            .rename(columns={y_fixed: "y"})
        )
        if not f.empty:
            fixed = f

    return main, teacher, fixed

series = [
    ("dkl_sparsity_bias_ckpt_perplexities.csv", "[0.2, 1.0]"),
    ("spread_perplexities.csv", "[0.1, 0.2, 0.4, 0.5, 0.8, 1.0]"),
]

fig, ax = plt.subplots(figsize=(6, 4))

for csv_path, label in series:
    main, teacher, fixed = load_curves(csv_path)

    (line,) = ax.plot(
        main["policy_keep_all"],
        main["y"],
        marker="o",
        linewidth=2,
        markersize=4,
        label=label,
    )
    color = line.get_color()

    if teacher is not None:
        ax.plot(
            teacher["policy_keep_all"],
            teacher["y"],
            linestyle="--",
            linewidth=1.2,
            marker=None,
            color=color,
            label="_nolegend_",
        )
    if fixed is not None:
        ax.plot(
            fixed["policy_keep_all"],
            fixed["y"],
            linestyle="-",
            linewidth=1.0,
            marker=None,
            color=color,
            alpha=0.9,
            label="_nolegend_",
        )

ax.axhline(y=9.810783548925864, linestyle="--", linewidth=2, color="black", label="_nolegend_")
ax.set_xlabel("Keep-Rate")
ax.set_ylabel("Perplexity")
ax.set_ylim(top=11.5)
ax.set_xlim(0.45, 1.05)
ax.grid(True, linewidth=0.5, alpha=0.5)
ax.legend(title="Actions")

plt.tight_layout()
plt.savefig("multiaction_pplx.pdf", bbox_inches="tight")
