import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# File names (adjust if needed)
# ------------------------------------------------------------------
FILES = {
    "prune": "dec6_prune_meffsc.csv",
    "quant": "dec6_quant_meffsc.csv",
    "toksparse": "dec6_toksparse_meffsc.csv",
    "all":   "dec6_all_meffsc.csv",
}

def maybe_read_csv(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df.dropna(how="all")
    return None

dfs = {k: maybe_read_csv(v) for k, v in FILES.items()}

# ------------------------------------------------------------------
# peraxis.pdf: 2 rows x 3 cols
#   row 0: ppl vs effective (actual)
#   row 1: ppl vs target (CLI)
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")

configs = [
    {
        "name": "prune",
        "title": "Prune",
        "x_top_policy": "policy_prune_keep_actual",
        "x_top_fixed":  "fixed_prune_keep",
        "x_bottom":     "tgt_prune_keep_cli",
    },
    {
        "name": "quant",
        "title": "Quant",
        "x_top_policy": "policy_quant_ratio_actual",
        "x_top_fixed":  "fixed_quant_ratio",
        "x_bottom":     "tgt_quant_ratio_cli",
    },
    {
        "name": "toksparse",
        "title": "TokSparse",
        "x_top_policy": "policy_keep_effective_actual",
        "x_top_fixed":  "fixed_keep_all",
        "x_bottom":     "tgt_keep_cli",
    },
]

for col_idx, cfg in enumerate(configs):
    df = dfs[cfg["name"]]
    ax_top = axes[0, col_idx]
    ax_bot = axes[1, col_idx]

    ax_top.set_title(cfg["title"])

    if df is None or df.empty:
        ax_top.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax_top.transAxes)
        ax_bot.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax_bot.transAxes)
        continue

    # ---------- Top row: ppl vs effective keep (actual) ----------
    if (
        cfg["x_top_policy"] in df.columns and
        cfg["x_top_fixed"] in df.columns and
        "policy_ppl" in df.columns and
        "fixed_ppl" in df.columns
    ):
        ax_top.plot(
            df[cfg["x_top_policy"]],
            df["policy_ppl"],
            marker="o",
            label="policy",
        )
        ax_top.plot(
            df[cfg["x_top_fixed"]],
            df["fixed_ppl"],
            marker="o",
            linestyle="--",
            label="fixed",
        )
        ax_top.set_xlabel(cfg["x_top_policy"])
        if col_idx == 0:
            ax_top.set_ylabel("Perplexity")
        ax_top.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_top.legend()
    else:
        ax_top.text(0.5, 0.5, "Missing cols", ha="center", va="center",
                    transform=ax_top.transAxes)

    # ---------- Bottom row: ppl vs target keep (CLI) ----------
    if (
        cfg["x_bottom"] in df.columns and
        "policy_ppl" in df.columns and
        "fixed_ppl" in df.columns
    ):
        ax_bot.plot(
            df[cfg["x_bottom"]],
            df["policy_ppl"],
            marker="o",
            label="policy",
        )
        ax_bot.plot(
            df[cfg["x_bottom"]],
            df["fixed_ppl"],
            marker="o",
            linestyle="--",
            label="fixed",
        )
        ax_bot.set_xlabel(cfg["x_bottom"])
        if col_idx == 0:
            ax_bot.set_ylabel("Perplexity")
        ax_bot.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_bot.legend()
    else:
        ax_bot.text(0.5, 0.5, "Missing cols", ha="center", va="center",
                    transform=ax_bot.transAxes)

plt.tight_layout()
plt.savefig("peraxis.pdf")
plt.close(fig)

# ------------------------------------------------------------------
# allaxis.pdf:
#   X = sum of all effective keep rates (policy / fixed separately)
#   Y = perplexity (policy_ppl / fixed_ppl)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
df_all = dfs["all"]

if df_all is None or df_all.empty:
    ax.text(0.5, 0.5, "No 'all' data", ha="center", va="center",
            transform=ax.transAxes)
    ax.set_xlabel("Sum of effective keep rates")
    ax.set_ylabel("Perplexity")
else:
    # Try to build sum-of-effective rates from actual columns
    cols_policy = ["policy_prune_keep_actual",
                   "policy_quant_ratio_actual",
                   "policy_keep_effective_actual"]
    cols_fixed = ["fixed_prune_keep",
                  "fixed_quant_ratio",
                  "fixed_keep_all"]

    have_policy_effective = all(c in df_all.columns for c in cols_policy)
    have_fixed_effective = all(c in df_all.columns for c in cols_fixed)

    x_policy = x_fixed = None

    if have_policy_effective and have_fixed_effective:
        x_policy = (
            df_all["policy_prune_keep_actual"] +
            df_all["policy_quant_ratio_actual"] +
            df_all["policy_keep_effective_actual"]
        )
        x_fixed = (
            df_all["fixed_prune_keep"] +
            df_all["fixed_quant_ratio"] +
            df_all["fixed_keep_all"]
        )
    elif "target_keep_effective" in df_all.columns:
        # Fallback: use same scalar for both if detailed pieces are missing
        x_policy = df_all["target_keep_effective"]
        x_fixed = df_all["target_keep_effective"]

    if (
        x_policy is not None and
        x_fixed is not None and
        "policy_ppl" in df_all.columns and
        "fixed_ppl" in df_all.columns
    ):
        ax.scatter(x_policy, df_all["policy_ppl"], label="policy", marker="o")
        ax.scatter(x_fixed,  df_all["fixed_ppl"],  label="fixed",  marker="x")

        ax.set_xlabel("Sum of effective keep rates")
        ax.set_ylabel("Perplexity")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Missing cols", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xlabel("Sum of effective keep rates")
        ax.set_ylabel("Perplexity")

plt.tight_layout()
plt.savefig("allaxis.pdf")
plt.close(fig)



# import os
# import pandas as pd
# import matplotlib.pyplot as plt


# def load_csv_safe(path):
#     """Read a CSV or return None if it doesn't exist / can't be read."""
#     if not os.path.exists(path):
#         print(f"[warn] {path} not found")
#         return None
#     try:
#         df = pd.read_csv(path)
#         if df.empty:
#             print(f"[warn] {path} is empty")
#             return None
#         return df
#     except Exception as e:
#         print(f"[warn] could not read {path}: {e}")
#         return None


# def plot_from_df(ax, df, x_col, title, x_label):
#     """
#     Plot policy_ppl and fixed_ppl vs x_col from a DataFrame.
#     Returns True if data was plotted, False otherwise.
#     """
#     ax.set_title(title)
#     ax.set_xlabel(x_label)
#     ax.grid(True, alpha=0.3)

#     if df is None or df.empty:
#         ax.text(
#             0.5, 0.5, "No data",
#             ha="center", va="center", transform=ax.transAxes
#         )
#         return False

#     required_cols = {x_col, "policy_ppl", "fixed_ppl"}
#     if not required_cols.issubset(df.columns):
#         ax.text(
#             0.5, 0.5, "Required columns\nmissing",
#             ha="center", va="center", transform=ax.transAxes
#         )
#         return False

#     df = df.sort_values(x_col)

#     x = df[x_col].values
#     policy_ppl = df["policy_ppl"].values
#     fixed_ppl = df["fixed_ppl"].values

#     ax.plot(x, policy_ppl, marker="o", linestyle="-", label="Policy")
#     ax.plot(x, fixed_ppl, marker="x", linestyle="--", label="Fixed")

#     return True


# def make_peraxis_ablation():
#     """Original per-axis ablation plot from *_only_* CSVs."""
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

#     plotted_any = False
#     legend_handles, legend_labels = None, None

#     # Sparsity-only: x = policy_keep_all
#     df_sparse = load_csv_safe("sparse_only_multi_eff_ppl_scan.csv")
#     if plot_from_df(
#         axes[0],
#         df_sparse,
#         x_col="policy_keep_all",
#         title="Sparsity only",
#         x_label="Sparsity keep rate (policy_keep_all)",
#     ):
#         if not plotted_any:
#             legend_handles, legend_labels = axes[0].get_legend_handles_labels()
#             plotted_any = True

#     # Prune-only: x = policy_prune_keep
#     # df_prune = load_csv_safe("prune_only_multi_eff_ppl_scan.csv")
#     df_prune = load_csv_safe("prune_only_45pc_multi_eff_ppl_scan.csv")
#     if plot_from_df(
#         axes[1],
#         df_prune,
#         x_col="policy_prune_keep",
#         title="Pruning only",
#         x_label="Pruning keep rate (policy_prune_keep)",
#     ):
#         if not plotted_any:
#             legend_handles, legend_labels = axes[1].get_legend_handles_labels()
#             plotted_any = True

#     # Quant-only: x = policy_quant_ratio
#     # df_quant = load_csv_safe("quant_only_multi_eff_ppl_scan.csv")
#     df_quant = load_csv_safe("quant_only_60pc_multi_eff_ppl_scan.csv")
#     if plot_from_df(
#         axes[2],
#         df_quant,
#         x_col="policy_quant_ratio",
#         title="Quantization only",
#         x_label="Quantization keep rate (policy_quant_ratio)",
#     ):
#         if not plotted_any:
#             legend_handles, legend_labels = axes[2].get_legend_handles_labels()
#             plotted_any = True

#     axes[0].set_ylabel("Perplexity (ppl)")
#     axes[1].set_xlim(0.4, 0.55)

#     if plotted_any and legend_handles:
#         fig.legend(legend_handles, legend_labels, loc="upper center", ncol=2)

#     fig.tight_layout(rect=[0, 0, 1, 0.90])
#     fig.savefig("peraxis_abl.pdf")
#     plt.close(fig)
#     print("[ok] saved peraxis_abl.pdf")


# def make_joint_peraxis():
#     """
#     Joint per-axis plot using joint_method_multi_eff_ppl_scan.csv.

#     For each axis:
#       - Sparsity: prune_bias = 0, quant_bias = 0, vary sparsity_bias
#       - Pruning: sparsity_bias = 0, quant_bias = 0, vary prune_bias
#       - Quant: sparsity_bias = 0, prune_bias = 0, vary quant_bias
#     """
#     csv_path = "joint_method_multi_eff_ppl_scan.csv"
#     df = load_csv_safe(csv_path)

#     fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
#     axes[0].set_ylabel("Perplexity (ppl)")

#     if df is None:
#         # Just mark all panels as no data and save
#         for ax in axes:
#             ax.text(
#                 0.5, 0.5, "No data",
#                 ha="center", va="center", transform=ax.transAxes
#             )
#             ax.grid(True, alpha=0.3)
#         fig.tight_layout()
#         fig.savefig("joint_peraxis.pdf")
#         plt.close(fig)
#         print("[warn] no joint data; saved empty joint_peraxis.pdf")
#         return

#     # Safely handle bias columns
#     needed_bias_cols = {"sparsity_bias", "prune_bias", "quant_bias"}
#     if not needed_bias_cols.issubset(df.columns):
#         for ax in axes:
#             ax.text(
#                 0.5, 0.5, "Bias columns\nmissing",
#                 ha="center", va="center", transform=ax.transAxes
#             )
#             ax.grid(True, alpha=0.3)
#         fig.tight_layout()
#         fig.savefig("joint_peraxis.pdf")
#         plt.close(fig)
#         print("[warn] bias columns missing in joint CSV; saved empty joint_peraxis.pdf")
#         return

#     tol = 1e-8

#     # Sparsity sweep: only sparsity_bias varies, others ~ 0
#     df_sparsity = df[
#         (df["prune_bias"].abs() < tol) &
#         (df["quant_bias"].abs() < tol)
#     ].copy()

#     # Pruning sweep: only prune_bias varies
#     df_prune = df[
#         (df["sparsity_bias"].abs() < tol) &
#         (df["quant_bias"].abs() < tol)
#     ].copy()

#     # Quantization sweep: only quant_bias varies
#     df_quant = df[
#         (df["sparsity_bias"].abs() < tol) &
#         (df["prune_bias"].abs() < tol)
#     ].copy()

#     plotted_any = False
#     legend_handles, legend_labels = None, None

#     if plot_from_df(
#         axes[0],
#         df_sparsity,
#         x_col="policy_keep_all",
#         title="Joint ckpt – Sparsity sweep",
#         x_label="Sparsity keep rate (policy_keep_all)",
#     ):
#         if not plotted_any:
#             legend_handles, legend_labels = axes[0].get_legend_handles_labels()
#             plotted_any = True

#     if plot_from_df(
#         axes[1],
#         df_prune,
#         x_col="policy_prune_keep",
#         title="Joint ckpt – Pruning sweep",
#         x_label="Pruning keep rate (policy_prune_keep)",
#     ):
#         if not plotted_any:
#             legend_handles, legend_labels = axes[1].get_legend_handles_labels()
#             plotted_any = True

#     if plot_from_df(
#         axes[2],
#         df_quant,
#         x_col="policy_quant_ratio",
#         title="Joint ckpt – Quantization sweep",
#         x_label="Quantization keep rate (policy_quant_ratio)",
#     ):
#         if not plotted_any:
#             legend_handles, legend_labels = axes[2].get_legend_handles_labels()
#             plotted_any = True

#     if plotted_any and legend_handles:
#         fig.legend(legend_handles, legend_labels, loc="upper center", ncol=2)

#     fig.tight_layout(rect=[0, 0, 1, 0.90])
#     fig.savefig("joint_peraxis.pdf")
#     plt.close(fig)
#     print("[ok] saved joint_peraxis.pdf")


# def make_joint_truejoint():
#     """
#     True joint plot: x-axis is effective decode compute defined as
#     linear sum of the three POLICY keep measures:

#         effective_compute = policy_keep_all
#                            + policy_prune_keep
#                            + policy_quant_ratio

#     y-axis is ppl (policy_ppl and fixed_ppl).
#     """
#     csv_path = "joint_method_multi_eff_ppl_scan.csv"
#     df = load_csv_safe(csv_path)

#     fig, ax = plt.subplots(figsize=(6, 4))

#     if df is None:
#         ax.text(
#             0.5, 0.5, "No data",
#             ha="center", va="center", transform=ax.transAxes
#         )
#         ax.set_xlabel("Effective decode compute")
#         ax.set_ylabel("Perplexity (ppl)")
#         ax.grid(True, alpha=0.3)
#         fig.tight_layout()
#         # ax.set_xlim(1.2, 1.7)
#         fig.savefig("joint_truejoint.pdf")
#         plt.close(fig)
#         print("[warn] no joint data; saved empty joint_truejoint.pdf")
#         return

#     required_cols = {
#         "policy_keep_all", "policy_prune_keep", "policy_quant_ratio",
#         "policy_ppl", "fixed_ppl"
#     }
#     if not required_cols.issubset(df.columns):
#         ax.text(
#             0.5, 0.5, "Required columns\nmissing",
#             ha="center", va="center", transform=ax.transAxes
#         )
#         ax.set_xlabel("Effective decode compute")
#         ax.set_ylabel("Perplexity (ppl)")
#         ax.grid(True, alpha=0.3)
#         # ax.set_xlim(1.2, 1.7)
#         fig.tight_layout()
#         fig.savefig("joint_truejoint.pdf")
#         plt.close(fig)
#         print("[warn] required columns missing in joint CSV; saved empty joint_truejoint.pdf")
#         return

#     # Define effective decode compute as linear sum of the three policy axes
#     df["effective_compute"] = (
#         df["policy_keep_all"].astype(float) +
#         df["policy_prune_keep"].astype(float) +
#         df["policy_quant_ratio"].astype(float)
#     )

#     df = df.sort_values("effective_compute")

#     x = df["effective_compute"].values
#     policy_ppl = df["policy_ppl"].values
#     fixed_ppl = df["fixed_ppl"].values

#     # Use small markers so dense sweeps don't look too crazy
#     ax.plot(x, policy_ppl, marker="o", linestyle="-", markersize=3, label="Policy")
#     ax.plot(x, fixed_ppl, marker="x", linestyle="--", markersize=3, label="Fixed")

#     ax.set_xlabel("Effective decode compute\n(policy_keep_all + policy_prune_keep + policy_quant_ratio)")
#     ax.set_ylabel("Perplexity (ppl)")
#     ax.grid(True, alpha=0.3)
#     ax.legend()

#     # ax.set_xlim(1.2, 1.7)
#     fig.tight_layout()
#     fig.savefig("joint_truejoint.pdf")
#     plt.close(fig)
#     print("[ok] saved joint_truejoint.pdf")


# def main():
#     make_peraxis_ablation()
#     make_joint_peraxis()
#     make_joint_truejoint()


# if __name__ == "__main__":
#     main()
