#!/usr/bin/env python3
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Config ----------
CSV_PATTERN = "indiv_policy_by_bias_{mode}.csv"  # modes: pruning, quantization4, sparsity
# CSV_PATTERN = "{mode}_11055_ckpt_perplexities.csv"  # modes: pruning, quantization4, sparsity
OUT_FIG = "indiv_policy_by_bias.pdf"

FIGSIZE = (12, 4.2)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
})

# Force policy to black across all plots
METHOD_COLORS = {
    "policy": "black",
}

# ---------- Helpers ----------

def find_methods(df: pd.DataFrame):
    """Infer method names from columns like '<method>_ppl'."""
    methods = set()
    for c in df.columns:
        m = re.fullmatch(r"(.+)_ppl", c)
        if m:
            methods.add(m.group(1))
    return sorted(methods)

def mode_conf(mode: str):
    """Returns (x_key, xlabel, title) for a given mode."""
    mode = mode.lower()
    if mode == "sparsity":
        # NOTE: sparsity uses *_keep_all (not keep_rate)
        return ("keep_all", "Token Keep-Rate", "Token Sparsity")
    if mode == "quantization4":
        return ("quant_ratio", "Quantization (Bits)", "Quantization")
    if mode == "pruning":
        return ("prune_keep", "Prune Keep-Rate", "Pruning")
    raise ValueError("mode must be one of {'sparsity','quantization4','pruning'}")

def filter_2x_min(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep only entries with y <= 2 * min(y) among finite y."""
    y_fin = y[mask]
    if y_fin.size == 0:
        return mask & False
    min_y = np.nanmin(y_fin)
    return mask & (y <= 2.0 * min_y)

def get_x_for_method(df_rowblock: pd.DataFrame, method: str, x_key: str) -> np.ndarray:
    """
    Return the X series for a method.
    Prefer <method>_<x_key> if present; otherwise fall back to policy_<x_key>.
    This lets us plot methods that don't carry their own X column (common in your CSVs).
    """
    meth_col = f"{method}_{x_key}"
    pol_col  = f"policy_{x_key}"
    if meth_col in df_rowblock.columns:
        x = pd.to_numeric(df_rowblock[meth_col], errors="coerce").to_numpy(float)
    elif pol_col in df_rowblock.columns:
        x = pd.to_numeric(df_rowblock[pol_col], errors="coerce").to_numpy(float)
    else:
        x = np.full(len(df_rowblock), np.nan, dtype=float)
    return x

method_map = {
    "fixed": "Fixed",
    "drift_aware": r"$\Delta_{\cos}(A)$",
    "dcp": r"$|A| \times \|W\|_{2}$",
    "ecov": r"$\|A\|_{1}$ Cov@90%",
    "policy": "Policy",
    "emc": r"$\hat H$",
    "random": "Random",
    "dense": "Dense",
    "dynr": "Dyn. Range",
    "margin": "Margin",
    "qnr": "Quant. Noise",
    # "dynr": r"$\frac{\|A\|_\infty}{\sigma(A)}$",
    # "qnr": r"$\frac{\sigma_{A - Q_b(A)}}{\sigma_{A}}$",
    "lrm": "LRM",
}

    # "ecov": r"L1Cov@0.90",
    # "ecov": r"$\frac{\|\,|Act|\,\|_{1,\rho}}{\|Act\|_{1}}\geq\tau$",
    # "ecov": r"$\|\,|Act|\,\|_{1,\rho}\ \geq\ \tau\,\|Act\|_{1}$",
    # "ecov": r"$\sum_{i=1}^{K}|Act|_{(i)}\ \geq\ 0.9\sum_{i=1}^{D}|Act|_{(i)}$",
    # "ecov": r"$\mathrm{TopP}_\rho(|Act|;\mathrm{L1})\geq\tau$" ,
def plot_mode(ax, df: pd.DataFrame, mode: str):
    x_key, xlabel, title = mode_conf(mode)
    methods = find_methods(df)

    if mode == "pruning":
        # ax.set_ylim(9.9, 11.0)
        # ax.set_xlim(0.65, 0.75)
        ax.set_ylim(9.75, 10.65)
        ax.set_xlim(0.65, 0.8)
    if mode == "quantization4":
        ax.set_xlim(6.8, 8.6)  # was 0.4–0.7 in fraction; now 6.4–11.2 bits
        ax.set_ylim(10.5, 13)
        # ax.set_ylim(9.5, 13)
    plotted_any = False
    for method in methods:
        # if method in ["dense", "dcp", "ecov", "qnr", "lrm"]:
        if method in ["dense", "dcp", "lrm"]:
            continue  # skip dense baseline
        ycol = f"{method}_ppl"
        if ycol not in df.columns:
            continue

        # Y values (perplexity)
        y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(float)
        # X values (fallback to policy_* if <method>_* missing)
        x = get_x_for_method(df, method, x_key)
        if "quantization4" in mode:
            x = x * 16.0  # convert quant_ratio to bits

        base_mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(base_mask):
            continue

        # Early outlier filter: drop points with PPL > 2x min PPL
        m = filter_2x_min(y, base_mask)
        if not np.any(m):
            continue

        xs, ys = x[m], y[m]
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]

        color = METHOD_COLORS.get(method, None)
        if color is None:
            try:
                # Works across many Matplotlib versions
                color = ax._get_lines.get_next_color()
            except Exception:
                # Fallback to the first color from the rc cycle
                cycle = plt.rcParams.get("axes.prop_cycle", None)
                color = (cycle.by_key().get("color", ["C0"])[0] if cycle else "C0")

        if xs.size >= 2:
            # Smoother, non-parametric trend: centered running average
            # win = 5 if xs.size >= 7 else 3
            win = 3
            # Use the point itself when the window can't be fully computed
            x_s = pd.Series(xs)
            y_s = pd.Series(ys)
            x_smooth = x_s.rolling(win, min_periods=1, center=True).mean().fillna(x_s).to_numpy()
            y_smooth = y_s.rolling(win, min_periods=1, center=True).mean().fillna(y_s).to_numpy()
            m_s = np.isfinite(x_smooth) & np.isfinite(y_smooth)
            if np.any(m_s):
                ax.plot(x_smooth[m_s], y_smooth[m_s], linewidth=2.0, alpha=1, color=color, label=method_map[method])
        ax.plot(xs, ys, linestyle="None", marker="o", markersize=4, alpha=0.6, label=None, color=color)
        # ax.plot(xs, ys, linewidth=1.0, marker="o", markersize=4, alpha=0.5, label=method, color=color)
        plotted_any = True

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if mode not in ["quantization4", "pruning"]:
        ax.set_ylabel("Perplexity (↓)")
    ax.grid(True, alpha=0.3)

    if plotted_any:
        ax.legend(loc="best", frameon=True, framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

# ---------- Main ----------

def main():
    """
    Optionally pass custom paths:
        python script.py sparsity.csv quantization4.csv pruning.csv
    Else, defaults to {mode}_ckpt_perplexities.csv in CWD.
    """
    # Resolve input CSVs from args or pattern
    if len(sys.argv) >= 4:
        paths = {
            "sparsity": Path(sys.argv[1]),
            "quantization4": Path(sys.argv[2]),
            "pruning": Path(sys.argv[3]),
        }
    else:
        paths = {
            "sparsity": Path(CSV_PATTERN.format(mode="sparsity")),
            "quantization4": Path(CSV_PATTERN.format(mode="quantization4")),
            "pruning": Path(CSV_PATTERN.format(mode="pruning")),
        }

    # Load dataframes (missing files → empty dataframe)
    dfs = {}
    for mode, p in paths.items():
        if p.exists():
            df = pd.read_csv(p)
            # --- NEW: sort by the appropriate policy_* X column and overwrite the CSV ---
            try:
                x_key, _, _ = mode_conf(mode)              # e.g., "keep_all", "quant_ratio", "prune_keep"
                policy_col = f"policy_{x_key}"             # e.g., "policy_keep_all"
                if policy_col in df.columns:
                    df = df.sort_values(
                        by=policy_col,
                        key=lambda s: pd.to_numeric(s, errors="coerce"),
                        kind="mergesort",                  # stable sort
                        na_position="last",
                    )
                    df.to_csv(p, index=False)
            except Exception:
                pass  # don't fail plotting if sorting/overwrite has an issue
            # ---- Minimal, hardcoded addition for pruning figure ----
            # If we're plotting from the *perpolicy* pruning CSV, also pull margin/ecov
            # points from pruning_11055_ckpt_perplexities.csv and add them to the dataframe.
            if mode == "pruning" and p.name == "perpolicy_pruning_main_ckpt_perplexities.csv":
                extra_p = Path("pruning_11055_ckpt_perplexities.csv")
                if extra_p.exists():
                    try:
                        df_extra = pd.read_csv(extra_p)
                        cols = [c for c in [
                            "margin_ppl", "margin_prune_keep",
                            "ecov_ppl",   "ecov_prune_keep",
                        ] if c in df_extra.columns]
                        if cols:
                            df = pd.concat([df, df_extra[cols]], axis=0, ignore_index=True)
                    except Exception:
                        pass  # stay minimal: ignore any issues with the extra file
            dfs[mode] = df
        else:
            dfs[mode] = pd.DataFrame()

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

    plot_mode(axes[0], dfs["sparsity"], "sparsity")
    plot_mode(axes[1], dfs["quantization4"], "quantization4")
    plot_mode(axes[2], dfs["pruning"], "pruning")

    fig.tight_layout(pad=1.2, w_pad=1.2)
    fig.savefig(OUT_FIG, bbox_inches="tight", dpi=300)
    print(f"Saved {OUT_FIG}")

if __name__ == "__main__":
    main()
