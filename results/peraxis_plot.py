#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# ---------- Config ----------
CSV_PATH = "early_multiact_ppl.csv"   # change if needed
OUT_FIG  = "delta_vs_edc_by_bias.pdf"

# Effective decode compute weights (same logic as your script)
W_ATTN   = 0.5
W_MLP    = 0.5
BITS_MAX = 16.0

# Plot/look settings (larger text + slightly smaller figure)
FIGSIZE = (10, 3.4)   # smaller width so text appears larger on screen
SCATTER_MAX = 400
MEDIAN_MIN_WINDOW = 11
YLIM = (-3, 18)      # % relative improvement range (tweak as needed)
RANDOM_SEED = 7
ZERO_EPS = 1e-12

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 18,         # base
    "axes.titlesize": 20,    # panel titles
    "axes.labelsize": 16,    # axis labels
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ----------------------------

rng = np.random.default_rng(RANDOM_SEED)

def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def compute_edc_frame(df, method: str) -> np.ndarray:
    keep_all = pd.to_numeric(df.get(f"{method}_keep_all"), errors="coerce").to_numpy(float)
    prune_k  = pd.to_numeric(df.get(f"{method}_prune_keep"), errors="coerce").to_numpy(float)
    bits     = pd.to_numeric(df.get(f"{method}_quant_ratio"), errors="coerce").to_numpy(float)
    edc = W_ATTN * keep_all + W_MLP * (prune_k * (bits / BITS_MAX))
    m = np.isfinite(edc)
    if m.any():
        lo, hi = np.nanmin(edc[m]), np.nanmax(edc[m])
        if hi > lo:
            edc = (edc - lo) / (hi - lo)
        else:
            edc[:] = 0.5
    else:
        edc[:] = np.nan
    return edc

def running_median(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) == 0:
        return np.array([]), np.array([])
    order = np.argsort(x)
    x, y = x[order], y[order]
    k = max(MEDIAN_MIN_WINDOW, max(3, len(x)//40))  # ~2.5% of data
    half = k // 2
    xm, ym = [], []
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        xm.append(x[i])
        ym.append(np.nanmedian(y[lo:hi]))
    return np.array(xm), np.array(ym)

def downsample(x, y, cap=SCATTER_MAX, rng=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    if n <= cap:
        return x, y
    idx = rng.choice(n, size=cap, replace=False)
    return x[idx], y[idx]

def pick_sparsity_col(df):
    if "sparsity_bias" in df.columns:
        return "sparsity_bias"
    if "sparisty_bias" in df.columns:  # common typo
        return "sparisty_bias"
    df["sparsity_bias"] = np.nan
    return "sparsity_bias"

def mask_only_one_bias(df, target: str) -> np.ndarray:
    s_col = pick_sparsity_col(df)
    q = pd.to_numeric(df.get("quant_bias"), errors="coerce").to_numpy(float)
    p = pd.to_numeric(df.get("prune_bias"), errors="coerce").to_numpy(float)
    s = pd.to_numeric(df.get(s_col), errors="coerce").to_numpy(float)

    is_q = np.abs(q) > ZERO_EPS
    is_p = np.abs(p) > ZERO_EPS
    is_s = np.abs(s) > ZERO_EPS

    if target == "quant":
        return is_q & (~is_p) & (~is_s)
    if target == "prune":
        return is_p & (~is_q) & (~is_s)
    if target == "sparsity":
        return is_s & (~is_q) & (~is_p)
    raise ValueError("target must be one of {'quant','prune','sparsity'}")

def panel(ax, title, xvals, yvals, xlabel, idx):
    xs, ys = downsample(xvals, yvals, cap=SCATTER_MAX, rng=rng)
    # if xlabel has "Quantization" scale by 16 to make bits
    if "Quantization" in xlabel:
        xs = xs * 16
    ax.scatter(xs, ys, s=20, alpha=0.28)        # bigger points
    ax.axhline(0.0, color="k", linewidth=1.2)   # thicker zero line
    xm, ym = running_median(xvals, yvals)
    if "Quantization" in xlabel:
        xm = xm * 16
    if len(xm):
        ax.plot(xm, ym, linewidth=2.0)          # thicker median line
    # ax.set_title(title)
    ax.set_xlabel(xlabel)
    if idx == 0:
        # ax.set_ylabel("Improvement @ ΔPPL (%) ↑")
        ax.set_ylabel("PPL Improvement (%)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(*YLIM)
    ax.tick_params(axis="both", which="major", length=5, width=1)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    df = pd.read_csv(path)

    to_numeric(df, [
        "policy_ppl","fixed_ppl",
        "policy_keep_all","policy_prune_keep","policy_quant_ratio",
        "fixed_keep_all","fixed_prune_keep","fixed_quant_ratio",
        "sparsity_bias","sparisty_bias","prune_bias","quant_bias",
    ])
    pol = pd.to_numeric(df["policy_ppl"], errors="coerce")
    # Optional outlier filter (keep commented unless you want it)
    # df = df.loc[pol < 2.0 * np.nanmin(pol)].copy()

    # ---- Y: Relative improvement vs fixed (%): positive = policy better ----
    fixed = pd.to_numeric(df["fixed_ppl"], errors="coerce")
    policy = pd.to_numeric(df["policy_ppl"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["rel_impr_pct"] = 100.0 * (fixed - policy) / fixed

    # ---- X: Per-panel raw axes rather than EDC ----
    # Quant panel → Quantization (bits)
    x_quant_bits = pd.to_numeric(df.get("policy_quant_ratio"), errors="coerce")
    # Prune panel → Prune Keep (%)  (kept fraction → %)
    x_prune_keep_pct = 100.0 * pd.to_numeric(df.get("policy_prune_keep"), errors="coerce")
    # Sparsity panel → Token Sparsity (%) (keep_all → % kept tokens)
    x_sparsity_keep_pct = 100.0 * pd.to_numeric(df.get("policy_keep_all"), errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

    # key, title, x-values series, x-axis label
    specs = (
        ("quant",    "Quantization",      x_quant_bits,        "Quantization (bits)"),
        ("prune",    "Activation Pruning",             x_prune_keep_pct,    "Pruning (keep %)"),
        ("sparsity", "Token Sparsity",    x_sparsity_keep_pct, "Token Sparsity (keep %)"),
    )
    idx = 0
    for ax, (key, title, xseries, xlabel) in zip(axes, specs):
        m = mask_only_one_bias(df, key)
        x = xseries[m].to_numpy(float)
        y = df.loc[m, "rel_impr_pct"].to_numpy(float)
        if (~np.isnan(x)).sum() == 0 or (~np.isnan(y)).sum() == 0:
            ax.set_title(title + " (no data)")
            ax.axis("off")
            continue
        panel(ax, title, x, y, xlabel, idx)
        idx += 1

    # No global subtitle; tighten spacing so labels are legible
    fig.tight_layout(pad=1.2, w_pad=1.2)
    fig.savefig(OUT_FIG, bbox_inches="tight", dpi=300)
    print(f"Saved {OUT_FIG}")

if __name__ == "__main__":
    main()
