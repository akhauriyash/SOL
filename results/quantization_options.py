#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

FILES = {
    "quantization4": "quantization4_ckpt_perplexities.csv",
    "quantization5": "quantization5_ckpt_perplexities.csv",
    "quantization":  "quantization_ckpt_perplexities.csv",
}

labelmap = {
    "quantization4": "4 Levels",
    "quantization5": "5 Levels",
    "quantization":  "12 Levels",
}

X_COLS = ["policy_quant_ratio", "quant_ratio", "policy_bits", "bits"]
Y_COL = "policy_ppl"
FIXED_COL = "fixed_ppl"
OUT = "quantization_options.pdf"

def load_series(path: Path):
    """
    Returns (x_sorted, y_sorted, pct_better_sorted) where:
      y = policy_ppl
      pct_better = (fixed_ppl - policy_ppl) / fixed_ppl * 100
    Applies finite filter and 2x-min(y) outlier filter.
    If FIXED_COL is missing, pct_better is None.
    """
    if not path.exists():
        return None, None, None
    df = pd.read_csv(path)

    # pick X
    x = None
    for c in X_COLS:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            break
    if x is None or Y_COL not in df.columns:
        return None, None, None

    y = pd.to_numeric(df[Y_COL], errors="coerce").to_numpy(float)

    # start mask with finite x,y
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return None, None, None

    # 2x-min filter on y (policy_ppl)
    min_y = np.nanmin(y[m])
    # m &= (y <= 2.0 * min_y)
    m &= (y <= 12)
    if not np.any(m):
        return None, None, None

    # optional fixed baseline
    pct_better = None
    if FIXED_COL in df.columns:
        fixed = pd.to_numeric(df[FIXED_COL], errors="coerce").to_numpy(float)
        m &= np.isfinite(fixed)
        if np.any(m):
            # recompute min filter after adding fixed finite constraint
            min_y = np.nanmin(y[m])
            m &= (y <= 2.0 * min_y)
            if np.any(m):
                pct_better = (fixed[m] - y[m]) / fixed[m] * 100.0
            else:
                pct_better = None
        else:
            pct_better = None

    # finalize sort
    if not np.any(m):
        return None, None, None
    xs, ys = x[m], y[m]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if pct_better is not None:
        pct_better = pct_better[order]

    return xs, ys, pct_better

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4))

for label, fname in FILES.items():
    x, y, pct = load_series(Path(fname))
    x = 16 * x  # convert quant_ratio to bits
    if x is None:
        continue
    # Left: policy perplexity vs bits
    ax_left.plot(x, y, marker="o", linewidth=2, label=labelmap[label])
    # Right: % better vs fixed_ppl (if available)
    if pct is not None:
        ax_right.plot(x, pct, marker="o", linewidth=2, label=labelmap[label])

# Left subplot styling
ax_left.set_xlabel("Quantization (Bits)")
ax_left.set_ylabel("policy_ppl")
ax_left.grid(True, alpha=0.3)
ax_left.legend(frameon=True, framealpha=0.9)

# Right subplot styling
ax_right.set_xlabel("Quantization (Bits)")
ax_right.set_ylabel("% better vs fixed_ppl")
ax_right.grid(True, alpha=0.3)
# ax_left.set_xscale("log", base=2)
# ax_right.set_xscale("log", base=2)

ax_right.set_xlim(right=10)
ax_left.set_xlim(right=10)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=300)