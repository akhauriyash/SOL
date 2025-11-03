import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving to file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

# --- Load ---
# df = pd.read_csv("multiaxis_ppl.csv")
# df = pd.read_csv("multi_eff_ppl_scan_v2.csv")


df = pd.read_csv("early_multiact_ppl.csv")

# --- Build the printed table (exact pairing of metric with its own run's policy_ppl) ---
def axis_view(df, vary_col, keep_zero_cols, metric_col, suffix):
    """Return bias, metric_col (renamed), and the respective policy_ppl_<suffix>."""
    mask = np.logical_and.reduce([df[c] == 0 for c in keep_zero_cols])
    sub = df.loc[mask, [vary_col, metric_col, "policy_ppl"]].rename(columns={vary_col: "bias"})
    sub = sub.groupby("bias", as_index=False).mean(numeric_only=True)  # average duplicates per bias
    sub = sub.rename(columns={"policy_ppl": f"policy_ppl_{suffix}"})
    return sub

keep_axis  = axis_view(df, "sparsity_bias", ["prune_bias", "quant_bias"], "policy_keep_all",    "keep")
prune_axis = axis_view(df, "prune_bias",     ["sparsity_bias", "quant_bias"], "policy_prune_keep", "prune")
quant_axis = axis_view(df, "quant_bias",     ["sparsity_bias", "prune_bias"], "policy_quant_ratio","quant")

table = keep_axis.merge(prune_axis, on="bias", how="outer").merge(quant_axis, on="bias", how="outer")
table = table.sort_values("bias")

cols = [
    "bias",
    "policy_keep_all",   "policy_ppl_keep",
    "policy_prune_keep", "policy_ppl_prune",
    "policy_quant_ratio","policy_ppl_quant",
]
cols = [c for c in cols if c in table.columns]
print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# --- 3D coverage scatter across ALL runs ---
pts = df[["policy_keep_all", "policy_prune_keep", "policy_quant_ratio"]].dropna()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pts["policy_keep_all"], pts["policy_prune_keep"], pts["policy_quant_ratio"], s=12)

ax.view_init(elev=25, azim=-100)
ax.set_xlabel("policy_keep_all")
ax.set_ylabel("policy_prune_keep")
ax.set_zlabel("policy_quant_ratio")
fig.tight_layout()
plt.savefig("mutliaxis_coverage.png", dpi=300)

import numpy as np
import pandas as pd

def infer_step(vals, min_step=1e-6):
    """Infer the intended grid step as the median positive adjacent difference."""
    uniq = np.unique(np.asarray(vals, dtype=float))
    diffs = np.diff(uniq)
    diffs = diffs[diffs > min_step]
    if len(diffs) == 0:
        return None
    # Round to a “nice” step to avoid float jitter
    step = np.median(diffs)
    # Normalize tiny float noise
    mag = max(1e-12, 10 ** np.floor(np.log10(step)))
    step = round(step / mag) * mag
    return step if step > min_step else None

def expected_grid(vmin, vmax, step):
    # Build inclusive grid; guard against floating residue
    n = int(round((vmax - vmin) / step))
    grid = vmin + step * np.arange(n + 1)
    return np.round(grid, 12)

def find_missing_points(series, name):
    vals = pd.Series(series).dropna().astype(float).values
    if len(vals) == 0:
        return {
            "action": name,
            "count": 0,
            "unique_values": [],
            "step_inferred": None,
            "missing_points": [],
            "adjacent_gaps": [],
            "min": None,
            "max": None,
        }

    uniq = np.unique(np.round(vals, 12))
    vmin, vmax = float(uniq.min()), float(uniq.max())
    step = infer_step(uniq)

    # Missing points on inferred grid
    missing = []
    if step is not None and vmin < vmax:
        grid = expected_grid(vmin, vmax, step)
        missing = [float(x) for x in grid if x not in uniq]

    # Adjacent gaps (just the raw big jumps)
    diffs = np.diff(uniq)
    # Consider a “big” gap if it’s > 1.5 * inferred step (or, if no step, use 10× median diff as heuristic)
    if step is not None:
        gap_thresh = 1.5 * step
    else:
        pos_diffs = diffs[diffs > 0]
        gap_thresh = (10 * np.median(pos_diffs)) if len(pos_diffs) else 0.0

    adjacent_gaps = []
    for a, b, d in zip(uniq[:-1], uniq[1:], diffs):
        if d > max(1e-9, gap_thresh):
            adjacent_gaps.append({"from": float(a), "to": float(b), "width": float(d)})

    return {
        "action": name,
        "count": int(len(vals)),
        "unique_count": int(len(uniq)),
        "min": vmin,
        "max": vmax,
        "step_inferred": step,
        "missing_points": missing,        # Points on the inferred grid that are absent
        "adjacent_gaps": adjacent_gaps,   # Large jumps in the observed uniques
        "sample_unique": [float(x) for x in uniq[:20]],  # a quick peek
    }

actions = {
    "policy_keep_all":    df["policy_keep_all"],
    "policy_prune_keep":  df["policy_prune_keep"],
    "policy_quant_ratio": df["policy_quant_ratio"],
}

reports = [find_missing_points(series, name) for name, series in actions.items()]

# Pretty print
import json
print(json.dumps(reports, indent=2, sort_keys=False))

# --- Visualize availability gaps per action as 0–1 line plots ---
# Produces a 1x3 grid of subplots, saved to 'scan_gaps.pdf'

import numpy as np
import matplotlib.pyplot as plt

def _infer_step(vals, min_step=1e-6):
    """Infer intended grid step as median positive adjacent diff (robust to float jitter)."""
    uniq = np.unique(np.asarray(vals, dtype=float))
    diffs = np.diff(uniq)
    diffs = diffs[diffs > min_step]
    if len(diffs) == 0:
        return None
    step = float(np.median(diffs))
    # Snap to a “nice” magnitude to reduce float noise
    mag = max(1e-12, 10 ** np.floor(np.log10(step)))
    step = round(step / mag) * mag
    return step if step > min_step else None

def _expected_grid(vmin, vmax, step):
    """Inclusive grid from vmin..vmax using step, with rounding to tame float residue."""
    n = int(round((vmax - vmin) / step))
    grid = vmin + step * np.arange(n + 1)
    return np.round(grid, 12)

def _availability_vector(vals):
    """Return grid (x) and availability (0/1) for plotting."""
    if len(vals) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    uniq = np.unique(np.round(np.asarray(vals, dtype=float), 12))
    vmin, vmax = float(uniq.min()), float(uniq.max())
    # If data within [0,1], plot across [0,1] to make axes comparable
    global_min, global_max = (0.0, 1.0) if (vmin >= 0.0 and vmax <= 1.0) else (vmin, vmax)

    step = _infer_step(uniq)
    if step is None or global_min == global_max:
        # Fallback: build a grid from observed points only
        x = uniq
        y = np.ones_like(x)
        return x, y

    # Use the inferred step across the global range
    x = _expected_grid(global_min, global_max, step)

    # Mark availability: point exists if it's close to any observed value (within 0.5*step)
    present = np.zeros_like(x, dtype=int)
    i_obs = 0
    for i, xi in enumerate(x):
        # advance observed pointer while less than xi - tol
        tol = 0.5 * step + 1e-12
        while i_obs < len(uniq) and uniq[i_obs] < xi - tol:
            i_obs += 1
        if i_obs < len(uniq) and abs(uniq[i_obs] - xi) <= tol:
            present[i] = 1
    return x, present

def _plot_availability(ax, x, y, title):
    """Plot as a step line from 0–1; shows coverage at y=1, gaps at y=0."""
    # Step plot with markers to see discrete grid hits
    ax.step(x, y, where="mid")
    ax.plot(x, y, "o", ms=3)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["missing", "present"])
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_xlabel("value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

# Prepare data for each action
actions = {
    "policy_keep_all":    df["policy_keep_all"].dropna().values,
    "policy_prune_keep":  df["policy_prune_keep"].dropna().values,
    "policy_quant_ratio": df["policy_quant_ratio"].dropna().values,
}

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)

for ax, (name, vals) in zip(axes, actions.items()):
    x, y = _availability_vector(vals)
    _plot_availability(ax, x, y, name)

fig.suptitle("Scan gaps (availability per action)", y=1.02, fontsize=12)
fig.savefig("scan_gaps.pdf", dpi=300)
print("Saved scan_gaps.pdf")

# --- Minimal printout of sparsity_bias near keep_all ≈ 0.85 and 0.95 ---

def _infer_step_min(vals, min_step=1e-6):
    v = np.unique(np.asarray(vals, dtype=float))
    d = np.diff(v)
    d = d[d > min_step]
    if len(d) == 0:
        return None
    step = float(np.median(d))
    mag = max(1e-12, 10 ** np.floor(np.log10(step)))
    step = round(step / mag) * mag
    return step if step > min_step else None

targets = [0.85, 0.94]
step = _infer_step_min(df["policy_keep_all"].dropna().values)
tol = (0.5 * step if step is not None else 1e-3) + 1e-12  # half-step, or small fallback

for t in targets:
    mask = df["policy_keep_all"].sub(t).abs() <= tol
    vals = np.unique(np.round(df.loc[mask, "sparsity_bias"].dropna().astype(float), 12))
    # concise, single-line print per target
    print(f"policy_keep_all≈{t:g} -> sparsity_bias: {', '.join(map(lambda x: f'{x:g}', vals)) or '(none)'}")
