import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def safe_get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


ROOT = Path(".")
scan_dirs = sorted([p for p in ROOT.glob("highset*") if p.is_dir()])

rows = []  # one row per (file, variant)

for d in scan_dirs:
    for jf in sorted(d.rglob("key_metrics_*.json")):
        try:
            obj = json.loads(jf.read_text())
        except Exception as e:
            print(f"Skipping unreadable JSON: {jf} ({e})")
            continue

        for variant in ("policy", "fixed_baseline"):
            v = obj.get(variant)
            if not isinstance(v, dict):
                continue

            token_keep = v.get("token_keep_avg")
            prune_keep = v.get("prune_keep_avg")
            prune_rate = v.get("prune_rate_avg")

            if any(x is None for x in (token_keep, prune_keep, prune_rate)):
                print(f"Skipping (missing keep/rate fields): {jf} [{variant}]")
                continue

            x_keep = (float(token_keep) + float(prune_keep) + float(prune_rate)) / 3.0

            per_task = safe_get(v, ["accuracy", "per_task"], default={})
            if not isinstance(per_task, dict):
                per_task = {}

            # Also plot macro accuracy as an extra subplot by treating it like a task
            macro = safe_get(v, ["accuracy", "macro"])
            if macro is not None:
                per_task = dict(per_task)  # don't mutate shared refs
                per_task["macro"] = float(macro)
            rows.append(
                {
                    "dir": d.name,
                    "file": jf.name,
                    "variant": variant,
                    "x": x_keep,
                    "per_task": per_task,  # dict task_name -> score (or None)
                }
            )

if not rows:
    raise RuntimeError(
        "No data found. Expected folders matching lmeval_scan_* containing key_metrics_*.json."
    )

# Discover all tasks present anywhere
tasks = sorted({t for r in rows for t in (r["per_task"] or {}).keys()})
if not tasks:
    raise RuntimeError("Found data rows but no tasks under accuracy.per_task.")

def collect_for_task(variant, task_name):
    xs, ys = [], []
    for r in rows:
        if r["variant"] != variant:
            continue
        y = (r["per_task"] or {}).get(task_name)
        if y is None:
            continue
        xs.append(r["x"])
        ys.append(float(y))
    return xs, ys

# Layout: choose a reasonable grid
n = len(tasks)
ncols = min(3, n)  # tweak if you prefer 4, etc.
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4.2 * ncols, 3.2 * nrows),
    sharex=True,
    sharey=True,
)
if nrows == 1 and ncols == 1:
    axes = [axes]
else:
    axes = axes.ravel()

# Plot per task
for i, task in enumerate(tasks):
    ax = axes[i]
    for variant in ("policy", "fixed_baseline"):
        xs, ys = collect_for_task(variant, task)
        if xs:
            sorted_pairs = sorted(zip(xs, ys))
            xs_sorted, ys_sorted = zip(*sorted_pairs)
            ax.plot(xs_sorted, ys_sorted, marker='o', label=variant, alpha=0.85)
    ax.set_title(f"{task}")
    ax.grid(True, alpha=0.3)

    if i % ncols == 0:
        ax.set_ylabel("accuracy")
    if i >= (nrows - 1) * ncols:
        ax.set_xlabel("net keep rate (avg of token_keep, prune_keep, prune_rate)")

# Hide any unused axes
for j in range(n, len(axes)):
    axes[j].set_visible(False)

# One legend for the whole figure (unique labels)
label_to_handle = {}
for ax in axes[:n]:
    h, l = ax.get_legend_handles_labels()
    for hh, ll in zip(h, l):
        label_to_handle.setdefault(ll, hh)

if label_to_handle:
    fig.legend(
        list(label_to_handle.values()),
        list(label_to_handle.keys()),
        loc="upper center",
        ncol=len(label_to_handle),
        frameon=False,
    )

plt.tight_layout(rect=[0, 0, 1, 0.94])
out = Path("downstreamacc.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out.resolve()}")
def summarize_task_scores(variant, task_name):
    vals = []
    for r in rows:
        if r["variant"] != variant:
            continue
        y = (r["per_task"] or {}).get(task_name)
        if y is None:
            continue
        vals.append(float(y))
    if not vals:
        return None
    return sum(vals) / len(vals)

def latex_escape(s: str) -> str:
    # minimal escaping for common LaTeX table issues
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("_", "\\_")
         .replace("%", "\\%")
         .replace("&", "\\&")
         .replace("#", "\\#")
    )

def fmt(x, nd=4):
    if x is None:
        return "--"
    return f"{x:.{nd}f}"

variants = [
    ("fixed_baseline", "Fixed"),
    ("policy", "Policy"),
]

# Build the table
col_tasks = tasks[:]  # already sorted
header = "Method & " + " & ".join(latex_escape(t) for t in col_tasks) + r" \\"
midrule = r"\hline"

lines = []
lines.append(r"\begin{tabular}{" + "l" + "c" * len(col_tasks) + "}")
lines.append(midrule)
lines.append(header)
lines.append(midrule)

for var_key, var_label in variants:
    row_cells = [var_label]
    for t in col_tasks:
        m = summarize_task_scores(var_key, t)
        row_cells.append(fmt(m))
    lines.append(" & ".join(row_cells) + r" \\")

lines.append(midrule)
lines.append(r"\end{tabular}")

latex_table = "\n".join(lines)
print(latex_table)

# --- Single macro-only plot (3b) ---
# (Append this to the end of the file)

import matplotlib as mpl

# Make EVERY font size 18 (literally everything)
mpl.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 18,
})

def collect_macro_points(variant_key: str):
    xs, ys = [], []
    for r in rows:
        if r["variant"] != variant_key:
            continue
        y = (r["per_task"] or {}).get("macro")
        if y is None:
            continue
        xs.append(float(r["x"]))
        ys.append(float(y) * 100.0)  # convert to %
    if not xs:
        return [], []
    pairs = sorted(zip(xs, ys))
    xs_s, ys_s = zip(*pairs)
    return list(xs_s), list(ys_s)

fig, ax = plt.subplots(figsize=(5, 5))

# Policy
xs, ys = collect_macro_points("policy")
if xs:
    ax.plot(xs, ys, marker="o", alpha=0.85, label="Policy")

# Fixed baseline
xs, ys = collect_macro_points("fixed_baseline")
if xs:
    ax.plot(xs, ys, marker="o", alpha=0.85, label="Fixed")

ax.set_xlabel("Net Keep-Rates")
ax.set_ylabel("Avg. Accuracy (%)")
ax.grid(True, alpha=0.3)

# No title; legend in figure
ax.legend(frameon=False)

out_3b = Path("downstream_3b.pdf")
fig.tight_layout()
fig.savefig(out_3b, bbox_inches="tight")
print(f"Saved: {out_3b.resolve()}")
