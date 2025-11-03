import os
import json
import math
from glob import glob
from statistics import mean
import matplotlib.pyplot as plt

# ---- Config ----
RECORDS_DIR = "records_full"  # change to "records" if that's your folder
OUTPUT_PATH = "downstream_comparison.pdf"

# ---- Helpers ----
def safe_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def extract_avg_accuracy(accuracy_block):
    """
    accuracy_block is expected to look like:
    {
      "macro": float|None,
      "per_task": {task: float, ...}
    }
    """
    if not isinstance(accuracy_block, dict):
        return None
    macro = safe_float(accuracy_block.get("macro"))
    if macro is not None:
        return macro
    per_task = accuracy_block.get("per_task") or {}
    vals = [safe_float(v) for v in per_task.values() if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip()!="")]
    return mean(vals) if vals else None

def load_key_metrics_json(path):
    with open(path, "r") as f:
        return json.load(f)

def detect_mode_from_filename(fname_lower):
    # Prefer exact keywords
    if "sparsity" in fname_lower:
        return "sparsity"
    if "prune" in fname_lower or "pruning" in fname_lower:
        return "prune"
    if "quant" in fname_lower or "quantization" in fname_lower or "quantise" in fname_lower:
        return "quant"
    return None

# ---- Collect data ----
modes_cfg = {
    "sparsity": {"x_key": "token_keep_avg", "label": "Token keep (avg)"},
    "prune":    {"x_key": "prune_keep_avg", "label": "Prune keep (avg)"},
    "quant":    {"x_key": "quant_ratio_avg","label": "Quant ratio (avg)"},
}
points = {k: [] for k in modes_cfg.keys()}      # mode -> list of (x, y, filename)
baselines = {k: None for k in modes_cfg.keys()} # mode -> dense baseline macro acc

pattern = os.path.join(RECORDS_DIR, "key_metrics_*.json")
for path in glob(pattern):
    fname = os.path.basename(path)
    mode = detect_mode_from_filename(fname.lower())
    if mode not in modes_cfg:
        continue

    data = load_key_metrics_json(path)

    # Policy point
    pol = data.get("policy", {})
    x = safe_float(pol.get(modes_cfg[mode]["x_key"]))
    y = extract_avg_accuracy(pol.get("accuracy"))
    if y is not None and y > 1.0:
        y /= 100.0
    if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
        points[mode].append((x, y, fname))

    # Dense baseline (take first encountered if multiple exist)
    if baselines[mode] is None and "dense_baseline" in data:
        db = data.get("dense_baseline") or {}
        db_acc = extract_avg_accuracy(db.get("accuracy"))
        # db_acc = 0.49
        if db_acc is not None and db_acc > 1.0:
            db_acc /= 100.0
        if db_acc is not None and not math.isnan(db_acc):
            baselines[mode] = db_acc

# ---- Plot ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

mode_order = ["sparsity", "prune", "quant"]
for ax, mode in zip(axes, mode_order):
    xs, ys, labels = zip(*sorted(points[mode], key=lambda t: t[0])) if points[mode] else ([], [], [])
    if xs:
        ax.scatter(xs, ys)
        # Optional: connect points to show trend
        ax.plot(xs, ys, linewidth=1)

    # Dense baseline line (if available)
    if baselines[mode] is not None:
        ax.axhline(baselines[mode], color="black", linewidth=0.8, linestyle="-", label="Dense baseline")

    # Labels / title
    ax.set_title(mode.capitalize())
    ax.set_xlabel(modes_cfg[mode]["label"])
    ax.set_ylabel("Average accuracy")

    # If we drew a baseline, add legend (avoid empty legends)
    if baselines[mode] is not None:
        ax.legend(frameon=False)

# Save
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
print(f"Saved → {OUTPUT_PATH}")
