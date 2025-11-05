
import os
import json
import math
from glob import glob
from statistics import mean
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# ---- Config ----
RECORDS_DIR = "records"          # directory containing key_metrics_*.json with policy + fixed_baseline
OUTPUT_PATH = "downstream_comparison.pdf"
SKIP_TASKS = []                  # e.g., ['winogrande', 'openbookqa', 'race']
RUNAVG_WINDOW = 3               # running-average window size (number of points, odd is best)
 
# ---- Helpers ----
def safe_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def running_average(xs, ys, k):
    """
    Centered running average with *partial windows* at the edges so the
    smoothed curve spans from the first to the last point.
    Returns (xs_ma, ys_ma) with len == len(xs).
    """
    if not xs or not ys or k is None or k <= 1:
        return list(xs), list(ys)
    n = len(xs)
    half = k // 2
    xs_ma, ys_ma = list(xs), []
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)  # exclusive
        window_y = ys[start:end]
        ys_ma.append(mean(window_y))
    return xs_ma, ys_ma


def extract_avg_accuracy(accuracy_block, skip_tasks=None):
    """
    accuracy_block:
    {
      "macro": float|None,
      "per_task": {task: float, ...}
    }
    Returns average accuracy in [0,1] or None.
    """
    if not isinstance(accuracy_block, dict):
        return None
    skip = {t.lower() for t in (skip_tasks or [])}
    per_task = accuracy_block.get("per_task") or {}
    # If skipping tasks, recompute from per_task.
    if skip:
        vals = []
        for k, v in per_task.items():
            if k.lower() in skip:
                continue
            fv = safe_float(v)
            if fv is not None:
                vals.append(fv)
        if vals:
            return mean(vals)
        return safe_float(accuracy_block.get("macro"))
    # Prefer macro if provided
    macro = safe_float(accuracy_block.get("macro"))
    if macro is not None:
        return macro
    vals = [safe_float(v) for v in per_task.values() if safe_float(v) is not None]
    return mean(vals) if vals else None

def load_key_metrics_json(path):
    with open(path, "r") as f:
        return json.load(f)

def detect_mode_from_filename(fname_lower):
    if "sparsity" in fname_lower:
        return "sparsity"
    if "prune" in fname_lower or "pruning" in fname_lower:
        return "prune"
    if "quant" in fname_lower or "quantization" in fname_lower or "quantise" in fname_lower:
        return "quant"
    return None

# ---- Collect data ----
modes_cfg = {
    "sparsity": {"x_key": "token_keep_avg", "label": "Token Keep-Rate"},
    "prune":    {"x_key": "prune_keep_avg", "label": "Prune Keep-Rate"},
    "quant":    {"x_key": "quant_ratio_avg","label": "Quantization (Bits)"},
}

policy_points = {k: [] for k in modes_cfg.keys()}     # mode -> list of (x, y, filename)
fixed_points   = {k: [] for k in modes_cfg.keys()}    # mode -> list of (x, y, filename)
baselines      = {k: None for k in modes_cfg.keys()}  # mode -> baseline y (rightmost policy)

# Track policy full dict per file so we can print all axes at baseline point
policy_by_file = {}  # filename -> policy dict

pattern = os.path.join(RECORDS_DIR, "key_metrics_*.json")
for path in glob(pattern):
    fname = os.path.basename(path)
    mode = detect_mode_from_filename(fname.lower())
    if mode not in modes_cfg:
        continue

    data = load_key_metrics_json(path)

    # --- Policy point ---
    pol = data.get("policy") or {}
    policy_by_file[fname] = pol
    x = safe_float(pol.get(modes_cfg[mode]["x_key"]))
    # For quant, show bits on X
    if mode == "quant" and x is not None:
        x *= 16.0
    # x = float(path.split("_")[-1].replace(".json", ""))
    print("path", path, "x", x)
    y = extract_avg_accuracy(pol.get("accuracy"), skip_tasks=SKIP_TASKS)
    if y is not None and y > 1.0:
        y /= 100.0
    # Filter: only show token_keep >= 0.6 for sparsity plot (as before)
    if mode == "sparsity" and x is not None and x < 0.6:
        x = None
    if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
        policy_points[mode].append((x, y * 100.0, fname))  # store Y as %

    # --- Fixed-from-policy point (same file) ---
    fx = data.get("fixed_baseline") or {}
    if fx:
        fx_x = safe_float(fx.get(modes_cfg[mode]["x_key"]))
        if mode == "quant" and fx_x is not None:
            fx_x *= 16.0
        fx_y = extract_avg_accuracy(fx.get("accuracy"), skip_tasks=SKIP_TASKS)
        if fx_y is not None and fx_y > 1.0:
            fx_y /= 100.0
        if mode == "sparsity" and fx_x is not None and fx_x < 0.6:
            fx_x = None
        # fx_x = float(path.split("_")[-1].replace(".json", ""))
        if fx_x is not None and fx_y is not None and not (math.isnan(fx_x) or math.isnan(fx_y)):
            fixed_points[mode].append((fx_x, fx_y * 100.0, fname))

# ---- Use rightmost policy point as the baseline (per mode) ----
for mode in modes_cfg:
    if policy_points[mode]:
        rx, ry, rfname = max(policy_points[mode], key=lambda t: t[0])  # rightmost by keep/ratio
        baselines[mode] = ry
        pp = policy_by_file.get(rfname, {}) or {}
        tk = pp.get("token_keep_avg")
        pk = pp.get("prune_keep_avg")
        qr = pp.get("quant_ratio_avg")
        tk_s = f"{tk:.6f}" if tk is not None else "n/a"
        pk_s = f"{pk:.6f}" if pk is not None else "n/a"
        qr_s = f"{qr:.6f}" if qr is not None else "n/a"
        print(
            f"[baseline] {mode}: rightmost {modes_cfg[mode]['label']}={rx:.6f} -> "
            f"baseline acc={ry:.6f} | at point: token_keep={tk_s}, "
            f"prune_keep={pk_s}, quant_ratio={qr_s} (from {rfname})"
        )

# ---- Plot ----
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True, sharey=False)
mode_order = ["sparsity", "prune", "quant"]

# Global Y based on sparsity panel (policy + fixed + baseline)
_ys_spars = [y for _, y, _ in policy_points["sparsity"]] + [y for _, y, _ in fixed_points["sparsity"]]
if baselines["sparsity"] is not None:
    _ys_spars.append(baselines["sparsity"])
y_min = y_max = None
if _ys_spars:
    y_min, y_max = min(_ys_spars), max(_ys_spars)

for ax, mode in zip(axes, mode_order):
    xs, ys, labels = zip(*sorted(policy_points[mode], key=lambda t: t[0])) if policy_points[mode] else ([], [], [])
    fxs, fys, flabels = zip(*sorted(fixed_points[mode], key=lambda t: t[0])) if fixed_points[mode] else ([], [], [])

    if xs:
        # scatter points (raw, potentially noisy)
        ax.scatter(xs, ys, label="Policy", zorder=3, alpha=0.6)
        # smooth running-average curve
        xs_ma, ys_ma = running_average(xs, ys, RUNAVG_WINDOW)
        if xs_ma:
            ax.plot(xs_ma, ys_ma, linewidth=1.5, label="_nolabel_", zorder=4)
    if fxs:
        # show fixed as scatter + smoothed line too
        ax.scatter(fxs, fys, marker="x", label="Fixed", zorder=2)
        fxs_ma, fys_ma = running_average(fxs, fys, RUNAVG_WINDOW)
        if fxs_ma:
            ax.plot(fxs_ma, fys_ma, linestyle="--", linewidth=1, label="_nolabel_", zorder=2)

    # Baseline = rightmost policy point
    if baselines[mode] is not None:
        ax.axhline(baselines[mode], linewidth=0.8, linestyle="--", label="Dense", color="black")

    # Labels / title
    ax.set_title(mode.capitalize())
    ax.set_xlabel(modes_cfg[mode]["label"])
    if mode == "sparsity":
        ax.set_ylabel("Avg. Accuracy (%)")

    if (xs or fxs) or (baselines[mode] is not None):
        pass  # single, figure-level legend added after the loop

    # Apply shared Y-lims from sparsity
    # if y_min is not None:
    #     ax.set_ylim(y_min * 0.9, y_max * 1.03)

    if mode == "quant":
        ax.set_xlim(6, 16)  # show bits scale
    ax.grid(True, linewidth=0.5, alpha=0.4, zorder=0)

# ---- Single, figure-level legend (top-center, one row) ----
# Gather unique handles/labels across all axes (skip placeholder labels)
all_handles, all_labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    for hh, ll in zip(h, l):
        if ll == "_nolabel_":
            continue
        if ll not in all_labels:
            all_handles.append(hh)
            all_labels.append(ll)
if all_handles:
    fig.legend(
        all_handles, all_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),  # above titles
        ncol=len(all_labels),        # one row
        frameon=True,
    )
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
print(f"Saved → {OUTPUT_PATH}")

# import os
# import json
# import math
# from glob import glob
# from statistics import mean
# import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "font.size": 16,
#     "axes.titlesize": 16,
#     "axes.labelsize": 16,
#     "xtick.labelsize": 16,
#     "ytick.labelsize": 16,
#     "legend.fontsize": 12,
# })

# # ---- Config ----
# RECORDS_DIR = "records_l"          # directory containing key_metrics_*.json with policy + fixed_baseline
# OUTPUT_PATH = "downstream_comparison.pdf"
# SKIP_TASKS = []                  # e.g., ['winogrande', 'openbookqa', 'race']

# # ---- Helpers ----
# def safe_float(x, default=None):
#     try:
#         return float(x)
#     except (TypeError, ValueError):
#         return default

# def extract_avg_accuracy(accuracy_block, skip_tasks=None):
#     """
#     accuracy_block:
#     {
#       "macro": float|None,
#       "per_task": {task: float, ...}
#     }
#     Returns average accuracy in [0,1] or None.
#     """
#     if not isinstance(accuracy_block, dict):
#         return None
#     skip = {t.lower() for t in (skip_tasks or [])}
#     per_task = accuracy_block.get("per_task") or {}
#     # If skipping tasks, recompute from per_task.
#     if skip:
#         vals = []
#         for k, v in per_task.items():
#             if k.lower() in skip:
#                 continue
#             fv = safe_float(v)
#             if fv is not None:
#                 vals.append(fv)
#         if vals:
#             return mean(vals)
#         return safe_float(accuracy_block.get("macro"))
#     # Prefer macro if provided
#     macro = safe_float(accuracy_block.get("macro"))
#     if macro is not None:
#         return macro
#     vals = [safe_float(v) for v in per_task.values() if safe_float(v) is not None]
#     return mean(vals) if vals else None

# def load_key_metrics_json(path):
#     with open(path, "r") as f:
#         return json.load(f)

# def detect_mode_from_filename(fname_lower):
#     if "sparsity" in fname_lower:
#         return "sparsity"
#     if "prune" in fname_lower or "pruning" in fname_lower:
#         return "prune"
#     if "quant" in fname_lower or "quantization" in fname_lower or "quantise" in fname_lower:
#         return "quant"
#     return None

# # ---- Collect data ----
# modes_cfg = {
#     "sparsity": {"x_key": "token_keep_avg", "label": "Token Keep-Rate"},
#     "prune":    {"x_key": "prune_keep_avg", "label": "Prune Keep-Rate"},
#     "quant":    {"x_key": "quant_ratio_avg","label": "Quantization (Bits)"},
# }

# policy_points = {k: [] for k in modes_cfg.keys()}     # mode -> list of (x, y, filename)
# fixed_points   = {k: [] for k in modes_cfg.keys()}    # mode -> list of (x, y, filename)
# baselines      = {k: None for k in modes_cfg.keys()}  # mode -> baseline y (rightmost policy)

# # Track policy full dict per file so we can print all axes at baseline point
# policy_by_file = {}  # filename -> policy dict

# pattern = os.path.join(RECORDS_DIR, "key_metrics_*.json")
# for path in glob(pattern):
#     fname = os.path.basename(path)
#     mode = detect_mode_from_filename(fname.lower())
#     if mode not in modes_cfg:
#         continue

#     data = load_key_metrics_json(path)

#     # --- Policy point ---
#     pol = data.get("policy") or {}
#     policy_by_file[fname] = pol
#     x = safe_float(pol.get(modes_cfg[mode]["x_key"]))
#     # For quant, show bits on X
#     if mode == "quant" and x is not None:
#         x *= 16.0
#     # x = float(path.split("_")[-1].replace(".json", ""))
#     print("path", path, "x", x)
#     y = extract_avg_accuracy(pol.get("accuracy"), skip_tasks=SKIP_TASKS)
#     if y is not None and y > 1.0:
#         y /= 100.0
#     # Filter: only show token_keep >= 0.6 for sparsity plot (as before)
#     if mode == "sparsity" and x is not None and x < 0.6:
#         x = None
#     if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
#         policy_points[mode].append((x, y * 100.0, fname))  # store Y as %

#     # --- Fixed-from-policy point (same file) ---
#     fx = data.get("fixed_baseline") or {}
#     if fx:
#         fx_x = safe_float(fx.get(modes_cfg[mode]["x_key"]))
#         if mode == "quant" and fx_x is not None:
#             fx_x *= 16.0
#         fx_y = extract_avg_accuracy(fx.get("accuracy"), skip_tasks=SKIP_TASKS)
#         if fx_y is not None and fx_y > 1.0:
#             fx_y /= 100.0
#         if mode == "sparsity" and fx_x is not None and fx_x < 0.6:
#             fx_x = None
#         # fx_x = float(path.split("_")[-1].replace(".json", ""))
#         if fx_x is not None and fx_y is not None and not (math.isnan(fx_x) or math.isnan(fx_y)):
#             fixed_points[mode].append((fx_x, fx_y * 100.0, fname))

# # ---- Use rightmost policy point as the baseline (per mode) ----
# for mode in modes_cfg:
#     if policy_points[mode]:
#         rx, ry, rfname = max(policy_points[mode], key=lambda t: t[0])  # rightmost by keep/ratio
#         baselines[mode] = ry
#         pp = policy_by_file.get(rfname, {}) or {}
#         tk = pp.get("token_keep_avg")
#         pk = pp.get("prune_keep_avg")
#         qr = pp.get("quant_ratio_avg")
#         tk_s = f"{tk:.6f}" if tk is not None else "n/a"
#         pk_s = f"{pk:.6f}" if pk is not None else "n/a"
#         qr_s = f"{qr:.6f}" if qr is not None else "n/a"
#         print(
#             f"[baseline] {mode}: rightmost {modes_cfg[mode]['label']}={rx:.6f} -> "
#             f"baseline acc={ry:.6f} | at point: token_keep={tk_s}, "
#             f"prune_keep={pk_s}, quant_ratio={qr_s} (from {rfname})"
#         )

# # ---- Plot ----
# fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True, sharey=False)
# mode_order = ["sparsity", "prune", "quant"]

# # Global Y based on sparsity panel (policy + fixed + baseline)
# _ys_spars = [y for _, y, _ in policy_points["sparsity"]] + [y for _, y, _ in fixed_points["sparsity"]]
# if baselines["sparsity"] is not None:
#     _ys_spars.append(baselines["sparsity"])
# y_min = y_max = None
# if _ys_spars:
#     y_min, y_max = min(_ys_spars), max(_ys_spars)

# for ax, mode in zip(axes, mode_order):
#     xs, ys, labels = zip(*sorted(policy_points[mode], key=lambda t: t[0])) if policy_points[mode] else ([], [], [])
#     fxs, fys, flabels = zip(*sorted(fixed_points[mode], key=lambda t: t[0])) if fixed_points[mode] else ([], [], [])

#     if xs:
#         ax.scatter(xs, ys, label="Policy")
#         ax.plot(xs, ys, linewidth=1)

#     if fxs:
#         ax.plot(fxs, fys, marker="x", linestyle="--", label="Fixed")

#     # Baseline = rightmost policy point
#     if baselines[mode] is not None:
#         ax.axhline(baselines[mode], linewidth=0.8, linestyle="--", label="Dense", color="black")

#     # Labels / title
#     ax.set_title(mode.capitalize())
#     ax.set_xlabel(modes_cfg[mode]["label"])
#     if mode == "sparsity":
#         ax.set_ylabel("Avg. Accuracy (%)")

#     # Legend only if something is drawn
#     if (xs or fxs) or (baselines[mode] is not None):
#         ax.legend(frameon=False)

#     # Apply shared Y-lims from sparsity
#     # if y_min is not None:
#     #     ax.set_ylim(y_min * 0.9, y_max * 1.03)

#     if mode == "quant":
#         ax.set_xlim(6, 16)  # show bits scale

# fig.savefig(OUTPUT_PATH, bbox_inches="tight")
# print(f"Saved → {OUTPUT_PATH}")





























# import os
# import json
# import math
# from glob import glob
# from statistics import mean
# import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "font.size": 16,
#     "axes.titlesize": 16,
#     "axes.labelsize": 16,
#     "xtick.labelsize": 16,
#     "ytick.labelsize": 16,
#     "legend.fontsize": 12,
# })


# # ---- Config ----
# RECORDS_DIR = "records_full"  # change to "records" if that's your folder
# OUTPUT_PATH = "downstream_comparison.pdf"
# FIXED_RECORDS_DIR = "records_fixed"  # directory for fixed-baseline runs
# # FIXED_RECORDS_DIR = "records_fixed_wrong"  # directory for fixed-baseline runs
# # SKIP_TASKS = ['winogrande', 'openbookqa', 'race']
# SKIP_TASKS=[]

# # ---- Helpers ----
# def safe_float(x, default=None):
#     try:
#         return float(x)
#     except (TypeError, ValueError):
#         return default

# def extract_avg_accuracy(accuracy_block, skip_tasks=None):
#     """
#     accuracy_block is expected to look like:
#     {
#       "macro": float|None,
#       "per_task": {task: float, ...}
#     }
#     """
#     if not isinstance(accuracy_block, dict):
#         return None
#     skip = {t.lower() for t in (skip_tasks or [])}
#     per_task = accuracy_block.get("per_task") or {}
#     # If skipping any tasks, recompute from per_task (prefer this over macro).
#     if skip:
#         vals = []
#         for k, v in per_task.items():
#             if k.lower() in skip:
#                 continue
#             fv = safe_float(v)
#             if fv is not None:
#                 vals.append(fv)
#         if vals:
#             return mean(vals)
#         # Fall back to macro if nothing left to average.
#         return safe_float(accuracy_block.get("macro"))
#     # No skips → keep existing behavior (prefer macro if provided).
#     macro = safe_float(accuracy_block.get("macro"))
#     if macro is not None:
#         return macro
#     vals = []
#     for v in per_task.values():
#         fv = safe_float(v)
#         if fv is not None:
#             vals.append(fv)
#     return mean(vals) if vals else None

# def load_key_metrics_json(path):
#     with open(path, "r") as f:
#         return json.load(f)

# def detect_mode_from_filename(fname_lower):
#     # Prefer exact keywords
#     if "sparsity" in fname_lower:
#         return "sparsity"
#     if "prune" in fname_lower or "pruning" in fname_lower:
#         return "prune"
#     if "quant" in fname_lower or "quantization" in fname_lower or "quantise" in fname_lower:
#         return "quant"
#     return None

# # ---- Collect data ----
# modes_cfg = {
#     "sparsity": {"x_key": "token_keep_avg", "label": "Token Keep-Rate"},
#     "prune":    {"x_key": "prune_keep_avg", "label": "Prune Keep-Rate"},
#     "quant":    {"x_key": "quant_ratio_avg","label": "Quantization (Bits)"},
# }
# points = {k: [] for k in modes_cfg.keys()}      # mode -> list of (x, y, filename)
# baselines = {k: None for k in modes_cfg.keys()} # mode -> dense baseline macro acc
# fixed_points = {k: [] for k in modes_cfg.keys()}  # mode -> list of (x, y, filename)

# policy_by_file = {}  # filename -> policy dict (to print all axes at baseline point)
# pattern = os.path.join(RECORDS_DIR, "key_metrics_*.json")
# for path in glob(pattern):
#     fname = os.path.basename(path)
#     mode = detect_mode_from_filename(fname.lower())
#     if mode not in modes_cfg:
#         continue

#     data = load_key_metrics_json(path)

#     # Policy point
#     pol = data.get("policy", {})
#     policy_by_file[fname] = pol
#     x = safe_float(pol.get(modes_cfg[mode]["x_key"]))
#     if mode == "quant" and x is not None:
#         x *= 16.0
#     y = extract_avg_accuracy(pol.get("accuracy"), skip_tasks=SKIP_TASKS)
#     if y is not None and y > 1.0:
#         y /= 100.0
#     # Limit sparsity X (token_keep_avg) to <= 0.6
#     if mode == "sparsity" and x is not None and x < 0.6:
#         x = None  # skip this point
#     y *= 100.0
#     if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
#         points[mode].append((x, y, fname))

#     # Dense baseline (take first encountered if multiple exist)
#     if baselines[mode] is None and "dense_baseline" in data:
#         db = data.get("dense_baseline") or {}
#         db_acc = extract_avg_accuracy(db.get("accuracy"), skip_tasks=SKIP_TASKS)
#         # db_acc = 0.49
#         if db_acc is not None and db_acc > 1.0:
#             db_acc /= 100.0
#         if db_acc is not None and not math.isnan(db_acc):
#             baselines[mode] = db_acc

# # ---- Collect fixed-baseline points ----
# pattern_fixed = os.path.join(FIXED_RECORDS_DIR, "key_metrics_*.json")
# for path in glob(pattern_fixed):
#     fname = os.path.basename(path)
#     mode = detect_mode_from_filename(fname.lower())
#     if mode not in modes_cfg:
#         continue

#     data = load_key_metrics_json(path)
#     fx = data.get("fixed_baseline") or {}

#     x = safe_float(fx.get(modes_cfg[mode]["x_key"]))
#     if mode == "quant" and x is not None:
#         x *= 16.0  # show bits on X for quant
#     y = extract_avg_accuracy(fx.get("accuracy"), skip_tasks=SKIP_TASKS)
#     if y is not None and y > 1.0:
#         y /= 100.0
#     # keep the same X filter used for policy points
#     if mode == "sparsity" and x is not None and x < 0.6:
#         x = None
#     if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
#         fixed_points[mode].append((x, y * 100.0, fname))

# # ---- Use rightmost policy point as the baseline (override any dense_baseline) ----
# for mode in modes_cfg:
#     if points[mode]:
#         rx, ry, rfname = max(points[mode], key=lambda t: t[0])  # rightmost by keep ratio
#         baselines[mode] = ry  # override
#         # print all axes' keep/quant ratios at that point for visibility
#         pp = policy_by_file.get(rfname, {}) or {}
#         tk = pp.get("token_keep_avg")
#         pk = pp.get("prune_keep_avg")
#         qr = pp.get("quant_ratio_avg")
#         tk_s = f"{tk:.6f}" if tk is not None else "n/a"
#         pk_s = f"{pk:.6f}" if pk is not None else "n/a"
#         qr_s = f"{qr:.6f}" if qr is not None else "n/a"
#         print(
#             f"[baseline] {mode}: rightmost {modes_cfg[mode]['label']}={rx:.6f} -> "
#             f"baseline acc={ry:.6f} | at point: token_keep={tk_s}, "
#             f"prune_keep={pk_s}, quant_ratio={qr_s} (from {rfname})"
#         )
# # ---- Plot ----
# fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True, sharey=True)

# mode_order = ["sparsity", "prune", "quant"]

# # Enforce global Y-range based on 'sparsity' only (include fixed points there too)
# _ys_spars = [y for _, y, _ in points["sparsity"]] + [y for _, y, _ in fixed_points["sparsity"]]
# if baselines["sparsity"] is not None:
#     _ys_spars.append(baselines["sparsity"])
# y_min = y_max = None
# if _ys_spars:
#     y_min, y_max = min(_ys_spars), max(_ys_spars)
# for ax, mode in zip(axes, mode_order):
#     xs, ys, labels = zip(*sorted(points[mode], key=lambda t: t[0])) if points[mode] else ([], [], [])
#     fxs, fys, flabels = zip(*sorted(fixed_points[mode], key=lambda t: t[0])) if fixed_points[mode] else ([], [], [])
#     if xs:
#         ax.scatter(xs, ys)
#         # Optional: connect points to show trend
#         ax.plot(xs, ys, linewidth=1)

#     if fxs:
#         # plot fixed baselines from records_fixed
#         ax.plot(fxs, fys, marker="x", label="Fixed baseline", linestyle="--")
#     # Dense baseline line (if available)
#     if baselines[mode] is not None or fixed_points[mode]:
#         ax.axhline(baselines[mode], color="black", linewidth=0.8, linestyle="--", label="Dense baseline")

#     # Labels / title
#     ax.set_title(mode.capitalize())
#     ax.set_xlabel(modes_cfg[mode]["label"])
#     if mode == "sparsity":
#         ax.set_ylabel("Avg. Accuracy (%)")

#     # If we drew a baseline, add legend (avoid empty legends)
#     if baselines[mode] is not None:
#         ax.legend(frameon=False)
#     # Apply Y-lims from sparsity to all axes (shared Y)
#     if y_min is not None:
#         ax.set_ylim(y_min*0.98, y_max*1.03)
    
#     if mode == "quant":
#         ax.set_xlim(6, 16)

# # Save
# fig.savefig(OUTPUT_PATH, bbox_inches="tight")
# print(f"Saved → {OUTPUT_PATH}")
