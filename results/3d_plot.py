import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

CSV_PATH = "multi_eff_ppl_scan.csv"

OUT_ENVELOPE = "edc_envelope.png"
OUT_PLANES   = "delta_planes.pdf"
OUT_DELTA_EDC = "delta_vs_edc.pdf"

W_ATTN = 0.5
W_MLP  = 0.5
BITS_MAX = 16.0

N_BINS = 40

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 22,
})

AXIS_NAME = {
    "sparsity": "Token Sparsity",
    "prune": "LLM Pruning",
    "quant": "Quantization (bits)",
}

COL = {
    "policy": {
        "sparsity": "policy_keep_all",
        "prune": "policy_prune_keep",
        "quant": "policy_quant_ratio",
        "ppl": "policy_ppl",
    },
    "fixed": {
        "sparsity": "fixed_keep_all",
        "prune": "fixed_prune_keep",
        "quant": "fixed_quant_ratio",
        "ppl": "fixed_ppl",
    },
}

def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def compute_edc_frame(df, method: str) -> np.ndarray:
    """A simple, normalized effective compute index per row for a method."""
    keep_all = df[COL[method]["sparsity"]].to_numpy(dtype=float)            # [0..1]
    prune_k  = df[COL[method]["prune"]].to_numpy(dtype=float)               # [0..1]
    bits     = df[COL[method]["quant"]].to_numpy(dtype=float)               # usually {4,8,16}
    edc = W_ATTN * keep_all + W_MLP * (prune_k * (bits / BITS_MAX))
    if np.nanmin(edc) != np.nanmax(edc):
        edc = (edc - np.nanmin(edc)) / (np.nanmax(edc) - np.nanmin(edc))
    else:
        edc[:] = 0.5
    return edc

def binned_frontier(x, y, nbins=40):
    """Given scattered (x=compute, y=ppl), return bin centers and min y per bin (envelope)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return np.array([]), np.array([])
    bins = np.linspace(np.min(x), np.max(x), nbins+1)
    idx = np.digitize(x, bins) - 1
    xs, ys = [], []
    for b in range(nbins):
        sel = (idx == b)
        if np.any(sel):
            xs.append(0.5*(bins[b] + bins[b+1]))
            ys.append(np.nanmin(y[sel]))
    return np.array(xs), np.array(ys)

def common_range(a, b):
    lo = max(np.nanmin(a), np.nanmin(b))
    hi = min(np.nanmax(a), np.nanmax(b))
    if lo >= hi:
        return None
    return lo, hi

def interp_on(x_src, y_src, x_new):
    if len(x_src) < 2:
        return np.full_like(x_new, np.nan, dtype=float)
    order = np.argsort(x_src)
    return np.interp(x_new, x_src[order], y_src[order], left=np.nan, right=np.nan)

def slice_leftout_max(df, method: str, leftout_key: str, quantile_fallback=0.9, min_rows=15):
    """Keep rows where the left-out dimension is at its max (or top quantile fallback)."""
    col = COL[method][leftout_key]
    vals = df[col].to_numpy(dtype=float)
    if not np.isfinite(vals).any():
        return df.copy()
    maxv = np.nanmax(vals)
    mask = np.isclose(vals, maxv, atol=1e-9)
    out = df.loc[mask].copy()
    if len(out) < min_rows:
        thr = np.nanquantile(vals, quantile_fallback)
        out = df.loc[vals >= thr].copy()
    return out

def tricontour_delta(ax, x, y, delta, title):
    """Δ-perplexity tricontourf plane (Fixed - Policy): blue = Policy better."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.asarray(delta, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(d))
    x, y, d = x[m], y[m], d[m]
    if len(x) == 0:
        ax.set_title(title + " (no data)")
        return
    xy = np.vstack([x, y]).T
    _, idx = np.unique(xy, axis=0, return_index=True)
    x, y, d = x[idx], y[idx], d[idx]

    if len(x) >= 3:
        tri = mtri.Triangulation(x, y)
        h = ax.tricontourf(tri, d, levels=15, cmap="RdBu_r")
        c = ax.tricontour(tri, d, levels=[0.0], linewidths=1.8, colors="k")
        ax.clabel(c, fmt="Δ=0", inline=True, fontsize=9)
        cb = plt.colorbar(h, ax=ax)
        cb.set_label("Fixed – Policy perplexity (↑ better)")
    else:
        ax.scatter(x, y, c=d, cmap="RdBu_r")
        ax.set_title(title + " (scatter)")

def label_axes(ax, x_key, y_key):
    ax.set_xlabel(AXIS_NAME[x_key])
    ax.set_ylabel(AXIS_NAME[y_key])

df = pd.read_csv(CSV_PATH)
to_numeric(df, [
    "policy_ppl","fixed_ppl",
    "policy_keep_all","policy_prune_keep","policy_quant_ratio",
    "fixed_keep_all","fixed_prune_keep","fixed_quant_ratio",
    "sparsity_bias","prune_bias","quant_bias",
])

df["delta_ppl"] = df["fixed_ppl"] - df["policy_ppl"]

# -----------------------------
# 1) Efficiency–Quality Envelope
# -----------------------------
edc_policy = compute_edc_frame(df, "policy")
edc_fixed  = compute_edc_frame(df, "fixed")

xs_p, ys_p = binned_frontier(edc_policy, df["policy_ppl"].to_numpy(dtype=float), nbins=N_BINS)
xs_f, ys_f = binned_frontier(edc_fixed,  df["fixed_ppl"].to_numpy(dtype=float), nbins=N_BINS)

fig1, ax1 = plt.subplots(figsize=(7,5))
ax1.plot(xs_p, ys_p, linewidth=2.2, label="SOL")
ax1.plot(xs_f, ys_f, linewidth=2.2, label="Fixed")

cr = common_range(xs_p, xs_f)
if cr is not None:
    lo, hi = cr
    x_fill = np.linspace(lo, hi, 300)
    yp = interp_on(xs_p, ys_p, x_fill)
    yf = interp_on(xs_f, ys_f, x_fill)
    m = ~(np.isnan(yp) | np.isnan(yf))
    ax1.fill_between(x_fill[m], yp[m], yf[m],
                     where=yf[m] > yp[m],
                     alpha=0.25, step="mid", label="")

ax1.set_xlabel("Effective Decode Compute")
ax1.set_ylabel("Perplexity ↓")
ax1.set_title("Quality–Efficiency Envelope")
ax1.grid(True, alpha=0.3)
ax1.legend(frameon=False)
fig1.tight_layout()
fig1.savefig(OUT_ENVELOPE, bbox_inches="tight", dpi=500)

# -----------------------------
# 2) Δ-Perplexity vs Compute (scatter + binned median)
# -----------------------------
edc_mid = 0.5*(edc_policy + edc_fixed)
deltas = df["delta_ppl"].to_numpy(dtype=float)
mask = ~(np.isnan(edc_mid) | np.isnan(deltas))

fig2, ax2 = plt.subplots(figsize=(7,5))
ax2.scatter(edc_mid[mask], deltas[mask], s=14, alpha=0.25)
ax2.axhline(0.0, color="k", linewidth=1)

bins = np.linspace(np.nanmin(edc_mid[mask]), np.nanmax(edc_mid[mask]), N_BINS+1)
idx = np.digitize(edc_mid[mask], bins) - 1
xb, mb = [], []
for b in range(N_BINS):
    sel = (idx == b)
    if np.any(sel):
        xb.append(0.5*(bins[b]+bins[b+1]))
        mb.append(np.nanmedian(deltas[mask][sel]))
ax2.plot(xb, mb, linewidth=2.2)
win_rate = 100.0*np.mean(deltas[mask] > 0.0)
ax2.set_title(f"Δ-Perplexity vs Compute")
ax2.set_xlabel("Effective Decode Compute")
ax2.set_ylabel("Δ perplexity (↑)")
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(OUT_DELTA_EDC, bbox_inches="tight")

print(f"Saved:\n  {OUT_ENVELOPE}\n  {OUT_PLANES}\n  {OUT_DELTA_EDC}")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string

def _alpha_series(n):
    """A, B, ..., Z, AA, AB, ... (Excel-style)"""
    labels = []
    i = 0
    while len(labels) < n:
        s = ""
        j = i
        while True:
            s = chr(ord('A') + (j % 26)) + s
            j = j // 26 - 1
            if j < 0:
                break
        labels.append(s)
        i += 1
    return labels

def _parse_pipe_floats(x):
    if not isinstance(x, str) or x.strip() == "":
        return None
    try:
        return [float(t) for t in x.split("|")]
    except Exception:
        return None

def _compute_edc_if_needed(df):
    w_attn = globals().get("W_ATTN", 0.5)
    w_mlp  = globals().get("W_MLP", 0.5)
    bits_max = globals().get("BITS_MAX", 16.0)

    keep_all = pd.to_numeric(df.get("policy_keep_all", pd.Series(np.nan)), errors="coerce").to_numpy(float)
    prune_k  = pd.to_numeric(df.get("policy_prune_keep", pd.Series(np.nan)), errors="coerce").to_numpy(float)
    bits     = pd.to_numeric(df.get("policy_quant_ratio", pd.Series(np.nan)), errors="coerce").to_numpy(float)

    edc = w_attn * keep_all + w_mlp * (prune_k * (bits / bits_max))
    m = np.isfinite(edc)
    if m.any():
        edc_min, edc_max = np.nanmin(edc[m]), np.nanmax(edc[m])
        if edc_max > edc_min:
            edc = (edc - edc_min) / (edc_max - edc_min)
        else:
            edc[:] = 0.5
    else:
        edc[:] = np.nan
    return edc

if "df" not in globals():
    df = pd.read_csv("multi_eff_ppl_scan.csv")

if "edc_policy" not in globals():
    edc_policy = _compute_edc_if_needed(df)

rows = []
for i, row in df.iterrows():
    levels = _parse_pipe_floats(row.get("sparsity_levels_kappa_order", ""))
    probs  = _parse_pipe_floats(row.get("policy_action_probs_kappa_order", ""))
    edc    = float(edc_policy[i]) if i < len(edc_policy) else np.nan

    rows.append((edc, np.array(probs, dtype=float)))

if not rows:
    print("No valid policy action-probability rows found; skipping heatmap.")
else:
    A = len(rows[0][1])
    N_BINS = 256
    edc_vals = np.array([r[0] for r in rows], dtype=float)
    edc_min, edc_max = float(np.nanmin(edc_vals)), float(np.nanmax(edc_vals))
    bins = np.linspace(edc_min, edc_max, N_BINS + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    sum_probs = np.zeros((A, N_BINS), dtype=float)
    counts    = np.zeros(N_BINS, dtype=int)

    for edc, pvec in rows:
        # guard
        if not np.isfinite(edc) or not np.all(np.isfinite(pvec)):
            continue
        b = np.searchsorted(bins, edc, side="right") - 1
        if 0 <= b < N_BINS:
            sum_probs[:, b] += pvec
            counts[b] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_probs = sum_probs / np.maximum(counts, 1)
    valid = counts > 0
    if not np.any(valid):
        print("No non-empty EDC bins; skipping heatmap.")
        raise SystemExit(0)
    avg_probs = avg_probs[:, valid]
    M = np.ma.masked_invalid(avg_probs)
    bin_centers = bin_centers[valid]
    action_letters = _alpha_series(A)

    ACTION_NAME_OVERRIDE = {} 
    yticklabels = [ACTION_NAME_OVERRIDE.get(letter, letter) for letter in action_letters]

    plt.figure(figsize=(7, 5))

    im = plt.imshow(
        M,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        extent=[0, M.shape[1], -0.5, A - 0.5],
    )
    keep_fracs = [0.2, 1.0]
    prune_keep = [1.0, 0.8, 0.6]
    q_bits     = [4, 8, 16]

    actions = [(k, p, q) for k in keep_fracs for p in prune_keep for q in q_bits]

    w_attn   = globals().get("W_ATTN", 0.5)
    w_mlp    = globals().get("W_MLP", 0.5)
    bits_max = float(globals().get("BITS_MAX", 16.0))

    def compute_density(k, p, q):
        return w_attn * k + w_mlp * (p * (q / bits_max))
    densities = np.array([compute_density(k, p, q) for (k, p, q) in actions], dtype=float)
    order = np.argsort(densities)
    avg_probs = avg_probs[order, :]
    actions   = [actions[i] for i in order]

    def fmt_label(k, p, q):
        tk = int(round(k * 100))
        tp = int(round(p * 100))
        tq = int(q)
        return f"t{tk:>3} | p{tp:<3} | q{tq:<2}"

    yticklabels = [fmt_label(k, p, q) for (k, p, q) in actions]

    fig, ax = plt.subplots(figsize=(7, 5))
    M = np.ma.masked_invalid(avg_probs)
    im = ax.imshow(
        M,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        extent=[0, M.shape[1], -0.5, A - 0.5],
    )
    ax.set_xlabel("Effective Decode Compute →")
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontfamily="monospace", ha="right", fontsize=12)
    fig.subplots_adjust(left=0.36)
    box = ax.get_position()
    ax.set_position([0.50, box.y0, 0.45, box.height])
    fig.subplots_adjust(right=0.98)
    plt.savefig("policy_action_histogram.pdf", bbox_inches="tight")
    print("Saved policy_action_histogram.pdf")
    print(f"Win-rate @ EDC<0.9: {100*np.mean((df['policy_ppl'] < df['fixed_ppl'])[edc_policy < 0.9]):.1f}%")













