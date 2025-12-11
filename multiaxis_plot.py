import pandas as pd
import matplotlib.pyplot as plt
# fontsize 18 with rcParam
plt.rcParams.update({'font.size': 18})


def parse_pipe_separated_floats(s):
    """Parse a '|' separated string of floats into a list; handle empty/NaN."""
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s:
        return []
    return [float(x) for x in s.split("|") if x != ""]


def extract_points(df):
    """
    From a dataframe with the logged columns, build:
        - policy points (x_policy, y_policy)
        - fixed  points (x_fixed,  y_fixed)
        - random points (x_rand,   y_rand)
    """
    x_policy, y_policy = [], []
    x_fixed,  y_fixed  = [], []
    x_rand,   y_rand   = [], []

    for _, row in df.iterrows():
        # ---- Policy ----
        x_p = (
            float(row["policy_keep_effective_actual"])
            + float(row["policy_prune_keep_actual"])
            + float(row["policy_quant_ratio_actual"])
        )
        y_p = float(row["policy_ppl"])
        x_policy.append(x_p)
        y_policy.append(y_p)

        # ---- Fixed baseline ----
        x_f = (
            float(row["fixed_keep_all"])
            + float(row["fixed_prune_keep"])
            + float(row["fixed_quant_ratio"])
        )
        y_f = float(row["fixed_ppl"])
        x_fixed.append(x_f)
        y_fixed.append(y_f)

        # ---- Random trials ----
        ppl_list   = parse_pipe_separated_floats(row.get("rand_ppl_trials", ""))
        keep_list  = parse_pipe_separated_floats(row.get("rand_keep_eff_trials", ""))
        prune_list = parse_pipe_separated_floats(row.get("rand_prune_keep_trials", ""))
        quant_list = parse_pipe_separated_floats(row.get("rand_quant_ratio_trials", ""))

        n = min(len(ppl_list), len(keep_list), len(prune_list), len(quant_list))
        for i in range(n):
            x_r = keep_list[i] + prune_list[i] + quant_list[i]
            y_r = ppl_list[i]
            x_rand.append(x_r)
            y_rand.append(y_r)

    return x_policy, y_policy, x_fixed, y_fixed, x_rand, y_rand


# ---- New helpers for axis-wise plots ----

def compute_maxima(df):
    """
    Compute per-axis maxima for policy, fixed, and random trials.
    Axes: 'keep' (token sparsity), 'prune', 'quant'.
    """
    # Policy & fixed: straight from columns
    policy_max = {
        "keep":  float(df["policy_keep_effective_actual"].max()),
        "prune": float(df["policy_prune_keep_actual"].max()),
        "quant": float(df["policy_quant_ratio_actual"].max()),
    }
    fixed_max = {
        "keep":  float(df["fixed_keep_all"].max()),
        "prune": float(df["fixed_prune_keep"].max()),
        "quant": float(df["fixed_quant_ratio"].max()),
    }

    # Random: need to scan through all trials
    rand_max = {"keep": None, "prune": None, "quant": None}

    for _, row in df.iterrows():
        keep_list  = parse_pipe_separated_floats(row.get("rand_keep_eff_trials", ""))
        prune_list = parse_pipe_separated_floats(row.get("rand_prune_keep_trials", ""))
        quant_list = parse_pipe_separated_floats(row.get("rand_quant_ratio_trials", ""))

        n = min(len(keep_list), len(prune_list), len(quant_list))
        for i in range(n):
            k = keep_list[i]
            p = prune_list[i]
            q = quant_list[i]

            if rand_max["keep"] is None or k > rand_max["keep"]:
                rand_max["keep"] = k
            if rand_max["prune"] is None or p > rand_max["prune"]:
                rand_max["prune"] = p
            if rand_max["quant"] is None or q > rand_max["quant"]:
                rand_max["quant"] = q

    return {"policy": policy_max, "fixed": fixed_max, "rand": rand_max}

def compute_pareto_front(xs, ys, return_mask=False):
    """
    2D Pareto front for (x, y) where both x and y are minimized.
    Returns:
        - front_x, front_y (sorted by x)
        - optionally: a boolean mask marking points that lie on the front
    """
    # sort by x
    pts = sorted(enumerate(zip(xs, ys)), key=lambda t: t[1][0])
    front_x, front_y = [], []
    mask = [False] * len(xs)
    best_y = float("inf")

    for idx, (x, y) in pts:
        if y < best_y:
            front_x.append(x)
            front_y.append(y)
            mask[idx] = True
            best_y = y

    if return_mask:
        return front_x, front_y, mask
    return front_x, front_y


def extract_axis_points_1d(df, axis_key, maxima):
    """
    For a given axis_key in {"keep", "prune", "quant"}, build
    1D scatter points (x vs ppl) for policy, fixed, random, while
    holding the *other two* axes near their maxima (>= 0.9 * max).

    Returns: (x_policy, y_policy, x_fixed, y_fixed, x_rand, y_rand)
    """
    assert axis_key in {"keep", "prune", "quant"}

    axis_cols_policy = {
        "keep":  "policy_keep_effective_actual",
        "prune": "policy_prune_keep_actual",
        "quant": "policy_quant_ratio_actual",
    }
    axis_cols_fixed = {
        "keep":  "fixed_keep_all",
        "prune": "fixed_prune_keep",
        "quant": "fixed_quant_ratio",
    }

    other_axes = [a for a in ("keep", "prune", "quant") if a != axis_key]

    x_pol, y_pol = [], []
    x_fix, y_fix = [], []
    x_rnd, y_rnd = [], []

    # Thresholds for "near max" per source
    policy_max = maxima["policy"]
    fixed_max  = maxima["fixed"]
    rand_max   = maxima["rand"]

    # Helper to check if a value is within [0.9 * max_val, max_val]
    def near_max(v, max_val):
        if max_val is None:
            return True  # no info -> don't filter
        # low = 0.9 * max_val
        low = 0.9 * max_val
        return (v >= low) and (v <= max_val)

    # ---- Policy & fixed points ----
    for _, row in df.iterrows():
        # Policy
        v_target = float(row[axis_cols_policy[axis_key]])
        v_other1 = float(row[axis_cols_policy[other_axes[0]]])
        v_other2 = float(row[axis_cols_policy[other_axes[1]]])

        if near_max(v_other1, policy_max[other_axes[0]]) and near_max(
            v_other2, policy_max[other_axes[1]]
        ):
            x_pol.append(v_target)
            y_pol.append(float(row["policy_ppl"]))

        # Fixed
        v_target_f = float(row[axis_cols_fixed[axis_key]])
        v_other1_f = float(row[axis_cols_fixed[other_axes[0]]])
        v_other2_f = float(row[axis_cols_fixed[other_axes[1]]])

        if near_max(v_other1_f, fixed_max[other_axes[0]]) and near_max(
            v_other2_f, fixed_max[other_axes[1]]
        ):
            x_fix.append(v_target_f)
            y_fix.append(float(row["fixed_ppl"]))

        # Random trials for this row
        ppl_list   = parse_pipe_separated_floats(row.get("rand_ppl_trials", ""))
        keep_list  = parse_pipe_separated_floats(row.get("rand_keep_eff_trials", ""))
        prune_list = parse_pipe_separated_floats(row.get("rand_prune_keep_trials", ""))
        quant_list = parse_pipe_separated_floats(row.get("rand_quant_ratio_trials", ""))

        n = min(len(ppl_list), len(keep_list), len(prune_list), len(quant_list))
        for i in range(n):
            k = keep_list[i]
            p = prune_list[i]
            q = quant_list[i]

            if axis_key == "keep":
                target = k
                other1 = p
                other2 = q
            elif axis_key == "prune":
                target = p
                other1 = k
                other2 = q
            else:  # "quant"
                target = q
                other1 = k
                other2 = p

            if near_max(other1, rand_max[other_axes[0]]) and near_max(
                other2, rand_max[other_axes[1]]
            ):
                x_rnd.append(target)
                y_rnd.append(ppl_list[i])

    return x_pol, y_pol, x_fix, y_fix, x_rnd, y_rnd


def main():
    csv_files = [
        # ("allaxis_v1.csv", "All-axis v1"),
        ("allaxis_v2.csv", "All-axis Binary"),
        ("allaxis_v3.csv", "All-axis Ternary"),
        # ("allaxis_v4.csv", "All-axis v4"),
    ]

    # ---- Load data & build original 1x2 plot (unchanged behaviour) ----
    results = []
    all_x, all_y = [], []
    datasets = []  # keep dfs around for the new 2x3 plot

    for path, title in csv_files:
        df = pd.read_csv(path)
        datasets.append((title, df))

        pts = extract_points(df)
        results.append((title, *pts))

        x_policy, y_policy, x_fixed, y_fixed, x_rand, y_rand = pts
        all_x.extend(x_policy)
        all_x.extend(x_fixed)
        all_x.extend(x_rand)
        all_y.extend(y_policy)
        all_y.extend(y_fixed)
        all_y.extend(y_rand)

    # Define fixed colors for each category using matplotlib's tab colors
    COLORS = {
        "Fixed": "tab:orange",
        "Random": "tab:red",
        "Policy": "tab:purple",
    }

    # 1x2 figure as before
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
    for ax, (title,
            x_policy, y_policy,
            x_fixed,  y_fixed,
            x_rand,   y_rand) in zip(axes, results):

        x_fixed_scaled  = [x / 3.0 for x in x_fixed]
        x_rand_scaled   = [x / 3.0 for x in x_rand]
        x_policy_scaled = [x / 3.0 for x in x_policy]

        # ---- Fixed ----
        if x_fixed_scaled:
            px_f, py_f, mask_f = compute_pareto_front(x_fixed_scaled, y_fixed, return_mask=True)

            x_fixed_pareto   = [x for x, m in zip(x_fixed_scaled, mask_f) if m]
            y_fixed_pareto   = [y for y, m in zip(y_fixed,       mask_f) if m]
            x_fixed_nonpareto = [x for x, m in zip(x_fixed_scaled, mask_f) if not m]
            y_fixed_nonpareto = [y for y, m in zip(y_fixed,        mask_f) if not m]

            # non-Pareto (faded)
            ax.scatter(
                x_fixed_nonpareto, y_fixed_nonpareto,
                alpha=0.4, marker="s", color=COLORS["Fixed"], label="_nolegend_"
            )
            # Pareto (full opacity, appears in legend)
            ax.scatter(
                x_fixed_pareto, y_fixed_pareto,
                alpha=1.0, marker="s", color=COLORS["Fixed"], label="Fixed"
            )

            # Pareto line (as before)
            if len(px_f) > 1:
                ax.plot(px_f, py_f, linewidth=1.6, linestyle="-",
                        label="_nolegend_", zorder=1000, color=COLORS["Fixed"])

        # ---- Random ----
        if x_rand_scaled:
            px_r, py_r, mask_r = compute_pareto_front(x_rand_scaled, y_rand, return_mask=True)
            x_rand_pareto   = [x for x, m in zip(x_rand_scaled, mask_r) if m]
            y_rand_pareto   = [y for y, m in zip(y_rand,       mask_r) if m]
            x_rand_nonpareto = [x for x, m in zip(x_rand_scaled, mask_r) if not m]
            y_rand_nonpareto = [y for y, m in zip(y_rand,        mask_r) if not m]

            ax.scatter(
                x_rand_nonpareto, y_rand_nonpareto,
                alpha=0.1, marker=".", color=COLORS["Random"], label="_nolegend_"
            )
            ax.scatter(
                x_rand_pareto, y_rand_pareto,
                alpha=1.0, marker=".", color=COLORS["Random"], label="Random"
            )
            if len(px_r) > 1:
                ax.plot(px_r, py_r, linewidth=1.6, linestyle="-",
                        label="_nolegend_", zorder=1000, color=COLORS["Random"])

        # ---- Policy ----
        if x_policy_scaled:
            px_p, py_p, mask_p = compute_pareto_front(x_policy_scaled, y_policy, return_mask=True)
            x_pol_pareto   = [x for x, m in zip(x_policy_scaled, mask_p) if m]
            y_pol_pareto   = [y for y, m in zip(y_policy,       mask_p) if m]
            x_pol_nonpareto = [x for x, m in zip(x_policy_scaled, mask_p) if not m]
            y_pol_nonpareto = [y for y, m in zip(y_policy,        mask_p) if not m]

            ax.scatter(
                x_pol_nonpareto, y_pol_nonpareto,
                alpha=0.4, marker="o", color=COLORS["Policy"], label="_nolegend_"
            )
            ax.scatter(
                x_pol_pareto, y_pol_pareto,
                alpha=1.0, marker="o", color=COLORS["Policy"], label="Policy"
            )

            if len(px_p) > 1:
                ax.plot(px_p, py_p, linewidth=1.6, linestyle="-",
                        label="_nolegend_", zorder=1000, color=COLORS["Policy"])

        ax.set_title(title)
        ax.set_xlabel("Net Keep Rates")

    axes[0].set_ylabel("Perplexity")
    axes[1].set_xlim(0.4, 0.6)
    axes[1].set_ylabel("Perplexity")

    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_pad = 0.02 * (x_max - x_min) if x_max > x_min else 0.1
        y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
        # for ax in axes:
        #     ax.set_xlim(x_min - x_pad, x_max + x_pad)
        #     ax.set_ylim(y_min - y_pad, y_max + y_pad)

    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("multiaxis_eff.pdf")
    print("Saved figure to multiaxis_eff.pdf")

    # ---- New 2x3 figure: per-axis sweeps, holding the other two near max ----
    axis_keys   = ["keep", "prune", "quant"]
    axis_titles = {
        "keep":  "Token sparsity",
        "prune": "Prune keep",
        "quant": "Quantization",
    }
    axis_xlabels = {
        "keep":  "Token sparsity rate",
        "prune": "Prune keep ratio",
        "quant": "Quantization ratio",
    }

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8), sharey=True)

    for row_idx, (title, df) in enumerate(datasets):
        maxima = compute_maxima(df)

        for col_idx, axis_key in enumerate(axis_keys):
            ax = axes2[row_idx, col_idx]

            x_pol, y_pol, x_fix, y_fix, x_rnd, y_rnd = extract_axis_points_1d(
                df, axis_key, maxima
            )
            # Fixed
            if len(x_fix) > 0:
                ax.scatter(x_fix, y_fix, label="Fixed", alpha=0.5, marker="s", color=COLORS["Fixed"])

            # # Random
            if len(x_rnd) > 0:
                ax.scatter(x_rnd, y_rnd, label="Random", alpha=0.5, marker=".", color=COLORS["Random"])

            # Policy
            if len(x_pol) > 0:
                ax.scatter(x_pol, y_pol, label="Policy", alpha=0.8, marker="o", color=COLORS["Policy"])

            # Pareto fronts (thin lines, no legend entry)
            for xs, ys, label in (
                (x_fix,  y_fix, "Fixed"),
                # (x_rnd,  y_rnd, "Random"),
                (x_pol,  y_pol, "Policy"),
            ):
                if len(xs) > 1:
                    px, py = compute_pareto_front(xs, ys)
                    if len(px) > 1:
                        ax.plot(px, py, linewidth=1.6, linestyle="-", label="_nolegend_", zorder=1000, color=COLORS[label])

            ax.set_xlabel(axis_xlabels[axis_key])
            ax.set_title(f"{title} – {axis_titles[axis_key]}")

            # Only left column gets a y-label
            if col_idx == 0:
                ax.set_ylabel("Perplexity")

            # Put legend once per row (on the rightmost subplot)
            # if col_idx == 2:
            ax.legend()

    fig2.tight_layout()
    fig2.savefig("multiaxis_axiswise_eff.pdf")
    print("Saved figure to multiaxis_axiswise_eff.pdf")


if __name__ == "__main__":
    main()
