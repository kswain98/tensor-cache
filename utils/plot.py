"""
NLL vs context-length plot CLI.

Foreground dataset is plotted with solid full-opacity lines; the optional
background dataset is overlaid as faded lines. Curves are smoothed with
monotone cubic interpolation (PCHIP) in log-x space.

Inputs are CSVs with at least these columns:
  - mode             one of {full_kv, window_kv, streaming_llm, infini, tc}
  - context_length   integer (alias: eval_tokens)
  - nll              float    (alias: nll_all)
Both bench.py long_context output and evaluate.py output match these aliases.

Usage:
    python utils/plot.py --foreground=results/owt_nll.csv \\
        [--background=results/shakespeare_nll.csv] \\
        --trained_ctx=1024 --out=results/fig_owt_nll_vs_ctx
"""

import os

# Visual style
COLORS = {
    "full_kv":       "#3454D1",
    "window_kv":     "#9FE3E1",
    "streaming_llm": "#3CB1A8",
    "infini":        "#5A87D8",
    "tc":            "#E8A33D",
}
LINESTYLES = {
    "full_kv":       "-",
    "window_kv":     "--",
    "streaming_llm": "-",
    "infini":        ":",
    "tc":            "-",
}
LINEWIDTHS = {
    "full_kv":       2.0,
    "window_kv":     2.0,
    "streaming_llm": 2.0,
    "infini":        2.4,
    "tc":            2.6,
}
LABELS = {
    "full_kv":       "Full KV",
    "window_kv":     "Window KV",
    "streaming_llm": "StreamingLLM",
    "infini":        "InfiniAttention",
    "tc":            "Tensor Cache",
}
ORDER = ["full_kv", "window_kv", "streaming_llm", "infini", "tc"]

CTX_ALIASES = ("context_length", "eval_tokens", "ctx", "length")
NLL_ALIASES = ("nll", "nll_all")
MODE_ALIASES = ("mode", "method", "kv_mode")


def _pick(row, aliases):
    for k in aliases:
        if k in row and row[k] != "":
            return row[k]
    raise KeyError(f"None of {aliases} found in CSV row: {list(row)}")


def load_curves(path):
    """Return {mode: (sorted_ctx_list, nll_list)} from a CSV."""
    import csv
    from collections import defaultdict

    by_mode_ctx = defaultdict(dict)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = str(_pick(row, MODE_ALIASES)).strip()
            if mode not in ORDER:
                continue
            ctx = int(float(_pick(row, CTX_ALIASES)))
            nll = float(_pick(row, NLL_ALIASES))
            if nll != nll or nll in (float("inf"), float("-inf")):
                continue
            by_mode_ctx[mode][ctx] = nll
    curves = {}
    for mode, ctx_to_nll in by_mode_ctx.items():
        ctxs = sorted(ctx_to_nll)
        curves[mode] = (ctxs, [ctx_to_nll[c] for c in ctxs])
    return curves


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")
    ax.tick_params(colors="#444", which="both")
    ax.grid(False)


def k_formatter(x, pos):
    if x >= 1000:
        return f"{int(x/1000)}K"
    return f"{int(x)}"


def smooth_curve(xs, ys, n_points=300):
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    log_x = np.log(xs)
    spl = PchipInterpolator(log_x, ys)
    log_x_dense = np.linspace(log_x.min(), log_x.max(), n_points)
    return np.exp(log_x_dense), spl(log_x_dense)


def draw_foreground(ax, xs, ys, *, color, linestyle, linewidth, label,
                    marker, markersize, zorder=4):
    x_smooth, y_smooth = smooth_curve(xs, ys)
    ax.plot(x_smooth, y_smooth,
            color=color, linestyle=linestyle, linewidth=linewidth,
            zorder=zorder, label=label)
    ax.plot(xs, ys, color=color, linestyle="None",
            marker=marker, markersize=markersize,
            markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=0.8, zorder=zorder + 1)


def draw_faded(ax, xs, ys, *, color, linestyle, linewidth, alpha=0.22,
               zorder=2):
    x_smooth, y_smooth = smooth_curve(xs, ys)
    ax.plot(x_smooth, y_smooth,
            color=color, linestyle=linestyle, linewidth=linewidth * 0.9,
            alpha=alpha, zorder=zorder)


def make_figure(foreground_csv, foreground_label, background_csv,
                background_label, trained_ctx, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib.lines import Line2D

    fg_curves = load_curves(foreground_csv)
    bg_curves = load_curves(background_csv) if background_csv else {}

    if not fg_curves:
        raise ValueError(f"No usable rows in foreground CSV: {foreground_csv}")

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    for m in ORDER:
        if m not in bg_curves:
            continue
        xs, ys = bg_curves[m]
        if len(xs) < 2:
            continue
        draw_faded(
            ax, xs, ys,
            color=COLORS[m], linestyle=LINESTYLES[m],
            linewidth=LINEWIDTHS[m],
            alpha=0.22, zorder=2,
        )

    if trained_ctx and trained_ctx > 0:
        ax.axvline(trained_ctx, color="#bbb", lw=0.9, ls=":", alpha=0.8, zorder=1)

    for m in ORDER:
        if m not in fg_curves:
            continue
        xs, ys = fg_curves[m]
        if len(xs) < 2:
            continue
        draw_foreground(
            ax, xs, ys,
            color=COLORS[m], linestyle=LINESTYLES[m],
            linewidth=LINEWIDTHS[m],
            marker="o" if m == "tc" else "s",
            markersize=6 if m == "tc" else 5,
            label=LABELS[m],
            zorder=4 if m == "tc" else 3,
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))

    all_ctx = set()
    for xs, _ in fg_curves.values():
        all_ctx.update(xs)
    for xs, _ in bg_curves.values():
        all_ctx.update(xs)
    all_ctx = sorted(all_ctx)
    ax.set_xticks(all_ctx)
    ax.set_xticklabels([k_formatter(c, None) for c in all_ctx])
    ax.set_xlabel("Context length", fontsize=11)
    ax.set_ylabel("NLL  (lower is better)", fontsize=11)

    method_legend = ax.legend(
        loc="upper left", fontsize=9, frameon=False,
        labelspacing=0.3, handlelength=2.4, title=None,
    )
    ax.add_artist(method_legend)

    if bg_curves:
        dataset_handles = [
            Line2D([0], [0], color="#444", linewidth=2.0, alpha=1.0,
                   marker="o", markersize=5, markerfacecolor="#444",
                   markeredgecolor="white", markeredgewidth=0.8,
                   label=foreground_label),
            Line2D([0], [0], color="#444", linewidth=1.8, alpha=0.30,
                   label=background_label),
        ]
        ax.legend(handles=dataset_handles, loc="lower right",
                  fontsize=8, frameon=False, labelspacing=0.3,
                  handlelength=2.4)

    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    fig.savefig(out_path + ".png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}.{{pdf,png}}")


def _plot_cli():
    import argparse
    p = argparse.ArgumentParser(
        description="Render NLL vs context-length figure from CSV(s).",
    )
    p.add_argument("--foreground", required=True,
                   help="CSV file for the foreground dataset (solid lines).")
    p.add_argument("--foreground_label", default="Foreground",
                   help="Label for the foreground dataset legend entry.")
    p.add_argument("--background", default="",
                   help="Optional CSV for the faded background dataset.")
    p.add_argument("--background_label", default="Background (faded)",
                   help="Label for the background dataset legend entry.")
    p.add_argument("--trained_ctx", type=int, default=1024,
                   help="Vertical guide line at this context length (0 to disable).")
    p.add_argument("--out", default="fig_nll_vs_ctx",
                   help="Output path stem; .pdf and .png are appended.")
    args = p.parse_args()
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    make_figure(
        foreground_csv=args.foreground,
        foreground_label=args.foreground_label,
        background_csv=args.background or None,
        background_label=args.background_label,
        trained_ctx=args.trained_ctx,
        out_path=args.out,
    )


if __name__ == "__main__":
    _plot_cli()
