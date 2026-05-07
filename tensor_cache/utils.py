"""
Shared helpers:
  - tqdm-aware printing, progress bars
  - CLI / config-file overrides (Karpathy-style)
  - NLL vs context-length plotting

The plotting code uses lazy imports for matplotlib / scipy so train.py,
sample.py, bench.py, evaluate.py don't pay their startup cost.

To render the figure: `python tensor_cache/utils.py --foreground=... [--background=...] --out=...`
"""

import os
import sys
from ast import literal_eval

from tqdm.auto import tqdm


# ----------------------------------------------------------------------
# Console / progress helpers
# ----------------------------------------------------------------------

def console_quiet() -> bool:
    mode = os.environ.get("TENSORCACHE_QUIET", "0").strip().lower()
    return mode in {"1", "on", "true", "yes", "quiet"}


def progress_enabled() -> bool:
    mode = os.environ.get("TENSORCACHE_TQDM", "auto").strip().lower()
    if mode in {"0", "off", "false", "disable", "disabled"}:
        return False
    if mode in {"1", "on", "true", "force"}:
        return True
    return sys.stderr.isatty()


def _base_position() -> int:
    raw = os.environ.get("TENSORCACHE_TQDM_POSITION", "0").strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def _prefixed_desc(desc: str) -> str:
    prefix = os.environ.get("TENSORCACHE_TQDM_DESC_PREFIX", "").strip()
    if prefix and desc:
        return f"{prefix} {desc}"
    if prefix:
        return prefix
    return desc


def _ascii_mode() -> bool:
    mode = os.environ.get("TENSORCACHE_TQDM_ASCII", "").strip().lower()
    return mode in {"1", "on", "true", "force", "yes"}


def make_progress(iterable=None, *, total=None, desc="", position_offset=0,
                  leave=True, disable=None, **kwargs):
    if disable is None:
        disable = not progress_enabled()
    mininterval = float(os.environ.get("TENSORCACHE_TQDM_MININTERVAL", "0.5"))
    return tqdm(
        iterable,
        total=total,
        desc=_prefixed_desc(desc),
        position=_base_position() + int(position_offset),
        leave=leave,
        disable=disable,
        dynamic_ncols=True,
        ascii=_ascii_mode(),
        mininterval=mininterval,
        smoothing=0.05,
        **kwargs,
    )


def tprint(msg="", end="\n"):
    """Print a message that coexists cleanly with any active tqdm bar."""
    tqdm.write(str(msg), end=end)


def cprint(*args, force=False, **kwargs):
    """Print only when console quiet mode is disabled unless force=True."""
    if force or not console_quiet():
        print(*args, **kwargs)


def ctprint(msg="", end="\n", force=False):
    """tqdm-aware print honoring console quiet mode unless force=True."""
    if force or not console_quiet():
        tprint(msg, end=end)


# ----------------------------------------------------------------------
# Config-file / CLI override system
#
# Karpathy-style: scripts declare module-level variables, then call
# apply_overrides(globals()) to absorb positional config files and --key=value
# CLI flags. Replaces the legacy exec(open('configurator.py').read()) pattern.
# ----------------------------------------------------------------------

def _config_quiet() -> bool:
    mode = os.environ.get("TENSORCACHE_CONFIG_QUIET", "").strip().lower()
    if mode:
        return mode in {"1", "on", "true", "yes", "quiet"}
    return console_quiet()


def apply_overrides(g):
    """Apply positional config files and --key=value CLI overrides to globals dict `g`."""
    quiet = _config_quiet()
    for arg in sys.argv[1:]:
        if "=" not in arg:
            assert not arg.startswith("--"), f"Unexpected flag without value: {arg}"
            config_file = arg
            if not quiet:
                print(f"Overriding config with {config_file}:")
                with open(config_file) as f:
                    print(f.read())
            with open(config_file) as f:
                exec(f.read(), g)
        else:
            assert arg.startswith("--"), f"Expected --key=value, got: {arg}"
            key, val = arg.split("=", 1)
            key = key[2:]
            if key not in g:
                raise ValueError(f"Unknown config key: {key}")
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            current = g[key]
            if type(attempt) is not type(current):
                if isinstance(current, str):
                    attempt = val
                elif isinstance(current, float) and isinstance(attempt, int):
                    attempt = float(attempt)
                else:
                    raise TypeError(
                        f"Config key '{key}': type mismatch, expected "
                        f"{type(current).__name__} but got "
                        f"{type(attempt).__name__} from value '{val}'"
                    )
            if not quiet:
                print(f"Overriding: {key} = {attempt}")
            g[key] = attempt


# ----------------------------------------------------------------------
# Plotting: NLL vs context length
#
# Foreground dataset is plotted with solid full-opacity lines; the optional
# background dataset is overlaid as faded lines. Curves are smoothed with
# monotone cubic interpolation (PCHIP) in log-x space.
#
# Inputs are CSVs with at least these columns:
#   - mode             one of {full_kv, window_kv, streaming_llm, infini, tc}
#   - context_length   integer (alias: eval_tokens)
#   - nll              float    (alias: nll_all)
# Both bench.py long_context output and evaluate.py output match these aliases.
# ----------------------------------------------------------------------

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
            # Skip non-finite without importing numpy.
            if nll != nll or nll in (float("inf"), float("-inf")):
                continue
            by_mode_ctx[mode][ctx] = nll  # last write wins for duplicates
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
