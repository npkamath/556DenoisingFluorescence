#!/usr/bin/env python3
"""
Plot 1: Bar chart comparing AP@0.5 across all denoising methods.

Reads per-image AP CSVs from results/ap_scores/ and plots mean AP@0.5
with ±1 std error bars. Methods are sorted by mean AP@0.5.

Usage:
    python src/plot_ap_comparison.py
    python src/plot_ap_comparison.py --ap_dir results/ap_scores --out_dir figures/comparison
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

AP_DIR  = Path("results/ap_scores")
OUT_DIR = Path("figures/comparison")
RUNTIME_SOURCES = {
    "cellpose3":   (Path("results/runtimes/cellpose3_denoise.csv"), "time_s"),
    "noise2void":  (Path("results/runtimes/noise2void.csv"),        "time_s"),
    "purelet_swt": (Path("results/runtimes/purelet_swt.csv"),       "time_s"),
    "wiener":      (Path("results/extension1/tuned_ap_scores/wiener_tuned_runtimes.csv"),     "denoise_time_s"),
    "bm3d":        (Path("results/extension1/tuned_ap_scores/bm3d_tuned_runtimes.csv"),       "denoise_time_s"),
    "poisson_tv":  (Path("results/extension1/tuned_ap_scores/poisson_tv_tuned_runtimes.csv"), "denoise_time_s"),
}

# Display names and color groups
DISPLAY = {
    "noisy":       "Noisy",
    "clean":       "Clean",
    "wiener":      "Wiener",
    "poisson_tv":  "Poisson-TV",
    "purelet_swt": "PURE-LET",
    "bm3d":        "BM3D",
    "noise2void":  "Noise2Void",
    "cellpose3":   "Cellpose3",
    "pnp_hqs":     "PnP-HQS",
}

# Color families by category
#   Baselines  — grays
#   Classical  — blue family
#   Deep learning — red/orange family
#   PnP-HQS   — purple family
CATEGORY = {
    "clean":       "baseline",
    "noisy":       "baseline",
    "wiener":      "classical",
    "poisson_tv":  "classical",
    "purelet_swt": "classical",
    "bm3d":        "classical",
    "noise2void":  "dl",
    "cellpose3":   "dl",
}

CATEGORY_LABELS = {
    "baseline":  "Baseline",
    "classical": "Classical",
    "dl":        "Deep Learning",
    "pnp":       "PnP-HQS",
}

CATEGORY_COLORS = {
    "baseline":  "#5c5c5c",
    "classical": "#3b7dd8",
    "dl":        "#e15759",
    "pnp":       "#9b59b6",
}

COLORS = {
    "noisy":       "#9a9a9a",
    "clean":       "#3c3c3c",
    "wiener":      "#6ba4e7",
    "poisson_tv":  "#3b7dd8",
    "bm3d":        "#1f5bb0",
    "purelet_swt": "#12407f",
    "noise2void":  "#f28e2b",
    "cellpose3":   "#e15759",
    "pnp_hqs":     "#9b59b6",
}
DEFAULT_COLOR = "#999999"


def category_for(method: str) -> str:
    if method in CATEGORY:
        return CATEGORY[method]
    if method.startswith("pnp_hqs"):
        return "pnp"
    return "classical"


def load_ap50(csv_path: Path) -> np.ndarray:
    """Load per-image AP@0.5 values from a method CSV."""
    scores = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        col = next((c for c in reader.fieldnames if "0.50" in c or "ap50" in c.lower()), None)
        if col is None:
            raise ValueError(f"No AP@0.5 column in {csv_path}")
        for row in reader:
            scores.append(float(row[col]))
    return np.array(scores)


def load_mean_runtime(method: str) -> float | None:
    """Return mean per-image denoise time in seconds, or None if unavailable."""
    if method not in RUNTIME_SOURCES:
        return None
    path, col = RUNTIME_SOURCES[method]
    if not path.exists():
        return None
    times = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                times.append(float(row[col]))
            except (KeyError, ValueError):
                continue
    return float(np.mean(times)) if times else None


def fmt_runtime(t: float | None) -> str:
    if t is None:
        return "—"
    if t >= 10:
        return f"{t:.1f}s"
    if t >= 1:
        return f"{t:.2f}s"
    return f"{t:.2f}s"


def pretty_label(m: str) -> str:
    if m in DISPLAY:
        return DISPLAY[m]
    if m.startswith("pnp_hqs_epoch"):
        epoch = m.replace("pnp_hqs_epoch", "")
        return f"PnP-HQS (epoch {epoch})"
    return m.replace("_", " ").title()


def render_plot(method_scores: dict, out_path: Path, title: str,
                keep_best_pnp: bool = False) -> None:
    scores = dict(method_scores)

    # If requested, keep only the best-performing PnP-HQS epoch
    if keep_best_pnp:
        pnp_methods = [m for m in scores if m.startswith("pnp_hqs")]
        if len(pnp_methods) > 1:
            best = max(pnp_methods, key=lambda m: scores[m].mean())
            for m in pnp_methods:
                if m != best:
                    scores.pop(m)

    sorted_methods = sorted(scores, key=lambda m: scores[m].mean())
    means = [scores[m].mean() for m in sorted_methods]
    labels = [pretty_label(m) for m in sorted_methods]
    colors = [CATEGORY_COLORS[category_for(m)] for m in sorted_methods]
    runtimes = [load_mean_runtime(m) for m in sorted_methods]
    cats_present = sorted({category_for(m) for m in sorted_methods},
                          key=lambda c: ["baseline", "classical", "dl", "pnp"].index(c))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8.2, max(4, len(sorted_methods) * 0.55)))
    y_pos = np.arange(len(sorted_methods))
    bars = ax.barh(y_pos, means, color=colors, edgecolor="white",
                   linewidth=0.6, height=0.72, alpha=0.92)

    for bar, mean, rt in zip(bars, means, runtimes):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{mean:.4f}", ha="left", va="center", fontsize=10,
                color="#333333")
        ax.text(bar.get_width() + 0.11, bar.get_y() + bar.get_height() / 2,
                fmt_runtime(rt), ha="left", va="center", fontsize=9,
                color="#777777", style="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Mean AP@0.5   (italic: mean denoise time / image)", fontsize=10.5)
    ax.set_title(title, fontsize=12.5, pad=12, loc="left")
    ax.set_xlim(0, min(1.0, max(means) + 0.26))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="x", which="major", linestyle="-", alpha=0.25, linewidth=0.6)
    ax.grid(axis="x", which="minor", linestyle=":", alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="both", length=0)
    ax.tick_params(axis="x", labelsize=10)

    # Category legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c], alpha=0.92)
        for c in cats_present
    ]
    legend_labels = [CATEGORY_LABELS[c] for c in cats_present]
    ax.legend(legend_handles, legend_labels, loc="lower right", frameon=False,
              fontsize=9, title="Category", title_fontsize=9.5,
              handlelength=1.2, handleheight=1.0, borderaxespad=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main(ap_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    method_scores = {}
    for csv_path in sorted(ap_dir.glob("*.csv")):
        if "_summary" in csv_path.stem:
            continue
        try:
            scores = load_ap50(csv_path)
            method_scores[csv_path.stem] = scores
        except Exception as e:
            print(f"  Skipping {csv_path.name}: {e}")

    if not method_scores:
        print(f"No AP score CSVs found in {ap_dir}")
        return

    # Main version: keep all PnP-HQS epochs
    render_plot(method_scores, out_dir / "ap_comparison.png",
                "Denoising Method Comparison — AP@0.5 on Segmentation")

    # No-PnP version: drop any pnp_hqs_* methods
    no_pnp = {m: s for m, s in method_scores.items() if not m.startswith("pnp_hqs")}
    if len(no_pnp) < len(method_scores):
        render_plot(no_pnp, out_dir / "ap_comparison_no_pnp.png",
                    "Denoising Method Comparison — AP@0.5 on Segmentation")

    # Print table (full set, best first)
    sorted_methods = sorted(method_scores, key=lambda m: -method_scores[m].mean())
    print(f"\n{'Method':<22} {'Mean AP@0.5':>12} {'Std':>8} {'N':>5}")
    print("-" * 50)
    for m in sorted_methods:
        scores = method_scores[m]
        print(f"{pretty_label(m):<22} {scores.mean():>12.4f} {scores.std():>8.4f} {len(scores):>5}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ap_dir",  type=Path, default=AP_DIR)
    p.add_argument("--out_dir", type=Path, default=OUT_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.ap_dir, args.out_dir)
