"""
Box-plot of per-image AP@0.5 distributions across all denoising methods.

Reads every per-image CSV in results/ap_scores/ (skipping *_summary.csv),
and renders two figures under figures/boxplot/:

  1. ap50_boxplot.png       — all methods on one axis, ordered by mean AP@0.5.
  2. ap50_boxplot_grouped.png — same data grouped by family (baseline / classical /
     learned DL / clean ceiling) for paper use.

Each box shows median, IQR, and whiskers; mean is drawn as a diamond and
annotated alongside σ. A jittered strip of per-image points is overlaid
to make distribution shape visible.

Usage:
    python src/boxplot_ap_distributions.py
    python src/boxplot_ap_distributions.py --methods noisy bm3d cellpose3 \
        pnp_hqs_epoch25 pnp_hqs_epoch50 pnp_hqs_epoch75 clean
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AP_DIR  = Path("results/ap_scores")
OUT_DIR = Path("figures/boxplot")

PRETTY = {
    "clean"            : "Clean\n(ceiling)",
    "noisy"            : "Noisy\nbaseline",
    "wiener"           : "Wiener\n(VST)",
    "poisson_tv"       : "Poisson-TV",
    "bm3d"             : "VST+BM3D",
    "purelet_swt"      : "PURE-LET",
    "noise2void"       : "Noise2Void",
    "cellpose3"        : "Cellpose3",
    "pnp_hqs_epoch25"  : "PnP-HQS\nep25",
    "pnp_hqs_epoch50"  : "PnP-HQS\nep50",
    "pnp_hqs_epoch75"  : "PnP-HQS\nep75",
}

FAMILY = {
    "noisy"           : ("Baseline",   "#9E9E9E"),
    "wiener"          : ("Classical",  "#FF9800"),
    "poisson_tv"      : ("Classical",  "#4CAF50"),
    "bm3d"            : ("Classical",  "#2196F3"),
    "purelet_swt"     : ("Classical",  "#9C27B0"),
    "noise2void"      : ("Learned",    "#E91E63"),
    "cellpose3"       : ("Learned",    "#F44336"),
    "pnp_hqs_epoch25" : ("Learned",    "#00BCD4"),
    "pnp_hqs_epoch50" : ("Learned",    "#0097A7"),
    "pnp_hqs_epoch75" : ("Learned",    "#006064"),
    "clean"           : ("Ceiling",    "#2ca02c"),
}

DEFAULT_COLOR = "#607D8B"


def load_ap50(path: Path) -> np.ndarray:
    vals = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            vals.append(float(row["ap@0.50"]))
    return np.asarray(vals, dtype=np.float64)


def discover_methods() -> list[str]:
    out = []
    for f in sorted(AP_DIR.glob("*.csv")):
        if f.stem.endswith("_summary"):
            continue
        out.append(f.stem)
    return out


def draw_boxplot(
    data: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    out_path: Path,
    title: str,
    figsize: tuple[float, float] = (11, 5.5),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    positions = np.arange(1, len(data) + 1)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="black",
                       markeredgecolor="black", markersize=5),
        medianprops=dict(color="black", linewidth=1.4),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="none",
                        markeredgecolor="gray", alpha=0.6),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    rng = np.random.default_rng(0)
    for i, (vals, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(positions[i] + jitter, vals,
                   color=color, edgecolors="white", linewidths=0.3,
                   s=14, alpha=0.7, zorder=3)

    for i, vals in enumerate(data):
        mean = float(np.mean(vals))
        std  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ax.text(positions[i], 1.02,
                f"μ={mean:.3f}\nσ={std:.2f}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Per-image AP@0.5", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=22)
    ax.set_ylim(-0.02, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Methods to include (default: all non-summary CSVs).")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    methods = args.methods if args.methods else discover_methods()

    data, labels, colors, families = [], [], [], []
    for m in methods:
        path = AP_DIR / f"{m}.csv"
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        vals = load_ap50(path)
        data.append(vals)
        labels.append(PRETTY.get(m, m))
        fam, col = FAMILY.get(m, ("Other", DEFAULT_COLOR))
        colors.append(col)
        families.append(fam)

    if not data:
        raise SystemExit("No AP CSVs found.")

    order = np.argsort([float(np.mean(v)) for v in data])
    data_s   = [data[i]   for i in order]
    labels_s = [labels[i] for i in order]
    colors_s = [colors[i] for i in order]
    draw_boxplot(
        data_s, labels_s, colors_s,
        out_dir / "ap50_boxplot.png",
        "Per-image AP@0.5 distributions — all methods (sorted by mean)",
    )

    family_order = ["Baseline", "Classical", "Learned", "Ceiling"]
    grouped_idx = []
    for fam in family_order:
        idxs = [i for i, f in enumerate(families) if f == fam]
        idxs.sort(key=lambda i: float(np.mean(data[i])))
        grouped_idx.extend(idxs)
    data_g   = [data[i]   for i in grouped_idx]
    labels_g = [labels[i] for i in grouped_idx]
    colors_g = [colors[i] for i in grouped_idx]
    draw_boxplot(
        data_g, labels_g, colors_g,
        out_dir / "ap50_boxplot_grouped.png",
        "Per-image AP@0.5 distributions — grouped by method family",
        figsize=(12, 5.5),
    )

    print("\nSummary:")
    print(f"  {'method':<22} {'n':>4}  {'mean':>6}  {'std':>6}  {'Q1':>6}  "
          f"{'med':>6}  {'Q3':>6}")
    for m, vals in zip(methods, data):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        print(f"  {m:<22} {len(vals):>4}  {vals.mean():>6.3f}  "
              f"{vals.std(ddof=1):>6.3f}  {q1:>6.3f}  {med:>6.3f}  {q3:>6.3f}")


if __name__ == "__main__":
    main()
