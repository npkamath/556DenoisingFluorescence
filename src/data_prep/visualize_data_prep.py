"""
Visualize data preparation results: noise generation quality and baseline AP scores.

Produces five figures under figures/data_prep/:
  1. triptych.png          — Clean vs Poisson vs Cellpose3 side-by-side (3 example images)
  2. pscale_distribution.png — Histogram of per-image noise scales
  3. ap_vs_iou.png         — AP vs IoU threshold curves (clean ceiling, both noisy floors)
  4. per_image_ap.png      — Per-image AP@0.5 distributions (box + strip)
  5. intensity_histograms.png — Pixel intensity distributions before/after noise

Usage:
    python src/data_prep/visualize_data_prep.py
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
CLEAN_DIR       = Path("data/clean_normed")
NOISY_POISSON   = Path("data/noisy/poisson")
NOISY_CELLPOSE3 = Path("data/noisy/cellpose3")
MASKS_DIR       = Path("data/masks")
NOISE_CSV       = Path("data/noise_params.csv")
AP_DIR          = Path("results/ap_scores")
OUT_DIR         = Path("figures/data_prep")

# Example images to show (spread across dataset)
EXAMPLE_STEMS = ["005", "030", "055"]


def to_display(img: np.ndarray) -> np.ndarray:
    """Clip to [0,1] and return RGB for imshow. Ch0=R=nuc, Ch1=G=cyto, Ch2=B=empty."""
    return np.clip(img, 0, 1)


def load_ap_csv(path: Path) -> dict:
    """Load a per-image AP CSV. Returns {stem: {threshold: ap}}."""
    if not path.exists():
        return {}
    result = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stem = row["image"]
            result[stem] = {k: float(v) for k, v in row.items() if k != "image"}
    return result


def load_summary_csv(path: Path) -> dict:
    """Load a summary CSV. Returns {threshold: mean_ap}."""
    if not path.exists():
        return {}
    result = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            result[float(row["threshold"])] = float(row["mean_ap"])
    return result


# ── Figure 1: Triptych ──────────────────────────────────────────────────────

def fig_triptych(out_path: Path):
    n = len(EXAMPLE_STEMS)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    for row, stem in enumerate(EXAMPLE_STEMS):
        clean = np.load(CLEAN_DIR / f"{stem}.npy")
        noisy_p = np.load(NOISY_POISSON / f"{stem}.npy")
        noisy_c = np.load(NOISY_CELLPOSE3 / f"{stem}.npy")

        for col, (img, title) in enumerate([
            (clean, "Clean"),
            (noisy_p, "Poisson noise"),
            (noisy_c, "Cellpose3 noise"),
        ]):
            axes[row, col].imshow(to_display(img))
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=13, fontweight="bold")
            if col == 0:
                axes[row, col].set_ylabel(f"Image {stem}", fontsize=11)

    fig.suptitle("Noise Generation: Clean vs. Poisson vs. Cellpose3",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1/5] {out_path}")


# ── Figure 2: Pscale distribution ───────────────────────────────────────────

def fig_pscale_distribution(out_path: Path):
    pscales = []
    with open(NOISE_CSV, newline="") as f:
        for row in csv.DictReader(f):
            pscales.append(float(row["pscale"]))
    pscales = np.array(pscales)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pscales, bins=20, edgecolor="white", color="#4C72B0", alpha=0.85)
    ax.axvline(pscales.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {pscales.mean():.2f}")
    ax.axvline(np.median(pscales), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median = {np.median(pscales):.2f}")
    ax.set_xlabel("Poisson noise scale (pscale)", fontsize=12)
    ax.set_ylabel("Number of images", fontsize=12)
    ax.set_title("Per-Image Noise Scale Distribution — Gamma(α=4, β=0.7)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.text(0.97, 0.85, f"n = {len(pscales)}\nmin = {pscales.min():.2f}\nmax = {pscales.max():.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2/5] {out_path}")


# ── Figure 3: AP vs IoU threshold curves ────────────────────────────────────

def fig_ap_vs_iou(out_path: Path):
    methods = {
        "Clean (ceiling)": ("clean_summary.csv", "#2ca02c"),
        "Noisy — Poisson": ("noisy_poisson_summary.csv", "#d62728"),
        "Noisy — Cellpose3": ("noisy_cellpose3_summary.csv", "#ff7f0e"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (fname, color) in methods.items():
        data = load_summary_csv(AP_DIR / fname)
        if not data:
            continue
        thresholds = sorted(data.keys())
        aps = [data[t] for t in thresholds]
        ax.plot(thresholds, aps, "o-", color=color, label=label, linewidth=2, markersize=5)

    ax.set_xlabel("IoU Threshold", fontsize=12)
    ax.set_ylabel("Mean AP", fontsize=12)
    ax.set_title("AP vs. IoU Threshold — Baseline Conditions",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(0.48, 0.97)
    ax.set_ylim(0, 0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3/5] {out_path}")


# ── Figure 4: Per-image AP@0.5 distributions ────────────────────────────────

def fig_per_image_ap(out_path: Path):
    conditions = {
        "Clean": "clean.csv",
        "Noisy\nPoisson": "noisy_poisson.csv",
        "Noisy\nCellpose3": "noisy_cellpose3.csv",
    }
    colors = ["#2ca02c", "#d62728", "#ff7f0e"]

    all_aps = []
    labels = []
    for label, fname in conditions.items():
        data = load_ap_csv(AP_DIR / fname)
        if not data:
            continue
        aps = [v["ap@0.50"] for v in data.values()]
        all_aps.append(aps)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(all_aps, tick_labels=labels, patch_artist=True, widths=0.5,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="black", markersize=5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    # Strip plot overlay
    for i, (aps, color) in enumerate(zip(all_aps, colors)):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(aps))
        ax.scatter(np.full(len(aps), i + 1) + jitter, aps,
                   color=color, alpha=0.5, s=18, edgecolors="none", zorder=3)

    ax.set_ylabel("AP@0.5", fontsize=12)
    ax.set_title("Per-Image AP@0.5 Distribution by Condition",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate means
    for i, aps in enumerate(all_aps):
        mean = np.mean(aps)
        ax.text(i + 1, mean + 0.03, f"{mean:.3f}", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4/5] {out_path}")


# ── Figure 5: Intensity histograms ──────────────────────────────────────────

def fig_intensity_histograms(out_path: Path):
    stem = EXAMPLE_STEMS[0]
    clean = np.load(CLEAN_DIR / f"{stem}.npy")
    noisy_p = np.load(NOISY_POISSON / f"{stem}.npy")
    noisy_c = np.load(NOISY_CELLPOSE3 / f"{stem}.npy")

    ch_names = ["Nucleus (ch0)", "Cytoplasm (ch1)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for col, ch in enumerate([0, 1]):
        ax = axes[col]
        for arr, label, color, ls in [
            (clean[:, :, ch], "Clean", "#2ca02c", "-"),
            (noisy_p[:, :, ch], "Poisson noise", "#d62728", "--"),
            (noisy_c[:, :, ch], "Cellpose3 noise", "#ff7f0e", ":"),
        ]:
            vals = arr.flatten()
            vals = vals[vals > 1e-6]  # skip background zeros
            ax.hist(vals, bins=100, density=True, alpha=0.4, color=color,
                    histtype="stepfilled", linewidth=0)
            ax.hist(vals, bins=100, density=True, histtype="step",
                    color=color, linewidth=1.5, linestyle=ls, label=label)

        ax.set_xlabel("Pixel intensity", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{ch_names[col]} — Image {stem}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(0, None)

    fig.suptitle("Pixel Intensity Distributions: Clean vs. Noisy",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5/5] {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating data preparation figures...\n")

    fig_triptych(OUT_DIR / "triptych.png")
    fig_pscale_distribution(OUT_DIR / "pscale_distribution.png")
    fig_ap_vs_iou(OUT_DIR / "ap_vs_iou.png")
    fig_per_image_ap(OUT_DIR / "per_image_ap.png")
    fig_intensity_histograms(OUT_DIR / "intensity_histograms.png")

    print(f"\nDone — all figures saved to {OUT_DIR}/")
