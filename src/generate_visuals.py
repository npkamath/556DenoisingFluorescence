"""
src/generate_visuals.py
------------------------
Generates publication-quality figures for the progress report.

Three figure types are produced:

  1. Per-image pair figure
     Left panel : denoised / input image (display channels: G=cyto, R=nuc)
     Right panel: predicted instance segmentation mask (coloured by instance)
                  with ground-truth cell boundaries overlaid in red

  2. Per-condition montage
     N sampled images stacked vertically, one column per panel type.
     Saved as  figures/montage_<method>.png

  3. Multi-method comparison grid  ← main figure for the progress report
     Rows = methods, columns = sampled images.
     Each cell is a side-by-side [image | mask] composite.
     Saved as  figures/comparison_grid.png

Reads:
    data/clean/              .npy float32 (H, W, 3)  — clean reference
    data/noisy/poisson/      .npy float32 (H, W, 3)  — noisy input
    results/denoised/<m>/    .npy float32 (H, W, 3)  — denoised images
    results/pred_masks/<m>/  .npy int32   (H, W)     — predicted masks
    data/masks/              .npy uint16  (H, W)     — ground-truth masks
    results/ap_scores/<m>.csv               — AP@0.5 for title annotation

Channel convention (matches segment.py):
    ch0 = Red  = nucleus
    ch1 = Green = cytoplasm   ← display as gray or composite
    ch2 = Blue  = empty

Usage
-----
    # Default: all methods found under results/pred_masks/
    python src/generate_visuals.py

    # Specific methods, 8 sample images, skip individual pair figures:
    python src/generate_visuals.py \
        --methods clean noisy cellpose3 wiener \
        --n_sample 8 \
        --skip_pairs

Output
------
    figures/<method>/<stem>_pair.png        (unless --skip_pairs)
    figures/montage_<method>.png
    figures/comparison_grid.png
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import binary_erosion


# ── Default paths ─────────────────────────────────────────────────────────────
CLEAN_DIR       = Path("data/clean")
NOISY_DIR       = Path("data/noisy/poisson")
GT_MASK_DIR     = Path("data/masks")
DENOISED_ROOT   = Path("results/denoised")
PRED_MASKS_ROOT = Path("results/pred_masks")
AP_SCORES_DIR   = Path("results/ap_scores")
FIGURES_DIR     = Path("figures")

# Methods that live directly under data/ rather than results/denoised/
BUILTIN_INPUT = {
    "clean" : CLEAN_DIR,
    "noisy" : NOISY_DIR,
}

PRETTY = {
    "clean"    : "Clean (reference)",
    "noisy"    : "Noisy (no restoration)",
    "cellpose3": "Cellpose3 (DL)",
    "wiener"   : "Wiener",
    "tv"       : "Poisson-TV",
    "bm3d"     : "VST + BM3D",
    "purelet"  : "PURE-LET",
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_npy(path: Path) -> np.ndarray:
    return np.load(str(path))


def load_ap50_for_method(method: str, ap_dir: Path) -> dict[str, float]:
    """Return {stem: ap@0.5} dict, or empty dict if CSV missing."""
    csv_path = ap_dir / f"{method}.csv"
    if not csv_path.exists():
        return {}
    result = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            result[row["image"]] = float(row["ap@0.50"])
    return result


def stems_for_dir(d: Path) -> list[str]:
    return sorted(f.stem for f in d.glob("*.npy"))


# ── Image display helpers ─────────────────────────────────────────────────────

def to_display(img: np.ndarray) -> np.ndarray:
    """
    Convert a float32 (H, W, 3) image to a displayable (H, W, 3) uint8.
    Uses channels [1, 0, 2] → [R=nuc, G=cyto, B=empty] already in RGB order.
    Stretches each channel independently to [0, 255] for visibility.
    """
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float64)
        lo, hi = ch.min(), ch.max()
        if hi > lo:
            out[:, :, c] = (ch - lo) / (hi - lo)
        else:
            out[:, :, c] = 0.0
    return out


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert integer instance mask → RGB float32 (H, W, 3), 0=black background."""
    cmap  = plt.get_cmap("tab20b")
    n     = int(mask.max())
    rgb   = np.zeros((*mask.shape, 3), dtype=np.float32)
    for i in range(1, n + 1):
        color = cmap((i % 20) / 20)[:3]
        rgb[mask == i] = color
    return rgb


def gt_boundary_overlay(gt_mask: np.ndarray) -> np.ndarray:
    """
    Return an RGBA float32 (H, W, 4) array with instance boundaries in red.
    Safe to imshow() on top of the mask panel.
    """
    boundary = np.zeros(gt_mask.shape, dtype=bool)
    for label in np.unique(gt_mask):
        if label == 0:
            continue
        obj     = gt_mask == label
        eroded  = binary_erosion(obj, iterations=1)
        boundary |= obj ^ eroded

    overlay = np.zeros((*gt_mask.shape, 4), dtype=np.float32)
    overlay[boundary] = [1.0, 0.15, 0.15, 0.85]
    return overlay


# ── Figure builders ───────────────────────────────────────────────────────────

def make_pair_figure(
    img      : np.ndarray,       # (H, W, 3) float32
    mask     : np.ndarray,       # (H, W) int32
    gt_mask  : np.ndarray | None,
    ap50     : float | None,
    label    : str,
    stem     : str,
    out_path : Path,
    dpi      : int = 150,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{label}  —  {stem}", fontsize=11, fontweight="bold")

    # Left: image
    axes[0].imshow(to_display(img), interpolation="nearest")
    axes[0].set_title("Input image", fontsize=9)
    axes[0].axis("off")

    # Right: mask + GT boundary
    axes[1].imshow(mask_to_rgb(mask), interpolation="nearest")
    if gt_mask is not None:
        axes[1].imshow(gt_boundary_overlay(gt_mask), interpolation="nearest")

    n_pred  = int(mask.max())
    ap_str  = f"AP@0.5 = {ap50:.3f}" if ap50 is not None else ""
    axes[1].set_title(
        f"Segmentation mask — {n_pred} cells  {ap_str}", fontsize=9
    )
    axes[1].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_montage(
    stems      : list[str],
    img_dir    : Path,
    mask_dir   : Path,
    gt_dir     : Path,
    ap_dict    : dict[str, float],
    label      : str,
    out_path   : Path,
    dpi        : int = 120,
) -> None:
    n = len(stems)
    fig, axes = plt.subplots(n, 2, figsize=(9, 4.0 * n), squeeze=False)
    fig.suptitle(f"Condition: {label}", fontsize=13, fontweight="bold")

    for row, stem in enumerate(stems):
        img     = load_npy(img_dir   / f"{stem}.npy")
        mask    = load_npy(mask_dir  / f"{stem}.npy").astype(np.int32)
        gt_path = gt_dir / f"{stem}.npy"
        gt_mask = load_npy(gt_path).astype(np.int32) if gt_path.exists() else None
        ap50    = ap_dict.get(stem)

        axes[row, 0].imshow(to_display(img), interpolation="nearest")
        axes[row, 0].set_title(stem, fontsize=7, pad=2)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask_to_rgb(mask), interpolation="nearest")
        if gt_mask is not None:
            axes[row, 1].imshow(gt_boundary_overlay(gt_mask), interpolation="nearest")
        ap_str = f"AP={ap50:.3f}" if ap50 is not None else ""
        axes[row, 1].set_title(
            f"{int(mask.max())} cells  {ap_str}", fontsize=7, pad=2
        )
        axes[row, 1].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Montage → {out_path}")


def make_comparison_grid(
    methods     : list[str],
    sampled     : list[str],           # image stems
    img_dirs    : dict[str, Path],
    mask_dirs   : dict[str, Path],
    gt_dir      : Path,
    ap_dicts    : dict[str, dict],
    out_path    : Path,
    dpi         : int = 110,
) -> None:
    """
    Grid: rows = methods, columns = sampled images.
    Each cell = [image | mask] composite side-by-side.
    """
    n_rows = len(methods)
    n_cols = len(sampled)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.0 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for r, method in enumerate(methods):
        img_dir  = img_dirs.get(method)
        mask_dir = mask_dirs.get(method)
        ap_dict  = ap_dicts.get(method, {})
        label    = PRETTY.get(method, method)

        for c, stem in enumerate(sampled):
            ax = axes[r, c]

            # Skip gracefully if files are missing
            if img_dir is None or not (img_dir / f"{stem}.npy").exists():
                ax.text(0.5, 0.5, f"missing\n{stem}", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=7)
                ax.axis("off")
                continue
            if mask_dir is None or not (mask_dir / f"{stem}.npy").exists():
                ax.text(0.5, 0.5, "no mask", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=7)
                ax.axis("off")
                continue

            img     = load_npy(img_dir  / f"{stem}.npy")
            mask    = load_npy(mask_dir / f"{stem}.npy").astype(np.int32)
            gt_path = gt_dir / f"{stem}.npy"
            gt_mask = load_npy(gt_path).astype(np.int32) if gt_path.exists() else None

            # Side-by-side composite: left = image, right = mask
            H, W = img.shape[:2]
            comp  = np.zeros((H, W * 2, 3), dtype=np.float32)
            comp[:, :W,  :] = to_display(img)
            comp[:, W:,  :] = mask_to_rgb(mask)

            # GT boundary on the right half
            if gt_mask is not None:
                bnd = gt_boundary_overlay(gt_mask)  # (H, W, 4)
                alpha = bnd[:, :, 3:4]
                comp[:, W:, :] = (
                    comp[:, W:, :] * (1 - alpha) + bnd[:, :, :3] * alpha
                )

            ax.imshow(comp, interpolation="nearest")
            ax.axvline(W - 0.5, color="white", linewidth=0.6, linestyle="--")

            # Annotations
            ap50   = ap_dict.get(stem)
            ap_str = f"AP={ap50:.3f}" if ap50 is not None else ""
            ax.text(0.99, 0.02, ap_str, transform=ax.transAxes, fontsize=7,
                    color="white", ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45))
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            if r == 0:
                ax.set_title(stem, fontsize=8)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle(
        "Left: input image   |   Right: Cellpose cyto2 segmentation mask\n"
        "(red lines = ground-truth boundaries)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison grid → {out_path}")


# ── Helpers to resolve directories for each method ────────────────────────────

def resolve_img_dir(method: str) -> Path | None:
    if method in BUILTIN_INPUT:
        return BUILTIN_INPUT[method]
    d = DENOISED_ROOT / method
    return d if d.exists() else None


def resolve_mask_dir(method: str) -> Path | None:
    d = PRED_MASKS_ROOT / method
    return d if d.exists() else None


def discover_methods() -> list[str]:
    """Find all methods that have both an image dir and a mask dir."""
    found = []
    candidates = list(BUILTIN_INPUT.keys())
    if DENOISED_ROOT.exists():
        candidates += [d.name for d in sorted(DENOISED_ROOT.iterdir()) if d.is_dir()]
    for m in candidates:
        if resolve_img_dir(m) and resolve_mask_dir(m):
            found.append(m)
    return found


def sample_stems(img_dir: Path, n: int) -> list[str]:
    all_stems = stems_for_dir(img_dir)
    if not all_stems:
        return []
    indices = np.linspace(0, len(all_stems) - 1, min(n, len(all_stems)), dtype=int)
    return [all_stems[i] for i in indices]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate denoised-image vs segmentation-mask figures "
                    "for the progress report."
    )
    p.add_argument("--methods",     nargs="+", default=None,
                   help="Methods to include (default: all found automatically)")
    p.add_argument("--n_sample",    type=int,  default=6,
                   help="Number of images to include in montages and grid")
    p.add_argument("--out_dir",     type=Path, default=FIGURES_DIR)
    p.add_argument("--skip_pairs",  action="store_true",
                   help="Skip per-image pair figures (faster for large datasets)")
    p.add_argument("--dpi",         type=int,  default=150)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    methods = args.methods if args.methods else discover_methods()
    if not methods:
        raise SystemExit(
            "No methods found. Make sure results/pred_masks/<method>/ exists."
        )
    print(f"Methods to visualise: {methods}\n")

    # Resolve dirs and AP dicts for every method
    img_dirs  = {m: resolve_img_dir(m)  for m in methods}
    mask_dirs = {m: resolve_mask_dir(m) for m in methods}
    ap_dicts  = {m: load_ap50_for_method(m, AP_SCORES_DIR) for m in methods}

    # Sampled stems — driven by the first method that has an image dir
    reference_method = next(m for m in methods if img_dirs[m] is not None)
    sampled = sample_stems(img_dirs[reference_method], args.n_sample)
    if not sampled:
        raise SystemExit(f"No .npy images found in {img_dirs[reference_method]}")
    print(f"Sampled images ({len(sampled)}): {sampled}\n")

    for method in methods:
        img_dir  = img_dirs[method]
        mask_dir = mask_dirs[method]
        label    = PRETTY.get(method, method)

        if img_dir is None or mask_dir is None:
            print(f"[skip] {method}: missing img or mask directory")
            continue

        print(f"[*] {label}")

        # ── Per-image pair figures ─────────────────────────────────────────
        if not args.skip_pairs:
            for stem in sampled:
                img_path  = img_dir  / f"{stem}.npy"
                mask_path = mask_dir / f"{stem}.npy"
                if not img_path.exists() or not mask_path.exists():
                    continue
                gt_path = GT_MASK_DIR / f"{stem}.npy"
                make_pair_figure(
                    img      = load_npy(img_path),
                    mask     = load_npy(mask_path).astype(np.int32),
                    gt_mask  = load_npy(gt_path).astype(np.int32)
                               if gt_path.exists() else None,
                    ap50     = ap_dicts[method].get(stem),
                    label    = label,
                    stem     = stem,
                    out_path = args.out_dir / method / f"{stem}_pair.png",
                    dpi      = args.dpi,
                )
            print(f"    Pair figures → {args.out_dir / method}/")

        # ── Montage ───────────────────────────────────────────────────────
        valid_stems = [s for s in sampled
                       if (img_dir / f"{s}.npy").exists()
                       and (mask_dir / f"{s}.npy").exists()]
        if valid_stems:
            make_montage(
                stems    = valid_stems,
                img_dir  = img_dir,
                mask_dir = mask_dir,
                gt_dir   = GT_MASK_DIR,
                ap_dict  = ap_dicts[method],
                label    = label,
                out_path = args.out_dir / f"montage_{method}.png",
                dpi      = args.dpi,
            )

    # ── Comparison grid ────────────────────────────────────────────────────
    valid_methods = [m for m in methods
                     if img_dirs[m] is not None and mask_dirs[m] is not None]
    grid_stems = [s for s in sampled
                  if all(
                      (img_dirs[m]  / f"{s}.npy").exists() and
                      (mask_dirs[m] / f"{s}.npy").exists()
                      for m in valid_methods
                  )]
    # Fall back to a per-method check if no stem is common to all
    if not grid_stems:
        grid_stems = sampled[:min(4, len(sampled))]

    if len(valid_methods) >= 2:
        make_comparison_grid(
            methods   = valid_methods,
            sampled   = grid_stems[:min(4, len(grid_stems))],
            img_dirs  = img_dirs,
            mask_dirs = mask_dirs,
            gt_dir    = GT_MASK_DIR,
            ap_dicts  = ap_dicts,
            out_path  = args.out_dir / "comparison_grid.png",
            dpi       = args.dpi,
        )

    print(f"\n[✓] All figures saved under {args.out_dir}/")
