"""
Evaluation: compute AP@0.5 and AP-IoU curves for segmentation results.

Compares predicted masks (from segment.py) against ground-truth masks to
compute Average Precision at IoU threshold 0.5, as well as AP at a range of
IoU thresholds (0.5 to 0.95) for AP-IoU curve plotting.

AP@0.5 definition (matching Cellpose3 paper):
  For predicted masks {R_i} and ground-truth masks {G_j}, a prediction is a
  True Positive iff IoU(R_i, G_j) > 0.5. Each ground-truth mask is matched
  at most once (via Hungarian algorithm). Per-image:
      AP = TP / (TP + FP + FN)
  Report mean AP over all images.

Usage:
    # Evaluate noisy baseline
    python src/pipeline/evaluate.py

    # Evaluate a specific method
    python src/pipeline/evaluate.py --pred_dir results/pred_masks/wiener --method_name wiener

    # Evaluate all methods in results/pred_masks/
    python src/pipeline/evaluate.py --all
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from cellpose.metrics import average_precision


# ── Default paths ─────────────────────────────────────────────────────────────
GT_DIR   = Path("data/masks")
PRED_DIR = Path("results/pred_masks/noisy")
OUT_DIR  = Path("results/ap_scores")

# IoU thresholds for AP-IoU curves (matches Cellpose3 paper Fig. 1j)
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05).tolist()


def load_masks(mask_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    """
    Load all .npy mask files from a directory.

    Args:
        mask_dir: directory containing .npy mask files.

    Returns:
        masks: list of 2D integer arrays.
        stems: list of filename stems.
    """
    npy_files = sorted(mask_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {mask_dir}")

    masks, stems = [], []
    for f in npy_files:
        masks.append(np.load(f).astype(np.int32))
        stems.append(f.stem)
    return masks, stems


def compute_ap(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    thresholds: list[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Average Precision at given IoU thresholds.

    Uses cellpose.metrics.average_precision which implements Hungarian matching
    (each ground-truth mask matched at most once).

    Args:
        gt_masks: list of ground-truth mask arrays (H, W), int, 0=background.
        pred_masks: list of predicted mask arrays (H, W), int, 0=background.
        thresholds: list of IoU thresholds. Defaults to [0.5, 0.55, ..., 0.95].

    Returns:
        ap: array of shape (n_images, n_thresholds), per-image AP at each threshold.
        tp: true positives, same shape.
        fp: false positives, same shape.
        fn: false negatives, same shape.
    """
    if thresholds is None:
        thresholds = IOU_THRESHOLDS
    return average_precision(gt_masks, pred_masks, threshold=thresholds)


def evaluate_method(
    gt_dir: Path,
    pred_dir: Path,
    out_dir: Path,
    method_name: str,
    thresholds: list[float] = None,
) -> float:
    """
    Evaluate a single denoising method's segmentation results.

    Loads ground-truth and predicted masks, computes AP at all IoU thresholds,
    prints a summary, and saves per-image results to CSV.

    Args:
        gt_dir: directory with ground-truth .npy masks.
        pred_dir: directory with predicted .npy masks from segment.py.
        out_dir: directory to save per-image AP CSV.
        method_name: name for this method (used in output filenames).
        thresholds: IoU thresholds for AP computation.

    Returns:
        mean_ap50: mean AP@0.5 across all images.
    """
    if thresholds is None:
        thresholds = IOU_THRESHOLDS
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_masks, gt_stems = load_masks(gt_dir)
    pred_masks, pred_stems = load_masks(pred_dir)

    # Match files by stem
    gt_dict = dict(zip(gt_stems, gt_masks))
    common_stems = [s for s in pred_stems if s in gt_dict]
    if not common_stems:
        raise ValueError(f"No matching stems between {gt_dir} and {pred_dir}")

    gt_matched = [gt_dict[s] for s in common_stems]
    pred_matched = [pred_masks[pred_stems.index(s)] for s in common_stems]

    print(f"Evaluating '{method_name}': {len(common_stems)} images")

    ap, tp, fp, fn = compute_ap(gt_matched, pred_matched, thresholds)

    # Print summary
    idx_05 = 0  # first threshold is 0.5
    mean_ap50 = float(ap[:, idx_05].mean())
    print(f"  Mean AP@0.5 = {mean_ap50:.4f}")
    print(f"  Mean AP@0.75 = {float(ap[:, 5].mean()):.4f}" if len(thresholds) > 5 else "")

    # Save per-image CSV
    csv_path = out_dir / f"{method_name}.csv"
    with open(csv_path, "w", newline="") as f:
        fields = ["image"] + [f"ap@{t:.2f}" for t in thresholds]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, stem in enumerate(common_stems):
            row = {"image": stem}
            for j, t in enumerate(thresholds):
                row[f"ap@{t:.2f}"] = f"{ap[i, j]:.4f}"
            writer.writerow(row)

    # Save mean summary
    summary_path = out_dir / f"{method_name}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        fields = ["threshold", "mean_ap", "mean_tp", "mean_fp", "mean_fn"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for j, t in enumerate(thresholds):
            writer.writerow({
                "threshold": f"{t:.2f}",
                "mean_ap": f"{ap[:, j].mean():.4f}",
                "mean_tp": f"{tp[:, j].mean():.1f}",
                "mean_fp": f"{fp[:, j].mean():.1f}",
                "mean_fn": f"{fn[:, j].mean():.1f}",
            })

    print(f"  Results saved to {csv_path}")
    return mean_ap50


def evaluate_all(
    gt_dir: Path,
    pred_base: Path,
    out_dir: Path,
) -> None:
    """
    Evaluate all methods found as subdirectories under pred_base.

    Expects structure:
        pred_base/
          noisy/     ← predicted masks from noisy images
          clean/     ← predicted masks from clean images
          wiener/    ← predicted masks from Wiener-denoised images
          ...

    Prints a comparison table at the end.

    Args:
        gt_dir: directory with ground-truth masks.
        pred_base: parent directory containing one subdirectory per method.
        out_dir: directory to save all results.
    """
    method_dirs = sorted(d for d in pred_base.iterdir() if d.is_dir())
    if not method_dirs:
        print(f"No method directories found under {pred_base}")
        return

    results = {}
    for method_dir in method_dirs:
        name = method_dir.name
        try:
            mean_ap = evaluate_method(gt_dir, method_dir, out_dir, name)
            results[name] = mean_ap
        except (FileNotFoundError, ValueError) as e:
            print(f"  Skipping {name}: {e}")
        print()

    # Print comparison table
    if results:
        print("=" * 40)
        print(f"{'Method':<20} {'Mean AP@0.5':>12}")
        print("-" * 40)
        for name, ap in sorted(results.items(), key=lambda x: -x[1]):
            print(f"{name:<20} {ap:>12.4f}")
        print("=" * 40)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate segmentation results against ground-truth masks"
    )
    p.add_argument("--gt_dir",      type=Path, default=GT_DIR,
                   help="directory with ground-truth .npy masks")
    p.add_argument("--pred_dir",    type=Path, default=PRED_DIR,
                   help="directory with predicted .npy masks")
    p.add_argument("--out_dir",     type=Path, default=OUT_DIR,
                   help="directory to save AP score CSVs")
    p.add_argument("--method_name", type=str, default="noisy",
                   help="name label for this method")
    p.add_argument("--all", action="store_true",
                   help="evaluate all subdirectories under --pred_dir as separate methods")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all:
        evaluate_all(args.gt_dir, args.pred_dir, args.out_dir)
    else:
        evaluate_method(args.gt_dir, args.pred_dir, args.out_dir, args.method_name)
