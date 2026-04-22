#!/usr/bin/env python3
"""
Extension 1: Task-Aware Hyperparameter Tuning for Classical Denoisers

Classical denoisers have hyperparameters typically tuned for pixel fidelity
(MSE/PSNR) rather than downstream segmentation accuracy. This extension
selects hyperparameters that maximize AP@0.5 on a held-out validation subset,
then evaluates the tuned settings on the full 68-image test set.

Hyperparameter grids:
  - Poisson-TV: weight (log-spaced around default 0.10, n_iter=200 fixed)
  - BM3D:       sigma_vst (multipliers around default 1.0)
  - Wiener:     mysize (odd window sizes around default 5)

Pipeline per grid point:
  denoise validation images -> segment (frozen cyto2) -> compute AP@0.5

Usage:
    python src/pipeline/task_aware_tuning.py                     # full pipeline
    python src/pipeline/task_aware_tuning.py --grid-only          # grid search only
    python src/pipeline/task_aware_tuning.py --skip-grid \\
        --tv-weight 0.06 --bm3d-sigma 0.8 --wiener-size 7

Output:
    results/extension1/
      validation_split.csv              validation/test image assignments
      grid_search_{method}.csv          per-setting AP@0.5 on validation
      best_params.csv                   selected hyperparameters
      tuned_denoised/{method}_tuned/    denoised images (tuned params)
      tuned_pred_masks/{method}_tuned/  segmentation masks (tuned params)
      tuned_ap_scores/{method}_tuned.csv  per-image AP scores
      tuned_summary.csv                 final comparison table
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from denoisers.classical.denoise_poisson_tv import denoise_one_image as _tv_denoise
from denoisers.classical.denoise_bm3d_vst    import denoise_one_image as _bm3d_denoise
from denoisers.classical.denoise_wiener_vst  import denoise_one_image as _wiener_denoise

from cellpose.models import CellposeModel
from cellpose.metrics import average_precision


# ── Constants ─────────────────────────────────────────────────────────────────

SEED = 42
N_VAL = 15  # ~22% of 68 images held out for validation

# Hyperparameter grids
TV_WEIGHTS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50]
TV_N_ITER = 200  # fixed across all grid points
BM3D_SIGMAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
WIENER_SIZES = [3, 5, 7, 9, 11, 13, 15]

# Default paths
NOISY_DIR = Path("data/noisy/poisson")
MASKS_DIR = Path("data/masks")
PSCALE_CSV = Path("data/noise_params.csv")
AP_SCORES_DIR = Path("results/ap_scores")
OUT_DIR = Path("results/extension1")


# ── Denoise wrappers (uniform interface: noisy, pscale, param) ────────────────

def _denoise_tv(noisy, pscale, param):
    return _tv_denoise(noisy, pscale, weight=param, n_iter=TV_N_ITER)


def _denoise_bm3d(noisy, pscale, param):
    return _bm3d_denoise(noisy, pscale, sigma_vst=param)


def _denoise_wiener(noisy, pscale, param):
    return _wiener_denoise(noisy, pscale, mysize=param)


# Method registry
METHODS = {
    "poisson_tv": {
        "denoise_fn": _denoise_tv,
        "param_name": "weight",
        "grid": TV_WEIGHTS,
        "default": 0.10,
    },
    "bm3d": {
        "denoise_fn": _denoise_bm3d,
        "param_name": "sigma_vst",
        "grid": BM3D_SIGMAS,
        "default": 1.0,
    },
    "wiener": {
        "denoise_fn": _denoise_wiener,
        "param_name": "mysize",
        "grid": WIENER_SIZES,
        "default": 5,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pscales(csv_path: Path) -> dict[str, float]:
    pscale = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            pscale[r["image"]] = float(r["pscale"])
    return pscale


def create_validation_split(
    stems: list[str], n_val: int, seed: int
) -> tuple[list[str], list[str]]:
    """Deterministic random split into validation and test subsets."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(stems))
    val_idx = sorted(perm[:n_val])
    test_idx = sorted(perm[n_val:])
    return [stems[i] for i in val_idx], [stems[i] for i in test_idx]


def segment_batch(
    images: list[np.ndarray], model: CellposeModel
) -> list[np.ndarray]:
    """Segment images with a pre-loaded Cellpose cyto2 model."""
    masks = []
    for img in images:
        m, _, _ = model.eval(
            img, diameter=None, channels=[2, 1],
            flow_threshold=0.4, cellprob_threshold=0.0,
        )
        masks.append(m.astype(np.int32))
    return masks


def mean_ap50(
    gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]
) -> float:
    """Compute mean AP at IoU threshold 0.5."""
    ap, _, _, _ = average_precision(gt_masks, pred_masks, threshold=[0.5])
    return float(ap[:, 0].mean())


def load_untuned_ap50(ap_dir: Path, method_name: str) -> float:
    """Load previously computed mean AP@0.5 from evaluate.py summary CSV."""
    summary_path = ap_dir / f"{method_name}_summary.csv"
    if not summary_path.exists():
        return float("nan")
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["threshold"] == "0.50":
                return float(row["mean_ap"])
    return float("nan")


# ── Phase 1: Grid search on validation ────────────────────────────────────────

def grid_search(
    method_name: str,
    val_stems: list[str],
    noisy_dict: dict[str, np.ndarray],
    pscales: dict[str, float],
    gt_dict: dict[str, np.ndarray],
    model: CellposeModel,
    out_dir: Path,
) -> tuple:
    """
    Sweep hyperparameter grid for one method on validation images.

    Returns (best_param_value, list_of_per_setting_results).
    """
    cfg = METHODS[method_name]
    denoise_fn = cfg["denoise_fn"]
    param_name = cfg["param_name"]
    grid = cfg["grid"]

    print(f"\n{'=' * 60}")
    print(f"  Grid search: {method_name} ({param_name})")
    print(f"  Validation images: {len(val_stems)}")
    print(f"  Grid ({len(grid)} points): {grid}")
    print(f"{'=' * 60}")

    results = []
    for gi, param_val in enumerate(grid, 1):
        t0 = time.time()
        marker = " (default)" if param_val == cfg["default"] else ""
        print(f"\n  [{gi}/{len(grid)}] {param_name}={param_val}{marker}")

        # Denoise validation images
        denoised = []
        for si, stem in enumerate(val_stems, 1):
            print(f"    denoising {si}/{len(val_stems)} ...", end="\r", flush=True)
            denoised.append(
                denoise_fn(noisy_dict[stem], pscales[stem], param_val)
            )
        t_denoise = time.time() - t0
        print(f"    denoised {len(val_stems)} images in {t_denoise:.1f}s     ")

        # Segment with frozen cyto2
        print(f"    segmenting ...", end="\r", flush=True)
        pred_masks = segment_batch(denoised, model)
        t_total = time.time() - t0

        # Evaluate AP@0.5
        gt_list = [gt_dict[s] for s in val_stems]
        ap = mean_ap50(gt_list, pred_masks)

        results.append({
            param_name: param_val,
            "mean_ap50": ap,
            "denoise_time_s": round(t_denoise, 2),
            "total_time_s": round(t_total, 2),
        })

        print(f"    => AP@0.5 = {ap:.4f}  (total {t_total:.1f}s)")

    # Save grid results CSV
    csv_path = out_dir / f"grid_search_{method_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {csv_path}")

    # Select best
    best = max(results, key=lambda r: r["mean_ap50"])
    print(f"\n  BEST: {param_name}={best[param_name]}  "
          f"AP@0.5={best['mean_ap50']:.4f}")

    return best[param_name], results


# ── Phase 2: Final evaluation on full test set ────────────────────────────────

def evaluate_tuned(
    method_name: str,
    best_param,
    all_stems: list[str],
    noisy_dict: dict[str, np.ndarray],
    pscales: dict[str, float],
    gt_dict: dict[str, np.ndarray],
    model: CellposeModel,
    out_dir: Path,
) -> dict:
    """Denoise + segment + evaluate with tuned params on full test set."""
    cfg = METHODS[method_name]
    denoise_fn = cfg["denoise_fn"]
    param_name = cfg["param_name"]

    denoised_dir = out_dir / "tuned_denoised" / f"{method_name}_tuned"
    masks_dir = out_dir / "tuned_pred_masks" / f"{method_name}_tuned"
    denoised_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {method_name}: {param_name}={best_param} on {len(all_stems)} images")

    # Denoise all images and measure per-image time
    denoised = []
    denoise_times = []
    for i, stem in enumerate(all_stems, 1):
        print(f"    denoising {i}/{len(all_stems)} ...", end="\r", flush=True)
        t0 = time.time()
        d = denoise_fn(noisy_dict[stem], pscales[stem], best_param)
        denoise_times.append(time.time() - t0)
        denoised.append(d)
        np.save(denoised_dir / f"{stem}.npy", d)
    print(f"    denoised {len(all_stems)} images in {sum(denoise_times):.1f}s     ")

    # Segment all
    print(f"    segmenting ...", end="")
    pred_masks = segment_batch(denoised, model)
    print(" done")
    for stem, mask in zip(all_stems, pred_masks):
        np.save(masks_dir / f"{stem}.npy", mask)

    # Evaluate AP@0.5 per image
    gt_list = [gt_dict[s] for s in all_stems]
    ap_arr, _, _, _ = average_precision(gt_list, pred_masks, threshold=[0.5])
    tuned_ap = float(ap_arr[:, 0].mean())
    mean_time = float(np.mean(denoise_times))

    # Save per-image AP scores
    ap_dir = out_dir / "tuned_ap_scores"
    ap_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ap_dir / f"{method_name}_tuned.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ap@0.50"])
        writer.writeheader()
        for stem, ap_val in zip(all_stems, ap_arr[:, 0]):
            writer.writerow({"image": stem, "ap@0.50": f"{ap_val:.4f}"})

    # Save per-image runtimes
    rt_path = ap_dir / f"{method_name}_tuned_runtimes.csv"
    with open(rt_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "denoise_time_s"])
        writer.writeheader()
        for stem, dt in zip(all_stems, denoise_times):
            writer.writerow({"image": stem, "denoise_time_s": f"{dt:.4f}"})

    print(f"    AP@0.5={tuned_ap:.4f}  mean_time={mean_time:.3f}s")

    return {
        "method": method_name,
        "param_name": param_name,
        "default_value": cfg["default"],
        "tuned_value": best_param,
        "tuned_ap50": round(tuned_ap, 4),
        "mean_denoise_time_s": round(mean_time, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extension 1: Task-aware hyperparameter tuning "
                    "for classical denoisers"
    )
    parser.add_argument("--noisy-dir", type=Path, default=NOISY_DIR)
    parser.add_argument("--masks-dir", type=Path, default=MASKS_DIR)
    parser.add_argument("--pscale-csv", type=Path, default=PSCALE_CSV)
    parser.add_argument("--ap-scores-dir", type=Path, default=AP_SCORES_DIR,
                        help="Directory with untuned AP score CSVs from evaluate.py")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-val", type=int, default=N_VAL,
                        help="Number of validation images (default 15)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-gpu", action="store_true",
                        help="Run Cellpose on CPU")
    parser.add_argument("--methods", nargs="+",
                        default=["poisson_tv", "bm3d", "wiener"],
                        choices=["poisson_tv", "bm3d", "wiener"],
                        help="Which methods to tune")
    parser.add_argument("--grid-only", action="store_true",
                        help="Run grid search only, skip final evaluation")
    parser.add_argument("--skip-grid", action="store_true",
                        help="Skip grid search; use --tv-weight/--bm3d-sigma/"
                             "--wiener-size for final evaluation")
    parser.add_argument("--tv-weight", type=float, default=None,
                        help="TV weight to use when --skip-grid is set")
    parser.add_argument("--bm3d-sigma", type=float, default=None,
                        help="BM3D sigma to use when --skip-grid is set")
    parser.add_argument("--wiener-size", type=int, default=None,
                        help="Wiener window size to use when --skip-grid is set")
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading data...")
    pscales = load_pscales(args.pscale_csv)
    all_stems = sorted(f.stem for f in args.noisy_dir.glob("*.npy"))
    if not all_stems:
        raise FileNotFoundError(f"No .npy files in {args.noisy_dir}")

    noisy_dict = {}
    gt_dict = {}
    for stem in all_stems:
        noisy_dict[stem] = np.load(args.noisy_dir / f"{stem}.npy").astype(np.float32)
        gt_dict[stem] = np.load(args.masks_dir / f"{stem}.npy").astype(np.int32)
    print(f"  Loaded {len(all_stems)} images")

    # ── Validation split ─────────────────────────────────────────────────────
    val_stems, test_stems = create_validation_split(
        all_stems, args.n_val, args.seed
    )
    split_path = args.out_dir / "validation_split.csv"
    with open(split_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split"])
        writer.writeheader()
        for s in all_stems:
            writer.writerow({
                "image": s,
                "split": "val" if s in val_stems else "test",
            })
    print(f"  Validation ({len(val_stems)}): {val_stems}")
    print(f"  Test ({len(test_stems)}): {len(test_stems)} images")
    print(f"  Split saved: {split_path}")

    # ── Load Cellpose model ──────────────────────────────────────────────────
    print("\nLoading Cellpose cyto2 model...")
    model = CellposeModel(gpu=not args.no_gpu, model_type="cyto2")

    # ── Phase 1: Grid search ─────────────────────────────────────────────────
    best_params = {}
    grid_results = {}

    if not args.skip_grid:
        t_grid_start = time.time()
        for mi, method_name in enumerate(args.methods, 1):
            print(f"\n>>> METHOD {mi}/{len(args.methods)}: {method_name.upper()} <<<")
            best_val, results = grid_search(
                method_name=method_name,
                val_stems=val_stems,
                noisy_dict=noisy_dict,
                pscales=pscales,
                gt_dict=gt_dict,
                model=model,
                out_dir=args.out_dir,
            )
            best_params[method_name] = best_val
            grid_results[method_name] = results

        t_grid = time.time() - t_grid_start
        print(f"\nGrid search completed in {t_grid:.0f}s")

        # Save best params
        bp_path = args.out_dir / "best_params.csv"
        with open(bp_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["method", "param_name", "default_value",
                               "tuned_value"]
            )
            writer.writeheader()
            for m in args.methods:
                writer.writerow({
                    "method": m,
                    "param_name": METHODS[m]["param_name"],
                    "default_value": METHODS[m]["default"],
                    "tuned_value": best_params[m],
                })
        print(f"Best parameters saved: {bp_path}")
    else:
        # Use manually specified params (fall back to defaults)
        manual = {
            "poisson_tv": args.tv_weight,
            "bm3d": args.bm3d_sigma,
            "wiener": args.wiener_size,
        }
        for m in args.methods:
            best_params[m] = manual.get(m) or METHODS[m]["default"]
        print(f"\nUsing specified params: {best_params}")

    if args.grid_only:
        print("\n--grid-only: skipping final evaluation.")
        return

    # ── Phase 2: Final evaluation on held-out test split only ───────────────
    print(f"\n{'=' * 60}")
    print(f"  Final evaluation: {len(test_stems)} held-out test images (val excluded)")
    print(f"{'=' * 60}")

    summary_rows = []
    for mi, method_name in enumerate(args.methods, 1):
        print(f"\n>>> FINAL EVAL {mi}/{len(args.methods)}: {method_name.upper()} <<<")
        result = evaluate_tuned(
            method_name=method_name,
            best_param=best_params[method_name],
            all_stems=test_stems,
            noisy_dict=noisy_dict,
            pscales=pscales,
            gt_dict=gt_dict,
            model=model,
            out_dir=args.out_dir,
        )

        # Load untuned AP@0.5 from existing results
        untuned_ap = load_untuned_ap50(args.ap_scores_dir, method_name)
        result["untuned_ap50"] = round(untuned_ap, 4)
        result["delta_ap50"] = round(result["tuned_ap50"] - untuned_ap, 4)
        summary_rows.append(result)

    # Save summary CSV
    summary_path = args.out_dir / "tuned_summary.csv"
    with open(summary_path, "w", newline="") as f:
        fields = ["method", "param_name", "default_value", "tuned_value",
                  "untuned_ap50", "tuned_ap50", "delta_ap50",
                  "mean_denoise_time_s"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # ── Print comparison table ───────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"  {'Method':<12} {'Param':<10} {'Default':>8} {'Tuned':>8} "
          f"{'Untuned AP':>11} {'Tuned AP':>9} {'Delta':>7} {'Time(s)':>8}")
    print(f"  {'-' * 72}")
    for r in summary_rows:
        print(f"  {r['method']:<12} {r['param_name']:<10} "
              f"{str(r['default_value']):>8} {str(r['tuned_value']):>8} "
              f"{r['untuned_ap50']:>11.4f} {r['tuned_ap50']:>9.4f} "
              f"{r['delta_ap50']:>+7.4f} {r['mean_denoise_time_s']:>8.3f}")
    print(f"{'=' * 78}")
    print(f"\nAll results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
