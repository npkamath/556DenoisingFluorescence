"""
pipeline_main.py  –  End-to-end PnP-HQS denoising + Cellpose segmentation pipeline.

Orchestrates full inference for Extension 2:

    1. Load noisy image        (data/noisy/poisson/ by default)
    2. PnP-HQS denoise         (VST → HQS loop → inverse VST, via pnp_solver.py)
    3. Cellpose segment        (frozen cyto2 model, channels=[2, 1])
    4. Compute AP metrics      (compare predicted mask against ground truth)
    5. Save outputs            (denoised .npy, mask .npy, per-image AP CSV)

Output CSV schema matches results/ap_scores/ produced by evaluate.py, so
PnP-HQS results are directly comparable with BM3D, PURE-LET, and Cellpose3
baselines in bootstrap_ci.py and generate_visuals.py.

Key differences from pipeline_main.py in the base project:
    - Uses the trained 1-channel CellposeDenoiserWrapper (not BM3D).
    - Forces blue channel (ch 2) to zero before denoising (it is empty).
    - Calls seg_model.eval() directly with SEG_CHANNELS=[2, 1] to guarantee
      correct channel assignment regardless of global defaults.

Usage:
    # Run on all 68 test images:
    python pipeline_main.py

    # Single image (debugging):
    python pipeline_main.py --single data/noisy/poisson/000.npy --pscale 5.7

    # Print per-iteration HQS residuals:
    python pipeline_main.py --verbose

    # Override HQS hyperparameters:
    python pipeline_main.py --n_iters 4 --mu_0 0.5 --mu_max 5.0 --rho 2.15

    # Use a specific checkpoint:
    python pipeline_main.py --denoiser_path models/pnp_denoiser/cellpose_vst_denoiser_epoch50.pth

Outputs (default paths):
    results/denoised/pnp_hqs/      denoised .npy images (H, W, 3) float32
    results/masks/pnp_hqs/         predicted instance masks (H, W) uint16
    results/ap_scores/pnp_hqs.csv  per-image AP at IoU 0.50–0.95
"""

import argparse
import csv
import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional
import imageio.v3 as iio

from cellpose import models as cp_models
from cellpose.metrics import average_precision

from pnp_solver import CellposeDenoiserWrapper, pnp_hqs_denoise, N_ITERS, MU_0, MU_MAX, RHO
from segment import run_cellpose
from evaluate import compute_ap as evaluate_compute_ap, IOU_THRESHOLDS

# ── Defaults ──────────────────────────────────────────────────────────────────
NOISY_DIR_DEFAULT    = Path("data/noisy/poisson")
PSCALE_CSV_DEFAULT   = Path("data/noise_params.csv")
MASKS_DIR_DEFAULT    = Path("data/masks")
DENOISED_OUT_DEFAULT = Path("results/denoised/pnp_hqs")
MASKS_OUT_DEFAULT    = Path("results/masks/pnp_hqs")
METRICS_CSV_DEFAULT  = Path("results/ap_scores/pnp_hqs.csv")
DENOISER_PATH_DEFAULT= "models/pnp_denoiser/cellpose_vst_denoiser_epoch50.pth"



# Cellpose segmentation settings  (cyto2, channels=[2,1] = Green cyto, Red nuc)
SEG_MODEL_TYPE = "cyto2"
SEG_CHANNELS   = [2, 1]
SEG_DIAMETER   = 0.0      # 0 → auto-estimate per image
SEG_FLOW_THRESH= 0.4
SEG_CELLPROB   = 0.0

# AP thresholds matching purelet.csv
AP_THRESHOLDS  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_pscales(csv_path: Path) -> dict[str, float]:
    result = {}
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            result[row["image"]] = float(row["pscale"])
    return result

def segment_image_standardized(img: np.ndarray, seg_model: cp_models.CellposeModel) -> np.ndarray:
    """
    Segment a denoised (H, W, 3) image using the loaded cyto2 model.

    Calls seg_model.eval() directly with SEG_CHANNELS=[2, 1] to ensure
    Green (ch 1) is treated as cytoplasm and Red (ch 0) as nucleus,
    matching the convention in segment.py.

    Returns (H, W) uint16 instance mask (0 = background).
    """
    # Grab all outputs into a single tuple, then just take the first item (masks)
    eval_results = seg_model.eval(
        img, 
        diameter=SEG_DIAMETER, 
        channels=SEG_CHANNELS,   # <--- Forces Cellpose to use [2, 1]
        flow_threshold=SEG_FLOW_THRESH,
        cellprob_threshold=SEG_CELLPROB
    )
    
    masks = eval_results[0]
    return masks.astype(np.uint16)


def compute_ap_standardized(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Compute AP at IoU thresholds 0.50–0.95 for a single image pair.

    Wraps evaluate.compute_ap and returns a flat dict keyed by "ap@T"
    matching the column format of the baseline CSVs in results/ap_scores/.
    """
    # evaluate.compute_ap returns (ap, tp, fp, fn)
    # ap shape: (n_images, n_thresholds)
    ap_array, _, _, _ = evaluate_compute_ap([gt_mask], [pred_mask], thresholds=AP_THRESHOLDS)
    
    return {
        f"ap@{t:.2f}": float(ap_array[0, i]) 
        for i, t in enumerate(AP_THRESHOLDS)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-image pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    noisy_path:   Path,
    pscale:       float,
    gt_mask_path: Path,
    denoiser:     CellposeDenoiserWrapper,
    seg_model:    cp_models.CellposeModel, # <--- Added to signature
    denoised_dir: Path,
    masks_dir:    Path,
    hqs_kwargs:   dict,
    verbose:      bool,
) -> dict:
    stem  = noisy_path.stem
    noisy = np.load(noisy_path).astype(np.float32)

    # Force the blue channel to be empty before passing to the 3-channel PnP
    if noisy.ndim == 3 and noisy.shape[-1] == 3:
        noisy[:, :, 2] = 0.0


    # 1. Denoise (PnP Logic)
    denoised = pnp_hqs_denoise(
        noisy    = noisy,
        pscale   = pscale,
        denoiser = denoiser,
        **hqs_kwargs,
    )
    np.save(denoised_dir / f"{stem}.npy", denoised)
    print(f"  [DEBUG] {stem} Denoised Min: {denoised.min():.6f}, Max: {denoised.max():.6f}")

    # 2. Segment (Now actually using seg_model and SEG_CHANNELS)
    pred_mask = segment_image_standardized(denoised, seg_model)
    np.save(masks_dir / f"{stem}.npy", pred_mask)

    # 3. Evaluation
    gt_mask = np.load(gt_mask_path).astype(np.uint16)
    ap_dict = compute_ap_standardized(pred_mask, gt_mask)

    result = {"image": stem, "pscale": f"{pscale:.4f}", **ap_dict}
    ap50 = ap_dict.get("ap@0.50", 0.0)
    print(f"  [✓] {stem:>12s}  AP@0.50={ap50:.4f}")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    noisy_dir:     Path,
    pscale_csv:    Path,
    masks_dir_gt:  Path,
    denoised_dir:  Path,
    masks_out_dir: Path,
    metrics_csv:   Path,
    denoiser_path: str,
    n_iters:       int,
    mu_0:          float,
    mu_max:        float,
    rho:           float,
    verbose:       bool,
    single:        Optional[Path],
    single_pscale: Optional[float],
) -> None:
    denoised_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load pscale lookup
    pscales = load_pscales(pscale_csv)

    # Load denoiser
    print(f"Loading PnP denoiser: {denoiser_path}")
    denoiser = CellposeDenoiserWrapper(
        denoiser_path, gpu=torch.cuda.is_available()
    )

    # Load segmentation model
    print(f"Loading Cellpose segmentation model: {SEG_MODEL_TYPE}\n")
    seg_model = cp_models.CellposeModel(
        model_type = SEG_MODEL_TYPE,
        gpu        = torch.cuda.is_available(),
    )

    hqs_kwargs = dict(n_iters=n_iters, mu_0=mu_0, mu_max=mu_max, rho=rho)

    # Determine which files to process
    if single is not None:
        files = [single]
        if single_pscale is None:
            stem = single.stem
            if stem not in pscales:
                raise ValueError(f"pscale for '{stem}' not found in {pscale_csv}. "
                                 "Pass --pscale explicitly.")
    else:
        files = sorted(noisy_dir.glob("*.npy"))
        if not files:
            print(f"No .npy files found in {noisy_dir}")
            return

    print(f"Processing {len(files)} image(s)\n")

    records = []
    for f in files:
        stem   = f.stem
        pscale = single_pscale if (single is not None and single_pscale) else pscales.get(stem)
        if pscale is None:
            print(f"  [skip] {stem} — pscale not found")
            continue

        gt_mask_path = masks_dir_gt / f"{stem}.npy"
        if not gt_mask_path.exists():
            print(f"  [skip] {stem} — ground-truth mask not found at {gt_mask_path}")
            continue

        rec = process_image(
            noisy_path   = f,
            pscale       = pscale,
            gt_mask_path = gt_mask_path,
            denoiser     = denoiser,
            seg_model    = seg_model,
            denoised_dir = denoised_dir,
            masks_dir    = masks_out_dir,
            hqs_kwargs   = hqs_kwargs,
            verbose      = verbose,
        )
        records.append(rec)

    if not records:
        print("No records produced — check your paths.")
        return

    # Write metrics CSV  (same schema as purelet.csv for easy comparison)
    fieldnames = ["image", "pscale"] + [f"ap@{t:.2f}" for t in AP_THRESHOLDS]
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Summary statistics
    ap50_vals = [float(r["ap@0.50"]) for r in records]
    print(f"\n{'─'*50}")
    print(f"  Images processed : {len(records)}")
    print(f"  Mean AP@0.50     : {np.mean(ap50_vals):.4f}")
    print(f"  Metrics CSV      : {metrics_csv}")
    print(f"  Denoised output  : {denoised_dir}")
    print(f"  Mask output      : {masks_out_dir}")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="PnP-HQS denoising + Cellpose segmentation pipeline"
    )
    p.add_argument("--noisy_dir",     type=Path,  default=NOISY_DIR_DEFAULT)
    p.add_argument("--pscale_csv",    type=Path,  default=PSCALE_CSV_DEFAULT)
    p.add_argument("--masks_dir",     type=Path,  default=MASKS_DIR_DEFAULT)
    p.add_argument("--denoised_out",  type=Path,  default=DENOISED_OUT_DEFAULT)
    p.add_argument("--masks_out",     type=Path,  default=MASKS_OUT_DEFAULT)
    p.add_argument("--metrics_csv",   type=Path,  default=METRICS_CSV_DEFAULT)
    p.add_argument("--denoiser_path", type=str,   default=DENOISER_PATH_DEFAULT)
    p.add_argument("--n_iters",       type=int,   default=N_ITERS)
    p.add_argument("--mu_0",          type=float, default=MU_0)
    p.add_argument("--mu_max",        type=float, default=MU_MAX)
    p.add_argument("--rho",           type=float, default=RHO)
    p.add_argument("--verbose",       action="store_true",
                   help="Print per-iteration HQS residuals")
    p.add_argument("--single",        type=Path,  default=None,
                   help="Process a single .npy file (for debugging)")
    p.add_argument("--pscale",        type=float, default=None,
                   help="pscale for --single mode (overrides CSV lookup)")
    args = p.parse_args()

    run_pipeline(
        noisy_dir     = args.noisy_dir,
        pscale_csv    = args.pscale_csv,
        masks_dir_gt  = args.masks_dir,
        denoised_dir  = args.denoised_out,
        masks_out_dir = args.masks_out,
        metrics_csv   = args.metrics_csv,
        denoiser_path = args.denoiser_path,
        n_iters       = args.n_iters,
        mu_0          = args.mu_0,
        mu_max        = args.mu_max,
        rho           = args.rho,
        verbose       = args.verbose,
        single        = args.single,
        single_pscale = args.pscale,
    )
