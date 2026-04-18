"""
data_prep_train.py  –  Training-pair generator for the PnP denoiser (Extension 2).

Reads clean _img.png files from train_cyto2/ (images 540–795), applies the
Anscombe VST, corrupts with synthetic Gaussian noise, and writes
(noisy_vst, clean_vst) pairs as single-channel .npy arrays.

This is SEPARATE from data_prep.py (which generates the test evaluation set).
Key differences from data_prep.py:
    Source  : data/raw/train_cyto2/, not data/raw/test/
    Domain  : VST domain (Gaussian, σ≈1), not raw Poisson counts
    Output  : single-channel (H, W) .npy per channel, not (H, W, 3)
    Purpose : supervised denoiser training, not segmentation evaluation

Pipeline per image:
    1. Load uint8 PNG, normalize to [0, 1] via /255 (matches data_prep.py).
    2. Extract Red (ch 0) and Green (ch 1) independently; skip empty channels.
    3. Percentile-normalize each channel slice to [0, 1] via cellpose img_norm.
    4. Convert to Poisson counts using PSCALE_REF (fixed reference scale).
    5. Apply Anscombe VST → clean_vst (H, W) float32.
    6. For each augmentation: add Gaussian noise sampled from
       Uniform(σ_min, σ_max) → noisy_vst (H, W) float32.
    7. Save paired .npy files and log to train_noise_params.csv.

Noise schedule:
    σ is sampled from Uniform(σ_min, σ_max) in the VST domain.
    The default range [0.5, 2.0] covers the σ≈1 expected from Anscombe
    output and ensures the denoiser generalises across pscale values at
    test time.

Output:
    data/train_pairs/
        noisy/  <stem>_ch<C>_aug<N>.npy   float32 (H, W), VST domain
        clean/  <stem>_ch<C>.npy          float32 (H, W), VST domain
    data/train_noise_params.csv            pair, clean_stem, sigma, pscale

Usage:
    python src/data_prep_train.py
    python src/data_prep_train.py --sigma_min 0.5 --sigma_max 2.0 --n_augments 4
    python src/data_prep_train.py --raw_dir data/raw/train_cyto2 --seed 0
"""

import argparse
import csv
import numpy as np
import imageio.v3 as iio
import torch
from pathlib import Path

from cellpose.denoise import img_norm
from vst_math import anscombe, to_counts

TRAIN_RAW_DIR   = Path("data/raw/train_cyto2")
TRAIN_PAIRS_DIR = Path("data/train_pairs")
TRAIN_CSV       = Path("data/train_noise_params.csv")

SIGMA_MIN   = 0.5   
SIGMA_MAX   = 2.0   
N_AUGMENTS  = 4     
SEED        = 42
PSCALE_REF  = 10.0  

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def normalize_to_float(img: np.ndarray) -> np.ndarray:
    """Normalize a uint8 image to float32 in [0, 1] via /255."""
    return img.astype(np.float32) / 255.0

def percentile_normalize(img_float: np.ndarray) -> np.ndarray:
    """Apply cellpose-style percentile normalization to a 1-channel [H, W] image."""
    # Add Channel and Batch dimensions -> [1, 1, H, W]
    t = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0).float()
    t = img_norm(t)
    # Remove extra dimensions -> [H, W]
    return t.squeeze(0).squeeze(0).numpy()

def main():
    p = argparse.ArgumentParser(description="Generate 1-Channel VST training pairs")
    p.add_argument("--raw_dir",    type=Path,  default=TRAIN_RAW_DIR)
    p.add_argument("--pairs_dir",  type=Path,  default=TRAIN_PAIRS_DIR)
    p.add_argument("--train_csv",  type=Path,  default=TRAIN_CSV)
    p.add_argument("--sigma_min",  type=float, default=SIGMA_MIN)
    p.add_argument("--sigma_max",  type=float, default=SIGMA_MAX)
    p.add_argument("--n_augments", type=int,   default=N_AUGMENTS)
    p.add_argument("--pscale",     type=float, default=PSCALE_REF)
    p.add_argument("--seed",       type=int,   default=SEED)
    args = p.parse_args()

    set_seed(args.seed)
    noisy_dir = args.pairs_dir / "noisy"
    clean_dir = args.pairs_dir / "clean"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total = 0

    files = sorted(list(args.raw_dir.glob("*_img.png")))
    for f in files:
        stem = f.stem.replace("_img", "")
        img = iio.imread(f)
        
        # Step 1: Divide by 255.0 EXACTLY like data_prep.py
        img_float = normalize_to_float(img)
        
        # Step 2: Extract R and G channels independently
        channels = []
        if img_float.ndim == 2:
            channels = [img_float] 
        else:
            channels = [img_float[:, :, 0], img_float[:, :, 1]]

        for ch_idx, img_ch in enumerate(channels):
            if np.ptp(img_ch) < 1e-8:
                continue
            
            # Step 3: Percentile normalize the 1-channel slice
            img_normed = percentile_normalize(img_ch)
            counts = np.maximum(img_normed * args.pscale, 0.0)
            clean_vst = anscombe(counts)
            
            clean_name = f"{stem}_ch{ch_idx}.npy"
            np.save(clean_dir / clean_name, clean_vst)

            for aug in range(args.n_augments):
                sigma = np.random.uniform(args.sigma_min, args.sigma_max)
                noise = np.random.randn(*clean_vst.shape) * sigma
                noisy_vst = clean_vst + noise.astype(np.float32)

                pair_name = f"{stem}_ch{ch_idx}_aug{aug}.npy"
                np.save(noisy_dir / pair_name, noisy_vst)

                records.append({
                    "pair": pair_name,
                    "clean_stem": clean_name,
                    "sigma": f"{sigma:.4f}",
                    "pscale": f"{args.pscale:.4f}",
                })
                total += 1

    with open(args.train_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "clean_stem", "sigma", "pscale"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {total} single-channel training pairs.")

if __name__ == "__main__":
    main()
