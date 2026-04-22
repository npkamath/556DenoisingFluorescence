#!/usr/bin/env python3
"""
VST-domain TV denoising baseline for Poisson noise.

Pipeline:
  noisy (float) -> counts = noisy * pscale
  Anscombe VST -> TV (Gaussian assumption) -> inverse Anscombe
  -> output float in [0,1] saved to results/denoised/poisson_tv/

Notes:
- We denoise only Red/Green channels (0,1). Blue channel is empty and left unchanged.
- TV is applied in the VST domain with a tunable weight parameter.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from skimage.restoration import denoise_tv_chambolle
except ImportError as e:
    raise ImportError(
        "Missing dependency: scikit-image.\n"
        "Install it in your active conda env:\n"
        "  pip install scikit-image\n"
        "Then re-run this script."
    ) from e

NOISY_DIR_DEFAULT = Path("data/noisy/poisson")
PSCALE_CSV_DEFAULT = Path("data/noise_params.csv")
OUT_DIR_DEFAULT = Path("results/denoised/poisson_tv")


def load_pscales(csv_path: Path) -> dict[str, float]:
    pscale = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pscale[r["image"]] = float(r["pscale"])
    return pscale


def anscombe_forward(counts: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(np.maximum(counts, 0.0) + 3.0 / 8.0)


def anscombe_inverse_safe(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Safe polynomial approx inverse Anscombe (avoids division by zero).
    For very small y, returns 0.
    """
    y = np.maximum(y, 0.0)
    y_safe = np.maximum(y, eps)

    inv = (
        (y_safe / 2.0) ** 2
        - 1.0 / 8.0
        + 1.0 / (4.0 * y_safe)
        - 5.0 / (8.0 * y_safe**2)
        + 1.0 / (8.0 * y_safe**3)
    )
    inv = np.where(y <= eps, 0.0, inv)
    return inv


def tv_denoise_channel(z: np.ndarray, weight: float, n_iter: int) -> np.ndarray:
    """
    Apply TV denoising to one 2D channel in the VST domain.
    We normalize to [0,1] for numerical stability, then map back.
    """
    if np.ptp(z) < 1e-8:
        return z.astype(np.float32)

    z_min = float(z.min())
    z_max = float(z.max())
    z01 = (z - z_min) / (z_max - z_min)

    z01_d = denoise_tv_chambolle(
        z01.astype(np.float32),
        weight=float(weight),
        max_num_iter=int(n_iter),
        channel_axis=None,
    )

    z_d = z01_d * (z_max - z_min) + z_min
    return z_d.astype(np.float32)


def denoise_one_image(noisy_float: np.ndarray, pscale: float, weight: float, n_iter: int) -> np.ndarray:
    """
    noisy_float: (H,W,3) float32 from data/noisy/poisson
    """
    counts = np.maximum(noisy_float.astype(np.float32) * float(pscale), 0.0)
    z = anscombe_forward(counts)  # (H,W,3)

    z_out = z.copy()

    # ch0=Red, ch1=Green, ch2=Blue(empty)
    for ch in [0, 1]:
        z_out[:, :, ch] = tv_denoise_channel(z[:, :, ch], weight=weight, n_iter=n_iter)

    # Keep blue channel unchanged
    z_out[:, :, 2] = z[:, :, 2]

    counts_d = anscombe_inverse_safe(z_out)
    counts_d = np.maximum(counts_d, 0.0)

    out = counts_d / float(pscale)
    out = np.clip(out, 0.0, 1.0)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description="VST-domain TV denoising for Poisson noise (baseline).")
    ap.add_argument("--noisy_dir", type=Path, default=NOISY_DIR_DEFAULT)
    ap.add_argument("--pscale_csv", type=Path, default=PSCALE_CSV_DEFAULT)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--weight", type=float, default=0.10,
                    help="TV weight in VST domain (larger = smoother). Try 0.05~0.20.")
    ap.add_argument("--n_iter", type=int, default=200,
                    help="TV iterations (Chambolle).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.noisy_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {args.noisy_dir}")

    pscales = load_pscales(args.pscale_csv)

    print(f"Denoising {len(files)} images from {args.noisy_dir} -> {args.out_dir}")
    print(f"TV params: weight={args.weight}, n_iter={args.n_iter}")

    for f in files:
        stem = f.stem
        noisy = np.load(f).astype(np.float32)
        out = denoise_one_image(noisy, pscales[stem], weight=args.weight, n_iter=args.n_iter)
        np.save(args.out_dir / f"{stem}.npy", out)
        if int(stem) % 10 == 0:
            print(f"  saved {stem}.npy")

    print("Done.")


if __name__ == "__main__":
    main()