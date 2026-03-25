import argparse
import csv
from pathlib import Path

import numpy as np
from bm3d import bm3d, BM3DProfileHigh


NOISY_DIR_DEFAULT = Path("data/noisy/poisson")
PSCALE_CSV_DEFAULT = Path("data/noise_params.csv")
OUT_DIR_DEFAULT = Path("results/denoised/bm3d")


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


def load_pscales(csv_path: Path) -> dict[str, float]:
    pscale = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pscale[r["image"]] = float(r["pscale"])
    return pscale


def bm3d_denoise_channel(y: np.ndarray, sigma: float) -> np.ndarray:
    """
    Run BM3D on a single 2D array. If it's near-constant, skip to avoid bm3d issues.
    """
    if np.ptp(y) < 1e-8:
        return y.astype(np.float32)

    # Normalize to [0,1] for numerical stability
    y_min = float(y.min())
    y_max = float(y.max())
    y01 = (y - y_min) / (y_max - y_min)
    sigma01 = sigma / (y_max - y_min)

    y01_d = bm3d(y01.astype(np.float32), sigma_psd=float(sigma01), profile=BM3DProfileHigh())
    y_d = y01_d * (y_max - y_min) + y_min
    return y_d.astype(np.float32)


def denoise_one_image(noisy_float: np.ndarray, pscale: float, sigma_vst: float) -> np.ndarray:
    """
    noisy_float: (H,W,3) float32 from data/noisy/poisson
    We denoise only R and G channels; keep B channel unchanged (it's empty).
    """
    counts = np.maximum(noisy_float.astype(np.float32) * float(pscale), 0.0)
    y = anscombe_forward(counts)  # (H,W,3)

    out_y = y.copy()

    # Channel mapping from your data convention:
    # ch0=Red, ch1=Green, ch2=Blue(empty)
    for ch in [0, 1]:
        out_y[:, :, ch] = bm3d_denoise_channel(y[:, :, ch], sigma=sigma_vst)

    # leave blue channel as-is
    out_y[:, :, 2] = y[:, :, 2]

    counts_d = anscombe_inverse_safe(out_y)
    counts_d = np.maximum(counts_d, 0.0)

    out = counts_d / float(pscale)
    out = np.clip(out, 0.0, 1.0)

    # hard safety: kill any remaining NaN/Inf
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description="BM3D denoising via Anscombe VST (safe)")
    ap.add_argument("--noisy_dir", type=Path, default=NOISY_DIR_DEFAULT)
    ap.add_argument("--pscale_csv", type=Path, default=PSCALE_CSV_DEFAULT)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--sigma_vst", type=float, default=1.0,
                    help="Gaussian sigma assumed in Anscombe domain (default 1.0)")
    args = ap.parse_args()

    pscales = load_pscales(args.pscale_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.noisy_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {args.noisy_dir}")

    print(f"Denoising {len(files)} images from {args.noisy_dir} -> {args.out_dir}")
    print(f"Using sigma_vst={args.sigma_vst}")

    for f in files:
        stem = f.stem
        noisy = np.load(f).astype(np.float32)
        out = denoise_one_image(noisy, pscales[stem], sigma_vst=args.sigma_vst)
        np.save(args.out_dir / f"{stem}.npy", out)
        if int(stem) % 10 == 0:
            print(f"  saved {stem}.npy")

    print("Done.")


if __name__ == "__main__":
    main()