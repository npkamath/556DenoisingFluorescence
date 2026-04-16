#!/usr/bin/env python3
"""
VST-domain Wiener filter baseline for Poisson noise.

Pipeline:
  noisy(float) -> counts = noisy * pscale
  Anscombe VST -> Wiener (Gaussian assumption) -> inverse Anscombe
  -> output float in [0,1] saved to results/denoised/wiener/

Notes:
- Denoise only Red/Green channels (0,1). Keep Blue (empty) unchanged.
- Wiener is applied in the VST domain.
"""

import argparse
import csv
from pathlib import Path
import numpy as np

try:
    from scipy.signal import wiener
except ImportError as e:
    raise ImportError(
        "Missing dependency: scipy.\n"
        "Install in your active conda env:\n"
        "  pip install scipy\n"
        "Then re-run this script."
    ) from e


NOISY_DIR_DEFAULT = Path("data/noisy/poisson")
PSCALE_CSV_DEFAULT = Path("data/noise_params.csv")
OUT_DIR_DEFAULT = Path("results/denoised/wiener")


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


def wiener_channel(z: np.ndarray, mysize: int) -> np.ndarray:
    if np.ptp(z) < 1e-8:
        return z.astype(np.float32)
    z_f = wiener(z.astype(np.float32), mysize=(mysize, mysize))
    return z_f.astype(np.float32)


def denoise_one_image(noisy_float: np.ndarray, pscale: float, mysize: int) -> np.ndarray:
    counts = np.maximum(noisy_float.astype(np.float32) * float(pscale), 0.0)
    z = anscombe_forward(counts)

    z_out = z.copy()
    for ch in (0, 1):
        z_out[:, :, ch] = wiener_channel(z[:, :, ch], mysize=mysize)
    z_out[:, :, 2] = z[:, :, 2]

    counts_d = anscombe_inverse_safe(z_out)
    counts_d = np.maximum(counts_d, 0.0)

    out = counts_d / float(pscale)
    out = np.clip(out, 0.0, 1.0)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description="VST-domain Wiener filter baseline for Poisson noise.")
    ap.add_argument("--noisy_dir", type=Path, default=NOISY_DIR_DEFAULT)
    ap.add_argument("--pscale_csv", type=Path, default=PSCALE_CSV_DEFAULT)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--mysize", type=int, default=5, help="Wiener window size (odd int), e.g., 3/5/7.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.noisy_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {args.noisy_dir}")

    pscales = load_pscales(args.pscale_csv)

    print(f"Denoising {len(files)} images from {args.noisy_dir} -> {args.out_dir}")
    print(f"Wiener params: mysize={args.mysize}")

    for f in files:
        stem = f.stem
        noisy = np.load(f).astype(np.float32)
        out = denoise_one_image(noisy, pscales[stem], mysize=args.mysize)
        np.save(args.out_dir / f"{stem}.npy", out)
        if stem.isdigit() and (int(stem) % 10 == 0):
            print(f"  saved {stem}.npy")

    print("Done.")


if __name__ == "__main__":
    main()