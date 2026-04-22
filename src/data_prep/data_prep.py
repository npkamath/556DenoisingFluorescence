"""
Phase 1: Data Preparation & Noise Generation

Reads the 68 cyto test images and masks from data/raw/test/ and produces:
  data/clean/              - original images as float32 .npy, shape (H, W, 3)
  data/clean_normed/       - percentile-normalized clean (for cellpose3 comparison)
  data/masks/              - ground-truth instance segmentation masks, uint16
  data/noisy/poisson/      - Poisson-only noise (preserves raw Poisson statistics)
  data/noisy/cellpose3/    - Full cellpose3 noise (blur + downsample + Poisson)

Two noise modes are generated:

1. Poisson-only (data/noisy/poisson/):
   Clean images are first percentile-normalized (img_norm) to [0, 1] with full
   dynamic range, then Poisson noise is applied to ALL channels:
     noisy = Poisson(pscale * img_normed) / pscale
   This preserves raw Poisson statistics. To recover integer counts for Anscombe
   VST: counts = noisy * pscale. The img_norm parameters are NOT re-applied after
   noise, so the noise is genuinely Poisson-distributed.

2. Cellpose3 (data/noisy/cellpose3/):
   Uses cellpose.denoise.add_noise which applies Gaussian blur + downsampling +
   Poisson noise, then renormalizes via img_norm. This matches the noise model in
   Stringer & Pachitariu (2025) and gives the reported ~50% AP@0.5 drop.

Channel convention for cyto test images:
  Channel 0 (Red)   = nucleus
  Channel 1 (Green) = cytoplasm
  Channel 2 (Blue)  = empty
  Cellpose cyto2 uses channels=[2, 1] (Green=cyto, Red=nuc).

Noise scale (pscale) is sampled per-image from Gamma(alpha=4, beta=0.7), clipped
to min 1. Mean pscale ~5.7. The SAME pscale is used for both noise modes.

Usage:
    python src/data_prep/data_prep.py                       # generate both noise types
    python src/data_prep/data_prep.py --noise_mode poisson   # Poisson-only
    python src/data_prep/data_prep.py --noise_mode cellpose3 # cellpose3 noise only

Output files are named by their original index (e.g., 000.npy, 001.npy, ...).
"""

import argparse
import csv
import numpy as np
import imageio.v3 as iio
import torch
from pathlib import Path

from cellpose.denoise import img_norm, add_noise


# ── Default paths (relative to repo root) ────────────────────────────────────
RAW_TEST_DIR = Path("data/raw/test")
CLEAN_DIR    = Path("data/clean")
MASKS_DIR    = Path("data/masks")
NOISY_DIR    = Path("data/noisy")
NOISE_CSV    = Path("data/noise_params.csv")

SEED = 42


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """Load a PNG image. Returns uint8 array of shape (H, W, 3)."""
    img = iio.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return img


def load_mask(path: Path) -> np.ndarray:
    """Load a segmentation mask PNG. Returns uint16 array of shape (H, W)."""
    return iio.imread(path).astype(np.uint16)


def normalize_to_float(img: np.ndarray) -> np.ndarray:
    """Normalize a uint8 image to float32 in [0, 1] via /255."""
    return img.astype(np.float32) / 255.0


def percentile_normalize(img_float: np.ndarray) -> np.ndarray:
    """Apply cellpose-style percentile normalization (img_norm) to a single image.

    IMPORTANT: Uses .copy() because img_norm modifies its tensor in-place via
    reshape (which creates a view) + in-place subtract/divide.  torch.from_numpy
    shares memory with the source array, so without the copy the caller's
    original numpy array would be silently mutated.
    """
    t = torch.from_numpy(img_float.copy().transpose(2, 0, 1)).unsqueeze(0).float()
    t = img_norm(t)
    return t.squeeze(0).permute(1, 2, 0).numpy()


# ── Noise generation ──────────────────────────────────────────────────────────

def sample_pscales(n: int, alpha: float = 4.0, beta: float = 0.7) -> list[float]:
    """
    Sample n Poisson noise scales from Gamma(alpha, beta), clipped to min 1.

    All pscales are sampled upfront to avoid RNG interference between noise modes.
    """
    m = torch.distributions.gamma.Gamma(alpha, beta)
    samples = torch.clamp(m.rsample(sample_shape=(n,)), min=1.0)
    return [float(s.item()) for s in samples]


def apply_poisson_noise(img_normed: np.ndarray, pscale: float) -> np.ndarray:
    """
    Apply Poisson noise to a percentile-normalized image (all channels).

    The input should be percentile-normalized (via img_norm) so pixel values use
    the full [0, 1] range. Noise is applied to all non-empty channels.

    Noise model: noisy = Poisson(pscale * img_normed) / pscale
    Preserves raw Poisson statistics. Recover counts via: counts = noisy * pscale.
    """
    noisy = img_normed.copy()
    for ch in range(img_normed.shape[2]):
        ch_data = img_normed[:, :, ch].astype(np.float64)
        if ch_data.max() > 1e-6:  # skip empty channels
            counts = np.random.poisson(np.maximum(pscale * ch_data, 0.0))
            noisy[:, :, ch] = (counts / pscale).astype(np.float32)
    return noisy


def apply_cellpose3_noise(img_uint8: np.ndarray, pscale: float) -> np.ndarray:
    """
    Apply cellpose3's full noise model using the actual add_noise function.

    Input must be the raw uint8 image (NOT /255 normalized).  Using raw [0,255]
    values is essential: after blur + bilinear-upsample, tiny /255 values can
    drift below zero due to float imprecision, which causes torch.poisson to
    fail with "invalid Poisson rate".  With [0,255] inputs the Poisson rates
    are large enough that any sub-pixel negative from interpolation is
    negligible.

    add_noise applies Gaussian blur + downsampling + Poisson, then calls
    img_norm (percentile-normalize) at the end, so the output is in [0,1]-ish
    range regardless of the input scale.
    """
    t = torch.from_numpy(img_uint8.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0)
    pscale_t = pscale * torch.ones((1, 1, 1, 1))

    noisy_t = add_noise(t, pscale=pscale_t, poisson=1.0)

    return noisy_t.squeeze(0).permute(1, 2, 0).numpy()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def prepare_data(
    raw_dir: Path,
    clean_dir: Path,
    masks_dir: Path,
    noisy_dir: Path,
    noise_csv: Path,
    seed: int,
    noise_mode: str = "both",
) -> None:
    """
    Process all images in raw_dir and write clean, masks, and noisy datasets.

    Args:
        raw_dir:    directory containing NNN_img.png / NNN_masks.png pairs.
        clean_dir:  output directory for clean images (uint8/255 float32).
        masks_dir:  output directory for instance masks.
        noisy_dir:  base output directory for noisy images (subfolders created).
        noise_csv:  path to write per-image noise parameter log.
        seed:       global random seed for reproducibility.
        noise_mode: "poisson", "cellpose3", or "both".
    """
    clean_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    noise_csv.parent.mkdir(parents=True, exist_ok=True)

    do_poisson = noise_mode in ("poisson", "both")
    do_cellpose3 = noise_mode in ("cellpose3", "both")

    if do_poisson:
        poisson_dir = noisy_dir / "poisson"
        poisson_dir.mkdir(parents=True, exist_ok=True)
    if do_cellpose3:
        cellpose3_dir = noisy_dir / "cellpose3"
        cellpose3_dir.mkdir(parents=True, exist_ok=True)

    # Percentile-normalized clean (used by both Poisson mode and cellpose3 comparison)
    clean_normed_dir = clean_dir.parent / "clean_normed"
    clean_normed_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    img_files = sorted(f for f in raw_dir.glob("*_img.png")
                       if not f.name.endswith(":Zone.Identifier"))
    if not img_files:
        raise FileNotFoundError(f"No *_img.png files found in {raw_dir}")

    # Filter to only images that have matching masks
    valid_files = []
    for img_path in img_files:
        stem = img_path.stem.replace("_img", "")
        mask_path = raw_dir / f"{stem}_masks.png"
        if mask_path.exists():
            valid_files.append((img_path, mask_path, stem))
        else:
            print(f"  [skip] no mask for {img_path.name}")

    print(f"Found {len(valid_files)} images in {raw_dir}")
    print(f"Noise mode: {noise_mode}\n")

    # Sample all pscales upfront (deterministic, independent of noise mode)
    pscales = sample_pscales(len(valid_files))

    records = []
    for idx, (img_path, mask_path, stem) in enumerate(valid_files):
        img = load_image(img_path)
        mask = load_mask(mask_path)
        img_float = normalize_to_float(img)
        pscale = pscales[idx]

        # Save raw clean (uint8/255) and mask
        np.save(clean_dir / f"{stem}.npy", img_float)
        np.save(masks_dir / f"{stem}.npy", mask)

        # Save percentile-normalized clean
        img_normed = percentile_normalize(img_float)
        np.save(clean_normed_dir / f"{stem}.npy", img_normed)

        # Poisson-only noise (on percentile-normalized image)
        if do_poisson:
            noisy_p = apply_poisson_noise(img_normed, pscale)
            np.save(poisson_dir / f"{stem}.npy", noisy_p)

        # Cellpose3 full noise (pass raw uint8, not /255 — see docstring)
        if do_cellpose3:
            noisy_c = apply_cellpose3_noise(img, pscale)
            np.save(cellpose3_dir / f"{stem}.npy", noisy_c)

        n_cells = len(np.unique(mask)) - 1
        records.append({"image": stem, "pscale": f"{pscale:.4f}"})
        print(f"  {stem}  pscale={pscale:.3f}  img{img.shape} mask_cells={n_cells}")

    # Save noise parameters CSV
    with open(noise_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "pscale"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\nDone. {len(records)} images processed.")
    print(f"  clean/        — raw clean images (uint8/255 float32)")
    print(f"  clean_normed/ — percentile-normalized clean")
    print(f"  masks/        — ground-truth instance masks")
    if do_poisson:
        print(f"  noisy/poisson/   — Poisson-only noise (raw stats preserved)")
    if do_cellpose3:
        print(f"  noisy/cellpose3/ — Full cellpose3 noise (blur+downsample+Poisson)")
    print(f"  noise parameters → {noise_csv}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1: prepare clean, noisy, and mask datasets"
    )
    p.add_argument("--raw_dir",   type=Path, default=RAW_TEST_DIR,
                   help="directory with *_img.png / *_masks.png files")
    p.add_argument("--clean_dir", type=Path, default=CLEAN_DIR)
    p.add_argument("--masks_dir", type=Path, default=MASKS_DIR)
    p.add_argument("--noisy_dir", type=Path, default=NOISY_DIR)
    p.add_argument("--noise_csv", type=Path, default=NOISE_CSV)
    p.add_argument("--seed",      type=int,  default=SEED)
    p.add_argument("--noise_mode", type=str, default="both",
                   choices=["poisson", "cellpose3", "both"],
                   help="which noise type(s) to generate")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_data(
        raw_dir=args.raw_dir,
        clean_dir=args.clean_dir,
        masks_dir=args.masks_dir,
        noisy_dir=args.noisy_dir,
        noise_csv=args.noise_csv,
        seed=args.seed,
        noise_mode=args.noise_mode,
    )
