#!/usr/bin/env python3
"""
Cellpose3 denoising baseline.

Runs the pretrained Cellpose3 denoising network on noisy images and saves
the restored outputs. No retraining — uses the released weights.

The CellposeDenoiseModel with restore_type='denoise_cyto3' applies the
perceptual + segmentation-loss denoising model from Stringer & Pachitariu
(2025). It also segments internally, but we only keep the restored images
here; segmentation is handled separately by segment.py for consistency.

Channel convention:
  Cellpose channels=[2, 1] means cytoplasm=Green(ch1), nucleus=Red(ch0).

Usage:
    python src/denoisers/learned/cellpose3_denoise.py
    python src/denoisers/learned/cellpose3_denoise.py --noisy_dir data/noisy/cellpose3
    python src/denoisers/learned/cellpose3_denoise.py --noisy_dir data/noisy/poisson
"""

import argparse
import csv
import time
import numpy as np
from pathlib import Path

from cellpose.denoise import CellposeDenoiseModel


NOISY_DIR_DEFAULT = Path("data/noisy/cellpose3")
OUT_DIR_DEFAULT = Path("results/denoised/cellpose3")
RUNTIME_DIR_DEFAULT = Path("results/runtimes")


def denoise_all(
    noisy_dir: Path,
    out_dir: Path,
    runtime_dir: Path,
    gpu: bool = True,
    batch_size: int = 8,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(noisy_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {noisy_dir}")

    print(f"Loading Cellpose3 denoise model (restore_type='denoise_cyto3')...")
    model = CellposeDenoiseModel(
        gpu=gpu,
        model_type="cyto2",
        restore_type="denoise_cyto3",
        chan2_restore=True,
    )

    print(f"Denoising {len(files)} images from {noisy_dir} -> {out_dir}\n")

    timings = []
    for f in files:
        stem = f.stem
        img = np.load(f).astype(np.float32)

        t0 = time.time()
        # eval returns (masks, flows, styles, imgs)
        # imgs contains the restored/denoised images
        masks, flows, styles, imgs = model.eval(
            img,
            channels=[2, 1],
            diameter=None,
            batch_size=batch_size,
            normalize=True,
        )
        elapsed = time.time() - t0

        # imgs is the denoised image — save it
        denoised = np.array(imgs).astype(np.float32)
        # Handle shape: eval may return (C, H, W) or (H, W, C)
        if denoised.ndim == 3 and denoised.shape[0] in (1, 2, 3):
            denoised = denoised.transpose(1, 2, 0)

        # CellposeDenoiseModel.eval with channels=[2,1] internally reorders
        # to [cyto, nuc] and returns imgs in that order. We need to map
        # back to our convention: ch0=Red(nuc), ch1=Green(cyto), ch2=Blue(empty).
        #
        # Complication: 52/68 cyto2 images are single-channel (Green/cyto only,
        # no nucleus). For those, the model outputs 2 channels but only the
        # cyto channel (index 0 in model output) has real data. We detect
        # which input channels were active and map accordingly.
        H, W = img.shape[:2]
        ch0_active = img[:, :, 0].max() > 0.01  # nucleus (Red)
        ch1_active = img[:, :, 1].max() > 0.01  # cytoplasm (Green)

        restored = np.zeros((H, W, 3), dtype=np.float32)
        if denoised.ndim == 3 and denoised.shape[2] >= 2:
            # Model output: [cyto, nuc]
            if ch0_active and ch1_active:
                # Dual-channel: swap back to [nuc, cyto]
                restored[:, :, 0] = denoised[:, :, 1]  # nuc -> ch0 (Red)
                restored[:, :, 1] = denoised[:, :, 0]  # cyto -> ch1 (Green)
            elif ch1_active:
                # Single-channel (cyto only): model output ch0 = cyto
                restored[:, :, 1] = denoised[:, :, 0]  # cyto -> ch1 (Green)
            else:
                # Fallback
                restored[:, :, 0] = denoised[:, :, 0]
        elif denoised.ndim == 3 and denoised.shape[2] == 1:
            if ch1_active:
                restored[:, :, 1] = denoised[:, :, 0]
            else:
                restored[:, :, 0] = denoised[:, :, 0]
        denoised = restored

        np.save(out_dir / f"{stem}.npy", denoised)
        timings.append({"image": stem, "time_s": f"{elapsed:.3f}"})
        print(f"  {stem}  shape={denoised.shape}  "
              f"range=[{denoised.min():.3f}, {denoised.max():.3f}]  "
              f"time={elapsed:.2f}s")

    # Save timing CSV
    timing_path = runtime_dir / "cellpose3_denoise.csv"
    with open(timing_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["image", "time_s"])
        writer.writeheader()
        writer.writerows(timings)

    mean_time = np.mean([float(t["time_s"]) for t in timings])
    print(f"\nDone. {len(files)} denoised images saved to {out_dir}/")
    print(f"  mean time per image: {mean_time:.2f}s")
    print(f"  timings saved to {timing_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run pretrained Cellpose3 denoising on noisy images"
    )
    p.add_argument("--noisy_dir", type=Path, default=NOISY_DIR_DEFAULT,
                   help="directory with noisy .npy images")
    p.add_argument("--out_dir", type=Path, default=OUT_DIR_DEFAULT,
                   help="directory to save denoised .npy images")
    p.add_argument("--runtime_dir", type=Path, default=RUNTIME_DIR_DEFAULT,
                   help="directory to save timing CSV")
    p.add_argument("--no_gpu", action="store_true",
                   help="run on CPU instead of GPU")
    p.add_argument("--batch_size", type=int, default=8,
                   help="batch size for 224x224 patch inference")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    denoise_all(
        noisy_dir=args.noisy_dir,
        out_dir=args.out_dir,
        runtime_dir=args.runtime_dir,
        gpu=not args.no_gpu,
        batch_size=args.batch_size,
    )
