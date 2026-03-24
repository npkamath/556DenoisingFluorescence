"""
Segmentation using the frozen Cellpose cyto2 model (cellpose v3.x).

Runs the Cellpose cyto2 model on all .npy images in an input directory and saves
predicted instance segmentation masks to an output directory.

The model handles its own internal normalisation (1st–99th percentile), so
input images can be in any reasonable float range — clean, noisy, or denoised.

Uses model_type='cyto2' (ResNet-based U-Net) from cellpose v3.x to match
the Cellpose3 paper (Stringer & Pachitariu, 2025). This is NOT the CPSAM
model from cellpose v4.0+, which gives different AP scores.

Channel convention for cyto test images:
  Channel 0 (Red)   = nucleus
  Channel 1 (Green) = cytoplasm
  Channel 2 (Blue)  = empty

Cellpose channel spec is [cytoplasm, nucleus] where 0=gray, 1=R, 2=G, 3=B.
So we use channels=[2, 1]: cytoplasm=Green(2), nucleus=Red(1).

Usage:
    # Segment noisy images (default)
    python src/segment.py

    # Segment denoised images from a specific method
    python src/segment.py --input_dir results/denoised/wiener --output_dir results/pred_masks/wiener

    # Segment clean images (upper bound)
    python src/segment.py --input_dir data/clean --output_dir results/pred_masks/clean
"""

import argparse
import time
import numpy as np
from pathlib import Path
from cellpose.models import CellposeModel


# ── Default paths ─────────────────────────────────────────────────────────────
INPUT_DIR  = Path("data/noisy")
OUTPUT_DIR = Path("results/pred_masks/noisy")


def load_images(input_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    """
    Load all .npy images from a directory.

    Args:
        input_dir: directory containing .npy files, each of shape (H, W, 3) float32.

    Returns:
        images: list of numpy arrays.
        stems: list of filename stems (e.g. ["000", "001", ...]).
    """
    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    images, stems = [], []
    for f in npy_files:
        images.append(np.load(f))
        stems.append(f.stem)
    return images, stems


def run_cellpose(
    images: list[np.ndarray],
    gpu: bool = True,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Run the frozen Cellpose cyto2 model on a list of images.

    Uses model_type='cyto2' with default parameters. The model normalises
    each image internally (1st–99th percentile per channel).

    channels=[2, 1]: in Cellpose convention, 2=Green (cytoplasm),
    1=Red (nucleus). Maps to img[:,:,1] and img[:,:,0].

    Args:
        images: list of float32 arrays, each (H, W, 3).
        gpu: whether to use GPU.

    Returns:
        pred_masks: list of int32 arrays, each (H, W), where 0 = background
                    and 1, 2, ... = cell instance labels.
        timings: per-image inference time in seconds.
    """
    model = CellposeModel(gpu=gpu, model_type='cyto2')

    pred_masks = []
    timings = []
    for i, img in enumerate(images):
        t0 = time.time()
        masks, flows, styles = model.eval(img, diameter=None, channels=[2, 1],
                                          flow_threshold=0.4, cellprob_threshold=0.0)
        elapsed = time.time() - t0
        pred_masks.append(masks.astype(np.int32))
        timings.append(elapsed)
    return pred_masks, timings


def segment_directory(
    input_dir: Path,
    output_dir: Path,
    gpu: bool = True,
) -> None:
    """
    Load images from input_dir, segment with Cellpose, save predicted masks.

    Each predicted mask is saved as a .npy int32 array with the same filename
    stem as the input image.

    Also saves timing information to output_dir/timings.csv.

    Args:
        input_dir: directory with .npy image files.
        output_dir: directory to write predicted mask .npy files.
        gpu: whether to use GPU for inference.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading images from {input_dir}...")
    images, stems = load_images(input_dir)
    print(f"  {len(images)} images loaded.\n")

    print("Running Cellpose segmentation...")
    pred_masks, timings = run_cellpose(images, gpu=gpu)

    for stem, mask, t in zip(stems, pred_masks, timings):
        np.save(output_dir / f"{stem}.npy", mask)
        n_cells = len(np.unique(mask)) - 1
        print(f"  {stem}  cells={n_cells:3d}  time={t:.2f}s")

    # Save timing log
    import csv
    timing_path = output_dir / "timings.csv"
    with open(timing_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "time_s"])
        writer.writeheader()
        for stem, t in zip(stems, timings):
            writer.writerow({"image": stem, "time_s": f"{t:.3f}"})

    mean_time = np.mean(timings)
    print(f"\nDone. {len(pred_masks)} masks saved to {output_dir}/")
    print(f"  mean time per image: {mean_time:.2f}s")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run frozen Cellpose model on a directory of images"
    )
    p.add_argument("--input_dir",  type=Path, default=INPUT_DIR,
                   help="directory with .npy image files to segment")
    p.add_argument("--output_dir", type=Path, default=OUTPUT_DIR,
                   help="directory to save predicted masks")
    p.add_argument("--no_gpu", action="store_true",
                   help="run on CPU instead of GPU")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    segment_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gpu=not args.no_gpu,
    )
