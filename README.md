# Denoising Methods in Fluorescence Microscopy (ECE 556)

- Benchmarks classical image denoising (Wiener, Poisson-TV, BM3D, PURE-LET) against Cellpose3 and self-supervised baselines on fluorescence microscopy data
- Evaluates denoising quality via downstream **instance segmentation** (AP@0.5) using the frozen Cellpose `cyto2` model
- Central question: how much of Cellpose3's advantage comes from task-specific learned restoration vs. what established model-based approaches already achieve?

## Setup

### Option A: pip only

```bash
pip install -r requirements.txt
```

### Option B: conda/mamba + pip (I use mamba personally, but pip should be simpler)

```bash
mamba create -n 556Proj python=3.10 -y
mamba activate 556Proj
mamba install numpy scipy scikit-image matplotlib seaborn pandas tifffile imageio -c conda-forge -y
pip install torch torchvision cellpose bm3d
```

---

## Dataset

Go to https://www.cellpose.org/dataset and download the following:

| Zip | Contents | Purpose |
|-----|----------|---------|
| `test.zip` | 68 images + masks | **Required** — main evaluation set |
| `cyto2` (181 MB) | 256 images + masks | Optional — used for hyperparameter tuning (Extension 1) |

> `train.zip` is not needed.

Extract and place the files into:
```
data/raw/
  test/    ← contents of test.zip (NNN_img.png + NNN_masks.png pairs)
  cyto2/   ← contents of cyto2 zip (optional)
```

Then run the data preparation script:
```bash
python src/data_prep.py                        # generate both noise types (default)
python src/data_prep.py --noise_mode poisson   # Poisson-only (sufficient for most work)
python src/data_prep.py --noise_mode cellpose3 # cellpose3 noise only
```

This produces:

| Directory | Format | Description |
|-----------|--------|-------------|
| `data/clean/` | `.npy` float32 (H, W, 3) | Original images (uint8 / 255) |
| `data/clean_normed/` | `.npy` float32 (H, W, 3) | Percentile-normalized clean (reference for residual analysis) |
| `data/noisy/poisson/` | `.npy` float32 (H, W, 3) | **Poisson-only noise** — primary input for all denoising methods |
| `data/noisy/cellpose3/` | `.npy` float32 (H, W, 3) | Cellpose3 noise (blur+downsample+Poisson) — optional comparison |
| `data/masks/` | `.npy` uint16 (H, W) | Ground-truth instance segmentation masks (0 = background) |
| `data/noise_params.csv` | CSV | Per-image pscale values (needed by classical filters) |

**Channel convention** (all images): ch0 = Red = nucleus, ch1 = Green = cytoplasm, ch2 = Blue = empty.

Loading in Python:
```python
import numpy as np
noisy = np.load("data/noisy/poisson/000.npy")  # float32 (H, W, 3) — input for denoising
clean = np.load("data/clean_normed/000.npy")   # float32 (H, W, 3) — clean reference
mask  = np.load("data/masks/000.npy")          # uint16  (H, W)    — ground truth
```

To recover integer Poisson counts for classical filters (Anscombe VST, etc.):
```python
import csv
with open("data/noise_params.csv") as f:
    pscale = {r["image"]: float(r["pscale"]) for r in csv.DictReader(f)}
counts = noisy * pscale["000"]  # non-negative integers
```

---

## Pipeline: Denoise → Segment → Evaluate

All methods follow the same pipeline so AP@0.5 scores are directly comparable:

```
data/noisy/poisson/  →  denoise  →  results/denoised/{method}/
                                            │
                                            ▼
                                     segment.py  →  results/pred_masks/{method}/
                                                            │
                                                            ▼
                                                     evaluate.py  →  AP@0.5
```

After saving denoised images to `results/denoised/<method>/`:

```bash
# Segment denoised images with frozen cyto2 model
python src/segment.py \
    --input_dir results/denoised/wiener \
    --output_dir results/pred_masks/wiener

# Compute AP@0.5 against ground truth
python src/evaluate.py \
    --pred_dir results/pred_masks/wiener \
    --method_name wiener

# Or evaluate all methods at once
python src/evaluate.py \
    --pred_dir results/pred_masks \
    --all
```

To establish baselines (clean ceiling, noisy floor):
```bash
# Clean ceiling
python src/segment.py --input_dir data/clean --output_dir results/pred_masks/clean
python src/evaluate.py --pred_dir results/pred_masks/clean --method_name clean

# Noisy floor
python src/segment.py --input_dir data/noisy/poisson --output_dir results/pred_masks/noisy
python src/evaluate.py --pred_dir results/pred_masks/noisy --method_name noisy
```

---

## Source Files

| File | Purpose |
|------|---------|
| `src/data_prep.py` | Generates clean, noisy, and mask datasets from raw PNGs. Produces two noise variants: **Poisson-only** (preserves raw Poisson statistics for classical filters) and **cellpose3** (blur + downsample + Poisson, matches the paper). Per-image noise scale (pscale) is logged to CSV. |
| `src/segment.py` | Runs the frozen Cellpose `cyto2` model on a directory of `.npy` images. Outputs predicted instance masks and per-image timing. Uses `channels=[2, 1]` (Green=cytoplasm, Red=nucleus). |
| `src/evaluate.py` | Computes AP@0.5 (and AP at IoU 0.5–0.95) by comparing predicted masks against ground truth. Outputs per-image CSV and summary statistics. Use `--all` to evaluate every method subdirectory at once. |
