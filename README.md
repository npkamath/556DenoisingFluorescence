# Denoising Methods in Fluorescence Microscopy (ECE 556)

- Benchmarks classical image denoising (Wiener, Poisson-TV, BM3D, PURE-LET) against Cellpose3 and self-supervised baselines on fluorescence microscopy data
- Evaluates denoising quality via downstream **instance segmentation** (AP@0.5) using the frozen Cellpose `cyto2` model
- Central question: how much of Cellpose3's advantage comes from task-specific learned restoration vs. what established model-based approaches already achieve?

## Setup

### Option A: pip only

```bash
pip install -r requirements.txt
```

### Option B: conda/mamba + pip (recommended)

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
  test/    ← contents of test.zip
  cyto2/   ← contents of cyto2 zip (optional)
```

Then run the data preparation script to generate `clean/`, `noisy/`, and `masks/`:
```bash
python src/data_prep.py
```

This produces:

| Directory | Format | Description |
|-----------|--------|-------------|
| `data/clean/` | `.npy` float32 (H, W, 3) | Original images, percentile-normalised |
| `data/noisy/` | `.npy` float32 (H, W, 3) | Poisson-degraded images (noise on cytoplasm channel only) |
| `data/masks/` | `.npy` uint16 (H, W) | Ground-truth instance segmentation masks (0 = background) |
| `data/noise_params.csv` | CSV | Per-image Poisson scale factor (pscale) used for noise generation |

Loading in Python:
```python
import numpy as np
clean = np.load("data/clean/000.npy")   # float32 (H, W, 3)
noisy = np.load("data/noisy/000.npy")   # float32 (H, W, 3)
mask  = np.load("data/masks/000.npy")   # uint16  (H, W)
```
