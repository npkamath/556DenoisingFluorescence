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
| `cyto2` (181 MB) | 256 images + masks | Optional — used for hyperparameter tuning (Extensions) |

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

## Classical Denoisers 

These two baselines apply **Anscombe VST** to convert Poisson counts into an approximately Gaussian (unit-variance) domain, run a Gaussian-assumption denoiser, then apply a **numerically-safe inverse VST** back to the original intensity range.

**Channel convention:** ch0 = Red (nucleus), ch1 = Green (cytoplasm), ch2 = Blue (empty).  
**Policy:** denoise Red/Green only; keep Blue unchanged to avoid degeneracies on an (almost) constant channel.

### 1) Poisson-TV (VST + ROF-TV via Chambolle)

- Script: `src/denoise_poisson_tv.py`
- Idea: in VST domain, solve a TV-regularized denoising problem (edge-preserving smoothing).
- Dependency: `scikit-image` (for `denoise_tv_chambolle`)

Run:
```bash
# Denoise (Poisson-only noise)
python src/denoise_poisson_tv.py --weight 0.10 --n_iter 200
```

Notes:

* --weight controls smoothness strength in the VST domain (larger = smoother).
* Implementation normalizes each channel to [0,1] inside the VST domain for stability, then maps back.



### 2) Wiener filter (VST + local Wiener smoothing)

* Script: src/denoise_wiener_vst.py
* Idea: local adaptive smoothing using neighborhood statistics (fast classical baseline).
* Dependency: scipy (for Wiener filtering)
```bash
# Denoise (Poisson-only noise)
python src/denoise_wiener_vst.py --mysize 5

# Segment + evaluate (frozen cyto2)
python src/segment.py --input_dir results/denoised/wiener --output_dir results/pred_masks/wiener --no_gpu
python src/evaluate.py --pred_dir results/pred_masks/wiener --method_name wiener
```
Notes:

* --mysize is the square window size (odd integer). Smaller windows preserve edges more but denoise less.
* Same safe inverse VST + clipping is applied to guarantee finite outputs in [0,1].



### 3) PURE-LET (VST + SWT + Stein's Unbiased Risk Estimator)

- Script: `src/denoise_purelet_swt.py`
- Idea: decomposes each channel with an undecimated (shift-invariant) SWT, then analytically solves for the optimal linear combination of threshold basis functions using Stein's unbiased Poisson risk estimate — no manual threshold tuning needed.
- Dependency: `pywt` (PyWavelets)

Run:
```bash
# Denoise (Poisson-only noise)
python src/denoise_purelet_swt.py --wavelet sym4 --n_levels 4

# Segment + evaluate (frozen cyto2)
python src/segment.py --input_dir results/denoised/purelet_swt --output_dir results/pred_masks/purelet_swt --no_gpu
python src/evaluate.py --pred_dir results/pred_masks/purelet_swt --method_name purelet_swt
```

Notes:

* `--wavelet` sets the mother wavelet (default `sym4`; `db4` and `bior2.2` are reasonable alternatives).
* `--n_levels` controls the decomposition depth (default 4). More levels capture lower-frequency structure but increase runtime.
* SWT avoids the ringing and shift-sensitivity artifacts of standard DWT; no cycle-spinning is needed.



## Segment + evaluate (frozen cyto2)
python src/segment.py --input_dir results/denoised/poisson_tv --output_dir results/pred_masks/poisson_tv --no_gpu
python src/evaluate.py --pred_dir results/pred_masks/poisson_tv --method_name poisson_tv

## Extension 2: Learned PnP Denoiser
Extension 2 replaces BM3D's hand-crafted prior with a neural denoiser trained
in the Anscombe VST domain and plugged into the same HQS framework.

### Training

```bash
# 1. Generate VST-domain training pairs from the cyto2 training set
python src/data_prep_train.py

# 2. Train the denoiser (checkpoints saved every 25 epochs)
python src/cellpose_trainer.py --n_epochs 100
```

### Inference and evaluation

```bash
# Denoise + segment all 68 test images with the epoch-75 checkpoint
python src/pipeline_main.py \
    --denoiser_path models/pnp_denoiser/cellpose_vst_denoiser_epoch75.pth \
    --metrics_csv results/ap_scores/pnp_hqs_epoch75.csv

# Compare against other methods via bootstrap CI
python src/bootstrap_ci.py --baseline noisy --method pnp_hqs_epoch75 bm3d purelet
python src/scatter_delta_plots.py --methods pnp_hqs_epoch75 bm3d purelet cellpose3
```

## Source Files

| File | Purpose |
|------|---------|
| `src/data_prep.py` | Generates clean, noisy, and mask datasets from raw PNGs. Produces two noise variants: **Poisson-only** (preserves raw Poisson statistics for classical filters) and **cellpose3** (blur + downsample + Poisson, matches the paper). Per-image noise scale (pscale) is logged to CSV. |
| `src/segment.py` | Runs the frozen Cellpose `cyto2` model on a directory of `.npy` images. Outputs predicted instance masks and per-image timing. Uses `channels=[2, 1]` (Green=cytoplasm, Red=nucleus). |
| `src/evaluate.py` | Computes AP@0.5 (and AP at IoU 0.5–0.95) by comparing predicted masks against ground truth. Outputs per-image CSV and summary statistics. Use `--all` to evaluate every method subdirectory at once. |
| `src/bootstrap_ci.py` | Paired bootstrap confidence intervals for mean AP@0.5 differences. Reads per-image CSVs from `evaluate.py`. For each (baseline, method) pair, computes the observed Δ = mean(method AP@0.5) − mean(baseline AP@0.5), a 95% percentile bootstrap CI (10,000 resamples at the image level), and a one-sided p-value for H₀: Δ ≤ 0. Writes a per-pair text report, a bootstrap Δ histogram PNG, and a `bootstrap_summary.csv` table. Use `--baseline noisy --method cellpose3 wiener ...` to compare any set of methods in one call. |
| `src/freq_residual_analysis.py` | Frequency-domain residual analysis for denoising methods. For each method, computes the residual (denoised − clean_normed) and produces: (1) a radially averaged power spectral density (PSD) plot vs. the noisy baseline; (2) a mean 2D log-power spectrum (DC-centred, Hann-windowed); (3) a cross-method PSD ratio figure (method / noisy — values < 1 indicate better noise removal at that frequency); and (4) a band-averaged power summary table and bar chart split into low (< 10% Nyquist), mid (10–30%), and high (> 30%) frequency bands. Only the active Red and Green channels are analysed; images are zero-padded to the next power of 2 before FFT. Reads from `data/clean_normed/`, `data/noisy/poisson/`, and `results/denoised/<method>/`; saves all figures under `figures/freq/`. |
| `src/scatter_delta_plots.py` | Per-image scatter and Δ bar plots for AP@0.5 comparisons. For each method produces: (1) a scatter plot of method vs. baseline AP@0.5, with points coloured by Δ on a diverging colormap and an annotation of how many images improved vs. degraded; (2) a per-image Δ sorted bar chart (green = improvement, red = degradation) with optional 95% bootstrap CI shading loaded from `bootstrap_summary.csv`; and (3) optionally, a scatter of each method vs. Cellpose3 (`--vs_cellpose3`). Also generates an all-methods summary panel. Reads AP CSVs from `results/ap_scores/` and saves figures under `figures/scatter/`. |
| `src/denoise_purelet_swt.py` | PURE-LET denoising via Stationary Wavelet Transform (SWT). Implements the PURE-LET algorithm (Poisson Unbiased Risk Estimator – Linear Expansion of Thresholds) using an undecimated, shift-invariant SWT rather than a standard DWT. For each non-empty channel, converts pixel values to photon counts via pscale, decomposes with `pywt.swt2` (sym4, 4 levels), and applies an interscale thresholding scheme: coefficients are modulated with a sigmoidal gate function jointly with an interscale predictor derived from same-scale LL subband gradients (horizontal → `grad_h(LL)`, vertical → `grad_v(LL)`, diagonal → both). Optimal LET coefficients are solved analytically via a 2×2 linear system (PURE divergence criterion) with Tikhonov regularisation for numerical stability in flat regions. The shape parameter t is set adaptively per image from the estimated noise floor. Outputs are clipped to `[0, 1]` and saved as float32 `.npy`. Usage: `python src/denoise_purelet_swt.py [--wavelet sym4] [--n_levels 4]`. |
| `src/vst_math.py` | Shared VST utilities (Extension 2). Centralises the forward Anscombe transform (`anscombe`) and the exact unbiased inverse (`anscombe_inverse_exact`, Makitalo & Foi 2013) used by both the classical BM3D pipeline and the PnP denoiser. Also provides to_counts / from_counts helpers for converting between normalised float images and photon counts. Imported by `denoise_bm3d_vst.py`, `data_prep_train.py`, and `pnp_solver.py` to guarantee mathematical consistency across the pipeline. |
|`src/data_prep_train.py` | Training-pair generator for the PnP denoiser (Extension 2). Reads clean images from `data/raw/train_cyto2/` (separate from the 68-image test set), extracts Red and Green channels independently, applies percentile normalisation then the Anscombe VST, and corrupts each channel with synthetic Gaussian noise sampled from Uniform(σ_min, σ_max) in the VST domain. Produces single-channel (`noisy_vst`, `clean_vst`) pairs as `.npy` arrays under `data/train_pairs/`, with a metadata CSV logging sigma and pscale per pair. Supports `--n_augments` (default 4) to generate multiple noise realisations per image. Must be run before `cellpose_trainer.py`. |
| `src/cellpose_trainer.py` | Learned denoiser training script (Extension 2). Fine-tunes a single-channel Cellpose `DenoiseModel` on the VST-domain pairs produced by `data_prep_train.py`. Initialises from `denoise_cyto3` weights and trains with AdamW + MSE loss. Checkpoints are saved every 25 epochs under `models/pnp_denoiser/`; the final weights are consumed by `pnp_solver.py` as the denoiser prior in the PnP-HQS loop. Usage: `python src/cellpose_trainer.py`. |
| `src/pnp_solver.py` | Plug-and-Play HQS solver (Extension 2). Implements the PnP-HQS algorithm for Poisson image restoration. Alternates between a closed-form Wiener-style data-fidelity step in the Anscombe VST domain (`z = (μ·v + y_vst) / (μ + 1)`) and a denoiser prior step via the trained CellposeDenoiserWrapper. The penalty parameter μ is annealed geometrically from `μ_0` to `μ_max` over `N_ITERS` (default 4) iterations. Red and Green channels are processed independently; the empty Blue channel is passed through unchanged. `CellposeDenoiserWrapper` pads inputs to multiples of 32 for the network and crops back after inference. |
| `src/pipeline_main.py` | End-to-end PnP inference pipeline (Extension 2). Orchestrates the full denoise → segment → evaluate flow for the learned PnP-HQS method. Loads noisy images and pscale values, runs `pnp_hqs_denoise` from `pnp_solver.py`, segments the denoised output with the frozen `cyto2` model (`channels=[2, 1]`), and computes AP at IoU thresholds 0.5–0.95 against ground-truth masks. Writes a metrics CSV in the same schema as the classical method CSVs for direct comparison via bootstrap_ci.py and scatter_delta_plots.py. Supports `--single` mode for per-image debugging and configurable HQS hyperparameters (`--n_iters`, `--mu_0`, `--mu_max`, `--rho`). Usage: `python src/pipeline_main.py`. |
