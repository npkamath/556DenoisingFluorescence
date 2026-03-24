# Denoising Methods in Fluorescence Microscopy (ECE 556)

- Benchmarks classical image denoising (Wiener, Poisson-TV, BM3D, PURE-LET) against Cellpose3 and self-supervised baselines on fluorescence microscopy data
- Evaluates denoising quality via downstream **instance segmentation** (AP@0.5) using the frozen Cellpose `cyto2` model
- Central question: how much of Cellpose3's advantage comes from task-specific learned restoration vs. what established model-based approaches already achieve?

## Dataset

This project uses the **cyto2** test set (68 images + ground-truth masks) from the Cellpose dataset.

1. Go to https://www.cellpose.org/dataset
2. Download the **cyto2** test set images and corresponding ground-truth masks
3. Place them in the project directory:
   ```
   data/
     clean/   ← original test images
     noisy/   ← generated in Phase 1 via Cellpose3 noise-calibration scripts
     masks/   ← ground-truth segmentation masks
   ```

Synthetic Poisson noise is then added using the noise-calibration scripts from the [Cellpose repo](https://github.com/MouseLand/cellpose), which degrade each image so that cyto2 AP@0.5 drops by roughly 50%.
