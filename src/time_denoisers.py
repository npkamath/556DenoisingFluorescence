#!/usr/bin/env python3
"""
Per-image runtime comparison for all implemented denoisers.

Runs each denoising method on every image in the noisy directory, records
wall-clock time per image, and saves results to a combined CSV plus a
per-method summary.

Methods timed:
  - Wiener (VST + Wiener filter)
  - Poisson-TV (VST + TV Chambolle)
  - BM3D (VST + BM3D)
  - PURE-LET (SWT-based)
  - Cellpose3 (pretrained denoising network)

Usage:
    python src/time_denoisers.py
    python src/time_denoisers.py --noisy_dir data/noisy/poisson --methods wiener bm3d
"""

import argparse
import csv
import time
import numpy as np
from pathlib import Path

# ── Method imports (lazy, so missing deps don't block other methods) ────────

NOISY_DIR_DEFAULT = Path("data/noisy/poisson")
PSCALE_CSV_DEFAULT = Path("data/noise_params.csv")
OUT_DIR_DEFAULT = Path("results/runtimes")


def load_pscales(csv_path: Path) -> dict[str, float]:
    pscale = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pscale[r["image"]] = float(r["pscale"])
    return pscale


# ── Denoiser wrappers (each returns denoised image, no I/O) ────────────────

def denoise_wiener(img: np.ndarray, pscale: float, **kwargs) -> np.ndarray:
    from scipy.signal import wiener as scipy_wiener

    mysize = kwargs.get("mysize", 5)
    counts = np.maximum(img.astype(np.float32) * pscale, 0.0)

    def anscombe_fwd(c):
        return 2.0 * np.sqrt(np.maximum(c, 0.0) + 3.0 / 8.0)

    def anscombe_inv(y, eps=1e-6):
        y = np.maximum(y, 0.0)
        ys = np.maximum(y, eps)
        inv = (ys / 2.0) ** 2 - 1.0 / 8.0 + 1.0 / (4.0 * ys) - 5.0 / (8.0 * ys**2) + 1.0 / (8.0 * ys**3)
        return np.where(y <= eps, 0.0, inv)

    z = anscombe_fwd(counts)
    z_out = z.copy()
    for ch in (0, 1):
        if np.ptp(z[:, :, ch]) > 1e-8:
            z_out[:, :, ch] = scipy_wiener(z[:, :, ch].astype(np.float32), mysize=(mysize, mysize))
    out = np.clip(np.maximum(anscombe_inv(z_out), 0.0) / pscale, 0.0, 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def denoise_poisson_tv(img: np.ndarray, pscale: float, **kwargs) -> np.ndarray:
    from skimage.restoration import denoise_tv_chambolle

    weight = kwargs.get("weight", 0.10)
    n_iter = kwargs.get("n_iter", 200)

    counts = np.maximum(img.astype(np.float32) * pscale, 0.0)

    def anscombe_fwd(c):
        return 2.0 * np.sqrt(np.maximum(c, 0.0) + 3.0 / 8.0)

    def anscombe_inv(y, eps=1e-6):
        y = np.maximum(y, 0.0)
        ys = np.maximum(y, eps)
        inv = (ys / 2.0) ** 2 - 1.0 / 8.0 + 1.0 / (4.0 * ys) - 5.0 / (8.0 * ys**2) + 1.0 / (8.0 * ys**3)
        return np.where(y <= eps, 0.0, inv)

    z = anscombe_fwd(counts)
    z_out = z.copy()
    for ch in (0, 1):
        zch = z[:, :, ch]
        if np.ptp(zch) < 1e-8:
            continue
        z_min, z_max = float(zch.min()), float(zch.max())
        z01 = (zch - z_min) / (z_max - z_min)
        z01_d = denoise_tv_chambolle(z01.astype(np.float32), weight=weight,
                                      max_num_iter=n_iter, channel_axis=None)
        z_out[:, :, ch] = z01_d * (z_max - z_min) + z_min
    out = np.clip(np.maximum(anscombe_inv(z_out), 0.0) / pscale, 0.0, 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def denoise_bm3d(img: np.ndarray, pscale: float, **kwargs) -> np.ndarray:
    from bm3d import bm3d as bm3d_func, BM3DProfileHigh

    sigma_vst = kwargs.get("sigma_vst", 1.0)
    counts = np.maximum(img.astype(np.float32) * pscale, 0.0)

    def anscombe_fwd(c):
        return 2.0 * np.sqrt(np.maximum(c, 0.0) + 3.0 / 8.0)

    def anscombe_inv(y, eps=1e-6):
        y = np.maximum(y, 0.0)
        ys = np.maximum(y, eps)
        inv = (ys / 2.0) ** 2 - 1.0 / 8.0 + 1.0 / (4.0 * ys) - 5.0 / (8.0 * ys**2) + 1.0 / (8.0 * ys**3)
        return np.where(y <= eps, 0.0, inv)

    z = anscombe_fwd(counts)
    z_out = z.copy()
    for ch in (0, 1):
        zch = z[:, :, ch]
        if np.ptp(zch) < 1e-8:
            continue
        y_min, y_max = float(zch.min()), float(zch.max())
        y01 = (zch - y_min) / (y_max - y_min)
        sigma01 = sigma_vst / (y_max - y_min)
        y01_d = bm3d_func(y01.astype(np.float32), sigma_psd=float(sigma01),
                           profile=BM3DProfileHigh())
        z_out[:, :, ch] = y01_d * (y_max - y_min) + y_min
    out = np.clip(np.maximum(anscombe_inv(z_out), 0.0) / pscale, 0.0, 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def denoise_purelet(img: np.ndarray, pscale: float, **kwargs) -> np.ndarray:
    import pywt

    n_levels = kwargs.get("n_levels", 4)
    wavelet = kwargs.get("wavelet", "sym4")

    def _gate(x, t):
        if t < 1e-12:
            return np.ones_like(x)
        return 1.0 - np.exp(-x ** 2 / (2.0 * t ** 2))

    def _dgate_dw(x, t):
        if t < 1e-12:
            return np.zeros_like(x)
        return (x / t ** 2) * np.exp(-x ** 2 / (2.0 * t ** 2))

    def _interscale_predictor(LL, sb):
        def _grad_h(a):
            return 0.5 * (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1))
        def _grad_v(a):
            return 0.5 * (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0))
        if sb == 'HL': return _grad_h(LL)
        elif sb == 'LH': return _grad_v(LL)
        elif sb == 'HH': return _grad_v(_grad_h(LL))
        else: raise ValueError(sb)

    def _pure_let_solve(F_list, dF_list, w):
        K = len(F_list)
        M = np.zeros((K, K), dtype=np.float64)
        rhs = np.zeros(K, dtype=np.float64)
        for i in range(K):
            for j in range(K):
                M[i, j] = float(np.sum(F_list[i] * F_list[j]))
            rhs[i] = float(np.sum(F_list[i] * w)) - float(np.sum(dF_list[i]))
        eps = 1e-6 * (float(np.sum(w ** 2)) + 1.0)
        for i in range(K):
            M[i, i] += eps
        try:
            a = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            a = np.zeros(K)
        return sum(a[i] * F_list[i] for i in range(K))

    def _process_subband(w, LL, sb, t):
        Tw = _gate(w, t)
        Tp = _gate(_interscale_predictor(LL, sb), t)
        dTw = _dgate_dw(w, t)
        F1, F2 = w * Tw, w * Tw * Tp
        dF1, dF2 = Tw + w * dTw, (Tw + w * dTw) * Tp
        return _pure_let_solve([F1, F2], [dF1, dF2], w)

    def _purelet_pass(z, n_levels, wavelet):
        H, W = z.shape
        dim_factor = 2 ** n_levels
        pad_h = (dim_factor - H % dim_factor) % dim_factor
        pad_w = (dim_factor - W % dim_factor) % dim_factor
        z_padded = np.pad(z, ((0, pad_h), (0, pad_w)), mode='reflect')
        mean_count = float(np.mean(z_padded))
        sigma_hat = np.sqrt(max(mean_count, 1e-6))
        t = sigma_hat * np.sqrt(2.0 * np.log(max(z_padded.size, 2)))
        coeffs = pywt.swt2(z_padded, wavelet, level=n_levels)
        denoised_coeffs = []
        for cA, (cH, cV, cD) in coeffs:
            r_H = _process_subband(cH, cA, 'HL', t)
            r_V = _process_subband(cV, cA, 'LH', t)
            r_D = _process_subband(cD, cA, 'HH', t)
            denoised_coeffs.append((cA, (r_H, r_V, r_D)))
        return pywt.iswt2(denoised_coeffs, wavelet)[:H, :W]

    out = img.copy().astype(np.float32)
    for ch in range(img.shape[2]):
        ch_data = img[:, :, ch].astype(np.float64)
        if ch_data.max() > 1e-6:
            z_counts = np.maximum(ch_data * pscale, 0.0)
            z_denoised = _purelet_pass(z_counts, n_levels, wavelet)
            out[:, :, ch] = np.clip(z_denoised / pscale, 0.0, 1.0).astype(np.float32)
    return out


def denoise_cellpose3(img: np.ndarray, pscale: float, **kwargs) -> np.ndarray:
    """Cellpose3 denoising. Model is loaded once and cached via kwargs['_model']."""
    model = kwargs.get("_model")
    if model is None:
        from cellpose.denoise import CellposeDenoiseModel
        model = CellposeDenoiseModel(
            gpu=True, model_type="cyto2",
            restore_type="denoise_cyto3", chan2_restore=True,
        )
        # Caller should cache this — but fallback works

    batch_size = kwargs.get("batch_size", 8)
    masks, flows, styles, imgs = model.eval(
        img, channels=[2, 1], diameter=None,
        batch_size=batch_size, normalize=True,
    )

    denoised = np.array(imgs).astype(np.float32)
    if denoised.ndim == 3 and denoised.shape[0] in (1, 2, 3):
        denoised = denoised.transpose(1, 2, 0)

    H, W = img.shape[:2]
    ch0_active = img[:, :, 0].max() > 0.01
    ch1_active = img[:, :, 1].max() > 0.01

    restored = np.zeros((H, W, 3), dtype=np.float32)
    if denoised.ndim == 3 and denoised.shape[2] >= 2:
        if ch0_active and ch1_active:
            restored[:, :, 0] = denoised[:, :, 1]
            restored[:, :, 1] = denoised[:, :, 0]
        elif ch1_active:
            restored[:, :, 1] = denoised[:, :, 0]
        else:
            restored[:, :, 0] = denoised[:, :, 0]
    elif denoised.ndim == 3 and denoised.shape[2] == 1:
        if ch1_active:
            restored[:, :, 1] = denoised[:, :, 0]
        else:
            restored[:, :, 0] = denoised[:, :, 0]
    return restored


# ── Registry ───────────────────────────────────────────────────────────────

METHODS = {
    "wiener": denoise_wiener,
    "poisson_tv": denoise_poisson_tv,
    "bm3d": denoise_bm3d,
    "purelet": denoise_purelet,
    "cellpose3": denoise_cellpose3,
}


# ── Main timing loop ──────────────────────────────────────────────────────

def run_timing(
    noisy_dir: Path,
    pscale_csv: Path,
    out_dir: Path,
    methods: list[str],
    sample: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pscales = load_pscales(pscale_csv)

    files = sorted(noisy_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {noisy_dir}")

    if sample is not None and sample < len(files):
        # Pick evenly spaced images to cover the range of sizes
        indices = np.linspace(0, len(files) - 1, sample, dtype=int)
        files = [files[i] for i in indices]
        print(f"Sampling {len(files)}/{sample} images (evenly spaced by index)")

    print(f"Timing {len(methods)} methods on {len(files)} images from {noisy_dir}\n")

    # Per-image rows: image, method, time_s
    all_rows = []

    for method_name in methods:
        denoise_fn = METHODS[method_name]
        print(f"── {method_name} ──")

        # Pre-load Cellpose3 model once (not timed)
        extra_kwargs = {}
        if method_name == "cellpose3":
            from cellpose.denoise import CellposeDenoiseModel
            print("  Loading Cellpose3 model (not timed)...")
            cp_model = CellposeDenoiseModel(
                gpu=True, model_type="cyto2",
                restore_type="denoise_cyto3", chan2_restore=True,
            )
            extra_kwargs["_model"] = cp_model

        times = []
        for f in files:
            stem = f.stem
            img = np.load(f).astype(np.float32)
            ps = pscales.get(stem, 1.0)

            t0 = time.time()
            _ = denoise_fn(img, ps, **extra_kwargs)
            elapsed = time.time() - t0

            times.append(elapsed)
            all_rows.append({"image": stem, "method": method_name, "time_s": f"{elapsed:.4f}"})

        mean_t = np.mean(times)
        std_t = np.std(times)
        print(f"  mean={mean_t:.3f}s  std={std_t:.3f}s  "
              f"min={np.min(times):.3f}s  max={np.max(times):.3f}s\n")

    # Save combined CSV
    combined_path = out_dir / "denoise_timings.csv"
    with open(combined_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["image", "method", "time_s"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Per-image timings saved to {combined_path}")

    # Save summary CSV
    summary_path = out_dir / "denoise_timings_summary.csv"
    with open(summary_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["method", "mean_s", "std_s", "min_s", "max_s"])
        writer.writeheader()
        for method_name in methods:
            method_times = [float(r["time_s"]) for r in all_rows if r["method"] == method_name]
            writer.writerow({
                "method": method_name,
                "mean_s": f"{np.mean(method_times):.4f}",
                "std_s": f"{np.std(method_times):.4f}",
                "min_s": f"{np.min(method_times):.4f}",
                "max_s": f"{np.max(method_times):.4f}",
            })
    print(f"Summary saved to {summary_path}")


def parse_args() -> argparse.Namespace:
    all_methods = list(METHODS.keys())
    p = argparse.ArgumentParser(description="Time all denoising methods per-image")
    p.add_argument("--noisy_dir", type=Path, default=NOISY_DIR_DEFAULT)
    p.add_argument("--pscale_csv", type=Path, default=PSCALE_CSV_DEFAULT)
    p.add_argument("--out_dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--methods", nargs="+", default=all_methods,
                   choices=all_methods,
                   help=f"Methods to time (default: all). Choices: {all_methods}")
    p.add_argument("--sample", type=int, default=None,
                   help="Number of images to sample (evenly spaced by size). Default: all.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_timing(
        noisy_dir=args.noisy_dir,
        pscale_csv=args.pscale_csv,
        out_dir=args.out_dir,
        methods=args.methods,
        sample=args.sample,
    )
