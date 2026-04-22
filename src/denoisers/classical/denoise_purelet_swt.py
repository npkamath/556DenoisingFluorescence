import argparse
import csv
from pathlib import Path
import numpy as np
import pywt

# ── Default paths ──────────────────────────
NOISY_DIR_DEFAULT  = Path("data/noisy/poisson")
PSCALE_CSV_DEFAULT = Path("data/noise_params.csv")
OUT_DIR_DEFAULT    = Path("results/denoised/purelet_swt")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Gate function and its derivative (for Stein divergence)
# ─────────────────────────────────────────────────────────────────────────────

def _gate(x: np.ndarray, t: float) -> np.ndarray:
    """Sigmoidal gate function T(x, t) as defined in Eq. (10) of the paper."""
    if t < 1e-12:
        return np.ones_like(x)
    return 1.0 - np.exp(-x ** 2 / (2.0 * t ** 2))


def _dgate_dw(x: np.ndarray, t: float) -> np.ndarray:
    """Derivative of T w.r.t. its argument for PURE divergence term."""
    if t < 1e-12:
        return np.zeros_like(x)
    return (x / t ** 2) * np.exp(-x ** 2 / (2.0 * t ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Interscale predictor (adapted for SWT)
# ─────────────────────────────────────────────────────────────────────────────

def _interscale_predictor(LL: np.ndarray, subband: str) -> np.ndarray:
    """Compute interscale predictor p from the same-scale LL subband gradients."""
    def _grad_h(a: np.ndarray) -> np.ndarray:
        return 0.5 * (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1))

    def _grad_v(a: np.ndarray) -> np.ndarray:
        return 0.5 * (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0))

    if subband == 'HL':
        return _grad_h(LL)
    elif subband == 'LH':
        return _grad_v(LL)
    elif subband == 'HH':
        return _grad_v(_grad_h(LL))
    else:
        raise ValueError(f"Unknown subband '{subband}'")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Stage 2: PURE-LET Coefficient Solver (Linear System)
# ─────────────────────────────────────────────────────────────────────────────

def _pure_let_solve(F_list, dF_list, w):
    """
    Automatically solves the linear system M * a = rhs to find optimal coefficients.
    Self-calibration addresses the limitation of manual thresholding.
    """
    K = len(F_list)
    M = np.zeros((K, K), dtype=np.float64)
    rhs = np.zeros(K, dtype=np.float64)
    
    # For Stationary Wavelet Transform (SWT), divergence factor is 1.0
    div_factor = 1.0

    for i in range(K):
        for j in range(K):
            M[i, j] = float(np.sum(F_list[i] * F_list[j]))
        # PURE formula for Poisson: <F, w> - sum(dF/dw)
        rhs[i] = (float(np.sum(F_list[i] * w)) - div_factor * float(np.sum(dF_list[i])))

    # Tikhonov regularization for numerical stability in flat regions
    eps = 1e-6 * (float(np.sum(w ** 2)) + 1.0)
    for i in range(K):
        M[i, i] += eps

    try:
        a = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        a = np.zeros(K)

    return sum(a[i] * F_list[i] for i in range(K))


def _process_subband_swt(w, LL, sb, t):
    """Denoise one subband using PURE-LET2 basis and adaptive solver."""
    Tw  = _gate(w, t)
    Tp  = _gate(_interscale_predictor(LL, sb), t)
    dTw = _dgate_dw(w, t)

    # LET basis: F1 (intra-scale), F2 (inter-scale)
    F1, F2 = w * Tw, w * Tw * Tp
    dF1, dF2 = Tw + w * dTw, (Tw + w * dTw) * Tp

    return _pure_let_solve([F1, F2], [dF1, dF2], w)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Stage 3: SWT Pipeline (Shift-Invariance)
# ─────────────────────────────────────────────────────────────────────────────

def _purelet_swt_pass(z: np.ndarray, n_levels: int, wavelet: str) -> np.ndarray:
    """
    Stationary Wavelet Transform denoising pass. 
    Provides shift-invariance natively without manual cycle-spinning.
    """
    H, W = z.shape
    # SWT requires size to be multiple of 2^n_levels
    dim_factor = 2**n_levels
    pad_h = (dim_factor - H % dim_factor) % dim_factor
    pad_w = (dim_factor - W % dim_factor) % dim_factor
    z_padded = np.pad(z, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    # Global adaptive shape parameter t based on noise floor
    mean_count = float(np.mean(z_padded))
    sigma_hat = np.sqrt(max(mean_count, 1e-6))
    t = sigma_hat * np.sqrt(2.0 * np.log(max(z_padded.size, 2)))

    # Forward SWT (undecimated)
    coeffs = pywt.swt2(z_padded, wavelet, level=n_levels)
    denoised_coeffs = []

    for i, (cA, (cH, cV, cD)) in enumerate(coeffs):
        # Process Horizontal, Vertical, and Diagonal subbands
        r_H = _process_subband_swt(cH, cA, 'HL', t) 
        r_V = _process_subband_swt(cV, cA, 'LH', t)
        r_D = _process_subband_swt(cD, cA, 'HH', t)
        denoised_coeffs.append((cA, (r_H, r_V, r_D)))

    # Inverse SWT and crop back to original size
    return pywt.iswt2(denoised_coeffs, wavelet)[:H, :W]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Execution Logic
# ─────────────────────────────────────────────────────────────────────────────

def purelet_swt_denoise_image(noisy, pscale, n_levels=4, wavelet='sym4'):
    """Apply PURE-LET via SWT to each non-empty channel."""
    out = noisy.copy().astype(np.float32)
    for ch in range(noisy.shape[2]):
        ch_data = noisy[:, :, ch].astype(np.float64)
        if ch_data.max() > 1e-6:
            z_counts = np.maximum(ch_data * pscale, 0.0)
            z_denoised = _purelet_swt_pass(z_counts, n_levels, wavelet)
            out[:, :, ch] = np.clip(z_denoised / pscale, 0.0, 1.0).astype(np.float32)
    return out


def _load_pscales(csv_path: Path) -> dict[str, float]:
    """Load pscales from CSV (Copied from your original denoise_purelet.py)."""
    result = {}
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            result[row["image"]] = float(row["pscale"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upgraded PURE-LET with SWT and Adaptive Coefficients.")
    parser.add_argument("--noisy_dir",  type=Path, default=NOISY_DIR_DEFAULT)
    parser.add_argument("--pscale_csv", type=Path, default=PSCALE_CSV_DEFAULT)
    parser.add_argument("--out_dir",    type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--n_levels",   type=int,  default=4)
    parser.add_argument("--wavelet",    type=str,  default="sym4")
    args = parser.parse_args()

    pscales = _load_pscales(args.pscale_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.noisy_dir.glob("*.npy"))
    if not files:
        print(f"No .npy files found in {args.noisy_dir}")
    else:
        print(f"Processing {len(files)} images -> {args.out_dir} (Wavelet: {args.wavelet})")

    for f in files:
        stem = f.stem
        if stem not in pscales: continue

        noisy = np.load(f).astype(np.float32)
        pscale = pscales[stem]

        denoised = purelet_swt_denoise_image(noisy, pscale, n_levels=args.n_levels, wavelet=args.wavelet)
        np.save(args.out_dir / f"{stem}.npy", denoised)
        print(f"  [✓] {f.name}")
