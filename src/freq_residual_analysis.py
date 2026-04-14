"""
src/freq_residual_analysis.py
------------------------------
Frequency-domain residual analysis for denoising methods.

For each method, the denoising residual is defined as:
    residual(x, y) = denoised(y) − clean_normed(x)

where x is the percentile-normalised clean image and y is the denoised output.
(Both have been through img_norm, so they live in the same intensity space.)

Three analysis products are produced per method:

  1. Radially averaged power spectral density (PSD) of the residual
     (figures/freq/<method>_psd.png)
     Averaged over all 68 images and both active channels (R=nuc, G=cyto).
     Compared against the noisy-image residual PSD as reference.
     Interpretation: high power at mid/high frequencies → the method leaves
     structured noise or blurs boundaries; low power = clean residual.

  2. Mean 2-D log-power spectrum of the residual
     (figures/freq/<method>_2d_psd.png)
     Log|FFT(residual)|² averaged over images.  DC removed.
     Reveals directional artifacts (ring artifacts, ringing, etc.)

  3. PSD ratio: method residual PSD / noisy residual PSD
     (figures/freq/psd_ratio_comparison.png)
     A ratio < 1 at frequency f means the method removes more noise at f
     than the noisy image had.  A ratio > 1 at mid/high f indicates the
     method introduces or preserves noise there relative to the baseline.
     All methods on one figure.

  4. Edge-band power table
     (figures/freq/edge_band_summary.csv  +  figures/freq/edge_band_bar.png)
     Breaks the radial frequency axis into three bands:
       low   : f < 0.1 * Nyquist  (large-scale structure)
       mid   : 0.1–0.3 * Nyquist  (cell-boundary scale)
       high  : f > 0.3 * Nyquist  (fine texture / noise)
     Reports mean residual power in each band, normalised by the noisy
     residual power.  Methods with low mid-band ratio better preserve
     cell boundaries; methods with low high-band ratio suppress shot noise
     most effectively.

Reads:
    data/clean_normed/         .npy float32 (H, W, 3)  — clean reference
    data/noisy/poisson/        .npy float32 (H, W, 3)  — noisy input
    results/denoised/<method>/ .npy float32 (H, W, 3)  — denoised output

Notes:
  - Only channels 0 (R=nucleus) and 1 (G=cytoplasm) are analysed.
    Channel 2 (B=empty) is skipped.
  - Images are zero-padded to the next power of 2 in each dimension before
    FFT to avoid spectral leakage edge effects.
  - A Hann window is applied before FFT to suppress spectral leakage.

Usage
-----
    # All methods found automatically under results/denoised/:
    python src/freq_residual_analysis.py

    # Specific methods:
    python src/freq_residual_analysis.py --methods bm3d wiener tv purelet

    # Also include the raw noisy residual as a reference method:
    python src/freq_residual_analysis.py --include_noisy

Output
------
    figures/freq/<method>_psd.png
    figures/freq/<method>_2d_psd.png
    figures/freq/psd_ratio_comparison.png
    figures/freq/edge_band_summary.csv
    figures/freq/edge_band_bar.png
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import zoom

# ── Default paths ──────────────────────────────────────────────────────────────
CLEAN_NORMED_DIR = Path("data/clean_normed")
NOISY_DIR        = Path("data/noisy/poisson")
DENOISED_ROOT    = Path("results/denoised")
FIGURES_DIR      = Path("figures/freq")

# Active channels: 0=Red(nucleus), 1=Green(cytoplasm)
ACTIVE_CHANNELS = [0, 1]

# Fixed number of radial frequency bins — all PSDs are interpolated onto this
# common grid so images of different sizes can be averaged together.
N_BINS = 256
FREQ_GRID = np.linspace(0.0, 0.5, N_BINS + 1)
FREQ_CENTRES = 0.5 * (FREQ_GRID[:-1] + FREQ_GRID[1:])

PRETTY = {
    "noisy"   : "Noisy (no restoration)",
    "bm3d"    : "VST + BM3D",
    "wiener"  : "Wiener",
    "tv"      : "Poisson-TV",
    "purelet" : "PURE-LET",
    "cellpose3": "Cellpose3 (DL)",
}

METHOD_COLORS = {
    "noisy"    : "#9E9E9E",
    "bm3d"     : "#2196F3",
    "wiener"   : "#FF9800",
    "tv"       : "#4CAF50",
    "purelet"  : "#9C27B0",
    "cellpose3": "#F44336",
}
DEFAULT_COLOR = "#607D8B"

# Frequency bands as fractions of Nyquist (0.5 cycles/pixel)
BANDS = {
    "low"  : (0.00, 0.10),
    "mid"  : (0.10, 0.30),
    "high" : (0.30, 0.50),
}


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_npy(path: Path) -> np.ndarray:
    return np.load(str(path)).astype(np.float32)


def stems_in_dir(d: Path) -> list[str]:
    return sorted(f.stem for f in d.glob("*.npy"))


def discover_methods() -> list[str]:
    if not DENOISED_ROOT.exists():
        return []
    return sorted(d.name for d in DENOISED_ROOT.iterdir() if d.is_dir())


# ── Signal processing helpers ──────────────────────────────────────────────────

def next_pow2(n: int) -> int:
    return int(2 ** np.ceil(np.log2(n)))


def hann2d(H: int, W: int) -> np.ndarray:
    """2-D Hann window of shape (H, W)."""
    wh = np.hanning(H).astype(np.float32)
    ww = np.hanning(W).astype(np.float32)
    return np.outer(wh, ww)


def compute_residual_psd(
    residual: np.ndarray,
) -> np.ndarray:
    """
    Compute radially averaged PSD of a 2D residual image, interpolated onto
    the global FREQ_CENTRES grid (length N_BINS) so every image returns the
    same-length array regardless of spatial dimensions.

    Args:
        residual: 2D float32 array (H, W).

    Returns:
        psd_fixed: (N_BINS,) float64 array aligned to FREQ_CENTRES.
    """
    H, W = residual.shape
    pH = next_pow2(H)
    pW = next_pow2(W)

    win = hann2d(H, W)
    padded = np.zeros((pH, pW), dtype=np.float32)
    padded[:H, :W] = residual * win

    F = np.fft.rfft2(padded)
    psd2d = (np.abs(F) ** 2) / float(H * W)

    # Radial frequency map for rfft2 output (pH, pW//2+1)
    freqs_row = np.fft.fftfreq(pH)
    freqs_col = np.fft.rfftfreq(pW)
    frow, fcol = np.meshgrid(freqs_row, freqs_col, indexing="ij")
    rad_freq = np.sqrt(frow ** 2 + fcol ** 2)

    # Bin on the image's native resolution, then interpolate to fixed grid.
    # Using np.histogram / bincount is much faster than the per-bin loop.
    n_native = min(pH, pW) // 2
    native_edges = np.linspace(0.0, 0.5, n_native + 1)

    rad_flat  = rad_freq.ravel()
    psd_flat  = psd2d.ravel()
    bin_idx   = np.searchsorted(native_edges, rad_flat, side="right") - 1
    bin_idx   = np.clip(bin_idx, 0, n_native - 1)

    sum_psd   = np.bincount(bin_idx, weights=psd_flat,  minlength=n_native)
    count_psd = np.bincount(bin_idx,                    minlength=n_native)
    with np.errstate(invalid="ignore"):
        psd_native = np.where(count_psd > 0, sum_psd / count_psd, 0.0)

    native_centres = 0.5 * (native_edges[:-1] + native_edges[1:])

    # Interpolate onto the fixed global FREQ_CENTRES grid
    psd_fixed = np.interp(FREQ_CENTRES, native_centres, psd_native)
    return psd_fixed


def compute_mean_2d_logpsd(residuals: list[np.ndarray]) -> np.ndarray:
    """
    Average 2D log-power spectra over a list of residual patches.
    All residuals are resized to a common shape (256×256) for averaging.
    Returns the mean log10(PSD + eps) 2D array, DC shifted to centre.
    """
    target = 256
    eps = 1e-12
    accum = np.zeros((target, target), dtype=np.float64)

    for res in residuals:
        # Resize to target
        if res.shape != (target, target):
            zy = target / res.shape[0]
            zx = target / res.shape[1]
            res_r = zoom(res.astype(np.float32), (zy, zx), order=1)
        else:
            res_r = res.astype(np.float32)

        win = hann2d(target, target)
        F = np.fft.fft2(res_r * win)
        psd = np.abs(F) ** 2 / (target * target)
        accum += np.log10(psd + eps)

    return np.fft.fftshift(accum / len(residuals))


# ── Per-method analysis ────────────────────────────────────────────────────────

def collect_residuals(
    clean_dir   : Path,
    denoised_dir: Path,
    common_stems: list[str],
) -> dict:
    """
    For each stem, compute the per-channel residual = denoised − clean_normed.
    Returns aggregated PSD data.

    Returns:
        dict with keys:
            mean_psd   : (n_bins,) radially averaged PSD, averaged over images & channels
            std_psd    : (n_bins,) std dev of per-image PSDs
            freq_bins  : (n_bins,) frequency centres
            residuals_2d: list of 2D residuals (one per image×channel, for 2D spectrum)
    """
    all_psds   = []
    residuals_2d = []
    freq_bins = None

    for stem in common_stems:
        clean_path    = clean_dir    / f"{stem}.npy"
        denoised_path = denoised_dir / f"{stem}.npy"

        if not clean_path.exists() or not denoised_path.exists():
            continue

        clean    = load_npy(clean_path)     # (H, W, 3)
        denoised = load_npy(denoised_path)  # (H, W, 3)
        residual = denoised - clean          # signed residual

        for ch in ACTIVE_CHANNELS:
            res_ch = residual[:, :, ch]
            psd_fixed = compute_residual_psd(res_ch)   # always (N_BINS,)
            all_psds.append(psd_fixed)
            residuals_2d.append(res_ch)

    if not all_psds:
        return {}

    # Stack is safe: every row is (N_BINS,)
    all_psds_arr = np.array(all_psds, dtype=np.float64)  # (n_samples, N_BINS)
    return dict(
        mean_psd     = all_psds_arr.mean(axis=0),
        std_psd      = all_psds_arr.std(axis=0, ddof=1),
        freq_bins    = FREQ_CENTRES,                   # fixed global grid
        residuals_2d = residuals_2d,
        n_samples    = len(all_psds),
    )


# ── Plot builders ──────────────────────────────────────────────────────────────

def plot_psd_single(
    psd_data      : dict,
    noisy_psd_data: dict,
    method        : str,
    out_path      : Path,
    dpi           : int = 150,
) -> None:
    """Radially averaged PSD: method residual vs noisy residual."""
    freq  = psd_data["freq_bins"]
    mean_ = psd_data["mean_psd"]
    std_  = psd_data["std_psd"]
    n     = psd_data["n_samples"]
    label = PRETTY.get(method, method)
    color = METHOD_COLORS.get(method, DEFAULT_COLOR)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Left: linear PSD ────────────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(freq, mean_ - std_, mean_ + std_,
                    alpha=0.20, color=color)
    ax.plot(freq, mean_, color=color, linewidth=1.8, label=label)

    if noisy_psd_data:
        nf, nm = noisy_psd_data["freq_bins"], noisy_psd_data["mean_psd"]
        ax.plot(nf, nm, color=METHOD_COLORS["noisy"], linewidth=1.2,
                linestyle="--", alpha=0.7, label="Noisy (reference)")

    # Band boundaries
    for bname, (blo, bhi) in BANDS.items():
        ax.axvspan(blo, bhi, alpha=0.04, label=f"{bname} band")

    ax.set_xlabel("Normalised radial frequency (cycles/pixel)", fontsize=10)
    ax.set_ylabel("Mean residual power", fontsize=10)
    ax.set_title(f"Radial PSD of residual — {label}\n(n={n} image×channel samples)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_xlim(0, 0.5)

    # ── Right: log PSD ────────────────────────────────────────────────────
    ax = axes[1]
    eps = 1e-12
    ax.fill_between(freq,
                    np.log10(np.maximum(mean_ - std_, eps)),
                    np.log10(mean_ + std_ + eps),
                    alpha=0.20, color=color)
    ax.plot(freq, np.log10(mean_ + eps), color=color, linewidth=1.8, label=label)

    if noisy_psd_data:
        ax.plot(nf, np.log10(nm + eps),
                color=METHOD_COLORS["noisy"], linewidth=1.2,
                linestyle="--", alpha=0.7, label="Noisy (reference)")

    ax.set_xlabel("Normalised radial frequency (cycles/pixel)", fontsize=10)
    ax.set_ylabel("log₁₀(Mean residual power)", fontsize=10)
    ax.set_title(f"Log-scale PSD — {label}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_xlim(0, 0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    PSD → {out_path}")


def plot_2d_logpsd(
    psd_data : dict,
    method   : str,
    out_path : Path,
    dpi      : int = 150,
) -> None:
    """Mean 2D log-PSD of residual (DC-centred)."""
    if not psd_data.get("residuals_2d"):
        return

    # Limit to 200 samples to keep runtime reasonable
    residuals = psd_data["residuals_2d"][:200]
    log_psd2d = compute_mean_2d_logpsd(residuals)
    label = PRETTY.get(method, method)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(log_psd2d, cmap="inferno", origin="lower",
                   extent=[-0.5, 0.5, -0.5, 0.5])
    ax.set_xlabel("Horizontal frequency (cycles/pixel)", fontsize=10)
    ax.set_ylabel("Vertical frequency (cycles/pixel)", fontsize=10)
    ax.set_title(
        f"Mean 2D log-PSD of residual — {label}\n"
        f"(n={len(residuals)} samples, Hann-windowed)",
        fontsize=10, fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("log₁₀(power)", fontsize=9)

    # Mark frequency band boundaries as rings (dashed circles)
    theta = np.linspace(0, 2 * np.pi, 300)
    for bname, (blo, bhi) in BANDS.items():
        for r in (blo, bhi):
            if r > 0:
                ax.plot(r * np.cos(theta), r * np.sin(theta),
                        linestyle="--", linewidth=0.8, color="white", alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    2D PSD → {out_path}")


def plot_psd_ratio_comparison(
    psd_by_method : dict[str, dict],
    noisy_psd_data: dict,
    out_path      : Path,
    dpi           : int = 150,
) -> None:
    """
    Ratio: method residual PSD / noisy residual PSD.
    All methods on one axes.  Ratio < 1 = better noise removal at that frequency.
    """
    if not noisy_psd_data:
        print("  [skip] PSD ratio plot — noisy PSD data not available")
        return

    noisy_mean = noisy_psd_data["mean_psd"]
    eps = 1e-12

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", zorder=1,
               label="Ratio = 1  (same as noisy)")

    for method, psd_data in psd_by_method.items():
        if not psd_data:
            continue
        freq   = psd_data["freq_bins"]
        ratio  = (psd_data["mean_psd"] + eps) / (noisy_mean[:len(psd_data["mean_psd"])] + eps)
        color  = METHOD_COLORS.get(method, DEFAULT_COLOR)
        label  = PRETTY.get(method, method)
        ax.plot(freq, ratio, color=color, linewidth=1.8, label=label)

    # Shade frequency bands
    band_alpha = 0.06
    band_colors = ["#CFD8DC", "#B0BEC5", "#90A4AE"]
    for (bname, (blo, bhi)), bc in zip(BANDS.items(), band_colors):
        ax.axvspan(blo, bhi, alpha=band_alpha * 2, color=bc,
                   label=f"{bname} band ({blo:.0%}–{bhi:.0%} Nyquist)")

    ax.set_xlabel("Normalised radial frequency (cycles/pixel)", fontsize=11)
    ax.set_ylabel("PSD ratio: method / noisy", fontsize=11)
    ax.set_title(
        "Residual PSD ratio vs. noisy baseline\n"
        "(< 1: method removes more noise; > 1: method amplifies at this frequency)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(0, 0.5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  PSD ratio comparison → {out_path}")


def compute_and_save_edge_band_summary(
    psd_by_method : dict[str, dict],
    noisy_psd_data: dict,
    out_dir       : Path,
    dpi           : int = 150,
) -> None:
    """
    For each band (low/mid/high), compute mean power ratio vs. noisy.
    Save CSV + bar chart.
    """
    eps = 1e-12
    rows = []

    for method, psd_data in psd_by_method.items():
        if not psd_data:
            continue
        freq      = psd_data["freq_bins"]
        mean_psd  = psd_data["mean_psd"]
        noisy_psd = noisy_psd_data["mean_psd"][:len(mean_psd)] if noisy_psd_data else None

        row = {"method": method, "label": PRETTY.get(method, method)}
        for bname, (blo, bhi) in BANDS.items():
            mask = (freq >= blo) & (freq < bhi)
            power_m = mean_psd[mask].mean() if mask.any() else np.nan
            row[f"{bname}_power"] = float(power_m)
            if noisy_psd is not None:
                power_n = noisy_psd[mask].mean() if mask.any() else np.nan
                row[f"{bname}_ratio"] = float((power_m + eps) / (power_n + eps))
            else:
                row[f"{bname}_ratio"] = np.nan
        rows.append(row)

    if not rows:
        return

    # Save CSV
    csv_path = out_dir / "edge_band_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "label"] + \
                 [f"{b}_{s}" for b in BANDS for s in ("power", "ratio")]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Edge-band CSV → {csv_path}")

    # Bar chart: ratio per band per method
    methods_  = [r["method"] for r in rows]
    labels_   = [r["label"]  for r in rows]
    bands     = list(BANDS.keys())
    n_methods = len(methods_)
    n_bands   = len(bands)
    x         = np.arange(n_methods)
    width     = 0.25

    band_plot_colors = {"low": "#78909C", "mid": "#42A5F5", "high": "#EF5350"}

    fig, ax = plt.subplots(figsize=(max(8, n_methods * 1.8), 5))
    for bi, bname in enumerate(bands):
        ratios = [r.get(f"{bname}_ratio", np.nan) for r in rows]
        bars = ax.bar(x + (bi - 1) * width, ratios, width=width * 0.95,
                      color=band_plot_colors[bname], alpha=0.85,
                      label=f"{bname} band ({BANDS[bname][0]:.0%}–{BANDS[bname][1]:.0%} Nyquist)",
                      edgecolor="white", linewidth=0.5)
        # Value labels on bars
        for bar, val in zip(bars, ratios):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=7, rotation=0)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
               label="Ratio = 1 (same as noisy)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Residual PSD ratio (method / noisy baseline)", fontsize=10)
    ax.set_title(
        "Band-averaged residual power ratio vs. noisy baseline\n"
        "(< 1: method removes noise in this band)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    bar_path = out_dir / "edge_band_bar.png"
    fig.savefig(str(bar_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Edge-band bar → {bar_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frequency-domain residual analysis for denoising methods."
    )
    p.add_argument("--methods",       nargs="+", default=None,
                   help="Denoising methods to analyse (default: all in results/denoised/)")
    p.add_argument("--clean_dir",     type=Path, default=CLEAN_NORMED_DIR)
    p.add_argument("--noisy_dir",     type=Path, default=NOISY_DIR)
    p.add_argument("--denoised_root", type=Path, default=DENOISED_ROOT)
    p.add_argument("--out_dir",       type=Path, default=FIGURES_DIR)
    p.add_argument("--include_noisy", action="store_true",
                   help="Include the raw noisy images as a reference method "
                        "(residual = noisy − clean_normed)")
    p.add_argument("--skip_2d",       action="store_true",
                   help="Skip 2D PSD plots (faster for large datasets)")
    p.add_argument("--dpi",           type=int, default=150)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Discover methods ──────────────────────────────────────────────────────
    methods = args.methods if args.methods else discover_methods()
    if not methods and not args.include_noisy:
        raise SystemExit(
            f"No denoised method directories found under {DENOISED_ROOT}.\n"
            f"Run a denoising script (e.g. denoise_bm3d_vst.py) first, or\n"
            f"pass --methods explicitly."
        )
    print(f"Methods to analyse: {methods}\n")

    # ── Find common stems (images present in clean_normed AND all methods) ───
    clean_stems = set(stems_in_dir(args.clean_dir))
    if not clean_stems:
        raise SystemExit(
            f"No .npy files in {args.clean_dir}.  Run data_prep.py first."
        )

    # Use the union of available stems across methods for robustness
    common_stems = sorted(clean_stems)
    print(f"Clean-normed images: {len(common_stems)}")

    # ── Compute noisy residual PSD (reference) ────────────────────────────────
    noisy_psd_data: dict = {}
    print("\nComputing noisy baseline PSD...")
    noisy_data = collect_residuals(args.clean_dir, args.noisy_dir, common_stems)
    if noisy_data:
        noisy_psd_data = noisy_data
        print(f"  Noisy PSD computed ({noisy_data['n_samples']} samples)")
    else:
        print("  [warn] Could not compute noisy PSD — noisy files may be missing")

    # ── Per-method analysis ───────────────────────────────────────────────────
    psd_by_method: dict[str, dict] = {}

    if args.include_noisy and noisy_psd_data:
        psd_by_method["noisy"] = noisy_psd_data

    for method in methods:
        denoised_dir = args.denoised_root / method
        if not denoised_dir.exists():
            print(f"[skip] {denoised_dir} not found")
            continue

        method_stems = set(stems_in_dir(denoised_dir))
        usable = sorted(clean_stems & method_stems)
        if not usable:
            print(f"[skip] {method}: no matching stems with clean_normed")
            continue

        label = PRETTY.get(method, method)
        print(f"\n[*] {label}  ({len(usable)} images)")

        psd_data = collect_residuals(args.clean_dir, denoised_dir, usable)
        if not psd_data:
            print(f"  [skip] Could not compute PSD for {method}")
            continue

        psd_by_method[method] = psd_data
        print(f"    PSD computed ({psd_data['n_samples']} image×channel samples)")

        # 1. Radial PSD plot
        plot_psd_single(
            psd_data, noisy_psd_data, method,
            out_path=args.out_dir / f"{method}_psd.png",
            dpi=args.dpi,
        )

        # 2. 2D log-PSD plot
        if not args.skip_2d:
            plot_2d_logpsd(
                psd_data, method,
                out_path=args.out_dir / f"{method}_2d_psd.png",
                dpi=args.dpi,
            )

    # ── Cross-method plots ────────────────────────────────────────────────────
    methods_to_plot = {k: v for k, v in psd_by_method.items() if k != "noisy"}

    if methods_to_plot:
        print("\nGenerating cross-method figures...")

        # 3. PSD ratio comparison
        plot_psd_ratio_comparison(
            psd_by_method=methods_to_plot,
            noisy_psd_data=noisy_psd_data,
            out_path=args.out_dir / "psd_ratio_comparison.png",
            dpi=args.dpi,
        )

        # 4. Edge-band power summary
        compute_and_save_edge_band_summary(
            psd_by_method=methods_to_plot,
            noisy_psd_data=noisy_psd_data,
            out_dir=args.out_dir,
            dpi=args.dpi,
        )

    print(f"\n[✓] All frequency-domain figures saved under {args.out_dir}/")
