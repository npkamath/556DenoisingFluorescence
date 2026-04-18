"""
vst_math.py  –  Variance-Stabilising Transform (VST) utilities.

Shared by pnp_solver.py, pipeline_main.py, and data_prep_train.py to
guarantee mathematical consistency between the forward and inverse transforms.

Functions:
    anscombe(x)                 Forward VST: counts → Gaussian-like domain
    anscombe_inverse_exact(y)   Exact unbiased inverse (Makitalo & Foi, 2013)
    to_counts(img_normed, p)    Normalized [0, 1] float → photon counts
    from_counts(counts, p)      Photon counts → normalized [0, 1] float

Forward transform:
    y = 2 * sqrt(max(x, 0) + 3/8)
    Maps Poisson(λ) → approximately N(2√λ, 1), stabilising variance to ~1.

Inverse transform (Makitalo & Foi, 2013):
    Closed-form approximation matching the LUT result to within 1e-5.
    Avoids bias introduced by the naive (y/2)² − 3/8 inversion at low counts.

Reference:
    Makitalo & Foi (2013), "Optimal Inversion of the Generalized Anscombe
    Transformation for Poisson-Gaussian Noise", IEEE TIP.
"""

import numpy as np


def anscombe(x: np.ndarray) -> np.ndarray:
    """
    Generalised Anscombe VST.
    Maps Poisson(λ) → approximately N(2√λ, 1).
    Input x should be non-negative photon counts (float64 recommended).

        y = 2 * sqrt(max(x, 0) + 3/8)
    """
    return 2.0 * np.sqrt(np.maximum(x, 0.0) + 3.0 / 8.0)


def anscombe_inverse_exact(y: np.ndarray) -> np.ndarray:
    """
    Exact unbiased inverse of the Anscombe VST (Makitalo & Foi 2013).

    Avoids the naive (y/2)^2 – 3/8 inverse which introduces bias for low
    counts. Closed-form approximation matches the LUT result to within 1e-5.
    """
    y_safe = np.maximum(y, anscombe(0.0))  # clamp to valid range
    exact = (
        (y_safe / 2.0) ** 2
        - 1.0 / 8.0
        + (1.0 / 4.0) * np.sqrt(3.0 / 2.0) / y_safe
        - (11.0 / 8.0) / (y_safe ** 2)
        + (5.0 / 8.0) * np.sqrt(3.0 / 2.0) / (y_safe ** 3)
    )
    return np.maximum(exact, 0.0)


# Alias kept for backward compatibility
anscombe_inverse_safe = anscombe_inverse_exact


def to_counts(img_normed: np.ndarray, pscale: float) -> np.ndarray:
    """Convert a [0, 1]-normalized image to photon counts via pscale."""
    return np.maximum(img_normed, 0.0) * pscale

def from_counts(counts: np.ndarray, pscale: float) -> np.ndarray:
    """Convert photon counts to [0, 1]-normalized float, clipped to [0, 1]."""
    return np.clip(counts / pscale, 0.0, 1.0)

if __name__ == "__main__":
    counts = np.array([0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0])
    vst    = anscombe(counts)
    recon  = anscombe_inverse_exact(vst)
    print("counts   :", counts)
    print("vst      :", np.round(vst, 4))
    print("recovered:", np.round(recon, 4))
    print("max |err|:", np.max(np.abs(recon - counts)))
