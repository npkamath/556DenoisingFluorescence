"""
pnp_solver.py  –  Plug-and-Play HQS solver for Poisson fluorescence images.

Solves  x* = argmin_x D(x; y) + λ R(x)  via Half-Quadratic Splitting (HQS),
where D is Poisson data fidelity and R is the implicit prior of a learned
single-channel Cellpose denoiser operating in the Anscombe VST domain.

HQS alternates two steps each iteration k:

    Step A  (data fidelity, closed form in VST domain):
        z = (μ·v + y_vst) / (μ + 1)

        Anscombe VST converts Poisson noise to approximate Gaussian(σ≈1),
        so the data fidelity term is quadratic and has a closed-form minimiser.

    Step B  (denoiser / prior):
        v = Dσ(z)

        Dσ is the trained 1-channel Cellpose denoiser applied to the current
        iterate z.

μ schedule:
    μ_k = μ_0 · ρ^k  (capped at μ_max)
    Geometric growth tightens coupling between data and prior over iterations.
    With μ_0=0.5, ρ=2.15, μ_max=5.0, N_ITERS=4, μ reaches 5.0 by iteration 4.

Channel convention:
    Red (ch 0) and Green (ch 1) are denoised independently.
    Blue (ch 2) is empty and passed through unchanged.

Reference:
    Chan, Wang & Elgendy (2017), "Plug-and-Play ADMM for Image Restoration",
    IEEE TCI.

Imports:
    CellposeDenoiserWrapper  — loads a trained .pth checkpoint, runs inference
                               on single (H, W) VST-domain arrays
    pnp_hqs_denoise          — full pipeline: counts → VST → HQS → inverse VST
                               → normalized [0, 1] output

Usage (imported by pipeline_main.py):
    from pnp_solver import CellposeDenoiserWrapper, pnp_hqs_denoise, N_ITERS, MU_0, MU_MAX, RHO

    denoiser = CellposeDenoiserWrapper("models/pnp_denoiser/cellpose_vst_denoiser_epoch50.pth")
    denoised = pnp_hqs_denoise(noisy, pscale, denoiser)
"""

import numpy as np
import torch
from pathlib import Path
from cellpose.denoise import DenoiseModel
from vst_math import anscombe, anscombe_inverse_exact

MU_0    = 0.5    
MU_MAX  = 5.0   
RHO     = 2.15   # Increased to mathematically reach MU_MAX
N_ITERS = 4      # 4 iterations is the sweet spot

class CellposeDenoiserWrapper:
    def __init__(self, model_path: str, gpu: bool = True):
        # Load as a 1-Channel model!
        self.model = DenoiseModel(pretrained_model=None, model_type=None, gpu=gpu, nchan=1)
        self.model.net.load_state_dict(torch.load(model_path, map_location="cuda" if gpu else "cpu"))
        self.model.net.eval()
        self.gpu = gpu

    @torch.no_grad()
    def eval(self, z: np.ndarray) -> np.ndarray:
        """
        Run the denoiser on a single (H, W) VST-domain array.

        Pads to the nearest multiple of 32 before inference (required by the
        Cellpose UNet's 4 downsampling stages) and crops back afterward.
        Reflection padding avoids hard edge artifacts at image borders.
        """
        H, W = z.shape
        
        # Calculate padding needed to make dimensions a multiple of 32
        pad_y = (32 - (H % 32)) % 32
        pad_x = (32 - (W % 32)) % 32
        
        # Pad using reflection to avoid hard edge artifacts
        if pad_y > 0 or pad_x > 0:
            z_padded = np.pad(z, ((0, pad_y), (0, pad_x)), mode='reflect')
        else:
            z_padded = z
            
        # PnP expects [H, W]. Cellpose expects [1, 1, H, W]
        t = torch.from_numpy(z_padded).unsqueeze(0).unsqueeze(0).float()
        if self.gpu: 
            t = t.cuda()
            
        out = self.model.net(t)[0] 
        out_np = out.squeeze().cpu().numpy()
        
        # Crop back to the original test image dimensions
        return out_np[:H, :W]

def pnp_hqs_denoise(noisy: np.ndarray, pscale: float, denoiser: CellposeDenoiserWrapper, 
                    n_iters: int = N_ITERS, mu_0: float = MU_0, mu_max: float = MU_MAX, 
                    rho: float = RHO) -> np.ndarray:
    """
    Full PnP-HQS denoising pipeline for a single (H, W, 3) float32 image.

    Steps:
        1. Convert normalized [0, 1] input to Poisson counts via pscale.
        2. Apply Anscombe VST to all channels.
        3. Run HQS on Red (ch 0) and Green (ch 1) independently;
           pass Blue (ch 2) through unchanged (empty channel).
        4. Apply exact inverse Anscombe and renormalize to [0, 1].

    Args:
        noisy   : (H, W, 3) float32, normalized [0, 1], from data/noisy/poisson/
        pscale  : Poisson scale factor logged in data/noise_params.csv
        denoiser: loaded CellposeDenoiserWrapper
        n_iters : HQS iterations
        mu_0    : initial μ
        mu_max  : μ cap
        rho     : geometric growth factor per iteration

    Returns:
        denoised (H, W, 3) float32, clipped to [0, 1]
    """
    counts = np.maximum(noisy * pscale, 0.0)
    y_vst_full = anscombe(counts)
    
    denoised_vst = np.zeros_like(y_vst_full)
    
    # Process Red and Green independently
    for c in [0, 1]:
        y_vst = y_vst_full[..., c].copy()
        
        # Initialize z with the noisy VST image
        z = y_vst.copy()
        
        for k in range(n_iters):
            mu = min(mu_0 * (rho ** k), mu_max)
            
            # Step B: Denoiser Prior
            v = denoiser.eval(z)
            
            # Step A: Soft-Wiener Data Fidelity 
            z = (mu * v + y_vst) / (mu + 1.0)
            
        denoised_vst[..., c] = z
        
    # Copy blue channel as-is (empty)
    if noisy.shape[-1] >= 3:
        denoised_vst[..., 2] = y_vst_full[..., 2]
        
    counts_d = anscombe_inverse_exact(denoised_vst)
    out = counts_d / pscale
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
