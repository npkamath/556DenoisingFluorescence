#!/usr/bin/env python3
"""
Noise2Void (N2V) baseline — per-image blind self-supervised denoising.

Follows the protocol from the Cellpose3 paper (Stringer & Pachitariu, 2025):
  - U-Net depth 2, kernel size 3
  - 100 epochs per image
  - Learning rate 0.0004
  - Batch size 128
  - 64x64 patches
  - Blind-spot masking (replace center pixel with random neighbour)

Each image is trained independently (no shared weights across images).
Only R (nucleus) and G (cytoplasm) channels are denoised; B is empty.

The N2V approach masks the center pixel of the receptive field during training
so the network cannot use it, forcing it to predict the clean value from
surrounding pixels. At inference time, no masking is applied.

Usage:
    python src/denoisers/learned/noise2void.py
    python src/denoisers/learned/noise2void.py --noisy_dir data/noisy/cellpose3
    python src/denoisers/learned/noise2void.py --epochs 50  # faster test run
"""

import argparse
import csv
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


NOISY_DIR_DEFAULT = Path("data/noisy/cellpose3")
OUT_DIR_DEFAULT = Path("results/denoised/noise2void")
RUNTIME_DIR_DEFAULT = Path("results/runtimes")

SEED = 42


# ── Blind-spot patch dataset ────────────────────────────────────────────────

class N2VPatchDataset(Dataset):
    """
    Extract random 64x64 patches from a single 2D image.
    Each patch has one blind-spot pixel replaced by a random neighbour.
    """

    def __init__(
        self,
        image: np.ndarray,
        patch_size: int = 64,
        n_patches: int = 12800,
        n_blind: int = 16,
        seed: int = SEED,
    ):
        """
        Args:
            image: 2D array (H, W), single channel.
            patch_size: side length of square patches.
            n_patches: total patches to extract.
            n_blind: number of blind-spot pixels per patch.
            seed: random seed.
        """
        self.image = image.astype(np.float32)
        self.H, self.W = image.shape
        self.ps = patch_size
        self.n_patches = n_patches
        self.n_blind = n_blind
        self.rng = np.random.RandomState(seed)

        # Pre-sample patch origins
        max_r = self.H - self.ps
        max_c = self.W - self.ps
        self.origins_r = self.rng.randint(0, max_r + 1, size=n_patches)
        self.origins_c = self.rng.randint(0, max_c + 1, size=n_patches)

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        r0 = self.origins_r[idx]
        c0 = self.origins_c[idx]
        patch = self.image[r0:r0 + self.ps, c0:c0 + self.ps].copy()

        # Create blind-spot mask and targets
        mask = np.zeros((self.ps, self.ps), dtype=np.float32)
        target = np.zeros((self.ps, self.ps), dtype=np.float32)

        # Sample blind-spot pixel locations
        br = self.rng.randint(1, self.ps - 1, size=self.n_blind)
        bc = self.rng.randint(1, self.ps - 1, size=self.n_blind)

        # 4-connected neighbour offsets
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for i in range(self.n_blind):
            r, c = br[i], bc[i]
            mask[r, c] = 1.0
            target[r, c] = patch[r, c]  # original value is the target

            # Replace with random neighbour
            dr, dc = offsets[self.rng.randint(4)]
            patch[r, c] = patch[r + dr, c + dc]

        # (1, H, W) tensors
        return (
            torch.from_numpy(patch[None]),
            torch.from_numpy(target[None]),
            torch.from_numpy(mask[None]),
        )


# ── U-Net (depth 2) ─────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3):
        super().__init__()
        pad = ks // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, ks, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetDepth2(nn.Module):
    """Minimal U-Net with depth 2 (2 downsampling levels), kernel size 3."""

    def __init__(self, in_ch=1, out_ch=1, base_filters=32, ks=3):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = ConvBlock(in_ch, f, ks)
        self.enc2 = ConvBlock(f, f * 2, ks)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 2, f * 4, ks)

        # Decoder
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2, ks)  # cat with enc2

        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f, ks)  # cat with enc1

        self.final = nn.Conv2d(f, out_ch, 1)

    def forward(self, x):
        # Pad to multiple of 4 for clean downsampling
        _, _, h, w = x.shape
        ph = (4 - h % 4) % 4
        pw = (4 - w % 4) % 4
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")

        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottleneck(F.max_pool2d(e2, 2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final(d1)

        # Remove padding
        if ph > 0 or pw > 0:
            out = out[:, :, :h, :w]
        return out


# ── Per-image N2V training ───────────────────────────────────────────────────

def train_and_denoise_channel(
    noisy_ch: np.ndarray,
    epochs: int = 100,
    patch_size: int = 64,
    batch_size: int = 128,
    lr: float = 0.0004,
    device: torch.device = torch.device("cpu"),
    seed: int = SEED,
) -> np.ndarray:
    """
    Train N2V on a single channel and return the denoised result.

    Args:
        noisy_ch: 2D array (H, W).
        epochs: number of training epochs.
        patch_size: patch side length.
        batch_size: training batch size.
        lr: learning rate.
        device: torch device.
        seed: random seed.

    Returns:
        denoised: 2D array (H, W), float32.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    H, W = noisy_ch.shape
    n_patches = max(128 * epochs // 10, 1280)  # enough patches for training

    dataset = N2VPatchDataset(
        noisy_ch, patch_size=patch_size, n_patches=n_patches, seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)

    model = UNetDepth2(in_ch=1, out_ch=1, base_filters=32, ks=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for patch, target, mask in loader:
            patch = patch.to(device)
            target = target.to(device)
            mask = mask.to(device)

            pred = model(patch)

            # Loss only at blind-spot pixels
            diff = (pred - target) ** 2 * mask
            loss = diff.sum() / mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

    # Inference: denoise the full image (no blind-spot masking)
    model.eval()
    with torch.no_grad():
        img_t = torch.from_numpy(noisy_ch[None, None]).float().to(device)
        denoised_t = model(img_t)
        denoised = denoised_t.squeeze().cpu().numpy()

    return denoised.astype(np.float32)


def denoise_one_image(
    noisy_img: np.ndarray,
    epochs: int,
    patch_size: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """Denoise R and G channels independently with N2V. Leave B unchanged."""
    out = noisy_img.copy()
    for ch in [0, 1]:
        ch_data = noisy_img[:, :, ch]
        if ch_data.max() - ch_data.min() < 1e-6:
            continue  # skip empty/constant channels
        out[:, :, ch] = train_and_denoise_channel(
            ch_data,
            epochs=epochs,
            patch_size=patch_size,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed + ch,  # different seed per channel
        )
    return out


# ── Main pipeline ────────────────────────────────────────────────────────────

def denoise_all(
    noisy_dir: Path,
    out_dir: Path,
    runtime_dir: Path,
    epochs: int = 100,
    patch_size: int = 64,
    batch_size: int = 128,
    lr: float = 0.0004,
    gpu: bool = True,
    seed: int = SEED,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    files = sorted(noisy_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {noisy_dir}")

    print(f"Noise2Void: {len(files)} images, {epochs} epochs/image, "
          f"patch={patch_size}, bs={batch_size}, lr={lr}")
    print(f"  {noisy_dir} -> {out_dir}\n")

    timings = []
    for f in files:
        stem = f.stem
        noisy = np.load(f).astype(np.float32)

        t0 = time.time()
        denoised = denoise_one_image(
            noisy, epochs=epochs, patch_size=patch_size,
            batch_size=batch_size, lr=lr, device=device, seed=seed,
        )
        elapsed = time.time() - t0

        np.save(out_dir / f"{stem}.npy", denoised)
        timings.append({"image": stem, "time_s": f"{elapsed:.3f}"})
        print(f"  {stem}  range=[{denoised.min():.3f}, {denoised.max():.3f}]  "
              f"time={elapsed:.1f}s")

    # Save timing CSV
    timing_path = runtime_dir / "noise2void.csv"
    with open(timing_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["image", "time_s"])
        writer.writeheader()
        writer.writerows(timings)

    mean_time = np.mean([float(t["time_s"]) for t in timings])
    print(f"\nDone. {len(files)} denoised images saved to {out_dir}/")
    print(f"  mean time per image: {mean_time:.1f}s")
    print(f"  timings saved to {timing_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Noise2Void per-image blind self-supervised denoising"
    )
    p.add_argument("--noisy_dir", type=Path, default=NOISY_DIR_DEFAULT,
                   help="directory with noisy .npy images")
    p.add_argument("--out_dir", type=Path, default=OUT_DIR_DEFAULT,
                   help="directory to save denoised .npy images")
    p.add_argument("--runtime_dir", type=Path, default=RUNTIME_DIR_DEFAULT,
                   help="directory to save timing CSV")
    p.add_argument("--epochs", type=int, default=100,
                   help="training epochs per image")
    p.add_argument("--patch_size", type=int, default=64,
                   help="patch side length")
    p.add_argument("--batch_size", type=int, default=128,
                   help="training batch size")
    p.add_argument("--lr", type=float, default=0.0004,
                   help="learning rate")
    p.add_argument("--no_gpu", action="store_true",
                   help="run on CPU instead of GPU")
    p.add_argument("--seed", type=int, default=SEED,
                   help="random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    denoise_all(
        noisy_dir=args.noisy_dir,
        out_dir=args.out_dir,
        runtime_dir=args.runtime_dir,
        epochs=args.epochs,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        lr=args.lr,
        gpu=not args.no_gpu,
        seed=args.seed,
    )
