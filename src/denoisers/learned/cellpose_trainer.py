"""
cellpose_trainer.py  –  Train a 1-channel Cellpose denoiser on VST-domain pairs.

Extension 2: replaces BM3D's hand-crafted prior with a learned denoiser
operating in the Anscombe VST domain. The trained model is used as the
prior D_σ in pnp_solver.py.

Architecture:
    Cellpose DenoiseModel initialized from denoise_cyto3 weights (1-channel
    variant, nchan=1). Only the net weights are saved; the DenoiseModel wrapper
    is reconstructed at inference time in CellposeDenoiserWrapper.

Training data:
    Single-channel (H, W) VST-domain arrays produced by data_prep_train.py.
    Each record in train_noise_params.csv maps a noisy pair file to its
    clean reference stem.

Training loop:
    MSE loss on (H, W) patches cropped from full images.
    Checkpoints saved every 25 epochs to <model_dir>/<model_name>_epoch<N>.pth.
    Final weights saved to <model_dir>/<model_name>_final.pth.

Prerequisites:
    python src/data_prep/data_prep_train.py   # generates data/train_pairs/

Usage:
    python src/denoisers/learned/cellpose_trainer.py
    python src/denoisers/learned/cellpose_trainer.py --n_epochs 200 --learning_rate 5e-5
    python src/denoisers/learned/cellpose_trainer.py --pretrained models/pnp_denoiser/cellpose_vst_denoiser.pth

Outputs:
    models/pnp_denoiser/cellpose_vst_denoiser_epoch25.pth   (checkpoint)
    models/pnp_denoiser/cellpose_vst_denoiser_epoch50.pth   (checkpoint)
    ...
    models/pnp_denoiser/cellpose_vst_denoiser_final.pth     (final weights)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from cellpose.denoise import DenoiseModel

TRAIN_PAIRS_DIR = Path("data/train_pairs")
MODEL_SAVE_DIR  = Path("models/pnp_denoiser")
MODEL_NAME      = "cellpose_vst_denoiser"

N_EPOCHS        = 100
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
BATCH_SIZE      = 8     # Increased since we are 1 channel
SEED            = 42
PATCH_SIZE      = 256
PRETRAINED_SRC  = "denoise_cyto3" # Changed from segmentation cyto3

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

class VSTPairDataset(Dataset):
    """
    Loads (noisy_vst, clean_vst) single-channel pairs from data/train_pairs/.

    Reads the pair index from train_noise_params.csv (written by
    data_prep_train.py). Each item is a random (patch_size × patch_size)
    crop returned as (1, H, W) float tensors, ready for the 1-channel UNet.
    Images smaller than patch_size are reflect-padded before cropping.
    """
    def __init__(self, pairs_dir: Path, patch_size: int = PATCH_SIZE):
        self.noisy_dir = pairs_dir / "noisy"
        self.clean_dir = pairs_dir / "clean"
        import csv
        self.records = []
        with open(pairs_dir.parent / "train_noise_params.csv", "r") as f:
            for row in csv.DictReader(f):
                self.records.append((row["pair"], row["clean_stem"]))
        self.patch_size = patch_size

    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        pair_f, clean_f = self.records[idx]
        noisy = np.load(self.noisy_dir / pair_f)
        clean = np.load(self.clean_dir / clean_f)
        
        H, W = noisy.shape
        
        # 1. Pad if the image is smaller than the patch size
        pad_y = max(0, self.patch_size - H)
        pad_x = max(0, self.patch_size - W)
        
        if pad_y > 0 or pad_x > 0:
            noisy = np.pad(noisy, ((0, pad_y), (0, pad_x)), mode='reflect')
            clean = np.pad(clean, ((0, pad_y), (0, pad_x)), mode='reflect')
            H, W = noisy.shape # Update H and W
            
        # 2. Random Crop (safe now since H, W >= patch_size)
        # Adding +1 handles the case where H == patch_size exactly
        y = np.random.randint(0, H - self.patch_size + 1)
        x = np.random.randint(0, W - self.patch_size + 1)
        
        noisy = noisy[y:y+self.patch_size, x:x+self.patch_size]
        clean = clean[y:y+self.patch_size, x:x+self.patch_size]
            
        # 3. Return as float tensors
        return torch.from_numpy(noisy).float().unsqueeze(0), \
               torch.from_numpy(clean).float().unsqueeze(0)

def train_denoiser(args):
    set_seed(args.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple 1-channel Denoising Model
    gpu = torch.cuda.is_available()
    model = DenoiseModel(pretrained_model=None, model_type=PRETRAINED_SRC, gpu=gpu, nchan=1)
    
    dataset = VSTPairDataset(args.pairs_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    print(f"Training 1-Channel Denoiser for {args.n_epochs} epochs...")
    for epoch in range(1, args.n_epochs + 1):
        model.net.train()
        epoch_loss = 0.0
        for noisy, clean in loader:
            noisy, clean = noisy.cuda(), clean.cuda()
            optimizer.zero_grad()
            # Predict
            pred = model.net(noisy)[0]
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"  epoch {epoch:3d}/{args.n_epochs}  loss={epoch_loss/len(loader):.6f}")
        if epoch % 25 == 0:
            torch.save(model.net.state_dict(), args.model_dir / f"{args.model_name}_epoch{epoch}.pth")
            
    torch.save(model.net.state_dict(), args.model_dir / f"{args.model_name}_final.pth")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir",     type=Path,  default=TRAIN_PAIRS_DIR)
    p.add_argument("--model_dir",     type=Path,  default=MODEL_SAVE_DIR)
    p.add_argument("--model_name",    type=str,   default=MODEL_NAME)
    p.add_argument("--n_epochs",      type=int,   default=N_EPOCHS)
    p.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    p.add_argument("--weight_decay",  type=float, default=WEIGHT_DECAY)
    p.add_argument("--batch_size",    type=int,   default=BATCH_SIZE)
    p.add_argument("--seed",          type=int,   default=SEED)
    train_denoiser(p.parse_args())
