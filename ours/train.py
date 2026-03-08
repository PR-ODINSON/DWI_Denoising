"""Training script for the Hybrid Multimodal DWI Denoising Network.

Usage
-----
    python ours/train.py \\
        --data_dir  /path/to/dataset \\
        --save_dir  /path/to/checkpoints \\
        --epochs    45 \\
        --batch_size 4 \\
        --lr        2e-4

The dataset directory must contain images directly or in train/ and val/
sub-directories produced by prepare_data/prepare_data.py.

Noise-Aware Curriculum
-----------------------
Noise levels are progressively expanded across epochs so the model first
learns easy (low-noise) restoration before tackling heavy corruption:

    Epochs  1-5  : σ ∈ {1%}
    Epochs  6-10 : σ ∈ {1, 3%}
    Epochs 11-15 : σ ∈ {1, 3, 5%}
    ...
    Epochs 36-45 : σ ∈ {1, 3, 5, 7, 9, 11, 13, 15%}
"""

import argparse
import glob
import os
import random
import sys

# Allow running as a top-level script: add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import piq

from ours.model import HybridMultiModal
from ours.dataset import DWIDataset
from ours.loss import StrongLoss

# ──────────────────────────────────────────────────────────────
# Noise curriculum schedule
# ──────────────────────────────────────────────────────────────
ALL_NOISE = [1, 3, 5, 7, 9, 11, 13, 15]

def _noise_schedule(epoch: int) -> list:
    """Return the active noise level list for *epoch* (0-indexed)."""
    thresholds = [5, 10, 15, 20, 25, 30, 35]
    for i, t in enumerate(thresholds):
        if epoch < t:
            return ALL_NOISE[: i + 1]
    return ALL_NOISE


# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train Hybrid Multimodal DWI Denoiser")
    p.add_argument("--data_dir",   type=str, required=True,
                   help="Root dataset directory (must contain PNG images).")
    p.add_argument("--save_dir",   type=str, default="checkpoints",
                   help="Directory in which to save model weights.")
    p.add_argument("--epochs",     type=int, default=45)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split",  type=float, default=0.2,
                   help="Fraction of files to reserve for validation (ignored if "
                        "val/ sub-directory already exists).")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data loading ───────────────────────────────────────────
    train_glob = os.path.join(args.data_dir, "train", "*.png")
    val_glob   = os.path.join(args.data_dir, "val",   "*.png")

    train_files = sorted(glob.glob(train_glob))
    val_files   = sorted(glob.glob(val_glob))

    # Fallback: single flat directory
    if not train_files:
        all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.png")))
        if not all_files:
            raise FileNotFoundError(f"No PNG files found in {args.data_dir}")
        random.shuffle(all_files)
        n_val = max(1, int(len(all_files) * args.val_split))
        train_files = all_files[n_val:]
        val_files   = all_files[:n_val]

    print(f"Train : {len(train_files)} images  |  Val : {len(val_files)} images")

    # ── Model, optimiser, scheduler, loss ─────────────────────
    model = HybridMultiModal().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = StrongLoss()

    best_psnr = 0.0

    # ── Training loop ─────────────────────────────────────────
    for ep in range(args.epochs):
        curr_noise = _noise_schedule(ep)
        print(f"\nEpoch {ep + 1:02d}/{args.epochs}  |  noise levels: {curr_noise}")

        model.train()
        train_loader = DataLoader(
            DWIDataset(train_files, train=True, noise_levels=curr_noise),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        running_loss = 0.0
        for noisy, clean, sigma in tqdm(train_loader, leave=False, desc="  train"):
            noisy = noisy.to(device)
            clean = clean.to(device)
            sigma = sigma.to(device)

            optimizer.zero_grad()
            pred = model(noisy, sigma)
            loss = criterion(pred, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        # ── Validation ──────────────────────────────────────
        model.eval()
        psnr_total, count = 0.0, 0

        val_loader = DataLoader(
            DWIDataset(val_files, train=False, noise_levels=ALL_NOISE),
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
        )

        with torch.no_grad():
            for noisy, clean, sigma in tqdm(val_loader, leave=False, desc="  val  "):
                noisy = noisy.to(device)
                clean = clean.to(device)
                sigma = sigma.to(device)

                out = torch.clamp(model(noisy, sigma), 0.0, 1.0)
                psnr_total += piq.psnr(out, clean, data_range=1.0).item()
                count += 1

        avg_psnr = psnr_total / count
        print(
            f"  Loss: {avg_loss:.4f}  |  Val PSNR: {avg_psnr:.2f} dB"
            f"  |  LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best checkpoint
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            ckpt_path = os.path.join(args.save_dir, "best.pth")
            torch.save(
                {
                    "epoch": ep + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "psnr": best_psnr,
                },
                ckpt_path,
            )
            print(f"  ✔ Saved best model  (PSNR {best_psnr:.2f} dB) → {ckpt_path}")

    print("\nTraining complete.")
    print(f"Best validation PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
