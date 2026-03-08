"""Training script for baseline denoising models (DnCNN, FFDNet).

Usage
-----
    # Train DnCNN
    python baselines/train_baseline.py \\
        --model     dncnn \\
        --data_dir  /path/to/dataset \\
        --save_dir  /path/to/baseline_checkpoints

    # Train FFDNet
    python baselines/train_baseline.py \\
        --model     ffdnet \\
        --data_dir  /path/to/dataset \\
        --save_dir  /path/to/baseline_checkpoints

Notes
-----
Both baselines are trained with all noise levels from the start (no curriculum),
using the same Rician noise model as the proposed method (see ours/utils.py).
The loss is a simple Charbonnier loss for pixel fidelity.
"""

import argparse
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import piq

from baselines.dncnn import DnCNN
from baselines.ffdnet import FFDNet
from ours.dataset import DWIDataset

ALL_NOISE = [1, 3, 5, 7, 9, 11, 13, 15]


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps))


def parse_args():
    p = argparse.ArgumentParser(description="Train a baseline DWI denoiser")
    p.add_argument("--model",        type=str, required=True,
                   choices=["dncnn", "ffdnet"],
                   help="Baseline model to train.")
    p.add_argument("--data_dir",     type=str, required=True,
                   help="Root dataset directory with PNG images.")
    p.add_argument("--save_dir",     type=str, default="baseline_checkpoints")
    p.add_argument("--epochs",       type=int, default=45)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split",    type=float, default=0.2)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def build_model(name: str) -> nn.Module:
    if name == "dncnn":
        return DnCNN()
    if name == "ffdnet":
        return FFDNet()
    raise ValueError(f"Unknown model: {name}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}  |  Model : {args.model.upper()}")

    save_dir = os.path.join(args.save_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)

    # ── Data loading ──────────────────────────────────────────
    train_files = sorted(glob.glob(os.path.join(args.data_dir, "train", "*.png")))
    val_files   = sorted(glob.glob(os.path.join(args.data_dir, "val",   "*.png")))

    if not train_files:
        all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.png")))
        if not all_files:
            raise FileNotFoundError(f"No PNG files found in {args.data_dir}")
        random.shuffle(all_files)
        n_val = max(1, int(len(all_files) * args.val_split))
        train_files = all_files[n_val:]
        val_files   = all_files[:n_val]

    print(f"Train : {len(train_files)}  |  Val : {len(val_files)}")

    # ── Model & optimiser ─────────────────────────────────────
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_psnr = 0.0

    # ── Training loop ─────────────────────────────────────────
    for ep in range(args.epochs):
        model.train()
        train_loader = DataLoader(
            DWIDataset(train_files, train=True, noise_levels=ALL_NOISE),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        running_loss = 0.0
        for noisy, clean, sigma in tqdm(train_loader, leave=False, desc=f"  ep {ep+1:02d}"):
            noisy = noisy.to(device)
            clean = clean.to(device)
            sigma = sigma.to(device)

            optimizer.zero_grad()
            # Both DnCNN and FFDNet accept (noisy, sigma); DnCNN ignores sigma
            if args.model == "ffdnet":
                pred = model(noisy, sigma)
            else:
                pred = model(noisy)

            loss = charbonnier(pred, clean)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────
        model.eval()
        psnr_total, count = 0.0, 0
        val_loader = DataLoader(
            DWIDataset(val_files, train=False, noise_levels=ALL_NOISE),
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
        )

        with torch.no_grad():
            for noisy, clean, sigma in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                sigma = sigma.to(device)

                if args.model == "ffdnet":
                    out = torch.clamp(model(noisy, sigma), 0.0, 1.0)
                else:
                    out = torch.clamp(model(noisy), 0.0, 1.0)

                psnr_total += piq.psnr(out, clean, data_range=1.0).item()
                count += 1

        avg_psnr = psnr_total / count
        print(
            f"Epoch {ep+1:02d}/{args.epochs}  |  "
            f"Loss: {avg_loss:.4f}  |  Val PSNR: {avg_psnr:.2f} dB"
        )

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            ckpt_path = os.path.join(save_dir, "best.pth")
            torch.save(
                {
                    "epoch": ep + 1,
                    "model_state_dict": model.state_dict(),
                    "psnr": best_psnr,
                },
                ckpt_path,
            )
            print(f"  ✔ Saved best model  (PSNR {best_psnr:.2f} dB) → {ckpt_path}")

    print(f"\n{args.model.upper()} training complete.")
    print(f"Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
