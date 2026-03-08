"""Evaluation script for the Hybrid Multimodal DWI Denoising Network.

Usage
-----
    python ours/test.py \\
        --data_dir   /path/to/test/images \\
        --checkpoint /path/to/checkpoints/best.pth \\
        --save_dir   /path/to/results

Outputs
-------
* Per-noise-level PSNR / SSIM printed to stdout.
* For each noise level, up to ``--save_per_noise`` (default 25) triplets of
  clean / noisy / denoised PNG images are written to:
      <save_dir>/noise_<NN>/<idx>_clean.png
      <save_dir>/noise_<NN>/<idx>_noisy.png
      <save_dir>/noise_<NN>/<idx>_denoised.png
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import piq

from ours.model import HybridMultiModal
from ours.utils import add_rician_noise

ALL_NOISE = [1, 3, 5, 7, 9, 11, 13, 15]
BASE_SIZE = 160


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Hybrid Multimodal DWI Denoiser")
    p.add_argument("--data_dir",       type=str, required=True,
                   help="Directory of test PNG images.")
    p.add_argument("--checkpoint",     type=str, required=True,
                   help="Path to the saved model checkpoint (best.pth).")
    p.add_argument("--save_dir",       type=str, default="test_results",
                   help="Directory in which to write result images.")
    p.add_argument("--save_per_noise", type=int, default=25,
                   help="Number of image triplets to save per noise level.")
    p.add_argument("--noise_levels",   type=int, nargs="+", default=ALL_NOISE,
                   help="Noise percentage levels to evaluate.")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load model ────────────────────────────────────────────
    model = HybridMultiModal().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Collect test files ────────────────────────────────────
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "*.png")))
    if not test_files:
        test_files = sorted(
            glob.glob(os.path.join(args.data_dir, "**", "*.png"), recursive=True)
        )
    if not test_files:
        raise FileNotFoundError(f"No PNG images found in {args.data_dir}")
    print(f"Test images: {len(test_files)}")

    os.makedirs(args.save_dir, exist_ok=True)
    resize    = transforms.Resize((BASE_SIZE, BASE_SIZE))
    to_tensor = transforms.ToTensor()

    results = {}

    with torch.no_grad():
        for nl in args.noise_levels:
            print(f"\n{'='*40}")
            print(f"Noise level : {nl}%")
            print(f"{'='*40}")

            noise_dir = os.path.join(args.save_dir, f"noise_{nl:02d}")
            os.makedirs(noise_dir, exist_ok=True)

            psnr_total = 0.0
            ssim_total = 0.0
            count      = 0
            saved      = 0

            for img_path in test_files:
                img = Image.open(img_path).convert("L")
                img = to_tensor(resize(img)).to(device)   # (1, H, W)

                clean  = img.unsqueeze(0)                 # (1, 1, H, W)
                sigma  = torch.tensor([nl / 100.0], device=device)
                noisy  = add_rician_noise(img, sigma.item()).unsqueeze(0)

                denoised = torch.clamp(model(noisy, sigma), 0.0, 1.0)

                psnr_val = piq.psnr(denoised, clean, data_range=1.0).item()
                ssim_val = piq.ssim(denoised, clean, data_range=1.0).item()

                psnr_total += psnr_val
                ssim_total += ssim_val
                count += 1

                if saved < args.save_per_noise:
                    prefix = os.path.join(noise_dir, f"{saved:02d}")
                    save_image(clean,    f"{prefix}_clean.png")
                    save_image(noisy,    f"{prefix}_noisy.png")
                    save_image(denoised, f"{prefix}_denoised.png")
                    saved += 1

            avg_psnr = psnr_total / count
            avg_ssim = ssim_total / count
            results[nl] = {"PSNR": avg_psnr, "SSIM": avg_ssim}
            print(f"  PSNR: {avg_psnr:.2f} dB  |  SSIM: {avg_ssim:.4f}")

    # ── Summary table ─────────────────────────────────────────
    print("\n" + "=" * 40)
    print("FINAL TEST SUMMARY")
    print("=" * 40)
    print(f"{'Noise (%)':>10} | {'PSNR (dB)':>10} | {'SSIM':>8}")
    print("-" * 36)
    for nl in args.noise_levels:
        r = results[nl]
        print(f"{nl:>10d} | {r['PSNR']:>10.2f} | {r['SSIM']:>8.4f}")

    overall_psnr = sum(r["PSNR"] for r in results.values()) / len(results)
    overall_ssim = sum(r["SSIM"] for r in results.values()) / len(results)
    print("-" * 36)
    print(f"{'Average':>10} | {overall_psnr:>10.2f} | {overall_ssim:>8.4f}")
    print("=" * 40)
    print(f"\nResult images saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
