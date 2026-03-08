"""Data preparation script for the Hybrid Multimodal DWI Denoising framework.

Converts raw DWI slices (PNG, JPG, TIFF, or DICOM) into the format expected
by the training pipeline:

    <out_dir>/
        train/   *.png
        val/     *.png
        test/    *.png

Each output image is:
    • Loaded and converted to grayscale (single-channel).
    • Resized to BASE_SIZE × BASE_SIZE (default 160).
    • Normalised to [0, 1] float, then saved as 8-bit PNG.

Subject-level splitting is supported: pass ``--subject_pattern`` as a glob
wildcard so that all slices from one subject stay in the same partition.

Usage examples
--------------
Flat directory of PNG slices
    python prepare_data/prepare_data.py \\
        --src_dir /raw/DWI_slices \\
        --out_dir /dataset

Subject-level split (each sub-directory = one subject)
    python prepare_data/prepare_data.py \\
        --src_dir   /raw/DWI_subjects \\
        --out_dir   /dataset \\
        --by_subject \\
        --split 0.7 0.1 0.2

DICOM support requires ``pydicom`` (``pip install pydicom``).
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ── Optional DICOM support ────────────────────────────────────────────────────
try:
    import pydicom
    _DICOM_AVAILABLE = True
except ImportError:
    _DICOM_AVAILABLE = False

BASE_SIZE: int = 160
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"}


# ─────────────────────────────────────────────────────────────────────────────
# File I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """Load any supported image file and return a float32 grayscale array [0,1]."""
    ext = path.suffix.lower()
    if ext == ".dcm":
        if not _DICOM_AVAILABLE:
            raise ImportError(
                "pydicom is required to load DICOM files: pip install pydicom"
            )
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
    else:
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32)

    # Normalise to [0, 1]
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    return arr


def save_png(arr: np.ndarray, out_path: Path, size: int = BASE_SIZE):
    """Resize and save a float32 [0,1] array as 8-bit PNG."""
    img = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    resample = getattr(Image, "Resampling", Image).BILINEAR
    img = img.resize((size, size), resample)
    img.save(str(out_path))


# ─────────────────────────────────────────────────────────────────────────────
# Splitting helpers
# ─────────────────────────────────────────────────────────────────────────────

def split_files(files: list, train: float, val: float, seed: int):
    """Split a flat list of files into train / val / test lists."""
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val   = int(n * val)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def split_subjects(subject_dirs: list, train: float, val: float, seed: int):
    """Split at subject level to prevent data leakage."""
    rng = random.Random(seed)
    shuffled = list(subject_dirs)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val   = int(n * val)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Prepare DWI dataset for DWI denoising")
    p.add_argument("--src_dir",    type=str, required=True,
                   help="Source directory containing raw DWI images.")
    p.add_argument("--out_dir",    type=str, required=True,
                   help="Output directory (will be created if absent).")
    p.add_argument("--size",       type=int, default=BASE_SIZE,
                   help=f"Output spatial resolution (default {BASE_SIZE}).")
    p.add_argument("--split",      type=float, nargs=3, default=[0.7, 0.1, 0.2],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Train/Val/Test proportions (must sum to 1.0).")
    p.add_argument("--by_subject", action="store_true",
                   help="Treat each sub-directory of src_dir as one subject "
                        "and split at subject level.")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def collect_files(directory: Path) -> list:
    """Recursively collect all supported image files under *directory*."""
    found = []
    for ext in SUPPORTED_EXT:
        found.extend(directory.rglob(f"*{ext}"))
        found.extend(directory.rglob(f"*{ext.upper()}"))
    return sorted(set(found))


def process_split(files: list, out_dir: Path, split_name: str, size: int):
    """Process and save all files belonging to one split."""
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for i, fp in enumerate(files):
        try:
            arr = load_image(fp)
            out_path = split_dir / f"img_{i:05d}.png"
            save_png(arr, out_path, size=size)
            n_ok += 1
        except Exception as exc:
            print(f"  [WARN] Skipped {fp.name}: {exc}", file=sys.stderr)

    return n_ok


def main():
    args = parse_args()

    train_frac, val_frac, test_frac = args.split
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("--split fractions must sum to 1.0")

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.by_subject:
        # Each immediate sub-directory is one subject
        subjects = sorted(
            d for d in src_dir.iterdir() if d.is_dir()
        )
        if not subjects:
            raise FileNotFoundError(f"No sub-directories found in {src_dir}")

        print(f"Found {len(subjects)} subjects.")
        train_subjects, val_subjects, test_subjects = split_subjects(
            subjects, train_frac, val_frac, args.seed
        )

        split_map = {
            "train": train_subjects,
            "val":   val_subjects,
            "test":  test_subjects,
        }
        for split_name, subj_list in split_map.items():
            files = []
            for s in subj_list:
                files.extend(collect_files(s))
            n = process_split(files, out_dir, split_name, args.size)
            print(
                f"  {split_name:5s}: {len(subj_list)} subjects "
                f"→ {n} images saved to {out_dir / split_name}"
            )
    else:
        # Flat file split
        all_files = collect_files(src_dir)
        if not all_files:
            raise FileNotFoundError(
                f"No supported images ({', '.join(SUPPORTED_EXT)}) found in {src_dir}"
            )

        print(f"Found {len(all_files)} images.")
        train_files, val_files, test_files = split_files(
            all_files, train_frac, val_frac, args.seed
        )

        for split_name, files in [
            ("train", train_files),
            ("val",   val_files),
            ("test",  test_files),
        ]:
            n = process_split(files, out_dir, split_name, args.size)
            print(f"  {split_name:5s}: {n} images → {out_dir / split_name}")

    print("\nDataset preparation complete.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
