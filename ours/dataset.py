"""DWI dataset with on-the-fly Rician noise and curriculum noise sampling."""

import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import add_rician_noise

# Spatial constants — must match the model's window-size requirements
BASE_SIZE: int = 160   # resize target before patch extraction
PATCH_SIZE: int = 96   # random crop size used during training


class DWIDataset(Dataset):
    """Single-channel DWI dataset with synthetic Rician noise augmentation.

    Each sample returns a (noisy, clean, sigma) triplet. During training a
    random spatial patch is extracted from the resized image; during validation
    or testing the full resized image is returned.

    Args:
        files:        List of absolute paths to PNG images.
        train:        If ``True``, apply random cropping to ``PATCH_SIZE``.
        noise_levels: List of integer noise percentages to sample from (e.g.
                      ``[1, 3, 5]`` means σ ∈ {0.01, 0.03, 0.05}).
    """

    def __init__(
        self,
        files: list,
        train: bool = True,
        noise_levels: list = None,
    ):
        if noise_levels is None:
            noise_levels = [1]
        self.files = files
        self.train = train
        self.noise_levels = noise_levels
        self.resize = transforms.Resize((BASE_SIZE, BASE_SIZE))
        self.to_tensor = transforms.ToTensor()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img = Image.open(self.files[idx]).convert("L")
        img = self.to_tensor(self.resize(img))  # (1, H, W)  float32 [0, 1]

        if self.train:
            top = random.randint(0, BASE_SIZE - PATCH_SIZE)
            left = random.randint(0, BASE_SIZE - PATCH_SIZE)
            img = img[:, top : top + PATCH_SIZE, left : left + PATCH_SIZE]

        nl = random.choice(self.noise_levels)
        sigma = torch.tensor(nl / 100.0, dtype=torch.float32)
        noisy = add_rician_noise(img, sigma)

        return noisy, img, sigma
