"""Shared utility functions: Rician noise simulation and 2-D wavelet transform."""

import numpy as np
import torch
import pywt


def add_rician_noise(x: torch.Tensor, sigma) -> torch.Tensor:
    """Add synthetic Rician noise to a clean image tensor.

    Models magnitude DWI reconstruction noise:
        Y = sqrt((X + n1)^2 + n2^2),  n1, n2 ~ N(0, sigma^2)

    Args:
        x:     Clean image tensor, arbitrary shape.
        sigma: Noise standard deviation — scalar float or broadcastable tensor.

    Returns:
        Noisy magnitude tensor with the same dtype and device as *x*.
    """
    n1 = torch.randn_like(x) * sigma
    n2 = torch.randn_like(x) * sigma
    return torch.sqrt((x + n1) ** 2 + n2 ** 2)


def dwt2d(x: torch.Tensor) -> torch.Tensor:
    """Apply a single-level 2-D Haar DWT to a batch of single-channel images.

    Args:
        x: Tensor of shape (B, 1, H, W).

    Returns:
        Wavelet coefficient tensor of shape (B, 4, H/2, W/2).
        Channel order: [LL (approx), LH (horiz detail), HL (vert detail), HH (diag detail)].
    """
    coeffs = []
    for i in range(x.shape[0]):
        img = x[i, 0].detach().cpu().numpy()
        cA, (cH, cV, cD) = pywt.dwt2(img, "haar")
        coeffs.append(np.stack([cA, cH, cV, cD], axis=0))
    out = torch.tensor(np.stack(coeffs), dtype=torch.float32)
    return out.to(x.device)
