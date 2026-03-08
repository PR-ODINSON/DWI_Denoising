"""Composite training loss for multimodal DWI denoising.

L = L_charb + λ1·(1 − SSIM) + λ2·L_freq + λ3·L_hf

Components
----------
L_charb : Charbonnier loss  — smooth L1 surrogate for pixel fidelity.
SSIM    : Structural Similarity — perceptual structural quality.
L_freq  : Wavelet L1 loss — full-band frequency consistency.
L_hf    : High-frequency wavelet penalty — detail preservation.
"""

import torch
import torch.nn as nn
import piq

from .utils import dwt2d


class StrongLoss(nn.Module):
    """Composite loss combining Charbonnier, SSIM, and wavelet-domain terms.

    Args:
        lambda_ssim: Weight for the SSIM term (default 0.3).
        lambda_freq: Weight for the full-band wavelet L1 term (default 0.2).
        lambda_hf:   Weight for the high-frequency wavelet L1 term (default 0.1).
        eps:         Charbonnier smoothing constant (default 1e-6).
    """

    def __init__(
        self,
        lambda_ssim: float = 0.3,
        lambda_freq: float = 0.2,
        lambda_hf: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_freq = lambda_freq
        self.lambda_hf = lambda_hf
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Charbonnier loss (smooth L1)
        charb = torch.mean(torch.sqrt((pred - target) ** 2 + self.eps))

        # SSIM loss (clamped to valid image range)
        pred_c = torch.clamp(pred, 0.0, 1.0)
        target_c = torch.clamp(target, 0.0, 1.0)
        ssim_val = piq.ssim(pred_c, target_c, data_range=1.0)

        # Wavelet-domain losses
        pred_w = dwt2d(pred)
        target_w = dwt2d(target)
        freq = torch.mean(torch.abs(pred_w - target_w))
        hf = torch.mean(torch.abs(pred_w[:, 1:] - target_w[:, 1:]))  # LH, HL, HH only

        return (
            charb
            + self.lambda_ssim * (1.0 - ssim_val)
            + self.lambda_freq * freq
            + self.lambda_hf * hf
        )
