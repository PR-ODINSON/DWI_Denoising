"""FFDNet: Fast and Flexible CNN-Based Image Denoiser.

Reference
---------
Zhang, K., Zuo, W., & Zhang, L. (2018).
FFDNet: Toward a Fast and Flexible Solution for CNN-Based Image Denoising.
IEEE Transactions on Image Processing, 27(9), 4608–4622.

Architecture
------------
FFDNet conditions the denoiser on an explicit noise-level map M of the same
spatial resolution as the input. The map is concatenated channel-wise with the
noisy image before being fed into a DnCNN-style backbone.

This enables a single trained model to handle a continuous range of noise
levels at inference time by simply varying M.

Forward interface
-----------------
    model(noisy, sigma)

where *sigma* is a per-sample scalar (shape [B]) that is broadcast to a
uniform noise map of shape (B, 1, H, W).
"""

import torch
import torch.nn as nn


class FFDNet(nn.Module):
    """FFDNet denoising network with explicit noise-level conditioning.

    Args:
        in_channels: Number of image input channels (1 for grayscale DWI).
        num_layers:  Total conv layers in the backbone (default 15).
        features:    Intermediate feature maps (default 64).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_layers: int = 15,
        features: int = 64,
    ):
        super().__init__()
        # Input to backbone = image + noise map
        backbone_in = in_channels + 1

        layers = []

        # First layer (no BN)
        layers += [
            nn.Conv2d(backbone_in, features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]

        # Intermediate layers
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ]

        # Last layer (outputs residual)
        layers += [
            nn.Conv2d(features, in_channels, kernel_size=3, padding=1, bias=True),
        ]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     Noisy image, shape (B, 1, H, W), values in [0, 1].
            sigma: Per-sample noise level scalar, shape (B,), values in (0, 1).

        Returns:
            Denoised image, shape (B, 1, H, W).
        """
        B, C, H, W = x.shape
        # Build spatially uniform noise-level map for each sample in the batch
        noise_map = sigma.view(B, 1, 1, 1).expand(B, 1, H, W)
        inp = torch.cat([x, noise_map], dim=1)        # (B, 2, H, W)
        residual = self.net(inp)
        return x - residual
