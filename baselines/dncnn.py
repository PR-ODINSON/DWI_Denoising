"""DnCNN: Beyond a Gaussian Denoiser.

Reference
---------
Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017).
Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising.
IEEE Transactions on Image Processing, 26(7), 3142–3155.

Architecture
------------
A plain 17-layer fully convolutional network with batch normalisation trained
to predict the noise residual (residual learning):

    denoised = noisy − model(noisy)

Layers
------
1:      Conv(1→64, 3×3) + ReLU
2–16:   Conv(64→64, 3×3) + BN + ReLU
17:     Conv(64→1, 3×3)
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """DnCNN denoising network.

    Args:
        in_channels:  Number of input channels (1 for grayscale DWI).
        num_layers:   Total number of convolutional layers (default 17).
        features:     Number of feature maps in intermediate layers (default 64).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_layers: int = 17,
        features: int = 64,
    ):
        super().__init__()
        layers = []

        # First layer: Conv + ReLU (no BN)
        layers += [
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]

        # Intermediate layers: Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ]

        # Last layer: Conv (outputs residual)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy image tensor, shape (B, 1, H, W), values in [0, 1].

        Returns:
            Denoised image, shape (B, 1, H, W).
        """
        residual = self.net(x)
        return x - residual
