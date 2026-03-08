"""Hybrid Multimodal DWI Denoising Network.

Architecture overview
---------------------
Network 1 — Spatial image encoder
    3×3 Conv  →  noise embedding (MLP)  →  SwinTransformer blocks (×3)

Network 2 — Signal-domain wavelet branch
    2D Haar DWT  →  3×3 Conv  →  Restormer blocks (×2)

Fusion & decoder
    Gated Cross-Modal Fusion  →  Restormer decoder blocks (×3)  →  residual head

Output:  denoised = noisy − residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import dwt2d

# ────────────────────────────────────────────────────────────
# Default architecture hyperparameters
# ────────────────────────────────────────────────────────────
DIM: int = 96
HEADS: int = 6
WINDOW: int = 8


# ════════════════════════════════════════════════════════════
# Swin-style windowed self-attention
# ════════════════════════════════════════════════════════════

class WindowAttention(nn.Module):
    """Non-overlapping window-based multi-head self-attention (Swin Transformer).

    Spatial tokens inside each window attend to one another, providing
    efficient local–global context within a fixed receptive field.

    Args:
        dim:         Feature dimension (must equal DIM).
        num_heads:   Number of attention heads.
        window_size: Side-length of each square attention window.
    """

    def __init__(self, dim: int = DIM, num_heads: int = HEADS, window_size: int = WINDOW):
        super().__init__()
        self.ws = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)  – channel-last layout expected by Swin
        B, H, W, C = x.shape
        ws = self.ws

        # Partition into non-overlapping windows: (B·nW, ws², C)
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, ws * ws, C)

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(q.size(0), q.size(1), self.num_heads, -1).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, -1).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, -1).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(x.size(0), ws * ws, C)
        out = self.proj(out)

        # Reverse window partition
        out = out.view(B, H // ws, W // ws, ws, ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        return out.view(B, H, W, C)


class SwinBlock(nn.Module):
    """Swin Transformer block: LayerNorm → WindowAttention + LayerNorm → MLP.

    Operates in channel-last (B, H, W, C) internally and converts back to
    channel-first (B, C, H, W) for compatibility with Conv layers.

    Args:
        dim: Feature channel dimension.
    """

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        # (B, H, W, C) → (B, C, H, W)
        return x.permute(0, 3, 1, 2)


# ════════════════════════════════════════════════════════════
# Restormer-style channel attention blocks
# ════════════════════════════════════════════════════════════

class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention (Restormer, CVPR 2022).

    Computes attention across channels (tokens = channels, keys = spatial
    positions) using depth-wise convolution to enrich Q/K/V projections.

    Args:
        dim:       Feature channel dimension.
        num_heads: Number of transposed attention heads.
    """

    def __init__(self, dim: int = DIM, num_heads: int = HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv_dw(self.qkv(x))          # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to (B, heads, C/heads, H*W)
        q = q.reshape(B, self.num_heads, -1, H * W)
        k = k.reshape(B, self.num_heads, -1, H * W)
        v = v.reshape(B, self.num_heads, -1, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, C/h, C/h)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(B, C, H, W)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network (Restormer, CVPR 2022).

    Applies a gated activation: fc2(GELU(dw(fc1_a)) * dw(fc1_b)).

    Args:
        dim: Feature channel dimension.
    """

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(
            dim * 4, dim * 4, kernel_size=3, padding=1, groups=dim * 4, bias=False
        )
        self.fc2 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(self.fc1(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.fc2(F.gelu(x1) * x2)


class RestormerBlock(nn.Module):
    """Single Restormer block: residual MDTA + residual GDFN.

    Args:
        dim: Feature channel dimension.
    """

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.attn = MDTA(dim)
        self.ffn = GDFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


# ════════════════════════════════════════════════════════════
# Wavelet branch
# ════════════════════════════════════════════════════════════

class WaveletBranch(nn.Module):
    """Signal-domain branch processing 2D DWT coefficients.

    Pipeline: noisy image → 2D Haar DWT → 3×3 Conv embed → Restormer blocks.

    The output spatial size is H/2 × W/2; the caller up-samples to match the
    spatial encoder before fusion.

    Args:
        dim: Embedding dimension (must match spatial encoder).
    """

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.embed = nn.Conv2d(4, dim, kernel_size=3, padding=1)
        self.encoder = nn.Sequential(
            RestormerBlock(dim),
            RestormerBlock(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) → coeffs: (B, 4, H/2, W/2)
        coeffs = dwt2d(x)
        feat = self.embed(coeffs)
        return self.encoder(feat)


# ════════════════════════════════════════════════════════════
# Gated cross-modal fusion
# ════════════════════════════════════════════════════════════

class CrossModalFusion(nn.Module):
    """Memory-efficient gated fusion of spatial and wavelet feature streams.

    Fused representation:
        F_fused = α · Fs + (1 − α) · Fc
    where Fc = reduce(cat(Fs, Fw)) and α is a learned channel gate.

    Args:
        dim: Feature channel dimension.
    """

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.reduce = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, Fs: torch.Tensor, Fw: torch.Tensor) -> torch.Tensor:
        Fc = self.reduce(torch.cat([Fs, Fw], dim=1))   # combined projection
        alpha = self.gate(Fc)                           # channel-wise gate ∈ (0,1)
        return alpha * Fs + (1.0 - alpha) * Fc


# ════════════════════════════════════════════════════════════
# Full hybrid model
# ════════════════════════════════════════════════════════════

class HybridMultiModal(nn.Module):
    """Hybrid Multimodal DWI Denoising Network.

    Jointly exploits spatial Swin Transformer features and signal-domain wavelet
    features via adaptive gated cross-modal fusion, followed by a Restormer
    decoder and residual reconstruction.

    Args:
        dim: Base feature dimension (default 96).

    Inputs:
        x     : Noisy DWI image, shape (B, 1, H, W), values in [0, 1].
        sigma : Per-sample noise level scalar, shape (B,), values in (0, 1).

    Returns:
        Denoised image, shape (B, 1, H, W).
    """

    def __init__(self, dim: int = DIM):
        super().__init__()

        # Spatial branch
        self.embed = nn.Conv2d(1, dim, kernel_size=3, padding=1)
        self.noise_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.spatial_encoder = nn.Sequential(
            SwinBlock(dim),
            SwinBlock(dim),
            SwinBlock(dim),
        )

        # Wavelet (signal-domain) branch
        self.wavelet_branch = WaveletBranch(dim)

        # Fusion
        self.fusion = CrossModalFusion(dim)

        # Transformer decoder
        self.decoder = nn.Sequential(
            RestormerBlock(dim),
            RestormerBlock(dim),
            RestormerBlock(dim),
        )

        # Residual reconstruction head
        self.recon = nn.Conv2d(dim, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # ── Spatial branch ─────────────────────────────────────
        Fs = self.embed(x)

        # Broadcast noise embedding spatially for conditioning
        noise_feat = (
            self.noise_embed(sigma.view(-1, 1))   # (B, dim)
            .unsqueeze(-1)
            .unsqueeze(-1)                         # (B, dim, 1, 1)
        )
        Fs = Fs + noise_feat
        Fs = self.spatial_encoder(Fs)             # (B, dim, H, W)

        # ── Wavelet branch ────────────────────────────────────
        Fw = self.wavelet_branch(x)               # (B, dim, H/2, W/2)
        Fw = F.interpolate(
            Fw, size=Fs.shape[-2:], mode="bilinear", align_corners=False
        )                                          # (B, dim, H, W)

        # ── Fusion & decoder ──────────────────────────────────
        F_fused = self.fusion(Fs, Fw)
        F_fused = self.decoder(F_fused)

        # Residual subtraction
        residual = self.recon(F_fused)
        return x - residual
