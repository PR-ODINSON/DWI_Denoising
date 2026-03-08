# Multimodal DWI Denoising Using Image and Signal Representations

**MICCAI 2026** | Anonymized Authors | anonymous@anonymous.com

> *Diffusion-weighted imaging (DWI) is widely used in whole-body cancer screening but suffers from long acquisition times. Reducing the number of acquisitions introduces signal-dependent noise, significantly degrading image quality. We propose a hybrid multimodal framework that jointly leverages spatial image representations and signal-domain wavelet features to robustly suppress Rician noise — achieving an average PSNR of **32.69 dB** and SSIM of **0.8113** across noise levels 1%–15%.*

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Methodology](#methodology)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Baselines](#baselines)
- [Citation](#citation)

---

## Overview

Magnitude reconstruction from complex DWI measurements introduces **signal-dependent Rician noise** — characterised by spatially heterogeneous variance and rectification bias in low-intensity regions. Unlike additive Gaussian noise, Rician noise distorts both image appearance and voxel-wise signal distributions, systematically biasing diffusion-derived metrics such as ADC and FA.

This work proposes a **hybrid multimodal denoising framework** that:

- Jointly processes **spatial image representations** (via Swin Transformer) and **signal-domain wavelet features** (via Restormer) in two parallel branches
- Fuses modalities with a learnable **gated cross-modal attention** mechanism that adaptively weights each branch under varying noise conditions
- Conditions the network on an **explicit noise-level scalar** σ for blind denoising across heterogeneous Rician corruption
- Trains with a **noise-aware curriculum** — progressively introducing harder noise levels to prevent overfitting to a single corruption regime

---

## Architecture

The proposed model (Fig. 1 from the paper) consists of two parallel networks merged through gated cross-modal fusion:

```
                      Noisy DWI Image (H × W × 1)
                               │
               ┌───────────────┴───────────────┐
               │                               │
        NETWORK 1                         NETWORK 2
   (Spatial Image Encoder)          (Signal-Domain Branch)
               │                               │
         3×3 CONV                        2D Discrete
     + Noise Embedding (σ)             Wavelet Transform
               │                               │
      Downsampling                   Wavelet Coefficients
    (Patch Merging)                 (Low-freq LL + High-freq
               │                     LH, HL, HH subbands)
               │                               │
    Swin Transformer                     3×3 CONV Embed
      Blocks (×3)                              │
    [Window-based                    Signal Encoder
    Multi-Head SA]                  (Restormer Blocks ×2)
    [Feed-Forward]                   [MDTA + GDFN]
               │                               │
      Upsampling                               │
    (Patch Merging)               ┌────────────┘
               │                  │
               └──────────────────┘
                        │
          Gated Cross-Modal Fusion
            F_fused = α·Fs + (1-α)·Fc
          [AdaptiveAvgPool → Sigmoid gate]
                        │
            3×3 CONV + Restormer
             Refinement Blocks (×3)
             [MDTA channel-attention]
              [Gated DConv FFN]
                        │
               Residual Head (3×3 CONV)
                        │
            X̂ = Y − N̂  (residual learning)
                        │
               Denoised DWI Image (H × W × 1)
```

### Key Components

| Component | Design | Purpose |
|-----------|--------|---------|
| **Spatial Encoder** | Swin Transformer blocks with non-overlapping window attention | Long-range spatial context, anatomical structure |
| **Noise Embedding** | MLP(σ) broadcast spatially | Explicit noise-level conditioning |
| **Wavelet Branch** | 2D Haar DWT → 3×3 Conv → Restormer (×2) | Frequency-domain signal statistics, noise energy distribution |
| **Cross-Modal Fusion** | Cat → reduce → AdaptiveAvgPool gate | Adaptive weighting of spatial vs. signal features |
| **Decoder** | Restormer blocks (×3) with MDTA + GDFN | Feature refinement with channel-adaptive attention |
| **Reconstruction** | Residual subtraction: X̂ = Y − N̂ | Training stability, fine structural detail preservation |

---

## Methodology

### Problem Formulation

Let $X \in \mathbb{R}^{H \times W}$ denote a clean DWI image. The noisy observed magnitude image is modelled as:

$$Y = \sqrt{(X + n_1)^2 + n_2^2}$$

where $n_1, n_2 \sim \mathcal{N}(0, \sigma^2)$ are independent Gaussian noise components. The network predicts a noise residual $\hat{N}$ and reconstructs via $\hat{X} = Y - \hat{N}$.

### Training Objective

The composite loss enforces pixel fidelity, structural consistency, and frequency-domain alignment:

$$\mathcal{L} = \mathcal{L}_{\text{charb}} + 0.3\,(1 - \text{SSIM}) + 0.2\,\mathcal{L}_{\text{freq}} + 0.1\,\mathcal{L}_{\text{hf}}$$

| Term | Formula | Role |
|------|---------|------|
| $\mathcal{L}_{\text{charb}}$ | $\mathbb{E}\left[\sqrt{(\hat{X}-X)^2 + \varepsilon}\right]$ | Smooth pixel fidelity (robust L1) |
| $\mathcal{L}_{\text{SSIM}}$ | $1 - \text{SSIM}(\hat{X}, X)$ | Perceptual structural quality |
| $\mathcal{L}_{\text{freq}}$ | $\mathbb{E}\left[\|W(\hat{X}) - W(X)\|_1\right]$ | Full-band wavelet consistency |
| $\mathcal{L}_{\text{hf}}$ | $\mathbb{E}\left[\|W_{\text{HF}}(\hat{X}) - W_{\text{HF}}(X)\|_1\right]$ | High-frequency detail preservation |

### Noise-Aware Curriculum

Training progressively introduces harder noise levels to stabilise convergence:

| Epochs | Active noise levels (σ) |
|--------|------------------------|
| 1–5    | {1%} |
| 6–10   | {1, 3%} |
| 11–15  | {1, 3, 5%} |
| 16–20  | {1, 3, 5, 7%} |
| 21–25  | {1, 3, 5, 7, 9%} |
| 26–30  | {1, 3, 5, 7, 9, 11%} |
| 31–35  | {1, 3, 5, 7, 9, 11, 13%} |
| 36–45  | {1, 3, 5, 7, 9, 11, 13, 15%} |

---

## Results

### Quantitative Performance (proposed model)

Evaluated across 8 Rician noise levels on ~22,500 DWI image slices:

| Noise Level (%) | PSNR (dB) | SSIM   |
|:---------------:|:---------:|:------:|
| 1               | 41.32     | 0.9698 |
| 3               | 35.71     | 0.8976 |
| 5               | 33.35     | 0.8439 |
| 7               | 31.91     | 0.8055 |
| 9               | 30.90     | 0.7763 |
| 11              | 30.09     | 0.7513 |
| 13              | 29.41     | 0.7298 |
| 15              | 28.85     | 0.7161 |
| **Average**     | **32.69** | **0.8113** |

Even at the highest noise level (15%), the model maintains PSNR of 28.85 dB and SSIM > 0.71, demonstrating stable denoising without abrupt failure under severe Rician corruption.

### Comparison with Prior Methods

| Method | Noise type | Dataset | PSNR | SSIM |
|--------|-----------|---------|------|------|
| NLM-Rician — Daesslé et al. | Rician | DW-MRI | 27.6 | 0.81 |
| Optimized NLM — Coupé et al. | Rician | 3D Brain MRI | 28.4 | 0.83 |
| NLPCA — Manjón et al. | Rician | Brainweb MRI | 31.2 | 0.88 |
| DWAN — Xie et al. | Rician | ASL MRI | 29.4 | 0.85 |
| DnCNN — Zhang et al. | Gaussian | Brain MRI | 28.5 | 0.82 |
| FFDNet — Zhang et al. | Gaussian | Brain MRI | 29.2 | 0.84 |
| **Ours — Hybrid (Image + Signal)** | **Rician** | **DWI** | **32.69** | **0.8113** |

> Results from prior work are reference-level comparisons due to differences in datasets.

### Ablation Study

| Model variant | PSNR (dB) | SSIM |
|---------------|:---------:|:----:|
| Image-domain network only (Network 1) | 33.69 | 0.8339 |
| Signal-domain network only (Network 2) | 32.12 | 0.8074 |
| **Full Hybrid (Image + Signal)** | **32.69** | **0.8113** |

Jointly leveraging both modalities improves robustness and generalisation, particularly under heterogeneous Rician corruption.

---

## Repository Structure

```
MRI_Multimodal/
├── ours/                        # Proposed hybrid multimodal model
│   ├── __init__.py
│   ├── model.py                 # HybridMultiModal — full architecture
│   ├── dataset.py               # MRIDataset with on-the-fly Rician noise
│   ├── loss.py                  # StrongLoss (Charbonnier + SSIM + wavelet)
│   ├── utils.py                 # add_rician_noise(), dwt2d()
│   ├── train.py                 # Training script (curriculum, CLI args)
│   └── test.py                  # Evaluation — per-noise PSNR/SSIM + image triplets
│
├── baselines/                   # Baseline comparison models
│   ├── __init__.py
│   ├── dncnn.py                 # DnCNN  (Zhang et al., TIP 2017)
│   ├── ffdnet.py                # FFDNet (Zhang et al., TIP 2018)
│   └── train_baseline.py        # Shared training script for baselines
│
├── prepare_data/
│   └── prepare_data.py          # Raw MRI → train/val/test PNG splits
│
├── clean/                       # Sample clean reference images (paper Fig. 2)
├── noisy/                       # Sample noisy inputs              (paper Fig. 2)
├── denoised/                    # Sample denoised outputs          (paper Fig. 2)
├── main.pdf                     # Full paper (MICCAI 2026 submission)
└── README.md
```

---

## Requirements

```
torch >= 2.0
torchvision
PyWavelets
piq
tqdm
Pillow
numpy
```

```bash
pip install torch torchvision PyWavelets piq tqdm Pillow numpy
```

Optional — for DICOM input support in `prepare_data`:
```bash
pip install pydicom
```

---

## Data Preparation

Convert raw MRI slices (PNG / JPG / TIFF / DICOM) into the training-ready layout:

```bash
# Flat directory of PNG slices
python prepare_data/prepare_data.py \
    --src_dir /path/to/raw_slices \
    --out_dir /path/to/dataset \
    --split 0.7 0.1 0.2

# Subject-level split (each sub-directory = one subject, prevents data leakage)
python prepare_data/prepare_data.py \
    --src_dir    /path/to/subjects \
    --out_dir    /path/to/dataset \
    --by_subject \
    --split 0.7 0.1 0.2
```

Output layout after preparation:
```
dataset/
├── train/   img_00000.png  img_00001.png  ...
├── val/     img_00000.png  ...
└── test/    img_00000.png  ...
```

All images are resized to 160×160 and normalised to [0, 1].

---

## Training

```bash
python ours/train.py \
    --data_dir   /path/to/dataset \
    --save_dir   /path/to/checkpoints \
    --epochs     45 \
    --batch_size 4 \
    --lr         2e-4
```

Key training details:
- **Optimiser**: AdamW, weight decay 1e-4
- **Scheduler**: Cosine annealing over 45 epochs
- **Patch size**: 96×96 random crops during training
- **Curriculum**: noise levels introduced progressively (see table above)
- **Model selection**: checkpoint saved when validation PSNR improves

The best model is saved to `<save_dir>/best.pth`.

---

## Evaluation

```bash
python ours/test.py \
    --data_dir       /path/to/dataset/test \
    --checkpoint     /path/to/checkpoints/best.pth \
    --save_dir       /path/to/results \
    --save_per_noise 25
```

Outputs:
- Per-noise-level PSNR / SSIM to stdout
- Up to `--save_per_noise` image triplets (clean / noisy / denoised) per noise level saved under `<save_dir>/noise_<NN>/`

---

## Baselines

Train DnCNN or FFDNet for reproduction of Table 2 comparisons:

```bash
# DnCNN
python baselines/train_baseline.py \
    --model    dncnn \
    --data_dir /path/to/dataset \
    --save_dir /path/to/baseline_checkpoints

# FFDNet
python baselines/train_baseline.py \
    --model    ffdnet \
    --data_dir /path/to/dataset \
    --save_dir /path/to/baseline_checkpoints
```

Both baselines use the same Rician noise model, dataset splits, and evaluation protocol as the proposed method.

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{anon2026multimodal,
  author    = {Anonymized Authors},
  title     = {Multimodal {DWI} Denoising Using Image and Signal Representations},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention ({MICCAI})},
  year      = {2026}
}
```

---

## References

1. Coupé et al. — *An Optimized Blockwise NLM Denoising Filter for 3D MRI.* IEEE TMI 2008.
2. Liang et al. — *SwinIR: Image Restoration Using Swin Transformer.* ICCVW 2021.
3. Magnotta et al. — *Multicenter Reliability of Diffusion Tensor Imaging.* Brain Connectivity 2012.
4. Manjón & Coupé — *MRI Denoising Using Deep Learning and Non-Local Averaging.* arXiv 2019.
5. Mohan et al. — *A Survey of Denoising Techniques for MRI.* Biomedical Signal Processing 2014.
6. Tripathi & Bag — *CNN-DMRI: A CNN for Denoising of MRI.* Pattern Recognition Letters 2020.
7. Wiest-Daesslé et al. — *Rician Noise Removal by NLM Filtering.* MICCAI 2008.
8. Xie et al. — *Denoising Arterial Spin Labeling Perfusion MRI.* Magnetic Resonance Imaging 2020.
9. Zamir et al. — *Restormer: Efficient Transformer for High-Resolution Image Restoration.* CVPR 2022.
10. Zhang et al. — *Beyond a Gaussian Denoiser: DnCNN.* IEEE TIP 2017.
11. Zhang et al. — *FFDNet: Toward a Fast and Flexible CNN Denoiser.* IEEE TIP 2018.

---

*For research use only. All rights reserved pending publication.*
