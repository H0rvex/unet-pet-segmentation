# U-Net: Image Segmentation from Scratch

PyTorch implementation of the U-Net architecture ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)) for semantic image segmentation, built from scratch.

## Architecture

```
Input (3, 128, 128)
    │
    ├── Encoder 1: 3 → 64 channels   ──── skip1 ────┐
    │   MaxPool 128→64                                │
    ├── Encoder 2: 64 → 128 channels  ──── skip2 ──┐ │
    │   MaxPool 64→32                               │ │
    ├── Encoder 3: 128 → 256 channels ──── skip3 ─┐│ │
    │   MaxPool 32→16                              ││ │
    │                                              ││ │
    ├── Bottleneck: 256 → 512 channels             ││ │
    │                                              ││ │
    ├── Decoder 1: 512 → 256  ← concat ← skip3 ───┘│ │
    ├── Decoder 2: 256 → 128  ← concat ← skip2 ────┘ │
    ├── Decoder 3: 128 → 64   ← concat ← skip1 ──────┘
    │
    └── 1×1 Conv: 64 → num_classes
Output (num_classes, 128, 128)
```

Each encoder/decoder block uses: `Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → ReLU`

Decoder blocks use transposed convolutions (`ConvTranspose2d`) for upsampling and concatenate encoder skip connections along the channel dimension to preserve spatial detail.

## Dataset

**Oxford-IIIT Pet Dataset** — 3,680 training / 3,669 test images with per-pixel segmentation masks.

3 classes: pet (foreground), background, boundary.

Images resized to 128×128. Training uses standard normalization.

## Results

| Metric | Value |
|--------|-------|
| **Mean IoU** | **0.7422** |
| Epochs | 25 |
| Optimizer | Adam (lr=0.001) |
| Batch size | 16 |
| GPU | NVIDIA GTX 1060 6GB |

## How to Run

```bash
# Install dependencies
pip install torch torchvision

# Train and evaluate
python u_net.py
```

The dataset downloads automatically on first run (~800MB).

## Key Concepts

- **Encoder-decoder architecture**: encoder compresses spatial information into high-level features, decoder reconstructs spatial resolution for dense prediction
- **Skip connections**: encoder features are concatenated with decoder features at matching spatial levels, preserving fine-grained detail lost during downsampling
- **Per-pixel classification**: output is a class probability map at the same resolution as the input — every pixel gets a predicted label

## Reference

He, K., et al. — *Deep Residual Learning for Image Recognition* (2015) was used as prior work for understanding skip connections.

Ronneberger, O., Fischer, P., & Brox, T. — *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015). [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
