"""Shared visualisation helpers used by both the training logger and scripts/visualize.py."""

from __future__ import annotations

import numpy as np
import torch

# Per-class colours for mask visualisation (foreground, background, boundary).
MASK_COLORS = np.array(
    [
        [64, 170, 64],  # foreground — green
        [30, 30, 30],  # background — near-black
        [220, 60, 60],  # boundary   — red
    ],
    dtype=np.uint8,
)

CLASS_NAMES = ["foreground", "background", "boundary"]

_IMG_MEAN = np.array([0.485, 0.456, 0.406])
_IMG_STD = np.array([0.229, 0.224, 0.225])


def unnormalize(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) ImageNet-normalized tensor -> (H,W,3) float in [0,1]."""
    arr = img.detach().cpu().float().numpy().transpose(1, 2, 0)
    return np.clip(arr * _IMG_STD + _IMG_MEAN, 0.0, 1.0)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """(H,W) int mask -> (H,W,3) uint8 colour image via the class palette."""
    return MASK_COLORS[mask.clip(0, len(MASK_COLORS) - 1)]


def overlay_mask(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend a colorized mask on top of a float image in [0,1]."""
    colour = colorize_mask(mask).astype(np.float32) / 255.0
    return np.clip((1.0 - alpha) * img + alpha * colour, 0.0, 1.0)
