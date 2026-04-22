"""Loss functions for segmentation training.

CE+Dice is the standard for class-imbalanced segmentation — Dice explicitly
optimises the overlap metric while CE provides stable gradients early in training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_pet_seg.config import Config


class DiceLoss(nn.Module):
    """Mean soft Dice loss over all classes (macro average)."""

    def __init__(self, num_classes: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        one_hot = F.one_hot(targets, self.num_classes)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice_per_cls = (2.0 * intersection + self.eps) / (cardinality + self.eps)

        # Average only over classes that appear in the batch; absent classes are
        # undefined (0/0) and artificially inflate the loss with near-zero probs.
        present = one_hot.sum(dim=dims) > 0
        if present.any():
            return 1.0 - dice_per_cls[present].mean()
        return dice_per_cls.new_zeros(())


class CEDiceLoss(nn.Module):
    """Cross-entropy + 0.5 * Dice — standard for imbalanced segmentation."""

    def __init__(self, num_classes: int, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets) + self.w * self.dice(logits, targets)


def build_loss(cfg: Config) -> nn.Module:
    """Construct the loss function from cfg.loss."""
    name = cfg.loss.lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "ce_dice":
        return CEDiceLoss(num_classes=cfg.num_classes)
    raise ValueError(f"Unknown loss {cfg.loss!r}. Choose 'ce' or 'ce_dice'.")
