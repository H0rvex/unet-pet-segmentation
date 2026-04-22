"""Torchvision semantic-segmentation baselines for apples-to-apples comparison.

Torchvision `fcn_resnet50` / `deeplabv3_resnet50` return a dict `{'out': logits, ...}`.
TVAdapter unwraps that so the rest of the codebase (Trainer, evaluate, visualize) can
treat every model as `tensor -> tensor` without branching.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50

from unet_pet_seg.config import Config
from unet_pet_seg.model import UNet

_TV_FACTORIES = {
    "fcn_resnet50": fcn_resnet50,
    "deeplabv3_resnet50": deeplabv3_resnet50,
}


class TVAdapter(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)["out"]


def _build_tv_baseline(name: str, num_classes: int) -> nn.Module:
    # weights="DEFAULT" is incompatible with a custom num_classes (head shape mismatch),
    # so we only load the pretrained backbone — the standard fine-tuning pattern.
    factory = _TV_FACTORIES[name]
    base = factory(
        weights=None,
        weights_backbone="DEFAULT",
        num_classes=num_classes,
        aux_loss=False,
    )
    return TVAdapter(base)


def build_model(cfg: Config) -> nn.Module:
    arch = cfg.arch.lower()
    if arch == "unet":
        return UNet(num_classes=cfg.num_classes)
    if arch in _TV_FACTORIES:
        return _build_tv_baseline(arch, cfg.num_classes)
    raise ValueError(
        f"Unknown arch {cfg.arch!r}. Expected 'unet' or one of {sorted(_TV_FACTORIES)}."
    )
