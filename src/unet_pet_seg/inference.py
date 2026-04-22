"""Image preprocessing and saving for deployment-style inference (no dataloader)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

_IMG_MEAN = (0.485, 0.456, 0.406)
_IMG_STD = (0.229, 0.224, 0.225)


def pil_to_model_input(pil: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    """RGB PIL image -> (1, 3, H, W) float tensor, ImageNet-normalized (eval pipeline, no aug)."""
    rgb = pil.convert("RGB")
    x = TF.pil_to_tensor(rgb).float() / 255.0
    x = TF.resize(x, [image_size, image_size])
    x = TF.normalize(x, list(_IMG_MEAN), list(_IMG_STD))
    return x.unsqueeze(0).to(device)


def mask_to_index_png(mask_hw: np.ndarray, path: Path) -> None:
    """Save class indices {0..C-1} as a single-channel uint8 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(mask_hw.astype(np.int64), 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


def list_image_paths(root: Path) -> list[Path]:
    """Sorted list of image files under ``root`` (non-recursive) or ``[root]`` if a file."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if root.is_file():
        return [root] if root.suffix.lower() in exts else []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)
