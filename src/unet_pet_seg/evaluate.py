import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unet_pet_seg.config import NUM_CLASSES


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, float]:
    """Return (per-class IoU tensor, mean IoU) over the given loader."""
    model.eval()
    TP = torch.zeros(NUM_CLASSES)
    FP = torch.zeros(NUM_CLASSES)
    FN = torch.zeros(NUM_CLASSES)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        predictions = model(images).argmax(dim=1)
        for c in range(NUM_CLASSES):
            TP[c] += ((predictions == c) & (masks == c)).sum().item()
            FP[c] += ((predictions == c) & (masks != c)).sum().item()
            FN[c] += ((predictions != c) & (masks == c)).sum().item()
    iou  = TP / (TP + FP + FN + 1e-8)
    miou = iou.mean().item()
    return iou, miou
