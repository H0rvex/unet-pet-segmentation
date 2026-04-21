import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> tuple[torch.Tensor, float]:
    """Return (per-class IoU tensor, mean IoU) over the given loader."""
    model.eval()
    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        predictions = model(images).argmax(dim=1)
        for c in range(num_classes):
            TP[c] += ((predictions == c) & (masks == c)).sum().item()
            FP[c] += ((predictions == c) & (masks != c)).sum().item()
            FN[c] += ((predictions != c) & (masks == c)).sum().item()
    iou  = TP / (TP + FP + FN + 1e-8)
    miou = iou.mean().item()
    return iou, miou
