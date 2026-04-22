import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> tuple[torch.Tensor, float]:
    """Return (per-class IoU tensor, mean IoU) over the given loader."""
    model.eval()
    TP = torch.zeros(num_classes, device=device)
    FP = torch.zeros(num_classes, device=device)
    FN = torch.zeros(num_classes, device=device)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        predictions = model(images).argmax(dim=1)
        for c in range(num_classes):
            TP[c] += ((predictions == c) & (masks == c)).sum()
            FP[c] += ((predictions == c) & (masks != c)).sum()
            FN[c] += ((predictions != c) & (masks == c)).sum()

    # If running under DDP, each rank sees a shard of the loader.
    # Reduce counts so metrics reflect the full dataset.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(TP, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(FP, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(FN, op=torch.distributed.ReduceOp.SUM)

    iou = TP / (TP + FP + FN + 1e-8)
    miou = iou.mean().item()
    return iou.detach().cpu(), miou
