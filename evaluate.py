import torch

from config import DEVICE, NUM_CLASSES


def evaluate(model, loader):
    model.eval()
    TP = torch.zeros(NUM_CLASSES)
    FP = torch.zeros(NUM_CLASSES)
    FN = torch.zeros(NUM_CLASSES)
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            predictions = model(images).argmax(dim=1)
            for c in range(NUM_CLASSES):
                TP[c] += ((predictions == c) & (masks == c)).sum().item()
                FP[c] += ((predictions == c) & (masks != c)).sum().item()
                FN[c] += ((predictions != c) & (masks == c)).sum().item()
    IoU  = TP / (TP + FP + FN + 1e-8)
    mIoU = IoU.mean()
    return mIoU
