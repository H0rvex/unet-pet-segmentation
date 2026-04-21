"""Evaluate a trained U-Net checkpoint and print per-class + mean IoU."""

import argparse
import torch

from unet_pet_seg.config import DEVICE, NUM_CLASSES
from unet_pet_seg.model import UNet
from unet_pet_seg.dataset import get_dataloaders
from unet_pet_seg.evaluate import evaluate

CLASS_NAMES = ["foreground", "background", "boundary"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate U-Net checkpoint on Oxford-IIIT Pet")
    p.add_argument("--checkpoint", required=True, metavar="PATH", help="Path to .pth saved by train.py")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    _, _, test_loader = get_dataloaders()
    iou, miou = evaluate(model, test_loader, device)

    print("Per-class IoU:")
    for name, score in zip(CLASS_NAMES, iou.tolist()):
        print(f"  {name:<12} {score:.4f}")
    print(f"\nmIoU: {miou:.4f}")


if __name__ == "__main__":
    main()
