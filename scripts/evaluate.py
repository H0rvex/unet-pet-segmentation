"""Evaluate a trained U-Net checkpoint and print per-class + mean IoU."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = (_REPO_ROOT / "src").resolve()
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from unet_pet_seg.config import Config
from unet_pet_seg.dataset import get_dataloaders
from unet_pet_seg.evaluate import evaluate
from unet_pet_seg.model import UNet

CLASS_NAMES = ["foreground", "background", "boundary"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate U-Net checkpoint on Oxford-IIIT Pet")
    p.add_argument("--checkpoint", required=True, metavar="PATH", help="Path to best.pth from train.py")
    p.add_argument("--data-dir", default=None, help="Override data dir from checkpoint config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = Config(**ckpt["config"]) if "config" in ckpt else Config()
    if args.data_dir is not None:
        cfg = dataclasses.replace(cfg, data_dir=args.data_dir)

    model = UNet(num_classes=cfg.num_classes).to(device)
    model.load_state_dict(ckpt["model"])

    _, _, test_loader = get_dataloaders(cfg)
    iou, miou = evaluate(model, test_loader, device, cfg.num_classes)

    print("Per-class IoU:")
    for name, score in zip(CLASS_NAMES, iou.tolist()):
        print(f"  {name:<12} {score:.4f}")
    print(f"\nmIoU: {miou:.4f}")


if __name__ == "__main__":
    main()
