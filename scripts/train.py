"""Entry point: train U-Net on Oxford-IIIT Pet and save the best checkpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow running this file directly without installing the package:
# `python scripts/train.py` and editor import resolution.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = (_REPO_ROOT / "src").resolve()
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from unet_pet_seg.config import (
    DEVICE, NUM_CLASSES, EPOCHS, LR,
    SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA, CHECKPOINT_PATH,
)
from unet_pet_seg.model import UNet
from unet_pet_seg.dataset import get_dataloaders
from unet_pet_seg.trainer import train_epoch
from unet_pet_seg.evaluate import evaluate
from unet_pet_seg.logger import Logger


if __name__ == "__main__":
    os.makedirs("runs/latest", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders()

    model     = UNet(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
    )

    logger = Logger("runs/latest")
    best_miou = 0.0
    try:
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
            scheduler.step()
            _, val_miou = evaluate(model, val_loader, DEVICE)
            lr_now = optimizer.param_groups[0]["lr"]
            logger.log(epoch, EPOCHS, train_loss, val_miou, lr_now)

            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(), CHECKPOINT_PATH)
    finally:
        logger.close()

    _, test_miou = evaluate(model, test_loader, DEVICE)
    print(f"\nTest mIoU: {test_miou:.4f}  (best val mIoU: {best_miou:.4f})")
