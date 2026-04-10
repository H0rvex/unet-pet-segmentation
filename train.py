"""UNet for Image Segmentation — training entry point"""

import torch
import torch.nn as nn

from config import (
    DEVICE, NUM_CLASSES, EPOCHS, LR,
    SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA, CHECKPOINT_PATH,
)
from model import UNet
from dataset import get_dataloaders
from evaluate import evaluate


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    model     = UNet(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
    )

    # training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            # forward
            predictions = model(images)
            # loss
            loss = loss_fn(predictions, masks)
            # backward
            optimizer.zero_grad()
            loss.backward()
            # update
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        val_miou = evaluate(model, val_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, loss = {total_loss / len(train_loader):.4f}, val mIoU = {val_miou:.4f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)

    test_miou = evaluate(model, test_loader)
    print(f"Test mIoU: {test_miou:.4f}")
