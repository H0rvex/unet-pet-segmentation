from __future__ import annotations

import dataclasses
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from unet_pet_seg.config import Config
from unet_pet_seg.evaluate import evaluate
from unet_pet_seg.logger import Logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: GradScaler,
        cfg: Config,
        device: torch.device,
        logger: Logger,
        run_dir: str,
    ) -> None:
        self.model     = model
        self.loss_fn   = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler    = scaler
        self.cfg       = cfg
        self.device    = device
        self.logger    = logger
        self.run_dir   = run_dir

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 1,
        best_miou: float = 0.0,
    ) -> float:
        """Run training loop. Returns best val mIoU achieved."""
        best_path = os.path.join(self.run_dir, "best.pth")
        last_path = os.path.join(self.run_dir, "last.pth")

        # Fixed sample batch for periodic pred grid visualisation.
        sample_images, sample_masks = next(iter(val_loader))
        sample_images = sample_images[:4].to(self.device)
        sample_masks  = sample_masks[:4].to(self.device)

        for epoch in range(start_epoch, self.cfg.epochs + 1):
            train_loss, grad_norm = self._train_epoch(train_loader)
            self.scheduler.step()

            per_class_iou, val_miou = evaluate(
                self.model, val_loader, self.device, self.cfg.num_classes
            )
            lr_now = self.optimizer.param_groups[0]["lr"]
            self.logger.log_metrics(
                epoch, self.cfg.epochs, train_loss, val_miou,
                per_class_iou, lr_now, grad_norm,
            )

            if epoch % self.cfg.log_pred_every == 0:
                with torch.no_grad():
                    preds = self.model(sample_images).argmax(dim=1)
                self.logger.log_pred_grid(epoch, sample_images, sample_masks, preds)

            self.save_checkpoint(last_path, epoch, best_miou)
            if val_miou > best_miou:
                best_miou = val_miou
                self.save_checkpoint(best_path, epoch, best_miou)

        return best_miou

    def save_checkpoint(self, path: str, epoch: int, best_miou: float) -> None:
        torch.save({
            "epoch":     epoch,
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler":    self.scaler.state_dict(),
            "best_miou": best_miou,
            "config":    dataclasses.asdict(self.cfg),
        }, path)

    def load_checkpoint(self, path: str) -> tuple[int, float]:
        """Restore all state. Returns (start_epoch, best_miou)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        best_miou = ckpt.get("best_miou", 0.0)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}  (best mIoU {best_miou:.4f})")
        return start_epoch, best_miou

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """One training epoch with AMP + grad clipping.
        Returns (mean_loss, mean_grad_norm).
        """
        self.model.train()
        total_loss = 0.0
        total_norm = 0.0
        use_amp = self.device.type == "cuda" and self.scaler.is_enabled()

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            amp_cm = autocast(enabled=True, dtype=torch.float16) if use_amp else nullcontext()
            with amp_cm:
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)

            if use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            # Unscale before clipping so the norm reflects actual gradients.
            if use_amp:
                self.scaler.unscale_(self.optimizer)
            max_norm = self.cfg.grad_clip if self.cfg.grad_clip > 0 else float("inf")
            total_norm += torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm
            ).item()
            if use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            total_loss += loss.item()

        n = len(loader)
        return total_loss / n, total_norm / n
