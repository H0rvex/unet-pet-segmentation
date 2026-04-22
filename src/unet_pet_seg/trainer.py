from __future__ import annotations

import dataclasses
import os
import typing
from contextlib import nullcontext
from typing import Protocol, cast

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from unet_pet_seg.checkpoint import expect_model_state, load_checkpoint, normalize_ddp_state_dict
from unet_pet_seg.config import Config
from unet_pet_seg.evaluate import evaluate


class TrainingLogger(Protocol):
    def log_metrics(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_miou: float,
        per_class_iou: torch.Tensor,
        lr: float,
        grad_norm: float,
    ) -> None: ...

    def log_pred_grid(
        self,
        epoch: int,
        images: torch.Tensor,
        masks: torch.Tensor,
        preds: torch.Tensor,
        n: int = 4,
    ) -> None: ...

    def close(self) -> None: ...


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
        logger: TrainingLogger,
        run_dir: str,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.run_dir = run_dir

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

        is_main = not (torch.distributed.is_available() and torch.distributed.is_initialized()) or (
            torch.distributed.get_rank() == 0
        )

        # Fixed sample batch for periodic pred grid visualisation (rank 0 only).
        sample_images = sample_masks = None
        if is_main:
            sample_images, sample_masks = next(iter(val_loader))
            sample_images = sample_images[:4].to(self.device)
            sample_masks = sample_masks[:4].to(self.device)

        for epoch in range(start_epoch, self.cfg.epochs + 1):
            # DDP shuffles need epoch to be set so each rank sees a different order.
            if hasattr(train_loader.sampler, "set_epoch"):
                cast(typing.Any, train_loader.sampler).set_epoch(epoch)

            train_loss, grad_norm = self._train_epoch(train_loader)
            self.scheduler.step()

            per_class_iou, val_miou = evaluate(
                self.model, val_loader, self.device, self.cfg.num_classes
            )
            lr_now = self.optimizer.param_groups[0]["lr"]
            if is_main:
                self.logger.log_metrics(
                    epoch,
                    self.cfg.epochs,
                    train_loss,
                    val_miou,
                    per_class_iou,
                    lr_now,
                    grad_norm,
                )

            if is_main and epoch % self.cfg.log_pred_every == 0 and sample_images is not None:
                with torch.no_grad():
                    preds = self.model(sample_images).argmax(dim=1)
                assert sample_masks is not None
                self.logger.log_pred_grid(epoch, sample_images, sample_masks, preds)

            if is_main:
                self.save_checkpoint(last_path, epoch, best_miou)
                if val_miou > best_miou:
                    best_miou = val_miou
                    self.save_checkpoint(best_path, epoch, best_miou)

        return best_miou

    def save_checkpoint(self, path: str, epoch: int, best_miou: float) -> None:
        model = cast(nn.Module, getattr(self.model, "module", self.model))
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_miou": best_miou,
                "config": dataclasses.asdict(self.cfg),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> tuple[int, float]:
        """Restore all state. Returns (start_epoch, best_miou)."""
        ckpt = load_checkpoint(path, map_location=self.device, full_training_state=True)
        target = cast(nn.Module, getattr(self.model, "module", self.model))
        target.load_state_dict(normalize_ddp_state_dict(expect_model_state(ckpt)))
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
            masks = masks.to(self.device, non_blocking=True)
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
            total_norm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm).item()
            if use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            total_loss += loss.item()

        n = len(loader)
        return total_loss / n, total_norm / n
