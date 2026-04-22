"""Three-sink logger: stdout, JSONL, TensorBoard."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from unet_pet_seg.viz import CLASS_NAMES, colorize_mask, unnormalize


class Logger:
    def __init__(self, run_dir: str) -> None:
        root = Path(run_dir)
        self._jsonl = open(root / "metrics.jsonl", "w", encoding="utf-8")
        self._tb = SummaryWriter(log_dir=str(root / "tb"))

    def __enter__(self) -> Logger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def log_metrics(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_miou: float,
        per_class_iou: torch.Tensor,
        lr: float,
        grad_norm: float,
    ) -> None:
        iou_vals = per_class_iou.tolist()
        iou_str = "  ".join(f"{n}={v:.3f}" for n, v in zip(CLASS_NAMES, iou_vals))
        print(
            f"Epoch {epoch:>3}/{total_epochs}"
            f"  lr={lr:.5f}"
            f"  loss={train_loss:.4f}"
            f"  val_mIoU={val_miou:.4f}"
            f"  grad={grad_norm:.3f}"
            f"  [{iou_str}]"
        )

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_miou": round(val_miou, 4),
            "lr": lr,
            "grad_norm": round(grad_norm, 4),
            **{f"iou_{n}": round(v, 4) for n, v in zip(CLASS_NAMES, iou_vals)},
        }
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()

        self._tb.add_scalar("loss/train", train_loss, epoch)
        self._tb.add_scalar("miou/val", val_miou, epoch)
        self._tb.add_scalar("lr", lr, epoch)
        self._tb.add_scalar("grad_norm", grad_norm, epoch)
        for name, val in zip(CLASS_NAMES, iou_vals):
            self._tb.add_scalar(f"iou/{name}", val, epoch)

    def log_pred_grid(
        self,
        epoch: int,
        images: torch.Tensor,
        masks: torch.Tensor,
        preds: torch.Tensor,
        n: int = 4,
    ) -> None:
        """Write an input | GT | pred grid to TensorBoard."""
        n = min(n, images.size(0))
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
        for i in range(n):
            axes[i, 0].imshow(unnormalize(images[i]))
            axes[i, 1].imshow(colorize_mask(masks[i].cpu().numpy()))
            axes[i, 2].imshow(colorize_mask(preds[i].cpu().numpy()))
        axes[0, 0].set_title("Input", fontsize=9)
        axes[0, 1].set_title("GT", fontsize=9)
        axes[0, 2].set_title("Pred", fontsize=9)
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        self._tb.add_figure("predictions/val", fig, global_step=epoch)
        plt.close(fig)

    def close(self) -> None:
        self._jsonl.close()
        self._tb.close()
