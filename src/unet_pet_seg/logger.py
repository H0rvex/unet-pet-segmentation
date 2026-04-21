"""Three-sink logger: stdout, JSONL, TensorBoard."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Per-class colours for mask visualisation (foreground, background, boundary).
_MASK_COLORS = np.array([
    [64,  170,  64],   # foreground — green
    [ 30,  30,  30],   # background — near-black
    [220,  60,  60],   # boundary   — red
], dtype=np.uint8)

_IMG_MEAN = np.array([0.485, 0.456, 0.406])
_IMG_STD  = np.array([0.229, 0.224, 0.225])

CLASS_NAMES = ["foreground", "background", "boundary"]


def _colorize(mask: np.ndarray) -> np.ndarray:
    return _MASK_COLORS[mask.clip(0, len(_MASK_COLORS) - 1)]


def _unnormalize(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) ImageNet-normalized tensor → (H,W,3) float32 in [0,1]."""
    arr = img.cpu().float().numpy().transpose(1, 2, 0)
    return np.clip(arr * _IMG_STD + _IMG_MEAN, 0.0, 1.0)


class Logger:
    def __init__(self, run_dir: str) -> None:
        root = Path(run_dir)
        self._jsonl = open(root / "metrics.jsonl", "w")
        self._tb    = SummaryWriter(log_dir=str(root / "tb"))

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
        iou_str  = "  ".join(f"{n}={v:.3f}" for n, v in zip(CLASS_NAMES, iou_vals))
        print(
            f"Epoch {epoch:>3}/{total_epochs}"
            f"  lr={lr:.5f}"
            f"  loss={train_loss:.4f}"
            f"  val_mIoU={val_miou:.4f}"
            f"  grad={grad_norm:.3f}"
            f"  [{iou_str}]"
        )

        record = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_miou":   round(val_miou, 4),
            "lr":         lr,
            "grad_norm":  round(grad_norm, 4),
            **{f"iou_{n}": round(v, 4) for n, v in zip(CLASS_NAMES, iou_vals)},
        }
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()

        self._tb.add_scalar("loss/train",  train_loss, epoch)
        self._tb.add_scalar("miou/val",    val_miou,   epoch)
        self._tb.add_scalar("lr",          lr,         epoch)
        self._tb.add_scalar("grad_norm",   grad_norm,  epoch)
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
            axes[i, 0].imshow(_unnormalize(images[i]))
            axes[i, 1].imshow(_colorize(masks[i].cpu().numpy()))
            axes[i, 2].imshow(_colorize(preds[i].cpu().numpy()))
        axes[0, 0].set_title("Input",  fontsize=9)
        axes[0, 1].set_title("GT",     fontsize=9)
        axes[0, 2].set_title("Pred",   fontsize=9)
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        self._tb.add_figure("predictions/val", fig, global_step=epoch)
        plt.close(fig)

    def close(self) -> None:
        self._jsonl.close()
        self._tb.close()
