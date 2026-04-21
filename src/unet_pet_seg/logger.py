"""Three-sink logger: stdout, JSONL, TensorBoard. Wired in Phase 3."""

import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, run_dir: str) -> None:
        root = Path(run_dir)
        self._jsonl = open(root / "metrics.jsonl", "w")
        self._tb = SummaryWriter(log_dir=str(root / "tb"))

    def log(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_miou: float,
        lr: float,
    ) -> None:
        print(
            f"Epoch {epoch:>3}/{total_epochs}"
            f"  lr={lr:.5f}"
            f"  loss={train_loss:.4f}"
            f"  val_mIoU={val_miou:.4f}"
        )
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_miou": round(val_miou, 4),
            "lr": lr,
        }
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()
        self._tb.add_scalar("loss/train", train_loss, epoch)
        self._tb.add_scalar("miou/val", val_miou, epoch)
        self._tb.add_scalar("lr", lr, epoch)

    def close(self) -> None:
        self._jsonl.close()
        self._tb.close()
