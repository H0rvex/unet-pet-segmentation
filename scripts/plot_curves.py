"""Parse runs/<ts>/metrics.jsonl into artifacts/curves.png.

Reads the JSONL that logger.py emits per epoch — avoids pulling in TensorBoard's
event-file parser just to plot three time series.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = (_REPO_ROOT / "src").resolve()
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from unet_pet_seg.viz import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training curves from a run's metrics.jsonl")
    p.add_argument("--run-dir", required=True, metavar="DIR",
                   help="Run directory containing metrics.jsonl (e.g. runs/20260421_120000)")
    p.add_argument("--out", type=str, default="artifacts/curves.png", metavar="PATH",
                   help="Output PNG path (default: artifacts/curves.png)")
    p.add_argument("--title", type=str, default=None,
                   help="Optional suptitle — defaults to the run-dir basename")
    return p.parse_args()


def _load_metrics(run_dir: Path) -> list[dict]:
    jsonl = run_dir / "metrics.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"No metrics.jsonl in {run_dir}")
    with open(jsonl) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    args     = parse_args()
    run_dir  = Path(args.run_dir)
    records  = _load_metrics(run_dir)
    if not records:
        raise SystemExit(f"metrics.jsonl in {run_dir} is empty — nothing to plot")

    epochs     = [r["epoch"]      for r in records]
    train_loss = [r["train_loss"] for r in records]
    val_miou   = [r["val_miou"]   for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs, train_loss, label="train loss", color="tab:blue")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_miou, label="val mIoU", color="tab:green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].set_title("Validation mIoU")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    for name in CLASS_NAMES:
        key = f"iou_{name}"
        if key in records[0]:
            axes[2].plot(epochs, [r[key] for r in records], label=name)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("IoU")
    axes[2].set_title("Per-class IoU (val)")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="lower right", fontsize=8)

    fig.suptitle(args.title or run_dir.name, fontsize=11)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  ({len(records)} epochs)")


if __name__ == "__main__":
    main()
