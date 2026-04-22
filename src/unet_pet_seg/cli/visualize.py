"""Save input | GT | pred | overlay grids for N test images."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from unet_pet_seg.baselines import build_model
from unet_pet_seg.checkpoint import load_checkpoint, load_model_weights
from unet_pet_seg.config import Config
from unet_pet_seg.dataset import get_dataloaders
from unet_pet_seg.viz import colorize_mask, overlay_mask, unnormalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write qualitative prediction grids from a checkpoint")
    p.add_argument("--checkpoint", required=True, metavar="PATH", help="best.pth from train.py")
    p.add_argument(
        "--num-samples",
        type=int,
        default=8,
        metavar="N",
        help="How many test images to visualise (default: 8)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/preds",
        metavar="DIR",
        help="Output directory for PNGs (default: artifacts/preds)",
    )
    p.add_argument(
        "--data-dir", type=str, default=None, help="Override data dir from checkpoint config"
    )
    return p.parse_args()


def _save_sample(image: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor, path: Path) -> None:
    img_np = unnormalize(image)
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[1].imshow(colorize_mask(gt_np))
    axes[1].set_title("Ground truth")
    axes[2].imshow(colorize_mask(pred_np))
    axes[2].set_title("Prediction")
    axes[3].imshow(overlay_mask(img_np, pred_np))
    axes[3].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main_cli() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.checkpoint, map_location=device, full_training_state=False)
    cfg = Config(**ckpt["config"]) if "config" in ckpt else Config()
    if args.data_dir is not None:
        cfg = dataclasses.replace(cfg, data_dir=args.data_dir)

    model = build_model(cfg).to(device)
    load_model_weights(model, ckpt)
    model.eval()

    _, _, test_loader = get_dataloaders(cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1)
            for i in range(images.size(0)):
                if saved >= args.num_samples:
                    break
                _save_sample(
                    images[i],
                    masks[i],
                    preds[i],
                    out_dir / f"sample_{saved:03d}.png",
                )
                saved += 1
            if saved >= args.num_samples:
                break

    print(f"Wrote {saved} grids to {out_dir}/")
