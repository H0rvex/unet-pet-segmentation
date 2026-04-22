"""Evaluate a trained checkpoint on the Oxford-IIIT Pet test split."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from unet_pet_seg.baselines import build_model
from unet_pet_seg.checkpoint import load_checkpoint, load_model_weights
from unet_pet_seg.config import Config
from unet_pet_seg.dataset import get_dataloaders
from unet_pet_seg.evaluate import evaluate

CLASS_NAMES = ["foreground", "background", "boundary"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate U-Net checkpoint on Oxford-IIIT Pet")
    p.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to best.pth from train.py",
    )
    p.add_argument("--data-dir", default=None, help="Override data dir from checkpoint config")
    p.add_argument(
        "--out-json",
        default=None,
        metavar="PATH",
        help="Optional output path for machine-readable metrics JSON",
    )
    return p.parse_args()


def main_cli() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.checkpoint, map_location=device, full_training_state=False)
    cfg = Config(**ckpt["config"]) if "config" in ckpt else Config()
    if args.data_dir is not None:
        cfg = dataclasses.replace(cfg, data_dir=args.data_dir)

    model = build_model(cfg).to(device)
    load_model_weights(model, ckpt)

    _, _, test_loader = get_dataloaders(cfg)
    iou, miou = evaluate(model, test_loader, device, cfg.num_classes)

    print("Per-class IoU:")
    for name, score in zip(CLASS_NAMES, iou.tolist()):
        print(f"  {name:<12} {score:.4f}")
    print(f"\nmIoU: {miou:.4f}")

    out_json = (
        Path(args.out_json)
        if args.out_json
        else Path(args.checkpoint).with_name("test_metrics.json")
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": "test",
        "miou": round(miou, 4),
        "per_class_iou": {name: round(score, 4) for name, score in zip(CLASS_NAMES, iou.tolist())},
    }
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote metrics JSON: {out_json}")
