"""Entry point: train U-Net on Oxford-IIIT Pet and save the best checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import os
import sys
import typing
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Allow running directly without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = (_REPO_ROOT / "src").resolve()
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from unet_pet_seg.config import Config
from unet_pet_seg.dataset import get_dataloaders
from unet_pet_seg.evaluate import evaluate
from unet_pet_seg.logger import Logger
from unet_pet_seg.model import UNet
from unet_pet_seg.trainer import train_epoch
from unet_pet_seg.utils.seeding import set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train U-Net on Oxford-IIIT Pet")
    p.add_argument("--config", type=str, default=None, metavar="PATH",
                   help="YAML config file; CLI flags override loaded values")
    p.add_argument("--resume", type=str, default=None, metavar="PATH",
                   help="Resume from a checkpoint saved by this script")
    p.add_argument("--out-dir", type=str, default="runs", metavar="DIR",
                   help="Root directory for run outputs (default: runs/)")
    hints = typing.get_type_hints(Config)
    for f in dataclasses.fields(Config):
        flag = f"--{f.name.replace('_', '-')}"
        ftype = hints[f.name]
        if ftype is bool:
            p.add_argument(flag, default=None, action=argparse.BooleanOptionalAction)
        else:
            p.add_argument(flag, type=ftype, default=None, metavar=ftype.__name__.upper())
    return p


def resolve_config(args: argparse.Namespace) -> Config:
    cfg_dict: dict = {f.name: f.default for f in dataclasses.fields(Config)}
    if args.config is not None:
        with open(args.config) as fh:
            cfg_dict.update(yaml.safe_load(fh) or {})
    for f in dataclasses.fields(Config):
        val = getattr(args, f.name, None)
        if val is not None:
            cfg_dict[f.name] = val
    return Config(**cfg_dict)


def setup_run(out_dir: str, cfg: Config) -> tuple[str, str, str]:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, ts)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as fh:
        yaml.dump(dataclasses.asdict(cfg), fh, default_flow_style=False, sort_keys=False)
    best_path = os.path.join(run_dir, "best.pth")
    last_path = os.path.join(run_dir, "last.pth")
    return run_dir, best_path, last_path


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_miou: float,
    cfg: Config,
) -> None:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_miou": best_miou,
        "config": dataclasses.asdict(cfg),
    }, path)


def main(cfg: Config, out_dir: str, resume: str | None = None) -> None:
    run_dir, best_path, last_path = setup_run(out_dir, cfg)
    print(f"Run dir : {run_dir}")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Config  : image_size={cfg.image_size}  epochs={cfg.epochs}  lr={cfg.lr}  amp={cfg.use_amp}")

    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    model     = UNet(num_classes=cfg.num_classes).to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma
    )

    start_epoch = 1
    best_miou   = 0.0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_miou   = ckpt["best_miou"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}  (best mIoU {best_miou:.4f})")

    logger = Logger(run_dir)
    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
            scheduler.step()
            _, val_miou = evaluate(model, val_loader, device, cfg.num_classes)
            lr_now = optimizer.param_groups[0]["lr"]
            logger.log(epoch, cfg.epochs, train_loss, val_miou, lr_now)

            save_checkpoint(last_path, epoch, model, optimizer, scheduler, best_miou, cfg)
            if val_miou > best_miou:
                best_miou = val_miou
                save_checkpoint(best_path, epoch, model, optimizer, scheduler, best_miou, cfg)
    finally:
        logger.close()

    _, test_miou = evaluate(model, test_loader, device, cfg.num_classes)
    print(f"\nTest mIoU  : {test_miou:.4f}")
    print(f"Best val   : {best_miou:.4f}  → {best_path}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    cfg  = resolve_config(args)
    main(cfg, out_dir=args.out_dir, resume=args.resume)
