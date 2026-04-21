"""Entry point: train U-Net on Oxford-IIIT Pet and save the best checkpoint."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import sys
import typing
from pathlib import Path

import torch
import yaml
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# Allow running directly without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = (_REPO_ROOT / "src").resolve()
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from unet_pet_seg.baselines import build_model
from unet_pet_seg.config import Config
from unet_pet_seg.dataset import get_dataloaders
from unet_pet_seg.evaluate import evaluate
from unet_pet_seg.logger import Logger
from unet_pet_seg.losses import build_loss
from unet_pet_seg.trainer import Trainer
from unet_pet_seg.utils.seeding import set_seed

CLASS_NAMES = ["foreground", "background", "boundary"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train U-Net on Oxford-IIIT Pet")
    p.add_argument("--config",  type=str, default=None, metavar="PATH",
                   help="YAML config file; CLI flags override loaded values")
    p.add_argument("--resume",  type=str, default=None, metavar="PATH",
                   help="Resume from a checkpoint saved by this script")
    p.add_argument("--out-dir", type=str, default="runs", metavar="DIR",
                   help="Root directory for timestamped run folders (default: runs/)")
    hints = typing.get_type_hints(Config)
    for f in dataclasses.fields(Config):
        flag  = f"--{f.name.replace('_', '-')}"
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


def setup_run(out_dir: str, cfg: Config) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, ts)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as fh:
        yaml.dump(dataclasses.asdict(cfg), fh, default_flow_style=False, sort_keys=False)
    return run_dir


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> torch.optim.lr_scheduler.LRScheduler:
    if cfg.lr_schedule == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        if cfg.warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                              total_iters=cfg.warmup_epochs)
            cosine = CosineAnnealingLR(optimizer,
                                       T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-6)
            return SequentialLR(optimizer, schedulers=[warmup, cosine],
                                milestones=[cfg.warmup_epochs])
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma
    )


def main(cfg: Config, out_dir: str, resume: str | None = None) -> None:
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = rank == 0

    if is_distributed:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # Create a single run dir and share it across ranks.
    if is_distributed:
        run_dir_obj: list[str | None] = [None]
        if is_main:
            run_dir_obj[0] = setup_run(out_dir, cfg)
        torch.distributed.broadcast_object_list(run_dir_obj, src=0)
        run_dir = typing.cast(str, run_dir_obj[0])
    else:
        run_dir = setup_run(out_dir, cfg)

    if is_main:
        print(f"Run dir   : {run_dir}")

    # Offset seed per-rank so DataLoader workers don't duplicate augmentations.
    set_seed(cfg.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp_on = cfg.use_amp and device.type == "cuda"
    if is_main:
        print(f"Device    : {device}")
        print(
            f"Config    : arch={cfg.arch}  size={cfg.image_size}  epochs={cfg.epochs}  lr={cfg.lr}"
            f"  loss={cfg.loss}  schedule={cfg.lr_schedule}  amp={amp_on}  grad_clip={cfg.grad_clip}"
            + (f"  ddp=1 world={torch.distributed.get_world_size()}" if is_distributed else "")
        )

    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    model = build_model(cfg).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    loss_fn   = build_loss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = GradScaler(enabled=amp_on)
    if is_main:
        logger: Logger | typing.Any = Logger(run_dir)
    else:
        class _NoOpLogger:
            def log_metrics(self, *args: typing.Any, **kwargs: typing.Any) -> None: ...
            def log_pred_grid(self, *args: typing.Any, **kwargs: typing.Any) -> None: ...
            def close(self) -> None: ...
        logger = _NoOpLogger()

    trainer = Trainer(model, loss_fn, optimizer, scheduler, scaler, cfg, device, logger, run_dir)

    start_epoch, best_miou = 1, 0.0
    if resume is not None:
        start_epoch, best_miou = trainer.load_checkpoint(resume)

    try:
        best_miou = trainer.fit(train_loader, val_loader, start_epoch, best_miou)
    finally:
        logger.close()

    test_iou, test_miou = evaluate(model, test_loader, device, cfg.num_classes)
    if is_main:
        best_path = os.path.join(run_dir, "best.pth")
        metrics_path = os.path.join(run_dir, "test_metrics.json")
        metrics_payload = {
            "checkpoint": str(Path(best_path).resolve()),
            "split": "test",
            "miou": round(test_miou, 4),
            "per_class_iou": {
                name: round(score, 4)
                for name, score in zip(CLASS_NAMES, test_iou.tolist())
            },
        }
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics_payload, fh, indent=2)
        print(f"\nTest mIoU : {test_miou:.4f}")
        print(f"Test JSON : {metrics_path}")
        print(f"Best val  : {best_miou:.4f}  → {best_path}")

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = build_parser().parse_args()
    cfg  = resolve_config(args)
    main(cfg, out_dir=args.out_dir, resume=args.resume)
