"""Validate merged YAML/CLI dicts before constructing ``Config``."""

from __future__ import annotations

import dataclasses
from typing import Any

from unet_pet_seg.config import Config

_ALLOWED_ARCH = frozenset({"unet", "fcn_resnet50", "deeplabv3_resnet50"})
_ALLOWED_LOSS = frozenset({"ce", "ce_dice"})
_ALLOWED_LR_SCHEDULE = frozenset({"cosine", "step"})


def validate_config_dict(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Return ``cfg_dict`` after checks; raises ``ValueError`` on invalid values or unknown keys."""
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(cfg_dict) - known
    if unknown:
        raise ValueError(f"Unknown config keys (typo?): {sorted(unknown)}")

    arch = cfg_dict.get("arch", Config.arch)
    if arch not in _ALLOWED_ARCH:
        raise ValueError(f"arch must be one of {sorted(_ALLOWED_ARCH)}, got {arch!r}")

    loss = cfg_dict.get("loss", Config.loss)
    if loss not in _ALLOWED_LOSS:
        raise ValueError(f"loss must be one of {sorted(_ALLOWED_LOSS)}, got {loss!r}")

    lr_schedule = cfg_dict.get("lr_schedule", Config.lr_schedule)
    if lr_schedule not in _ALLOWED_LR_SCHEDULE:
        raise ValueError(
            f"lr_schedule must be one of {sorted(_ALLOWED_LR_SCHEDULE)}, got {lr_schedule!r}"
        )

    val_split = cfg_dict.get("val_split", Config.val_split)
    if not isinstance(val_split, int | float) or not (0.0 < float(val_split) < 1.0):
        raise ValueError(f"val_split must be in (0, 1), got {val_split!r}")

    image_size = cfg_dict.get("image_size", Config.image_size)
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError(f"image_size must be a positive int, got {image_size!r}")

    num_classes = cfg_dict.get("num_classes", Config.num_classes)
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f"num_classes must be an int >= 2, got {num_classes!r}")

    epochs = cfg_dict.get("epochs", Config.epochs)
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"epochs must be a positive int, got {epochs!r}")

    batch_size = cfg_dict.get("batch_size", Config.batch_size)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")

    num_workers = cfg_dict.get("num_workers", Config.num_workers)
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError(f"num_workers must be a non-negative int, got {num_workers!r}")

    lr = cfg_dict.get("lr", Config.lr)
    if not isinstance(lr, int | float) or float(lr) <= 0.0:
        raise ValueError(f"lr must be > 0, got {lr!r}")

    grad_clip = cfg_dict.get("grad_clip", Config.grad_clip)
    if not isinstance(grad_clip, int | float) or float(grad_clip) < 0.0:
        raise ValueError(f"grad_clip must be >= 0 (0 disables), got {grad_clip!r}")

    warmup_epochs = cfg_dict.get("warmup_epochs", Config.warmup_epochs)
    if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be a non-negative int, got {warmup_epochs!r}")

    scheduler_step_size = cfg_dict.get("scheduler_step_size", Config.scheduler_step_size)
    if not isinstance(scheduler_step_size, int) or scheduler_step_size < 1:
        raise ValueError(f"scheduler_step_size must be a positive int, got {scheduler_step_size!r}")

    scheduler_gamma = cfg_dict.get("scheduler_gamma", Config.scheduler_gamma)
    if not isinstance(scheduler_gamma, int | float) or float(scheduler_gamma) <= 0.0:
        raise ValueError(f"scheduler_gamma must be > 0, got {scheduler_gamma!r}")

    seed = cfg_dict.get("seed", Config.seed)
    if not isinstance(seed, int):
        raise ValueError(f"seed must be int, got {seed!r}")

    log_pred_every = cfg_dict.get("log_pred_every", Config.log_pred_every)
    if not isinstance(log_pred_every, int) or log_pred_every < 1:
        raise ValueError(f"log_pred_every must be a positive int, got {log_pred_every!r}")

    use_aug = cfg_dict.get("use_aug", Config.use_aug)
    if not isinstance(use_aug, bool):
        raise ValueError(f"use_aug must be a bool (YAML: true/false), got {use_aug!r}")

    use_amp = cfg_dict.get("use_amp", Config.use_amp)
    if not isinstance(use_amp, bool):
        raise ValueError(f"use_amp must be a bool (YAML: true/false), got {use_amp!r}")

    return cfg_dict
