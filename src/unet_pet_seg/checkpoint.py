"""Checkpoint I/O, DDP-aware state dict loading, and safer inference loads."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _weights_only_safe_load_failed(exc: BaseException) -> bool:
    """True if ``exc`` is the kind of failure where falling back to legacy unpickling is intended."""
    if isinstance(exc, pickle.UnpicklingError):
        return True
    if isinstance(exc, RuntimeError):
        low = str(exc).lower()
        return "weights" in low and "only" in low
    return False


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device | dict[str, str] | None = None,
    *,
    full_training_state: bool = False,
) -> dict[str, Any]:
    """Load a ``.pth`` written by ``Trainer.save_checkpoint``.

    Training resumes need ``full_training_state=True`` (optimizer, scheduler, scaler).

    For evaluation and inference, ``full_training_state=False`` tries ``weights_only=True``
    first so only tensors and plain metadata are unpickled. If that fails with a
    known safe-load restriction error, falls back to ``weights_only=False`` — use only
    with **trusted** local files. Other errors are re-raised.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if full_training_state:
        return torch.load(path, map_location=map_location, weights_only=False)

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        if not _weights_only_safe_load_failed(exc):
            raise
        warnings.warn(
            "Loaded checkpoint with weights_only=False (trusted local file only). "
            "Re-save with the current trainer if you need strict safe unpickling.",
            UserWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def normalize_ddp_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    """Strip ``module.`` prefix from keys when a checkpoint was saved under ``DistributedDataParallel``."""
    as_dict = dict(state)
    if not as_dict:
        return as_dict
    if any(str(k).startswith("module.") for k in as_dict):
        return {str(k).removeprefix("module."): v for k, v in as_dict.items()}
    return as_dict


def expect_model_state(ckpt: dict[str, Any], *, key: str = "model") -> Mapping[str, Any]:
    """Return the model state dict from a training checkpoint, validating shape of the payload."""
    if key not in ckpt:
        raise ValueError(
            f"Checkpoint missing {key!r}; expected keys from Trainer.save_checkpoint "
            f"(including 'model' and 'config')."
        )
    raw = ckpt[key]
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"Checkpoint[{key!r}] must be a state_dict mapping, got {type(raw).__name__}"
        )
    return raw


def load_model_weights(module: nn.Module, ckpt: dict[str, Any], *, key: str = "model") -> None:
    """Load ``ckpt[key]`` into ``module``, normalizing DDP-prefixed keys when needed."""
    state = normalize_ddp_state_dict(expect_model_state(ckpt, key=key))
    module.load_state_dict(state)
