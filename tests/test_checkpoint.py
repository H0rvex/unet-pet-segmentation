"""Checkpoint loading helpers."""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from unet_pet_seg.checkpoint import (
    expect_model_state,
    load_checkpoint,
    load_model_weights,
    normalize_ddp_state_dict,
)
from unet_pet_seg.config import Config


def test_load_checkpoint_inference_roundtrip_state_dict() -> None:
    model = nn.Linear(4, 2)
    cfg = dataclasses.asdict(Config(image_size=32, num_classes=3))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.pth"
        torch.save({"model": model.state_dict(), "config": cfg, "epoch": 0}, path)
        ckpt = load_checkpoint(path, map_location="cpu", full_training_state=False)
        assert "model" in ckpt
        m2 = nn.Linear(4, 2)
        load_model_weights(m2, ckpt)
        for p2, p1 in zip(m2.parameters(), model.parameters(), strict=True):
            assert torch.allclose(p2, p1)


def test_load_checkpoint_full_training_state() -> None:
    model = nn.Linear(3, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "full.pth"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": GradScaler(enabled=False).state_dict(),
                "epoch": 1,
                "best_miou": 0.1,
                "config": dataclasses.asdict(Config()),
            },
            path,
        )
        ckpt = load_checkpoint(path, map_location="cpu", full_training_state=True)
        assert ckpt["epoch"] == 1


def test_expect_model_state_missing_key() -> None:
    with pytest.raises(ValueError, match="missing 'model'"):
        expect_model_state({"epoch": 0})


def test_expect_model_state_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="state_dict mapping"):
        expect_model_state({"model": [1, 2, 3]})


def test_normalize_ddp_strips_module_prefix() -> None:
    raw = {"module.weight": torch.ones(2, 3), "module.bias": torch.zeros(2)}
    out = normalize_ddp_state_dict(raw)
    assert set(out) == {"weight", "bias"}
    assert torch.equal(out["weight"], raw["module.weight"])


def test_load_model_weights_accepts_ddp_prefixed_state_dict() -> None:
    base = nn.Linear(3, 2)
    ddp_like = {f"module.{k}": v for k, v in base.state_dict().items()}
    ckpt = {"model": ddp_like, "config": dataclasses.asdict(Config())}
    m = nn.Linear(3, 2)
    load_model_weights(m, ckpt)
    for p2, p1 in zip(m.parameters(), base.parameters(), strict=True):
        assert torch.allclose(p2, p1)
