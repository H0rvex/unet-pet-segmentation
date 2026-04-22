"""Config dict validation before ``Config`` construction."""

from __future__ import annotations

import dataclasses

import pytest

from unet_pet_seg.config import Config
from unet_pet_seg.config_validate import validate_config_dict


def test_validate_accepts_defaults_dict() -> None:
    d = {f.name: f.default for f in dataclasses.fields(Config)}
    validate_config_dict(d)


def test_validate_rejects_unknown_key() -> None:
    d = {f.name: f.default for f in dataclasses.fields(Config)}
    d["typo_epochs"] = 1
    with pytest.raises(ValueError, match="Unknown config keys"):
        validate_config_dict(d)


def test_validate_rejects_bad_arch() -> None:
    d = {f.name: f.default for f in dataclasses.fields(Config)}
    d["arch"] = "resnet"
    with pytest.raises(ValueError, match="arch"):
        validate_config_dict(d)


def test_validate_rejects_val_split_out_of_range() -> None:
    d = {f.name: f.default for f in dataclasses.fields(Config)}
    d["val_split"] = 1.0
    with pytest.raises(ValueError, match="val_split"):
        validate_config_dict(d)


def test_validate_rejects_non_bool_use_amp() -> None:
    d = {f.name: f.default for f in dataclasses.fields(Config)}
    d["use_amp"] = "yes"
    with pytest.raises(ValueError, match="use_amp"):
        validate_config_dict(d)


def test_validate_rejects_non_bool_use_aug() -> None:
    d = {f.name: f.default for f in dataclasses.fields(Config)}
    d["use_aug"] = 1
    with pytest.raises(ValueError, match="use_aug"):
        validate_config_dict(d)
