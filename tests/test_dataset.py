import os

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from unet_pet_seg.dataset import PetSegDataset

_PET_DATA_DIR = os.path.join(".", "data", "oxford-iiit-pet")


class _FakePetData:
    """Minimal stand-in for OxfordIIITPet — no download required."""

    def __init__(self, n: int = 8, size: tuple[int, int] = (60, 80)) -> None:
        self._n    = n
        self._size = size  # (H, W)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image]:
        rng      = np.random.default_rng(idx)
        img_arr  = rng.integers(0, 255, (*self._size, 3), dtype=np.uint8)
        mask_arr = rng.choice([1, 2, 3], size=self._size).astype(np.uint8)
        return Image.fromarray(img_arr), Image.fromarray(mask_arr, mode="L")


def _make_dataset(augment: bool = False, image_size: int = 64) -> PetSegDataset:
    return PetSegDataset(_FakePetData(), image_size=image_size, augment=augment)


def test_image_output_shape():
    ds = _make_dataset()
    img, _ = ds[0]
    assert img.shape == (3, 64, 64)


def test_mask_output_shape():
    ds = _make_dataset()
    _, mask = ds[0]
    assert mask.shape == (64, 64)


def test_image_dtype_is_float32():
    ds = _make_dataset()
    img, _ = ds[0]
    assert img.dtype == torch.float32


def test_mask_dtype_is_int64():
    ds = _make_dataset()
    _, mask = ds[0]
    assert mask.dtype == torch.int64


def test_mask_label_range():
    ds = _make_dataset()
    for i in range(len(ds)):
        _, mask = ds[i]
        assert mask.min() >= 0 and mask.max() <= 2, (
            f"Sample {i}: unexpected labels {mask.unique().tolist()}"
        )


def test_image_is_normalized():
    """After ImageNet normalisation, values should extend outside [0, 1]."""
    ds = _make_dataset()
    img, _ = ds[0]
    assert img.min().item() < 0.0 or img.max().item() > 1.0


def test_no_aug_is_deterministic():
    """Same index, no augmentation → identical outputs."""
    ds = _make_dataset(augment=False)
    img1, mask1 = ds[0]
    img2, mask2 = ds[0]
    assert torch.equal(img1, img2)
    assert torch.equal(mask1, mask2)


def test_aug_produces_varied_outputs():
    """Same index, augmentation on → outputs should differ across calls."""
    ds = _make_dataset(augment=True, image_size=128)
    # At least one pair must differ over 10 samples (p≈1 for any real aug)
    outputs = [ds[0][0] for _ in range(10)]
    assert not all(torch.equal(outputs[0], o) for o in outputs[1:]), (
        "Augmentation produced identical outputs on 10 consecutive calls"
    )


def test_aug_mask_labels_stay_valid():
    """Augmentation must not introduce out-of-range label values."""
    ds = _make_dataset(augment=True, image_size=128)
    for i in range(len(ds)):
        _, mask = ds[i]
        assert mask.min() >= 0 and mask.max() <= 2, (
            f"Augmented sample {i}: unexpected labels {mask.unique().tolist()}"
        )


def test_dataloader_batch_shapes():
    ds     = _make_dataset()
    loader = DataLoader(ds, batch_size=4)
    imgs, masks = next(iter(loader))
    assert imgs.shape  == (4, 3, 64, 64)
    assert masks.shape == (4, 64, 64)
    assert imgs.dtype  == torch.float32
    assert masks.dtype == torch.int64


@pytest.mark.skipif(
    not os.path.isdir(_PET_DATA_DIR),
    reason="Oxford-IIIT Pet data not present; run train.py once to download",
)
def test_real_dataset_label_range():
    from torchvision.datasets import OxfordIIITPet

    raw = OxfordIIITPet(root="./data", split="test", target_types="segmentation", download=False)
    ds  = PetSegDataset(raw, image_size=128)
    for i in range(0, min(len(ds), 50), 5):
        _, mask = ds[i]
        assert mask.min() >= 0 and mask.max() <= 2
