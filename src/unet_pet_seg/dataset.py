from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Protocol
from torch.utils.data.distributed import DistributedSampler
from torchvision import tv_tensors
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2

from unet_pet_seg.config import Config
from unet_pet_seg.utils.seeding import make_generator, worker_init_fn

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

class _PetLikeDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image]: ...


def _build_spatial_transform(image_size: int, augment: bool) -> v2.Compose:
    """Build spatial (+photometric) transforms applied jointly to image and mask.

    tv_tensors.Mask automatically receives NEAREST interpolation for all geometric
    ops; tv_tensors.ColorJitter only affects tv_tensors.Image, not Mask.
    """
    if augment:
        return v2.Compose([
            v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=(10, 10), translate=(0.05, 0.05)),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        ])
    return v2.Compose([v2.Resize((image_size, image_size))])

class PetSegDataset(Dataset):
    def __init__(self, dataset: _PetLikeDataset, image_size: int, augment: bool = False) -> None:
        self.dataset   = dataset
        self.transform = _build_spatial_transform(image_size, augment)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_pil, mask_pil = self.dataset[idx]

        # PIL mask is a palette image with values {1,2,3} for foreground/background/boundary.
        mask_np = np.array(mask_pil, dtype=np.int64) - 1  # {1,2,3} -> {0,1,2}

        # Wrap as tv_tensors so v2 geometric transforms apply consistently to both.
        image_tv = tv_tensors.Image(TF.pil_to_tensor(image_pil))               # (3,H,W) uint8
        mask_tv  = tv_tensors.Mask(torch.from_numpy(mask_np).unsqueeze(0))     # (1,H,W) int64

        image_tv, mask_tv = self.transform(image_tv, mask_tv)

        # Normalize image to float; keep mask as long integer.
        image = TF.normalize(image_tv.float() / 255.0, list(_MEAN), list(_STD))
        mask  = mask_tv.squeeze(0).long()
        return image, mask


def get_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_data = OxfordIIITPet(root=cfg.data_dir, split="trainval", target_types="segmentation", download=True)
    test_data  = OxfordIIITPet(root=cfg.data_dir, split="test",     target_types="segmentation", download=True)

    # Val and test are NEVER augmented — deterministic eval is non-negotiable.
    full_train_set = PetSegDataset(train_data, cfg.image_size, augment=False)
    test_set       = PetSegDataset(test_data,  cfg.image_size, augment=False)

    val_size   = int(cfg.val_split * len(full_train_set))
    train_size = len(full_train_set) - val_size
    train_subset, val_subset = random_split(
        full_train_set, [train_size, val_size],
        generator=make_generator(cfg.seed),
    )

    # Re-wrap with augmentation flag now that the split is fixed.
    aug_train_set = PetSegDataset(train_data, cfg.image_size, augment=cfg.use_aug)
    train_set = torch.utils.data.Subset(aug_train_set, train_subset.indices)
    val_set   = torch.utils.data.Subset(full_train_set, val_subset.indices)

    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    train_sampler = DistributedSampler(train_set, shuffle=True) if is_distributed else None
    val_sampler   = DistributedSampler(val_set, shuffle=False) if is_distributed else None
    test_sampler  = DistributedSampler(test_set, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, test_loader
