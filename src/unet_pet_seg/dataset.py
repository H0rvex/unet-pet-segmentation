import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from unet_pet_seg.config import (
    DATA_ROOT, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    VAL_SPLIT, RANDOM_SEED,
)


class PetSegDataset(Dataset):
    def __init__(self, dataset: OxfordIIITPet) -> None:
        self.dataset = dataset
        self.img_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[idx]
        image = self.img_transform(image)
        mask  = self.mask_transform(mask)
        mask  = (mask * 255).round().long().squeeze(0) - 1  # shifts 1,2,3 -> 0,1,2
        return image, mask


def get_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    train_data = OxfordIIITPet(root=DATA_ROOT, split="trainval", target_types="segmentation", download=True)
    test_data  = OxfordIIITPet(root=DATA_ROOT, split="test",     target_types="segmentation", download=True)

    full_train_set = PetSegDataset(train_data)
    test_set       = PetSegDataset(test_data)

    val_size   = int(VAL_SPLIT * len(full_train_set))
    train_size = len(full_train_set) - val_size
    train_set, val_set = random_split(
        full_train_set, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader
