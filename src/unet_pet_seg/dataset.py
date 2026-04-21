import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from unet_pet_seg.config import Config
from unet_pet_seg.utils.seeding import make_generator, worker_init_fn


class PetSegDataset(Dataset):
    def __init__(self, dataset: OxfordIIITPet, image_size: int) -> None:
        self.dataset = dataset
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[idx]
        image = self.img_transform(image)
        mask  = self.mask_transform(mask)
        mask  = (mask * 255).round().long().squeeze(0) - 1  # shifts labels 1,2,3 -> 0,1,2
        return image, mask


def get_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_data = OxfordIIITPet(root=cfg.data_dir, split="trainval", target_types="segmentation", download=True)
    test_data  = OxfordIIITPet(root=cfg.data_dir, split="test",     target_types="segmentation", download=True)

    full_train_set = PetSegDataset(train_data, cfg.image_size)
    test_set       = PetSegDataset(test_data,  cfg.image_size)

    val_size   = int(cfg.val_split * len(full_train_set))
    train_size = len(full_train_set) - val_size
    train_set, val_set = random_split(
        full_train_set, [train_size, val_size],
        generator=make_generator(cfg.seed),
    )

    loader_kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                         pin_memory=True, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_set, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_set,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
