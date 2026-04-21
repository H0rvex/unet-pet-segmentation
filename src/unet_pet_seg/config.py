from dataclasses import dataclass


@dataclass
class Config:
    # Data
    data_dir: str = "./data"
    image_size: int = 128
    num_classes: int = 3
    val_split: float = 0.1

    # Training
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 25
    lr: float = 1e-3

    # LR schedule: StepLR (replaced by cosine in Phase 3)
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1

    # Mixed precision — wired into trainer in Phase 3
    use_amp: bool = False

    # Reproducibility
    seed: int = 42
