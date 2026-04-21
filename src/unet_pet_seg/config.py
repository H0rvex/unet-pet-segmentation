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
    grad_clip: float = 1.0  # max gradient norm; 0.0 to disable

    # LR schedule: "cosine" (with optional linear warmup) or "step"
    lr_schedule: str = "cosine"
    warmup_epochs: int = 3      # linear warmup before cosine; ignored for "step"
    scheduler_step_size: int = 10  # used only when lr_schedule="step"
    scheduler_gamma: float = 0.1   # used only when lr_schedule="step"

    # Augmentation (training set only; val/test never augmented)
    use_aug: bool = False

    # Mixed precision (CUDA only; silently disabled on CPU)
    use_amp: bool = True

    # Reproducibility
    seed: int = 42

    # Logging
    log_pred_every: int = 5  # write prediction grid to TensorBoard every N epochs
