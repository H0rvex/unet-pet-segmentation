import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs and optionally enable cuDNN determinism.

    deterministic=True enables torch.use_deterministic_algorithms (warn_only),
    which catches non-deterministic ops without aborting the run. Set to False
    to trade reproducibility for speed on ops that lack deterministic kernels.
    Note: DataLoader worker non-determinism is handled separately via worker_init_fn.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed: int) -> torch.Generator:
    """Return a seeded CPU Generator for DataLoader shuffle reproducibility."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    """Propagate per-worker torch seed to Python random and NumPy."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
