import math
import tempfile

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset

from unet_pet_seg.config import Config
from unet_pet_seg.evaluate import evaluate
from unet_pet_seg.logger import Logger
from unet_pet_seg.losses import CEDiceLoss, DiceLoss, build_loss
from unet_pet_seg.model import UNet
from unet_pet_seg.trainer import Trainer
from unet_pet_seg.utils.seeding import set_seed

_IMG_SIZE = 32  # small spatial size keeps tests fast on CPU
_N = 32
_BATCH = 32  # single batch so loss decrease is reliable
_CLASSES = 3


def _make_loader(seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(_N, 3, _IMG_SIZE, _IMG_SIZE, generator=g)
    y = torch.randint(0, _CLASSES, (_N, _IMG_SIZE, _IMG_SIZE), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=_BATCH)


def _build_trainer(tmp_path: str, seed: int = 42) -> tuple[Trainer, Logger]:
    set_seed(seed)
    device = torch.device("cpu")
    cfg = Config(
        image_size=_IMG_SIZE,
        num_classes=_CLASSES,
        epochs=2,
        lr=1e-2,
        use_amp=False,
        grad_clip=1.0,
        log_pred_every=1,
        lr_schedule="step",
        scheduler_step_size=10,
        scheduler_gamma=0.1,
    )
    model = UNet(num_classes=_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = GradScaler(enabled=False)
    logger = Logger(tmp_path)
    trainer = Trainer(model, loss_fn, optimizer, scheduler, scaler, cfg, device, logger, tmp_path)
    return trainer, logger


def test_loss_decreases_on_fixed_batch():
    """Two epochs on the same fixed batch: loss must fall."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer, logger = _build_trainer(tmp)
        loader = _make_loader()
        loss1, _ = trainer._train_epoch(loader)
        loss2, _ = trainer._train_epoch(loader)
        logger.close()
    assert loss2 < loss1, f"Loss did not decrease: {loss1:.4f} → {loss2:.4f}"


def test_initial_loss_near_log_num_classes():
    """Random-init cross-entropy should be close to log(num_classes) = log(3) ≈ 1.099."""
    model = UNet(num_classes=_CLASSES)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(4, 3, _IMG_SIZE, _IMG_SIZE)
    y = torch.randint(0, _CLASSES, (4, _IMG_SIZE, _IMG_SIZE))
    with torch.no_grad():
        loss = loss_fn(model(x), y).item()
    assert abs(loss - math.log(_CLASSES)) < 0.5, f"Unexpected initial loss: {loss:.4f}"


def test_evaluate_returns_miou_in_range():
    model = UNet(num_classes=_CLASSES)
    device = torch.device("cpu")
    loader = _make_loader()
    iou, miou = evaluate(model, loader, device, _CLASSES)
    assert iou.shape == (3,)
    assert 0.0 <= miou <= 1.0


def test_grad_norm_is_finite():
    with tempfile.TemporaryDirectory() as tmp:
        trainer, logger = _build_trainer(tmp)
        loader = _make_loader()
        _, grad_norm = trainer._train_epoch(loader)
        logger.close()
    assert torch.isfinite(torch.tensor(grad_norm))


def test_checkpoint_roundtrip():
    """Save and reload a checkpoint; forward output must be bit-exact."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer, logger = _build_trainer(tmp)
        loader = _make_loader()
        trainer._train_epoch(loader)

        ckpt_path = f"{tmp}/test.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1, best_miou=0.42)

        # Build a fresh trainer and load the checkpoint.
        trainer2, logger2 = _build_trainer(tmp, seed=99)  # different seed → different init
        trainer2.load_checkpoint(ckpt_path)

        x = torch.randn(2, 3, _IMG_SIZE, _IMG_SIZE)
        with torch.no_grad():
            out1 = trainer.model(x)
            out2 = trainer2.model(x)

        logger.close()
        logger2.close()

    assert torch.allclose(out1, out2), "Checkpoint reload produced different model weights"


def test_fit_runs_and_returns_best_miou():
    """Full fit() over 2 epochs on synthetic data must complete without error."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer, logger = _build_trainer(tmp)
        loader = _make_loader()
        best_miou = trainer.fit(loader, loader, start_epoch=1, best_miou=0.0)
        logger.close()
    assert 0.0 <= best_miou <= 1.0


def test_dice_loss_perfect_prediction_is_zero():
    """Dice loss should be ~0 when predicted probabilities match ground truth."""
    loss_fn = DiceLoss(num_classes=_CLASSES)
    logits = torch.zeros(2, _CLASSES, 4, 4)
    targets = torch.zeros(2, 4, 4, dtype=torch.long)
    logits[:, 0, :, :] = 10.0  # near-certain class 0 prediction → targets are all 0
    assert loss_fn(logits, targets).item() < 0.05


def test_ce_dice_loss_shape_and_range():
    """CE+Dice loss must return a scalar >= 0."""
    loss_fn = CEDiceLoss(num_classes=_CLASSES)
    logits = torch.randn(2, _CLASSES, _IMG_SIZE, _IMG_SIZE)
    targets = torch.randint(0, _CLASSES, (2, _IMG_SIZE, _IMG_SIZE))
    val = loss_fn(logits, targets)
    assert val.shape == torch.Size([]), "loss must be a scalar"
    assert val.item() >= 0.0


def test_build_loss_dispatch():
    cfg_ce = Config(loss="ce", num_classes=_CLASSES)
    cfg_cedice = Config(loss="ce_dice", num_classes=_CLASSES)
    assert isinstance(build_loss(cfg_ce), nn.CrossEntropyLoss)
    assert isinstance(build_loss(cfg_cedice), CEDiceLoss)


def test_determinism():
    """Same seed must produce bit-exact loss on two independent runs."""

    def one_run() -> float:
        with tempfile.TemporaryDirectory() as tmp:
            trainer, logger = _build_trainer(tmp, seed=42)
            loader = _make_loader(seed=7)
            loss, _ = trainer._train_epoch(loader)
            logger.close()
        return loss

    assert one_run() == one_run(), "Training is not deterministic under the same seed"
