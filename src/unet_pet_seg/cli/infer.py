"""Batch inference on arbitrary images: masks, overlays, and latency stats."""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from unet_pet_seg.baselines import build_model
from unet_pet_seg.checkpoint import load_checkpoint, load_model_weights
from unet_pet_seg.config import Config
from unet_pet_seg.inference import list_image_paths, mask_to_index_png, pil_to_model_input
from unet_pet_seg.viz import colorize_mask, overlay_mask, unnormalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run segmentation on image files or a folder (deployment-style path)"
    )
    p.add_argument(
        "--checkpoint", required=True, metavar="PATH", help="Path to best.pth / last.pth"
    )
    p.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Image file or directory of images (.jpg, .png, …)",
    )
    p.add_argument(
        "--out-dir",
        default="artifacts/infer",
        metavar="DIR",
        help="Output directory for mask PNG, overlay PNG, optional grid (default: artifacts/infer)",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        metavar="N",
        help="Override square size (default: from checkpoint config)",
    )
    p.add_argument(
        "--warmup", type=int, default=10, metavar="N", help="Warmup forwards on first image"
    )
    p.add_argument(
        "--save-grids",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save a 1×3 PNG (input | color mask | overlay) per image",
    )
    return p.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _save_grid(inp: torch.Tensor, pred_hw: torch.Tensor, path: Path) -> None:
    img_np = unnormalize(inp.squeeze(0))
    pred_np = pred_hw.cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[1].imshow(colorize_mask(pred_np))
    axes[1].set_title("Prediction")
    axes[2].imshow(overlay_mask(img_np, pred_np))
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main_cli() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    ckpt_path = Path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path, map_location=device, full_training_state=False)
    cfg = Config(**ckpt["config"]) if "config" in ckpt else Config()
    if args.image_size is not None:
        cfg = dataclasses.replace(cfg, image_size=args.image_size)

    model = build_model(cfg).to(device)
    load_model_weights(model, ckpt)
    model.eval()

    in_root = Path(args.input)
    paths = list_image_paths(in_root)
    if not paths:
        raise FileNotFoundError(f"No images found under {in_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latencies_ms: list[float] = []

    with torch.no_grad():
        for idx, img_path in enumerate(paths):
            with Image.open(img_path) as pil:
                x = pil_to_model_input(pil, cfg.image_size, device)

            if idx == 0:
                for _ in range(args.warmup):
                    _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            t0 = time.perf_counter()
            logits = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            pred = logits.argmax(dim=1).squeeze(0)
            pred_np = pred.cpu().numpy()
            stem = img_path.stem
            mask_path = out_dir / f"{stem}_mask.png"
            overlay_path = out_dir / f"{stem}_overlay.png"
            mask_to_index_png(pred_np, mask_path)

            img_np = unnormalize(x.squeeze(0))
            ol = (overlay_mask(img_np, pred_np) * 255.0).clip(0, 255).astype("uint8")
            Image.fromarray(ol).save(overlay_path)

            if args.save_grids:
                _save_grid(x, pred, out_dir / f"{stem}_grid.png")

    lat_sorted = sorted(latencies_ms)
    p50 = _percentile(lat_sorted, 0.50)
    p90 = _percentile(lat_sorted, 0.90)
    mean_ms = statistics.mean(latencies_ms)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    print(f"Wrote outputs for {len(paths)} image(s) to {out_dir}/")
    print(
        f"Latency ms/img: mean={mean_ms:.3f}  p50={p50:.3f}  p90={p90:.3f}  "
        f"(warmup={args.warmup} on first image only)"
    )
    print(f"Throughput: {fps:.2f} images/s (mean batch=1)")
