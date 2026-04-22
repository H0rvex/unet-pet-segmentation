"""Benchmark inference latency (PyTorch fp32/amp, optional ONNX Runtime)."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from unet_pet_seg.baselines import build_model
from unet_pet_seg.checkpoint import load_checkpoint, load_model_weights
from unet_pet_seg.config import Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark segmentation model inference")
    p.add_argument("--checkpoint", required=True, metavar="PATH", help="Path to checkpoint (.pth)")
    p.add_argument("--batch-size", type=int, default=1, metavar="N", help="Batch size for timing")
    p.add_argument("--warmup", type=int, default=20, metavar="N", help="Warmup iterations")
    p.add_argument("--iters", type=int, default=100, metavar="N", help="Measured iterations")
    p.add_argument(
        "--mode",
        choices=["fp32", "amp", "both"],
        default="both",
        help="Benchmark mode(s) for PyTorch inference",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Benchmark device",
    )
    p.add_argument(
        "--onnx",
        default=None,
        metavar="PATH",
        help="Optional ONNX file path to benchmark with ONNX Runtime",
    )
    p.add_argument("--out-json", default=None, metavar="PATH", help="Output JSON metrics path")
    return p.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _time_pytorch(
    model: torch.nn.Module,
    sample: torch.Tensor,
    warmup: int,
    iters: int,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    durations_ms: list[float] = []
    amp_on = use_amp and sample.device.type == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_on):
                _ = model(sample)
        if sample.device.type == "cuda":
            torch.cuda.synchronize()

        for _ in range(iters):
            if sample.device.type == "cuda":
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_on):
                    _ = model(sample)
                end_evt.record()
                torch.cuda.synchronize()
                durations_ms.append(float(start_evt.elapsed_time(end_evt)))
            else:
                start = time.perf_counter()
                _ = model(sample)
                end = time.perf_counter()
                durations_ms.append((end - start) * 1000.0)

    mean_ms = statistics.mean(durations_ms)
    std_ms = statistics.pstdev(durations_ms) if len(durations_ms) > 1 else 0.0
    ms_per_img = mean_ms / sample.shape[0]
    fps = 1000.0 / ms_per_img if ms_per_img > 0 else 0.0
    return {
        "iterations": iters,
        "batch_size": int(sample.shape[0]),
        "mean_ms_per_batch": round(mean_ms, 4),
        "std_ms_per_batch": round(std_ms, 4),
        "mean_ms_per_image": round(ms_per_img, 4),
        "fps": round(fps, 2),
    }


def _time_onnxruntime(
    onnx_path: Path,
    sample: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    try:
        import onnxruntime as ort  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnxruntime is not installed") from exc

    providers = ["CPUExecutionProvider"]
    if sample.device.type == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    inp = sample.detach().cpu().numpy()

    for _ in range(warmup):
        _ = session.run(None, {input_name: inp})

    durations_ms: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = session.run(None, {input_name: inp})
        end = time.perf_counter()
        durations_ms.append((end - start) * 1000.0)

    mean_ms = statistics.mean(durations_ms)
    std_ms = statistics.pstdev(durations_ms) if len(durations_ms) > 1 else 0.0
    ms_per_img = mean_ms / sample.shape[0]
    fps = 1000.0 / ms_per_img if ms_per_img > 0 else 0.0
    return {
        "iterations": iters,
        "batch_size": int(sample.shape[0]),
        "mean_ms_per_batch": round(mean_ms, 4),
        "std_ms_per_batch": round(std_ms, 4),
        "mean_ms_per_image": round(ms_per_img, 4),
        "fps": round(fps, 2),
    }


def main_cli() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = _resolve_device(args.device)
    ckpt = load_checkpoint(ckpt_path, map_location=device, full_training_state=False)
    cfg = Config(**ckpt["config"]) if "config" in ckpt else Config()
    model = build_model(cfg).to(device)
    load_model_weights(model, ckpt)
    model.eval()

    sample = torch.randn(args.batch_size, 3, cfg.image_size, cfg.image_size, device=device)
    results: dict[str, Any] = {
        "checkpoint": str(ckpt_path.resolve()),
        "arch": cfg.arch,
        "image_size": cfg.image_size,
        "device": str(device),
        "warmup": args.warmup,
        "iters": args.iters,
        "pytorch": {},
    }

    modes = ["fp32", "amp"] if args.mode == "both" else [args.mode]
    for mode in modes:
        if mode == "amp" and device.type != "cuda":
            print("Skipping AMP benchmark on CPU")
            continue
        run = _time_pytorch(model, sample, args.warmup, args.iters, use_amp=(mode == "amp"))
        results["pytorch"][mode] = run
        print(
            f"[PyTorch/{mode}] {run['mean_ms_per_image']:.4f} ms/img"
            f" ({run['mean_ms_per_batch']:.4f} ms/batch, fps={run['fps']:.2f})"
        )

    if args.onnx:
        onnx_path = Path(args.onnx)
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        onnx_run = _time_onnxruntime(onnx_path, sample, args.warmup, args.iters)
        results["onnxruntime"] = onnx_run
        print(
            f"[ONNX Runtime] {onnx_run['mean_ms_per_image']:.4f} ms/img"
            f" ({onnx_run['mean_ms_per_batch']:.4f} ms/batch, fps={onnx_run['fps']:.2f})"
        )

    out_json = (
        Path(args.out_json)
        if args.out_json
        else Path("artifacts/benchmarks") / f"{cfg.arch}_{cfg.image_size}_{ckpt_path.stem}.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Wrote benchmark JSON: {out_json}")
