"""Export a trained checkpoint to ONNX for deployment/runtime validation."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = (_REPO_ROOT / "src").resolve()
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from unet_pet_seg.baselines import build_model
from unet_pet_seg.config import Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a segmentation checkpoint to ONNX")
    p.add_argument("--checkpoint", required=True, metavar="PATH", help="Path to checkpoint (.pth)")
    p.add_argument("--out", default=None, metavar="PATH", help="Output ONNX path")
    p.add_argument("--batch-size", type=int, default=1, metavar="N", help="Dummy input batch size")
    p.add_argument("--opset", type=int, default=17, metavar="N", help="ONNX opset version")
    p.add_argument(
        "--dynamic-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dynamic batch dimension in ONNX export",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"]) if "config" in ckpt else Config()

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_path = (
        Path(args.out)
        if args.out
        else Path("artifacts/onnx") / f"{cfg.arch}_{cfg.image_size}_{ckpt_path.stem}.onnx"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample = torch.randn(args.batch_size, 3, cfg.image_size, cfg.image_size, device=device)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}} if args.dynamic_batch else None

    with torch.no_grad():
        torch.onnx.export(
            model,
            sample,
            out_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            export_params=True,
            do_constant_folding=True,
            opset_version=args.opset,
        )

    print(f"Exported ONNX: {out_path}")
    print(
        "Input shape hint: "
        f"(N, 3, {cfg.image_size}, {cfg.image_size})"
        + (" with dynamic N" if args.dynamic_batch else "")
    )

    try:
        import onnx  # type: ignore

        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX validation: OK")
    except ModuleNotFoundError:
        print("ONNX validation skipped (install optional dependency: pip install onnx)")


if __name__ == "__main__":
    main()
