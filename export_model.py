"""Export a trained YOLO-OBB model into all formats we ship:

  export/
    pytorch/best.pt     <- copied from --weights (used by Python services)
    onnx/best.onnx      <- ONNX export (used by Node/Bun/Rust/Python services)
    tfjs/model.json     <- TFJS graph export (used by the browser/Bun TFJS worker)
    tfjs/group1-shard*.bin

The Bun TFJS worker loads `tfjs/model.json`. A Python or Node microservice loads
`onnx/best.onnx`. PyTorch weights are kept around for retraining / fine-tuning.

Install the export deps first:

    uv sync --extra export
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def export_tfjs(model: YOLO, imgsz: int, out_dir: Path) -> None:
    print(f"\n=== TFJS export -> {out_dir}")
    exported = Path(model.export(format="tfjs", imgsz=imgsz))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(exported, out_dir)


def export_onnx(model: YOLO, imgsz: int, out_dir: Path) -> None:
    print(f"\n=== ONNX export -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    exported = Path(model.export(format="onnx", imgsz=imgsz, simplify=True, opset=17))
    target = out_dir / "best.onnx"
    if target.exists():
        target.unlink()
    shutil.copy(exported, target)


def copy_pytorch(weights: Path, out_dir: Path) -> None:
    print(f"\n=== PyTorch weights -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "best.pt"
    if target.exists():
        target.unlink()
    shutil.copy(weights, target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--output-dir", type=Path, default=Path("export"))
    p.add_argument("--skip-tfjs", action="store_true", help="don't run TFJS export")
    p.add_argument("--skip-onnx", action="store_true", help="don't run ONNX export")
    p.add_argument("--skip-pytorch", action="store_true", help="don't copy best.pt")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_pytorch:
        copy_pytorch(args.weights, args.output_dir / "pytorch")

    # YOLO loads from the original weights — each export reuses the same model.
    model = YOLO(str(args.weights))

    if not args.skip_onnx:
        export_onnx(model, args.imgsz, args.output_dir / "onnx")

    if not args.skip_tfjs:
        # Reload model since `model.export` mutates state across formats.
        model_tfjs = YOLO(str(args.weights))
        export_tfjs(model_tfjs, args.imgsz, args.output_dir / "tfjs")

    print(f"\ndone. Contents of {args.output_dir}:")
    for sub in sorted(args.output_dir.iterdir()):
        if sub.is_dir():
            for f in sorted(sub.iterdir()):
                print(f"  {sub.name}/{f.name}")
        else:
            print(f"  {sub.name}")


if __name__ == "__main__":
    main()
