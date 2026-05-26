"""Export a trained YOLO-pose card-corner model.

Install export deps first:

    uv sync --extra export
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


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
    p.add_argument("--output-dir", type=Path, default=Path("export/pose"))
    p.add_argument("--skip-onnx", action="store_true", help="don't run ONNX export")
    p.add_argument("--skip-pytorch", action="store_true", help="don't copy best.pt")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_pytorch:
        copy_pytorch(args.weights, args.output_dir / "pytorch")

    if not args.skip_onnx:
        model = YOLO(str(args.weights))
        export_onnx(model, args.imgsz, args.output_dir / "onnx")

    print(f"\ndone. Contents of {args.output_dir}:")
    for sub in sorted(args.output_dir.iterdir()):
        if sub.is_dir():
            for f in sorted(sub.iterdir()):
                print(f"  {sub.name}/{f.name}")
        else:
            print(f"  {sub.name}")


if __name__ == "__main__":
    main()
