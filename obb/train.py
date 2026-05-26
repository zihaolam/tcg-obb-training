"""Train YOLO-OBB on the synthesized card dataset.

Defaults target a single RTX 4090 / A100 with a yolo11n-obb (nano) model — the
smallest OBB variant, which is plenty for single-class card detection and the
fastest at inference time. Bump --model to yolo11s-obb.pt or yolo11m-obb.pt
if you have headroom and want a bit more accuracy.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/dataset.yaml"))
    p.add_argument("--model", default="yolo11n-obb.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--device",
        default="0",
        help='GPU index (e.g. "0"), "cpu", or "mps" for Apple Silicon',
    )
    p.add_argument("--project", default="runs/obb")
    p.add_argument("--name", default="train")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Aggressive rotation aug — belt-and-suspenders on top of our 360° synthetic data.
        # If we ever swap the synth pipeline for real labels, this still keeps the OBB
        # head trained across the full angle range.
        degrees=180,
        fliplr=0.5,
        flipud=0.5,
        # Looser color aug than defaults since phone cameras vary a lot in white balance.
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        # Mosaic helps OBB generalization, then back off for the last epochs so the model
        # fine-tunes on naturally-composed scenes.
        mosaic=1.0,
        close_mosaic=10,
        single_cls=True,
        patience=20,
        plots=True,
        save=True,
        cache="ram",
    )


if __name__ == "__main__":
    main()
