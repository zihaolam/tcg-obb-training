"""Train YOLO-pose to detect Pokemon card corners.

The model predicts one `card` object per visible card plus four keypoints:
top-left, top-right, bottom-right, bottom-left. Use those keypoints for
perspective normalization instead of OBB rectangle corners.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/pose/dataset.yaml"))
    p.add_argument("--model", default="yolo11n-pose.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--device",
        default="0",
        help='GPU index (e.g. "0"), "cpu", or "mps" for Apple Silicon',
    )
    p.add_argument("--project", default="runs/pose")
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
        # Full rotation coverage is still useful because cards can appear at any angle.
        # Flip aug is disabled: the keypoints are physical card corners, and a mirrored
        # card is not a real camera view.
        degrees=180,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
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
