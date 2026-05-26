"""Verify a trained YOLO-pose card-corner model on synthetic perspective samples."""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from generate_dataset import (
    color_jitter,
    list_images,
    load_background,
    load_card,
    maybe_blur,
    maybe_noise,
    order_corners_for_display,
    sample_dest_corners,
    warp_card_onto_bg,
)

LABEL_COLOR = (255, 0, 255)  # magenta (BGR)
PRED_COLOR = (255, 255, 0)  # cyan (BGR)


def point_error(pred: np.ndarray, label: np.ndarray) -> float:
    return float(np.linalg.norm(pred - label, axis=1).mean())


def draw_corners(img, corners, color, label):
    display_corners = order_corners_for_display(corners)
    pts = display_corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)
    for i, (x, y) in enumerate(corners.astype(int)):
        cv2.circle(img, (int(x), int(y)), 6, color, -1)
        cv2.putText(
            img,
            f"{label}{i}",
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def make_sample(card_path, bg_path, size, rng):
    card = load_card(card_path)
    bg = load_background(bg_path, size, rng)
    if card is None or bg is None:
        return None
    dest = sample_dest_corners(size, rng)
    composite = warp_card_onto_bg(card, bg, dest)
    composite = color_jitter(composite, rng)
    composite = maybe_blur(composite, rng)
    composite = maybe_noise(composite, rng)
    return composite, dest


def run_inference(model: YOLO, image_bgr: np.ndarray, conf: float, imgsz: int):
    result = model(image_bgr, conf=conf, verbose=False, imgsz=imgsz)[0]
    if result.keypoints is None or result.keypoints.xy.shape[0] == 0:
        return None, None
    pred = result.keypoints.xy[0].cpu().numpy().astype(np.float32)
    pred_conf = None
    if result.boxes is not None and result.boxes.conf.shape[0] > 0:
        pred_conf = float(result.boxes.conf[0].cpu().numpy())
    return pred, pred_conf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--cards-dir", type=Path, default=Path("data/sources/cards"))
    p.add_argument("--backgrounds-dir", type=Path, default=Path("data/sources/backgrounds"))
    p.add_argument("--output-dir", type=Path, default=Path("verify_pose_output"))
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    cards = list_images(args.cards_dir)
    bgs = list_images(args.backgrounds_dir)
    if not cards:
        raise SystemExit(f"no cards found under {args.cards_dir}")
    if not bgs:
        raise SystemExit(f"no backgrounds found under {args.backgrounds_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print(f"loading model from {args.weights}...")
    model = YOLO(str(args.weights))

    errors = []
    misses = 0
    for i in range(args.num_samples):
        sample = make_sample(
            rng.choice(cards), rng.choice(bgs), args.image_size, rng
        )
        if sample is None:
            continue
        image, label = sample
        pred, pred_conf = run_inference(model, image, args.conf, args.image_size)

        overlay = image.copy()
        draw_corners(overlay, label, LABEL_COLOR, "L")
        if pred is None:
            misses += 1
            cv2.putText(
                overlay,
                "no detection",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            err = None
        else:
            draw_corners(overlay, pred, PRED_COLOR, "P")
            err = point_error(pred, label)
            errors.append(err)
            cv2.putText(
                overlay,
                f"conf={pred_conf:.2f} mean_px_err={err:.1f}",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                PRED_COLOR,
                2,
            )

        cv2.imwrite(str(args.output_dir / f"sample_{i:02d}.png"), overlay)
        status = "miss" if err is None else f"mean_px_err={err:.2f}"
        print(f"  sample_{i:02d}: {status}")

    mean_err = float(np.mean(errors)) if errors else 0.0
    print("\n=== summary ===")
    print(f"  detections: {len(errors)}/{args.num_samples}")
    print(f"  misses: {misses}")
    print(f"  mean corner error: {mean_err:.2f}px")
    print(f"  overlays written to {args.output_dir}/")


if __name__ == "__main__":
    main()
