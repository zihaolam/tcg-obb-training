"""Verify whether the OBB model's predictions match its training labels.

Generates synthetic samples through the same pipeline as generate_dataset.py,
runs the trained model on them, and overlays predicted vs label corners.

Two variants:
  - "jittered" : current training distribution (per-corner jitter up to 0.18 * card_h)
  - "clean"    : same pipeline with jitter disabled (label is the true card rectangle)

Interpretation of the summary metrics:
  - jittered IoU high + clean IoU high   -> model is fine
  - jittered IoU high + clean IoU low    -> labels in training are inflated by jitter;
                                            model learned the wrong target
  - jittered IoU low                     -> model is undertrained

Usage:
  uv run python obb/verify_obb.py --weights export/pytorch/best.pt
  uv run python obb/verify_obb.py --weights export/onnx/best.onnx
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from generate_dataset import (
    CARD_ASPECT,
    color_jitter,
    list_images,
    load_background,
    load_card,
    maybe_blur,
    maybe_noise,
    warp_card_onto_bg,
)

LABEL_COLOR = (255, 0, 255)  # magenta (BGR)
PRED_COLOR = (255, 255, 0)  # cyan (BGR)


def sample_dest_corners(size: int, rng: random.Random, *, jitter_scale: float):
    """Same as generate_dataset.sample_dest_corners but jitter is configurable."""
    scale = rng.uniform(0.18, 0.7)
    card_h = scale * size
    card_w = card_h * CARD_ASPECT
    hw, hh = card_w / 2, card_h / 2
    corners = np.array(
        [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32
    )

    angle = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(angle), np.sin(angle)
    corners = corners @ np.array([[c, -s], [s, c]], dtype=np.float32).T

    if jitter_scale > 0:
        tilt = rng.uniform(0.0, jitter_scale)
        jitter = (
            np.random.uniform(-tilt, tilt, size=(4, 2)) * card_h
        ).astype(np.float32)
        corners = corners + jitter

    pad = card_h * 0.55
    cx = rng.uniform(pad, size - pad)
    cy = rng.uniform(pad, size - pad)
    return corners + np.array([cx, cy], dtype=np.float32)


def make_sample(card_path, bg_path, size, rng, *, jitter_scale):
    card = load_card(card_path)
    bg = load_background(bg_path, size, rng)
    if card is None or bg is None:
        return None
    dest = sample_dest_corners(size, rng, jitter_scale=jitter_scale)
    composite = warp_card_onto_bg(card, bg, dest)
    composite = color_jitter(composite, rng)
    composite = maybe_blur(composite, rng)
    composite = maybe_noise(composite, rng)
    return composite, dest


def run_inference(model: YOLO, image_bgr: np.ndarray, conf: float = 0.25):
    """Run the model on a BGR numpy image. Returns the top predicted 4x2 corners or None."""
    result = model(image_bgr, conf=conf, verbose=False, imgsz=640)[0]
    if result.obb is None or result.obb.xyxyxyxy.shape[0] == 0:
        return None
    # xyxyxyxy is (N, 4, 2) sorted by confidence descending
    return result.obb.xyxyxyxy[0].cpu().numpy()


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_hull = cv2.convexHull(a.astype(np.float32))
    b_hull = cv2.convexHull(b.astype(np.float32))
    inter_area, _ = cv2.intersectConvexConvex(a_hull, b_hull)
    a_area = cv2.contourArea(a_hull)
    b_area = cv2.contourArea(b_hull)
    union = a_area + b_area - inter_area
    return 0.0 if union <= 0 else float(inter_area / union)


def quad_area(a: np.ndarray) -> float:
    return float(cv2.contourArea(cv2.convexHull(a.astype(np.float32))))


def draw_quad(img, corners, color, label):
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)
    for x, y in corners.astype(int):
        cv2.circle(img, (int(x), int(y)), 6, color, -1)
    tl = corners[np.argmin(corners.sum(axis=1))].astype(int)
    cv2.putText(
        img,
        label,
        (int(tl[0]) + 10, int(tl[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to best.pt (preferred) or best.onnx",
    )
    p.add_argument(
        "--cards-dir", type=Path, default=Path("data/sources/cards")
    )
    p.add_argument(
        "--backgrounds-dir", type=Path, default=Path("data/sources/backgrounds")
    )
    p.add_argument("--output-dir", type=Path, default=Path("verify_output"))
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--jitter-scale",
        type=float,
        default=0.18,
        help="Jitter scale for the jittered variant. Defaults match current training.",
    )
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

    summary: dict[str, list[dict]] = {"jittered": [], "clean": []}

    for variant, jitter in [
        ("jittered", args.jitter_scale),
        ("clean", 0.0),
    ]:
        print(f"\n=== {variant} samples (jitter_scale={jitter}) ===")
        for i in range(args.num_samples):
            card = rng.choice(cards)
            bg = rng.choice(bgs)
            sample = make_sample(
                card, bg, args.image_size, rng, jitter_scale=jitter
            )
            if sample is None:
                print(f"  {variant}_{i:02d}: failed to make sample, skipping")
                continue
            composite, label_corners = sample
            pred = run_inference(model, composite)

            overlay = composite.copy()
            draw_quad(overlay, label_corners, LABEL_COLOR, "label")
            if pred is not None:
                draw_quad(overlay, pred, PRED_COLOR, "pred")
                iou = quad_iou(pred, label_corners)
                area_ratio = quad_area(pred) / max(quad_area(label_corners), 1e-6)
            else:
                iou = 0.0
                area_ratio = 0.0

            out_path = args.output_dir / f"{variant}_{i:02d}.png"
            cv2.imwrite(str(out_path), overlay)

            tag = "no-detection" if pred is None else ""
            print(
                f"  {variant}_{i:02d}: IoU={iou:.3f}  "
                f"pred_area/label_area={area_ratio:.2f} {tag}"
            )
            summary[variant].append({"iou": iou, "area_ratio": area_ratio})

    print("\n=== summary ===")
    for variant, results in summary.items():
        ious = [r["iou"] for r in results]
        ratios = [r["area_ratio"] for r in results if r["iou"] > 0]
        mean_iou = float(np.mean(ious)) if ious else 0.0
        mean_ratio = float(np.mean(ratios)) if ratios else 0.0
        print(
            f"  {variant}: mean IoU = {mean_iou:.3f}, "
            f"mean pred/label area ratio = {mean_ratio:.2f}"
        )

    print(f"\noverlays written to {args.output_dir}/")
    print("\ninterpretation:")
    print("  - jittered IoU high + clean IoU high   -> model is fine, labels are fine")
    print("  - jittered IoU high + clean IoU low    -> labels are inflated by jitter")
    print("  - jittered IoU low                     -> model is undertrained")


if __name__ == "__main__":
    main()
