"""Synthesize a YOLO-OBB training set by compositing card images onto random backgrounds.

Each composite is:
  - background (random crop from --backgrounds-dir, sized to --image-size)
  - one card (random pick from --cards-dir), perspective-warped onto the background
  - random rotation across the full 360 degrees, random scale and perspective tilt
  - color / blur / JPEG-compression jitter to mimic phone-camera diversity

Labels are written in YOLO OBB format: `class x1 y1 x2 y2 x3 y3 x4 y4` (normalized,
clockwise from top-left). Run with --help for CLI options.
"""

import argparse
import multiprocessing as mp
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Pokémon TCG card aspect ratio (2.5" x 3.5")
CARD_ASPECT = 2.5 / 3.5

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def load_card(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def load_background(path: Path, size: int, rng: random.Random):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w) * 1.1
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]
    y = rng.randint(0, h - size)
    x = rng.randint(0, w - size)
    return img[y : y + size, x : x + size]


def sample_dest_corners(size: int, rng: random.Random):
    """Return 4 corner points (in image coords) for the card in this composite.

    The card is randomly scaled, rotated through the full 360°, perspective-jittered,
    and translated. Full-circle rotation is the whole point — it forces the model
    to be orientation-invariant.
    """
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

    # Per-corner jitter — cheap proxy for perspective tilt without a 3D camera model
    tilt = rng.uniform(0.0, 0.18)
    jitter = (np.random.uniform(-tilt, tilt, size=(4, 2)) * card_h).astype(np.float32)
    corners = corners + jitter

    pad = card_h * 0.55
    cx = rng.uniform(pad, size - pad)
    cy = rng.uniform(pad, size - pad)
    return corners + np.array([cx, cy], dtype=np.float32)


def warp_card_onto_bg(card: np.ndarray, bg: np.ndarray, dest: np.ndarray):
    h, w = card.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dest.astype(np.float32))
    bh, bw = bg.shape[:2]

    if card.shape[2] == 4:
        warped = cv2.warpPerspective(card, H, (bw, bh), flags=cv2.INTER_LINEAR)
        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
        rgb = warped[:, :, :3].astype(np.float32)
        return (bg.astype(np.float32) * (1 - alpha) + rgb * alpha).astype(np.uint8)

    # No alpha channel — derive a mask from the destination quad
    warped = cv2.warpPerspective(card, H, (bw, bh), flags=cv2.INTER_LINEAR)
    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dest.astype(np.int32), 255)
    alpha = (mask[..., None] / 255.0).astype(np.float32)
    return (bg.astype(np.float32) * (1 - alpha) + warped.astype(np.float32) * alpha).astype(np.uint8)


def color_jitter(img: np.ndarray, rng: random.Random):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-10, 10)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.55, 1.3), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    a = rng.uniform(0.85, 1.15)
    b = rng.uniform(-15, 15)
    return np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


def maybe_blur(img: np.ndarray, rng: random.Random):
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    return img


def maybe_noise(img: np.ndarray, rng: random.Random):
    if rng.random() < 0.3:
        sigma = rng.uniform(2, 8)
        noise = np.random.normal(0, sigma, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def yolo_obb_label(corners: np.ndarray, img_size: int, class_id: int = 0):
    """Ordered clockwise starting from the top-left-most corner, then normalized."""
    c = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
    ordered = corners[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    normed = np.clip(ordered / img_size, 0, 1)
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in normed.flatten())


def generate_one(task):
    idx, card_path, bg_path, out_img, out_lbl, size, seed = task
    rng = random.Random(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    card = load_card(card_path)
    bg = load_background(bg_path, size, rng)
    if card is None or bg is None:
        return False

    dest = sample_dest_corners(size, rng)
    composite = warp_card_onto_bg(card, bg, dest)
    composite = color_jitter(composite, rng)
    composite = maybe_blur(composite, rng)
    composite = maybe_noise(composite, rng)

    quality = rng.randint(60, 95)
    ok, buf = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    Path(out_img).write_bytes(buf.tobytes())
    Path(out_lbl).write_text(yolo_obb_label(dest, size) + "\n")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cards-dir", type=Path, required=True)
    p.add_argument("--backgrounds-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument("--num-train", type=int, default=20000)
    p.add_argument("--num-val", type=int, default=2000)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cards = list_images(args.cards_dir)
    bgs = list_images(args.backgrounds_dir)
    if not cards:
        raise SystemExit(f"no cards found under {args.cards_dir}")
    if not bgs:
        raise SystemExit(f"no backgrounds found under {args.backgrounds_dir}")
    print(f"found {len(cards)} card images, {len(bgs)} backgrounds")

    rng = random.Random(args.seed)

    for split, n in [("train", args.num_train), ("val", args.num_val)]:
        img_dir = args.output_dir / split / "images"
        lbl_dir = args.output_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for i in range(n):
            card = rng.choice(cards)
            bg = rng.choice(bgs)
            stem = f"{split}_{i:07d}"
            tasks.append(
                (
                    i,
                    card,
                    bg,
                    img_dir / f"{stem}.jpg",
                    lbl_dir / f"{stem}.txt",
                    args.image_size,
                    rng.randint(0, 2**31 - 1),
                )
            )

        ok_count = 0
        with mp.Pool(args.workers) as pool:
            for ok in tqdm(
                pool.imap_unordered(generate_one, tasks, chunksize=8),
                total=len(tasks),
                desc=f"generating {split}",
            ):
                ok_count += int(ok)
        print(f"{split}: wrote {ok_count}/{n} composites")

    yaml_path = args.output_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {args.output_dir.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        "  0: card\n"
    )
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
