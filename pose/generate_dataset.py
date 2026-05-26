"""Synthesize a YOLO-pose training set for true card-corner detection.

Each composite contains one or more cards perspective-warped onto a background.
Labels are written in YOLO pose format:

    class bbox_x bbox_y bbox_w bbox_h kp1_x kp1_y kp1_v ... kp4_x kp4_y kp4_v

All coordinates are normalized. Keypoints are ordered by the source card's
physical corners: top-left, top-right, bottom-right, bottom-left. Visibility is
always 2 because synthetic cards are fully visible.
"""

import argparse
import multiprocessing as mp
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Pokemon TCG card aspect ratio (2.5" x 3.5")
CARD_ASPECT = 2.5 / 3.5

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def load_image(path: Path, flags: int):
    img = cv2.imread(str(path), flags)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def load_card(path: Path):
    return load_image(path, cv2.IMREAD_UNCHANGED)


def load_background(path: Path, size: int, rng: random.Random):
    img = load_image(path, cv2.IMREAD_COLOR)
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


def order_corners_for_display(corners: np.ndarray) -> np.ndarray:
    """Return corners clockwise, starting with the top-left-most point."""
    c = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
    ordered = corners[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_hull = cv2.convexHull(a.astype(np.float32))
    b_hull = cv2.convexHull(b.astype(np.float32))
    inter_area, _ = cv2.intersectConvexConvex(a_hull, b_hull)
    union = cv2.contourArea(a_hull) + cv2.contourArea(b_hull) - inter_area
    return 0.0 if union <= 0 else float(inter_area / union)


def sample_dest_corners(size: int, rng: random.Random):
    """Sample a true perspective quadrilateral inside the output image."""
    for _ in range(100):
        scale = rng.uniform(0.2, 0.72)
        card_h = scale * size
        card_w = card_h * CARD_ASPECT
        hw, hh = card_w / 2, card_h / 2
        corners = np.array(
            [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32
        )

        # One-sided foreshortening is the failure mode this pose pipeline must learn.
        if rng.random() < 0.8:
            strength = rng.uniform(0.04, 0.35)
            side = rng.choice(["left", "right", "top", "bottom"])
            if side == "left":
                idx = [0, 3]
                corners[idx, 0] += card_w * strength
                corners[idx, 1] *= rng.uniform(0.72, 0.95)
            elif side == "right":
                idx = [1, 2]
                corners[idx, 0] -= card_w * strength
                corners[idx, 1] *= rng.uniform(0.72, 0.95)
            elif side == "top":
                idx = [0, 1]
                corners[idx, 1] += card_h * strength
                corners[idx, 0] *= rng.uniform(0.72, 0.95)
            else:
                idx = [2, 3]
                corners[idx, 1] -= card_h * strength
                corners[idx, 0] *= rng.uniform(0.72, 0.95)

        tilt = rng.uniform(0.0, 0.08)
        jitter = (np.random.uniform(-tilt, tilt, size=(4, 2)) * card_h).astype(
            np.float32
        )
        corners += jitter

        angle = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        corners = corners @ np.array([[c, -s], [s, c]], dtype=np.float32).T
        if not cv2.isContourConvex(corners.reshape(-1, 1, 2).astype(np.float32)):
            continue

        min_xy = corners.min(axis=0)
        max_xy = corners.max(axis=0)
        margin = 8.0
        if max_xy[0] - min_xy[0] > size - 2 * margin:
            continue
        if max_xy[1] - min_xy[1] > size - 2 * margin:
            continue

        cx = rng.uniform(-min_xy[0] + margin, size - max_xy[0] - margin)
        cy = rng.uniform(-min_xy[1] + margin, size - max_xy[1] - margin)
        return corners + np.array([cx, cy], dtype=np.float32)

    raise RuntimeError("failed to sample an in-bounds card quad")


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

    warped = cv2.warpPerspective(card, H, (bw, bh), flags=cv2.INTER_LINEAR)
    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dest.astype(np.int32), 255)
    alpha = (mask[..., None] / 255.0).astype(np.float32)
    return (bg.astype(np.float32) * (1 - alpha) + warped.astype(np.float32) * alpha).astype(np.uint8)


def color_jitter(img: np.ndarray, rng: random.Random):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-10, 10)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.35), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.5, 1.35), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    a = rng.uniform(0.82, 1.18)
    b = rng.uniform(-18, 18)
    return np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


def maybe_blur(img: np.ndarray, rng: random.Random):
    if rng.random() < 0.35:
        k = rng.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    return img


def maybe_noise(img: np.ndarray, rng: random.Random):
    if rng.random() < 0.35:
        sigma = rng.uniform(2, 10)
        noise = np.random.normal(0, sigma, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def yolo_pose_label(corners: np.ndarray, img_size: int, class_id: int = 0):
    min_xy = corners.min(axis=0)
    max_xy = corners.max(axis=0)
    bbox_c = (min_xy + max_xy) / 2
    bbox_wh = max_xy - min_xy
    bbox = np.concatenate([bbox_c, bbox_wh]) / img_size

    keypoints = []
    for x, y in corners / img_size:
        keypoints.extend([float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1)), 2.0])

    values = [class_id, *np.clip(bbox, 0, 1).tolist(), *keypoints]
    return " ".join(
        str(int(v)) if i == 0 or (i >= 5 and (i - 7) % 3 == 0) else f"{v:.6f}"
        for i, v in enumerate(values)
    )


def choose_card_count(rng: random.Random, min_cards: int, max_cards: int) -> int:
    if min_cards == max_cards:
        return min_cards
    return rng.randint(min_cards, max_cards)


def generate_one(task):
    (
        _idx,
        card_paths,
        bg_path,
        out_img,
        out_lbl,
        size,
        min_cards,
        max_cards,
        seed,
    ) = task
    rng = random.Random(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    bg = load_background(bg_path, size, rng)
    if bg is None:
        return False

    composite = bg
    labels = []
    placed = []
    target_count = choose_card_count(rng, min_cards, max_cards)

    for _ in range(target_count):
        card = load_card(rng.choice(card_paths))
        if card is None:
            continue

        dest = None
        for _attempt in range(60):
            candidate = sample_dest_corners(size, rng)
            if all(quad_iou(candidate, prev) < 0.08 for prev in placed):
                dest = candidate
                break
        if dest is None:
            continue

        composite = warp_card_onto_bg(card, composite, dest)
        labels.append(yolo_pose_label(dest, size))
        placed.append(dest)

    if not labels:
        return False

    composite = color_jitter(composite, rng)
    composite = maybe_blur(composite, rng)
    composite = maybe_noise(composite, rng)

    quality = rng.randint(60, 95)
    ok, buf = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    Path(out_img).write_bytes(buf.tobytes())
    Path(out_lbl).write_text("\n".join(labels) + "\n")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cards-dir", type=Path, default="data/sources/cards")
    p.add_argument("--backgrounds-dir", type=Path, default="data/sources/backgrounds")
    p.add_argument("--output-dir", type=Path, default=Path("data/pose"))
    p.add_argument("--num-train", type=int, default=20000)
    p.add_argument("--num-val", type=int, default=2000)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--min-cards", type=int, default=1)
    p.add_argument("--max-cards", type=int, default=3)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.min_cards < 1:
        raise SystemExit("--min-cards must be >= 1")
    if args.max_cards < args.min_cards:
        raise SystemExit("--max-cards must be >= --min-cards")

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
            bg = rng.choice(bgs)
            stem = f"{split}_{i:07d}"
            tasks.append(
                (
                    i,
                    cards,
                    bg,
                    img_dir / f"{stem}.jpg",
                    lbl_dir / f"{stem}.txt",
                    args.image_size,
                    args.min_cards,
                    args.max_cards,
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
        "kpt_shape: [4, 3]\n"
        "flip_idx: [1, 0, 3, 2]\n"
        "names:\n"
        "  0: card\n"
    )
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
