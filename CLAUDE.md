# tcg-obb-training

Synthetic training pipelines for Pokemon TCG card detection.

The repo now has two model families:

| Directory | Purpose |
|---|---|
| `pose/` | Current path. Trains a YOLO pose model that detects each card and predicts four independent physical card-corner keypoints: top-left, top-right, bottom-right, bottom-left. Use these points for perspective normalization. |
| `obb/` | Legacy path. Trains a YOLO-OBB model. Useful for rough rotated-card detection, but OBB outputs are rotated rectangles, not true perspective corners. |

## Why Pose Replaces OBB

Ultralytics OBB accepts 8-point labels, but internally converts them to
`xywhr` (`center_x`, `center_y`, `width`, `height`, `rotation`) via
`cv2.minAreaRect`. That means the model cannot represent a one-sided perspective
quad. Its returned `xyxyxyxy` points are reconstructed rectangle corners.

For card scanning, the runtime needs true projected card corners so it can run a
perspective warp. YOLO pose is a better fit because each corner is an independent
keypoint.

## RunPod Commands

```bash
# fresh setup
uv sync

# generate a pose dataset
uv run python pose/generate_dataset.py \
  --cards-dir data/sources/cards \
  --backgrounds-dir data/sources/backgrounds \
  --num-train 20000 --num-val 2000

# train pose model
uv run python pose/train.py --epochs 100 --batch 32

# export PyTorch + ONNX artifacts
uv sync --extra export
uv run python pose/export_model.py --weights runs/pose/train/weights/best.pt

# optional upload to R2
./pose/upload_model.sh
```

The pose dataset generator writes `data/pose/dataset.yaml` with:

```yaml
kpt_shape: [4, 3]
flip_idx: [1, 0, 3, 2]
names:
  0: card
```

Each label row is:

```text
class bbox_x bbox_y bbox_w bbox_h tl_x tl_y tl_v tr_x tr_y tr_v br_x br_y br_v bl_x bl_y bl_v
```

All coordinates are normalized. The keypoint order is the source card's physical
corner order, not "top-left-most in the image." Visibility is `2` for synthetic
samples.

## Legacy OBB Commands

```bash
uv run python obb/generate_dataset.py
uv run python obb/train.py
uv run python obb/verify_obb.py --weights runs/obb/train/weights/best.pt
uv run python obb/export_model.py --weights runs/obb/train/weights/best.pt
```

## Notes

- Input size remains `640` by default.
- Source card art should live under `data/sources/cards`.
- Backgrounds should live under `data/sources/backgrounds`.
- Large generated data, runs, exports, and model weights are gitignored.
