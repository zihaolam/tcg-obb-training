# Pose Card-Corner Training

This pipeline trains YOLO pose to detect each card and predict four independent
corner keypoints:

1. source card top-left
2. source card top-right
3. source card bottom-right
4. source card bottom-left

Use these keypoints with `cv2.getPerspectiveTransform` to normalize perspective
photos. Unlike YOLO-OBB, this model is not limited to rotated rectangles.

## RunPod

```bash
export R2_ACCOUNT_ID="..."
export R2_ACCESS_KEY_ID="..."
export R2_SECRET_ACCESS_KEY="..."
export R2_REMOTE_NAME="r2"

./setup_pod.sh

./sync_cards.sh data/sources/cards 20000
./sync_backgrounds.sh data/sources/backgrounds
if [ -d data/sources/cards/cards ]; then ./flatten_cards.sh data/sources/cards; fi

uv sync

uv run python pose/generate_dataset.py \
  --cards-dir data/sources/cards \
  --backgrounds-dir data/sources/backgrounds \
  --output-dir data/pose \
  --num-train 20000 \
  --num-val 2000 \
  --min-cards 1 \
  --max-cards 3 \
  --workers 16

uv run python pose/train.py \
  --data data/pose/dataset.yaml \
  --model yolo11n-pose.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --device 0 \
  --workers 8

uv sync --extra export
uv run python pose/export_model.py \
  --weights runs/pose/train/weights/best.pt \
  --output-dir export/pose

./pose/upload_model.sh export/pose
```

The root `pyproject.toml` pins `torch==2.8.0` and `torchvision==0.23.0` from
PyTorch's CUDA 12.8 wheel index, which is the intended setup for RTX 5090
RunPod instances.

The generator defaults to `--min-cards 1 --max-cards 3`, so the model sees
multi-card scenes. Set both to `1` if you want single-card-only training.

## Checkpoint Upload For Resume

On ephemeral pods, upload the YOLO run directory before shutting the pod down:

```bash
cd /tcg-obb-training

./pose/upload_checkpoint.sh \
  /tcg-obb-training/runs/pose/runs/pose/train-2 \
  train-2
```

Restore and resume on a new pod:

```bash
cd /tcg-obb-training

./pose/download_checkpoint.sh \
  train-2 \
  /tcg-obb-training/runs/pose/runs/pose/train-2

nohup uv run python pose/train.py \
  --model /tcg-obb-training/runs/pose/runs/pose/train-2/weights/last.pt \
  --resume \
  > train_pose_resume.log 2>&1 &
```

The dataset path from the original run must exist before resume. If the pod is
fresh, download sources and regenerate `data/pose` first.

Checkpoint names are explicit. Use the matching name and destination when
switching between runs:

```bash
./pose/download_checkpoint.sh train-1 /tcg-obb-training/runs/pose/runs/pose/train-1
./pose/download_checkpoint.sh train-2 /tcg-obb-training/runs/pose/runs/pose/train-2
./pose/download_checkpoint.sh train-3 /tcg-obb-training/runs/pose/runs/pose/train-3
```

## Label Format

```text
class bbox_x bbox_y bbox_w bbox_h tl_x tl_y tl_v tr_x tr_y tr_v br_x br_y br_v bl_x bl_y bl_v
```

The bbox is axis-aligned and only supports detection. The keypoints are the
source card corners to use for perspective correction.
