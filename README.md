# Pokemon TCG Card Training

This repo trains card detectors from synthetic composites.

- `pose/` is the current pipeline. It trains YOLO pose to predict true card
  corners for perspective correction.
- `obb/` is the legacy YOLO-OBB pipeline. It predicts rotated rectangles, not
  true perspective corners.

## Fast RunPod Recipe

Run these from the repo root on RunPod. This assumes you have the R2 env vars
available in the pod:

```bash
export R2_ACCOUNT_ID="..."
export R2_ACCESS_KEY_ID="..."
export R2_SECRET_ACCESS_KEY="..."
export R2_REMOTE_NAME="r2"
```

Then run:

```bash
./setup_pod.sh

# Download training sources from R2.
# Args: destination, number of sampled card images.
./sync_cards.sh data/sources/cards 20000
./sync_backgrounds.sh data/sources/backgrounds

# Only needed if the card bucket downloads into data/sources/cards/cards/... .
if [ -d data/sources/cards/cards ]; then ./flatten_cards.sh data/sources/cards; fi

# Generate synthetic pose dataset.
uv run python pose/generate_dataset.py \
  --cards-dir data/sources/cards \
  --backgrounds-dir data/sources/backgrounds \
  --output-dir data/pose \
  --num-train 20000 \
  --num-val 2000 \
  --min-cards 1 \
  --max-cards 3 \
  --workers 16

# Train the nano pose model.
uv run python pose/train.py \
  --data data/pose/dataset.yaml \
  --model yolo11n-pose.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --device 0 \
  --workers 8

# Export artifacts.
uv sync --extra export
uv run python pose/export_model.py \
  --weights runs/pose/train/weights/best.pt \
  --output-dir export/pose

# Upload to R2: tcg-models/ptcg-detector-yolo-pose-v1/
./pose/upload_model.sh export/pose
```

`setup_pod.sh` updates `uv`, then runs `uv sync` against the PyTorch CUDA 12.8
wheel index. The project pins `torch==2.8.0` and `torchvision==0.23.0`, which is
the official PyTorch 2.8 CUDA 12.8 pair for RTX 5090-class GPUs.

## Cheap Smoke Test

Use this before a full paid run if you changed code:

```bash
./setup_pod.sh
./sync_cards.sh data/sources/cards 200
./sync_backgrounds.sh data/sources/backgrounds
if [ -d data/sources/cards/cards ]; then ./flatten_cards.sh data/sources/cards; fi

uv run python pose/generate_dataset.py \
  --cards-dir data/sources/cards \
  --backgrounds-dir data/sources/backgrounds \
  --output-dir data/pose-smoke \
  --num-train 100 \
  --num-val 20 \
  --workers 8

uv run python pose/train.py \
  --data data/pose-smoke/dataset.yaml \
  --epochs 1 \
  --batch 16 \
  --name smoke
```

## R2 Buckets Used By Scripts

- `./sync_cards.sh`: downloads a random sample from `r2:tcg-assets`.
- `./sync_backgrounds.sh`: downloads all backgrounds from
  `r2:tcg-training-background`.
- `./pose/upload_model.sh`: uploads to
  `r2:tcg-models/ptcg-detector-yolo-pose-v1` by default.

Override the upload prefix if needed:

```bash
POSE_MODEL_PREFIX=ptcg-detector-yolo-pose-v2 ./pose/upload_model.sh export/pose
```

## Saving Training Checkpoints From Ephemeral Pods

Use this before shutting down a pod, or after a long run reaches a useful
epoch. Upload the whole YOLO run directory, not just `best.pt`, so resume has
`weights/last.pt` and the run metadata:

```bash
cd /tcg-obb-training

# Use a specific checkpoint name like train-1, train-2, train-3, etc.
./pose/upload_checkpoint.sh \
  /tcg-obb-training/runs/pose/runs/pose/train-2 \
  train-2
```

On a fresh pod, restore it to the same path and resume:

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

The training dataset still needs to exist at the path stored in `args.yaml`,
usually `data/pose/dataset.yaml`. Re-run the source download and
`pose/generate_dataset.py` first if the fresh pod does not have it.

To restore another checkpoint, change both instances of the name:

```bash
./pose/download_checkpoint.sh train-1 /tcg-obb-training/runs/pose/runs/pose/train-1
./pose/download_checkpoint.sh train-2 /tcg-obb-training/runs/pose/runs/pose/train-2
./pose/download_checkpoint.sh train-3 /tcg-obb-training/runs/pose/runs/pose/train-3
```
