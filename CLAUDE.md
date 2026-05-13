# tcg-obb-training

Synthetic-data generation and YOLO-OBB training pipeline for the Pokémon TCG card scanner. Sister project to `../tcg` (Bun/TypeScript). The trained model is consumed at runtime by `tcg/backend/src/card/CardImageOBBWorker.ts`.

## Why this exists

The production OBB model can't detect rotated cards in a single pass, so the runtime worker brute-forces up to 7 rotations of the input image (0°, ±15°, ±30°, ±45°) — see `tcg/backend/src/card/CardImageOBBWorker.ts:68-116` and `YOLO_OBB_SEARCH_ROTATIONS` in `CardImageYoloObb.ts`. That makes a single detection take ~7-8 seconds for tilted cards.

Root cause is on the training side: the previous model wasn't trained with full 360° rotation augmentation, so its OBB angle head only generalizes to a narrow range. This repo retrains it the right way and removes the runtime workaround.

## Tech stack

- **Python 3.10-3.12**, managed by [`uv`](https://github.com/astral-sh/uv)
- **ultralytics** for YOLO-OBB training (`yolo11n-obb` by default — nano variant)
- **opencv-python + numpy** for synthetic compositing
- **tensorflow + tensorflowjs** (optional extra) for exporting the trained model to TFJS graph-model format, which is the format the Bun worker loads

GPU training is intended to run on a cloud RTX 4090 (RunPod). CPU/MPS works for smoke tests but is impractically slow for full runs.

## Files

| File | Purpose |
|---|---|
| `generate_dataset.py` | Composites cards onto backgrounds with full-360° rotation, perspective tilt, scale jitter, color/blur/noise/JPEG aug. Writes YOLO-OBB labels (`class x1 y1 x2 y2 x3 y3 x4 y4`, normalized, clockwise from top-left). |
| `train.py` | Wraps `ultralytics.YOLO.train()` with our defaults (degrees=180, fliplr/flipud, mosaic, single_cls=True). |
| `export_tfjs.py` | Wraps `model.export(format="tfjs")` and copies the result to `export/tfjs/`. |
| `setup_pod.sh` | One-shot bootstrap for a fresh RunPod pod: installs `uv`, runs `uv sync`, sanity-checks the GPU. |
| `pyproject.toml` | Core deps + optional `export` extra (heavy TF deps, install only when exporting). |

## Commands

```bash
# fresh setup (anywhere)
uv sync

# generate a 20k/2k train/val dataset (defaults shown)
uv run python generate_dataset.py \
  --cards-dir data/sources/cards \
  --backgrounds-dir data/sources/backgrounds \
  --num-train 20000 --num-val 2000

# train (defaults to yolo11n-obb, RunPod GPU 0; use --device mps on Mac)
uv run python train.py --epochs 100 --batch 32

# export trained weights to TFJS (requires extras)
uv sync --extra export
uv run python export_tfjs.py --weights runs/obb/train/weights/best.pt
```

The exported `export/tfjs/` directory's contents go to `s3://tcg-models/ptcg-detector-yolo-obb/`, which is where `CardImageOBBManager.ensureModelAvailable()` in the `tcg` repo pulls from. The manager downloads `model.json`, `metadata.yaml`, and three `group1-shardNof3.bin` files — make sure the export produces matching names (Ultralytics' TFJS export normally does).

## Conventions

- **Single-class detection**: class 0 is `card`. Set in the generated `dataset.yaml`.
- **Input size**: 640×640 to match the existing worker (`OBB_INPUT_SIZE = 640` in `CardImageOBBManager.ts:13`).
- **Card aspect**: 2.5 / 3.5 ≈ 0.714 (Pokémon TCG standard).
- **Label format**: YOLO OBB — 8 normalized coords, clockwise starting from the top-left-most corner.
- **Full 360° rotation in synthesis**: this is the whole point. Don't soften it. The whole reason for retraining is that the previous model lacked this coverage.

## What does NOT belong in this repo

- Anything Bun/TypeScript — that's in `../tcg`.
- Real photos for validation should live in `data/real_val/` (gitignored). Do not commit large image dirs.
- Model weights (`*.pt`, `runs/`, `export/`) are gitignored.

## Plan and history

Active plan: `.claude/plans/01-replace-obb-rotation-search.md`. Read it before starting work — it has the full context, current status, and the followup edits needed in the `tcg` repo once the new model is deployed.
