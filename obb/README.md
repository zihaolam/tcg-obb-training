# Legacy OBB Training

This directory contains the old YOLO-OBB pipeline. It is useful for rotated
rectangle detection, but it cannot predict true perspective card corners.

Ultralytics converts 8-point OBB labels to `xywhr` rotated boxes during
training, then reconstructs rectangle corners at inference time. For perspective
normalization, prefer the `pose/` pipeline.

```bash
uv run python obb/generate_dataset.py
uv run python obb/train.py
uv run python obb/verify_obb.py --weights runs/obb/train/weights/best.pt
uv run python obb/export_model.py --weights runs/obb/train/weights/best.pt
```
