#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod for YOLO-OBB training. Run from training/.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync

echo
echo "--- GPU check ---"
uv run python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PY

cat <<'EOF'

--- ready ---
Next steps (run from training/):
  1. Upload sources to data/sources/{cards,backgrounds}/
  2. Generate dataset:
       uv run python generate_dataset.py \
         --cards-dir data/sources/cards \
         --backgrounds-dir data/sources/backgrounds
  3. Train:
       uv run python train.py
  4. Export to TFJS:
       uv sync --extra export
       uv run python export_tfjs.py --weights runs/obb/train/weights/best.pt
EOF
