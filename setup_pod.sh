#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod for card detector training. Run from repo root.
set -euo pipefail

echo "installing/updating uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

uv sync

if ! command -v rclone >/dev/null 2>&1; then
    echo "installing rclone..."
    curl -LsSf https://rclone.org/install.sh | sudo bash
fi

# Configure R2 remote from env vars if all four are set.
# Expected: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (R2_REMOTE_NAME optional, defaults to "r2")
if [[ -n "${R2_ACCOUNT_ID:-}" && -n "${R2_ACCESS_KEY_ID:-}" && -n "${R2_SECRET_ACCESS_KEY:-}" ]]; then
    remote="${R2_REMOTE_NAME:-r2}"
    conf_dir="${HOME}/.config/rclone"
    conf_file="${conf_dir}/rclone.conf"
    mkdir -p "$conf_dir"
    if grep -q "^\[${remote}\]" "$conf_file" 2>/dev/null; then
        echo "rclone remote '${remote}' already configured, skipping"
    else
        echo "writing rclone remote '${remote}' to ${conf_file}"
        cat >> "$conf_file" <<EOF

[${remote}]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY_ID}
secret_access_key = ${R2_SECRET_ACCESS_KEY}
endpoint = https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
region = auto
EOF
        chmod 600 "$conf_file"
    fi
else
    echo "skipping rclone R2 config (set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY to enable)"
fi

echo
echo "--- GPU check ---"
uv run python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check that uv installed torch 2.8.0 from the cu128 index and that the pod has an NVIDIA GPU.")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"capability: {torch.cuda.get_device_capability(0)}")
    print(f"vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PY

cat <<'EOF'

--- ready ---
Next steps (run from repo root):
  1. Upload sources to data/sources/{cards,backgrounds}/
  2. Generate pose dataset:
       uv run python pose/generate_dataset.py \
         --cards-dir data/sources/cards \
         --backgrounds-dir data/sources/backgrounds
  3. Train:
       uv run python pose/train.py
  4. Export:
       uv sync --extra export
       uv run python pose/export_model.py --weights runs/pose/train/weights/best.pt
  5. Optional upload:
       ./pose/upload_model.sh

Legacy OBB commands are under obb/:
  uv run python obb/generate_dataset.py
  uv run python obb/train.py
EOF
