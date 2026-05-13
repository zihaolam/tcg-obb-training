#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod for YOLO-OBB training. Run from training/.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

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
