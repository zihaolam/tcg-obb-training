#!/usr/bin/env bash
# Download all background images from R2 bucket tcg-training-background.
set -euo pipefail

REMOTE="${R2_REMOTE_NAME:-r2}"
BUCKET="tcg-training-background"
DEST="${1:-data/sources/backgrounds}"

mkdir -p "$DEST"

echo "syncing ${REMOTE}:${BUCKET} -> ${DEST}"
rclone copy "${REMOTE}:${BUCKET}" "$DEST" \
    --transfers 16 \
    --checkers 32 \
    --progress

count=$(find "$DEST" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.bmp" \) | wc -l | tr -d ' ')
echo "done: ${count} images in ${DEST}"
