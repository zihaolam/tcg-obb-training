#!/usr/bin/env bash
# Sample N card images from R2 bucket tcg-cards and download them locally.
# Defaults to 20000 of the ~70000 available.
set -euo pipefail

REMOTE="${R2_REMOTE_NAME:-r2}"
BUCKET="tcg-cards"
DEST="${1:-data/sources/cards}"
SAMPLE_N="${2:-20000}"

mkdir -p "$DEST"
picked="$(mktemp -t tcg-cards-picked.XXXXXX)"
trap 'rm -f "$picked"' EXIT

echo "listing ${REMOTE}:${BUCKET}..."
rclone lsf "${REMOTE}:${BUCKET}" --files-only -R | shuf -n "$SAMPLE_N" > "$picked"

picked_count=$(wc -l < "$picked" | tr -d ' ')
echo "sampled ${picked_count} files; downloading -> ${DEST}"

rclone copy "${REMOTE}:${BUCKET}" "$DEST" \
    --files-from "$picked" \
    --transfers 16 \
    --checkers 32 \
    --progress

count=$(find "$DEST" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.bmp" \) | wc -l | tr -d ' ')
echo "done: ${count} images in ${DEST}"
