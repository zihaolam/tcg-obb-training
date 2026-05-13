#!/usr/bin/env bash
# Upload an exported TFJS model to R2 at tcg-models/ptcg-detector-yolo-obb-v2.
set -euo pipefail

REMOTE="${R2_REMOTE_NAME:-r2}"
BUCKET="tcg-models"
PREFIX="ptcg-detector-yolo-obb-v2"
SRC="${1:-export/tfjs}"

if [ ! -d "$SRC" ]; then
    echo "error: source dir '${SRC}' not found. Run export_tfjs.py first." >&2
    exit 1
fi

if [ ! -f "${SRC}/model.json" ]; then
    echo "error: '${SRC}/model.json' not found — is this really a TFJS export?" >&2
    exit 1
fi

echo "uploading ${SRC} -> ${REMOTE}:${BUCKET}/${PREFIX}"
echo "contents:"
ls -lh "$SRC"
echo

rclone copy "$SRC" "${REMOTE}:${BUCKET}/${PREFIX}" \
    --transfers 8 \
    --checkers 16 \
    --progress

echo
echo "done. Verify with:"
echo "  rclone ls ${REMOTE}:${BUCKET}/${PREFIX}"
