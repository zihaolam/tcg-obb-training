#!/usr/bin/env bash
# Upload trained model artifacts to R2 under tcg-models/ptcg-detector-yolo-obb-v2/.
# Uploads BOTH the TFJS export (for browser/Bun) and the PyTorch weights (for Python).
#
# Layout in R2:
#   tcg-models/ptcg-detector-yolo-obb-v2/tfjs/    model.json + *.bin shards
#   tcg-models/ptcg-detector-yolo-obb-v2/pytorch/ best.pt + args.yaml + results.png
set -euo pipefail

REMOTE="${R2_REMOTE_NAME:-r2}"
BUCKET="tcg-models"
PREFIX="ptcg-detector-yolo-obb-v2"

TFJS_SRC="${TFJS_SRC:-export/tfjs}"
RUN_DIR="${RUN_DIR:-runs/obb/runs/obb/train-3}"

# --- TFJS export -----------------------------------------------------------
if [ -d "$TFJS_SRC" ] && [ -f "${TFJS_SRC}/model.json" ]; then
    echo "=== uploading TFJS export: ${TFJS_SRC} -> ${REMOTE}:${BUCKET}/${PREFIX}/tfjs"
    ls -lh "$TFJS_SRC"
    rclone copy "$TFJS_SRC" "${REMOTE}:${BUCKET}/${PREFIX}/tfjs" \
        --transfers 8 --checkers 16 --progress
    echo
else
    echo "skipping TFJS (no '${TFJS_SRC}/model.json'). Run export_tfjs.py to create it."
    echo
fi

# --- PyTorch weights -------------------------------------------------------
if [ -f "${RUN_DIR}/weights/best.pt" ]; then
    echo "=== uploading PyTorch weights from ${RUN_DIR} -> ${REMOTE}:${BUCKET}/${PREFIX}/pytorch"
    # best.pt is the required one; args.yaml + results.png are nice-to-haves
    rclone copy "${RUN_DIR}/weights/best.pt" \
        "${REMOTE}:${BUCKET}/${PREFIX}/pytorch" \
        --progress
    for opt in args.yaml results.png results.csv; do
        if [ -f "${RUN_DIR}/${opt}" ]; then
            rclone copy "${RUN_DIR}/${opt}" \
                "${REMOTE}:${BUCKET}/${PREFIX}/pytorch" \
                --progress
        fi
    done
    echo
else
    echo "skipping PyTorch (no '${RUN_DIR}/weights/best.pt')."
    echo "  set RUN_DIR=path/to/run env var if your run is elsewhere."
    echo
fi

echo "done. Verify with:"
echo "  rclone tree ${REMOTE}:${BUCKET}/${PREFIX}"
