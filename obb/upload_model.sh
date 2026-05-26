#!/usr/bin/env bash
# Upload all trained model artifacts to R2 under tcg-models/ptcg-detector-yolo-obb-v2/.
# Expects `obb/export_model.py` to have run, populating ./export/{pytorch,onnx,tfjs}.
#
# Layout in R2 mirrors local:
#   tcg-models/ptcg-detector-yolo-obb-v2/pytorch/best.pt
#   tcg-models/ptcg-detector-yolo-obb-v2/onnx/best.onnx
#   tcg-models/ptcg-detector-yolo-obb-v2/tfjs/{model.json, *.bin}
set -euo pipefail

REMOTE="${R2_REMOTE_NAME:-r2}"
BUCKET="tcg-models"
PREFIX="ptcg-detector-yolo-obb-v2"
SRC="${1:-export}"

if [ ! -d "$SRC" ]; then
    echo "error: '${SRC}' not found. Run obb/export_model.py first." >&2
    exit 1
fi

uploaded_any=0

upload_subdir() {
    local subdir="$1"
    local expected_file="$2"
    local path="${SRC}/${subdir}"
    if [ -d "$path" ] && [ -f "${path}/${expected_file}" ]; then
        echo "=== ${subdir}: ${path} -> ${REMOTE}:${BUCKET}/${PREFIX}/${subdir}"
        ls -lh "$path"
        rclone copy "$path" "${REMOTE}:${BUCKET}/${PREFIX}/${subdir}" \
            --transfers 8 --checkers 16 --progress
        echo
        uploaded_any=1
    else
        echo "skipping ${subdir} (missing ${path}/${expected_file})"
        echo
    fi
}

upload_subdir "pytorch" "best.pt"
upload_subdir "onnx"    "best.onnx"
upload_subdir "tfjs"    "model.json"

if [ "$uploaded_any" -eq 0 ]; then
    echo "error: nothing uploaded — did obb/export_model.py produce any output?" >&2
    exit 1
fi

echo "done. Verify with:"
echo "  rclone tree ${REMOTE}:${BUCKET}/${PREFIX}"
