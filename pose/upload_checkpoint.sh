#!/usr/bin/env bash
# Upload a YOLO pose training run directory to R2 so it can be resumed later.
#
# Example:
#   ./pose/upload_checkpoint.sh /tcg-obb-training/runs/pose/runs/pose/train-2 train-2
set -euo pipefail

REMOTE="${R2_REMOTE_NAME:-r2}"
BUCKET="tcg-models"
PREFIX="${POSE_MODEL_PREFIX:-ptcg-detector-yolo-pose-v1}"

usage() {
    cat <<EOF
Usage:
  $0 <run-dir> [checkpoint-name]

Examples:
  $0 runs/pose/train
  $0 /tcg-obb-training/runs/pose/runs/pose/train-2 train-2

Environment:
  R2_REMOTE_NAME       rclone remote name (default: r2)
  POSE_MODEL_PREFIX    R2 prefix under ${BUCKET}/ (default: ptcg-detector-yolo-pose-v1)
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ] || [ -z "${1:-}" ]; then
    usage
    exit 0
fi

RUN_DIR="$1"
CHECKPOINT_NAME="${2:-$(basename "$RUN_DIR")}"
DST="${REMOTE}:${BUCKET}/${PREFIX}/checkpoints/${CHECKPOINT_NAME}"

if [ ! -d "$RUN_DIR" ]; then
    echo "error: run dir '${RUN_DIR}' not found." >&2
    exit 1
fi

if [ ! -f "${RUN_DIR}/weights/last.pt" ]; then
    echo "error: '${RUN_DIR}/weights/last.pt' not found. This is required for resume." >&2
    exit 1
fi

echo "=== checkpoint: ${RUN_DIR} -> ${DST}"
echo "including full run metadata plus weights/last.pt for resume"
ls -lh "${RUN_DIR}/weights"

rclone copy "$RUN_DIR" "$DST" \
    --transfers 8 --checkers 16 --progress

echo
echo "done. To restore on a fresh pod, run:"
echo "  mkdir -p ${RUN_DIR}"
echo "  rclone copy ${DST} ${RUN_DIR} --transfers 8 --checkers 16 --progress"
echo "  uv run python pose/train.py --model ${RUN_DIR}/weights/last.pt --resume"
