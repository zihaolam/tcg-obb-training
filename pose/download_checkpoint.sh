#!/usr/bin/env bash
# Download a YOLO pose training checkpoint from R2 so it can be resumed.
#
# Examples:
#   ./pose/download_checkpoint.sh train-1
#   ./pose/download_checkpoint.sh train-2 /tcg-obb-training/runs/pose/runs/pose/train-2
set -euo pipefail

REMOTE="${R2_REMOTE_NAME:-r2}"
BUCKET="tcg-models"
PREFIX="${POSE_MODEL_PREFIX:-ptcg-detector-yolo-pose-v1}"

usage() {
    cat <<EOF
Usage:
  $0 <checkpoint-name> [run-dir]

Examples:
  $0 train-1
  $0 train-2
  $0 train-2 /tcg-obb-training/runs/pose/runs/pose/train-2

Environment:
  R2_REMOTE_NAME       rclone remote name (default: r2)
  POSE_MODEL_PREFIX    R2 prefix under ${BUCKET}/ (default: ptcg-detector-yolo-pose-v1)
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ] || [ -z "${1:-}" ]; then
    usage
    exit 0
fi

CHECKPOINT_NAME="$1"
RUN_DIR="${2:-runs/pose/runs/pose/${CHECKPOINT_NAME}}"
SRC="${REMOTE}:${BUCKET}/${PREFIX}/checkpoints/${CHECKPOINT_NAME}"

echo "=== checkpoint: ${SRC} -> ${RUN_DIR}"
mkdir -p "$RUN_DIR"

rclone copy "$SRC" "$RUN_DIR" \
    --transfers 8 --checkers 16 --progress

if [ ! -f "${RUN_DIR}/weights/last.pt" ]; then
    echo "error: '${RUN_DIR}/weights/last.pt' not found after download." >&2
    echo "Check that checkpoint '${CHECKPOINT_NAME}' exists under ${REMOTE}:${BUCKET}/${PREFIX}/checkpoints/." >&2
    exit 1
fi

echo
echo "downloaded checkpoint '${CHECKPOINT_NAME}'. Resume with:"
echo "  uv run python pose/train.py --model ${RUN_DIR}/weights/last.pt --resume"
