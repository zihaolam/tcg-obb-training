"""Export a trained YOLO-OBB model to TensorFlow.js graph-model format.

The Bun worker (backend/src/card/CardImageOBBWorker.ts) loads the exported
model with `tf.loadGraphModel` from `model.json` plus the shard `.bin` files.
After export, upload the contents of --output-dir to:

    s3://tcg-models/ptcg-detector-yolo-obb-v2/

Install the export-only deps first:

    uv sync --extra export
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--output-dir", type=Path, default=Path("export/tfjs"))
    args = p.parse_args()

    model = YOLO(str(args.weights))
    exported = Path(model.export(format="tfjs", imgsz=args.imgsz))

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    shutil.copytree(exported, args.output_dir)

    print(f"exported to {args.output_dir}")
    print("upload these to s3://tcg-models/ptcg-detector-yolo-obb-v2/:")
    for f in sorted(args.output_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
