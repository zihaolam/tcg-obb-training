"""Tiny FastAPI app for interactive OBB inference testing.

Run on the VPS where the model lives:
  uv sync --extra serve
  uv run python obb/serve.py --weights export/pytorch/best.pt

Tunnel from your laptop:
  ssh -L 8000:localhost:8000 user@vps

Open http://localhost:8000 and drop an image in.
"""

import argparse
import base64
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO

# Output dimensions of the warped crop (matches the Bun normalization step).
WARP_W = 320
WARP_H = 448


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Clockwise, starting at the corner closest to (0, 0). Matches the Bun pipeline."""
    c = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
    ordered = corners[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def encode_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def draw_overlay(img: np.ndarray, all_corners: np.ndarray, confs: np.ndarray):
    out = img.copy()
    for i, (corners, conf) in enumerate(zip(all_corners, confs)):
        color = (0, 255, 255) if i == 0 else (180, 180, 180)
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, 3)
        for j, (x, y) in enumerate(corners.astype(int)):
            cv2.circle(out, (int(x), int(y)), 8, color, -1)
            cv2.putText(
                out,
                str(j),
                (int(x) + 10, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        cx, cy = corners.mean(axis=0).astype(int)
        cv2.putText(
            out,
            f"{conf:.2f}",
            (int(cx) - 20, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
    return out


INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>OBB tester</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; }
    h1 { margin: 0 0 0.5em; }
    .drop { border: 2px dashed #999; padding: 2.5em; text-align: center; cursor: pointer; border-radius: 8px; }
    .drop:hover { background: rgba(127, 127, 127, 0.08); }
    .drop input { display: none; }
    .status { margin: 1em 0; font-family: ui-monospace, monospace; }
    .row { display: flex; gap: 1em; flex-wrap: wrap; }
    .panel { flex: 1 1 360px; }
    .panel h3 { margin: 0 0 0.5em; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.7; }
    .panel img { max-width: 100%; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.15); }
    .warp img { max-width: 320px; }
    pre { background: rgba(127,127,127,0.1); padding: 1em; border-radius: 6px; overflow: auto; font-size: 0.85em; }
  </style>
</head>
<body>
  <h1>OBB inference tester</h1>
  <label class="drop">
    <input type="file" accept="image/*" id="file">
    <div>Click or drop an image here</div>
  </label>
  <div class="status" id="status">no image yet</div>
  <div class="row">
    <div class="panel"><h3>Overlay</h3><div id="overlay"></div></div>
    <div class="panel warp"><h3>Warped crop</h3><div id="warp"></div></div>
  </div>
  <h3>Detections</h3>
  <pre id="json">(none)</pre>

  <script>
    const file = document.getElementById('file');
    const status = document.getElementById('status');
    const overlayEl = document.getElementById('overlay');
    const warpEl = document.getElementById('warp');
    const jsonEl = document.getElementById('json');

    async function send(f) {
      if (!f) return;
      const fd = new FormData();
      fd.append('image', f);
      status.textContent = 'running inference...';
      overlayEl.innerHTML = '';
      warpEl.innerHTML = '';
      const t0 = performance.now();
      try {
        const res = await fetch('/detect', { method: 'POST', body: fd });
        if (!res.ok) {
          status.textContent = 'error: HTTP ' + res.status;
          return;
        }
        const data = await res.json();
        const ms = Math.round(performance.now() - t0);
        status.textContent = `done in ${ms} ms — ${data.detections.length} detection(s)`;
        if (data.overlay) overlayEl.innerHTML = `<img src="${data.overlay}">`;
        if (data.warp) warpEl.innerHTML = `<img src="${data.warp}">`;
        jsonEl.textContent = JSON.stringify(data.detections, null, 2);
      } catch (err) {
        status.textContent = 'error: ' + err.message;
      }
    }

    file.addEventListener('change', () => send(file.files[0]));
    const drop = document.querySelector('.drop');
    drop.addEventListener('dragover', e => { e.preventDefault(); });
    drop.addEventListener('drop', e => {
      e.preventDefault();
      send(e.dataTransfer.files[0]);
    });
  </script>
</body>
</html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    print(f"loading model from {args.weights}...")
    model = YOLO(str(args.weights))

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return INDEX_HTML

    @app.post("/detect")
    async def detect(image: UploadFile = File(...)):
        data = await image.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "could not decode image")

        result = model(img, conf=args.conf, verbose=False, imgsz=args.imgsz)[0]

        if result.obb is None or result.obb.xyxyxyxy.shape[0] == 0:
            return JSONResponse(
                {"detections": [], "overlay": encode_png(img), "warp": None}
            )

        all_corners = result.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
        confs = result.obb.conf.cpu().numpy()

        overlay = draw_overlay(img, all_corners, confs)

        top = order_corners(all_corners[0])
        dst = np.array(
            [[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]],
            dtype=np.float32,
        )
        H = cv2.getPerspectiveTransform(top, dst)
        warped = cv2.warpPerspective(img, H, (WARP_W, WARP_H))

        return JSONResponse(
            {
                "detections": [
                    {
                        "confidence": float(c),
                        "corners": corners.tolist(),
                    }
                    for c, corners in zip(confs, all_corners)
                ],
                "overlay": encode_png(overlay),
                "warp": encode_png(warped),
            }
        )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
