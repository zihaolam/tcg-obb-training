# Plan: replace the OBB rotation-search workaround with a properly-trained model

## Background

In the `tcg` repo, `backend/src/card/CardImageOBBWorker.ts` runs the YOLO-OBB card detector. Because the production model can only detect roughly upright cards, the worker brute-forces up to 7 rotations of the input image and keeps whichever rotation produced a confident detection.

Concrete cost (from production logs, 2026-05-13):

- Single card scan: **7,865 ms** of OBB normalization (the rotation loop), versus ~419 ms once a detection is actually found.
- Successful detection was at `-45°`, the second rotation tried — meaning all 7 rotations still ran, because the early-exit guard in `CardImageOBBWorker.ts:108-115` only fires at `rotationDegrees === 0`.

Two issues compound:

1. **Bad early-exit guard** — the loop should break when *any* rotation finds a confident box, not only the 0° pass. This is a 5-line fix, but is a symptom, not a cure.
2. **The OBB model is doing the wrong job** — an OBB head predicts `(cx, cy, w, h, angle)` and should not require pre-rotated input at all. The rotation loop exists only because the original model was trained without full-circle rotation augmentation, so its angle head only generalizes near 0°.

This plan addresses (2). The fix is to retrain with proper augmentation. Once the new model is shipped, (1) becomes moot because the whole rotation loop deletes.

## Goal

Train a YOLO-OBB model that detects cards at *any* orientation in one forward pass, then replace the production model and delete the rotation loop in the worker.

## Approach

1. **Generate a synthetic dataset.** We have ~70k card images and an open-source background image repo. The synth pipeline composites a random card onto a random background with random scale, perspective tilt, full 360° rotation, color jitter, blur, noise, and JPEG-compression artifacts. See `generate_dataset.py`.

   Recommended starting size: 20k train + 2k val. Picking from ~2-5k unique cards (not all 70k) is fine — diversity comes from augmentation, not card catalog size.

2. **Train `yolo11n-obb`** at 640×640 input. Defaults in `train.py` are tuned for this: `degrees=180, fliplr=0.5, flipud=0.5, mosaic=1.0, close_mosaic=10, single_cls=True`. 100 epochs on an RTX 4090 ≈ 2-3 hours.

3. **Validate on real photos.** Synthetic-only validation lies. Before trusting any metrics, collect ~200-500 real hand-held phone photos of cards at varied angles/lighting and label corners (CVAT or Label Studio export to YOLO OBB format). Put these in `data/real_val/` and re-run `model.val()` with that as the val set.

4. **Export to TFJS** and upload to `s3://tcg-models/ptcg-detector-yolo-obb/`. The bucket layout must match what `CardImageOBBManager.ensureModelAvailable()` expects in `tcg/backend/src/card/CardImageOBBManager.ts:19-25`:
   ```
   model.json
   metadata.yaml
   group1-shard1of3.bin
   group1-shard2of3.bin
   group1-shard3of3.bin
   ```
   Ultralytics' TFJS export usually produces matching names; verify before uploading.

5. **Smoke test in staging.** Confirm a few diverse phone photos (various angles, backgrounds, lighting) return correct boxes in a single forward pass.

6. **Clean up the runtime.** Once production is healthy on the new model, edit the `tcg` repo (see "Followup edits in tcg" below).

## Status as of 2026-05-13

- [x] Pipeline code written: `generate_dataset.py`, `train.py`, `export_tfjs.py`, `setup_pod.sh`, `pyproject.toml`.
- [x] RunPod setup walked through with the user (RTX 4090 Secure Cloud, ~$0.69/hr, ~$2 per training run).
- [ ] Source card images and backgrounds **not yet located/uploaded**. The user mentioned ~70k card images and an open-source background repo, but neither location is recorded. **First task for the next agent: ask the user where these live** (R2 bucket? local dir? URL?). Update this plan once known.
- [ ] No real-photo validation set yet. This is a non-blocking risk — first training run can use synthetic-only val just to confirm the pipeline works. But before declaring the model production-ready, collect 200-500 real labeled photos.
- [ ] No training run has been executed yet.
- [ ] No model deployed.

## Followup edits in tcg (after the new model is live)

Once the new model handles full 360° rotation in one pass, the following are dead code and should be deleted:

1. **`backend/src/card/CardImageOBBWorker.ts:68-116`** — the entire `for (const rotationDegrees of YOLO_OBB_SEARCH_ROTATIONS)` loop. Replace with a single `createYoloObbInput(buffer, config.inputSize)` call (no rotation arg) and one inference.
2. **`backend/src/card/CardImageYoloObb.ts:13-15`** — `YOLO_OBB_SEARCH_ROTATIONS` constant.
3. **`backend/src/card/CardImageYoloObb.ts:56-79`** — the `rotationDegrees` parameter on `decodeYoloObbImage`, plus the now-redundant rotate-and-redecode step. The image only needs one EXIF-orient pass.
4. **`backend/src/card/CardImageYoloObb.ts:81-203`** — the `sourceRotationDegrees` plumbing through `detectYoloObbBoxes` and `OrientedCardBox` becomes unnecessary; corners are already in the original image's coordinate frame.
5. **`backend/src/card/CardImageYoloObb.ts:205-301`** — `suppressOverlappingYoloObbDetections` and `mapDetectionCornersToOriginalImage` can be simplified or replaced with stock NMS since detections no longer come from multiple rotated frames.

Do these as a single PR after the new model is verified, not before — keeping the rotation loop alive during rollout means we can flip the model without simultaneously shipping a behavioral change.

## Open questions for the user

- **Where are the 70k card images?** R2 bucket? Local dir? URL prefix to download from? (Required before the first training run.)
- **Where is the background image repo?** Same questions.
- **Should the dataset live in R2 once generated**, so subsequent training runs don't have to re-synthesize? (Optional optimization.)
- **Is anyone going to label real validation photos?** This needs to happen before declaring the model production-ready. ~200-500 photos with corner annotations.

## Don'ts

- **Don't soften the rotation augmentation** in `generate_dataset.py`. Full 360° coverage is the whole reason this project exists. If model quality is bad, the fix is more/better data, not less rotation.
- **Don't delete the runtime rotation loop in `tcg` before the new model is verified in production.** The old workaround is ugly but functional; keep it until the replacement is proven.
- **Don't commit large image directories or model weights.** `.gitignore` already covers `data/`, `runs/`, `export/`, `*.pt`. Keep it that way.
- **Don't run full training on a Mac.** MPS works for a 5-minute smoke test (small dataset, few epochs) to confirm the pipeline runs, but real training belongs on a cloud RTX 4090.
