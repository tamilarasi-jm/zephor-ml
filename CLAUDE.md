# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the API

The server runs via the project virtualenv at `.venv/`, **not** a system Python install. System `python3` does not have the required packages.

```bash
# Start the API server
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

# Or with auto-reload during development
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server is production-running as a background process (PID under ubuntu user). Check `ss -tlnp | grep 8000` to confirm it's live.

## Data Preparation Pipeline

Adding a new product requires running these scripts in order:

1. **Add raw reference images** to `reference_images/<barcode>/` (original photos of the bottle)
2. **Crop bottles from backgrounds:**
   ```bash
   .venv/bin/python3 crop_reference.py
   ```
   Outputs YOLO-cropped images to `cropped_reference_images/<barcode>/`
3. **Generate CLIP embeddings:**
   ```bash
   .venv/bin/python3 generate_embeddings.py
   ```
   Outputs `.npy` embedding files to `embeddings/<barcode>/`
4. **Restart the API** so it reloads the embeddings cache from disk into RAM.

## Architecture

This is a **reverse vending machine (RVM) AI verification backend** — it accepts bottle images and decides whether a returned bottle matches a claimed product barcode.

**Inference pipeline (per request to `POST /detect`):**
1. Receive image bytes + barcode string (multipart form)
2. YOLOv8n (`yolov8n.pt`) detects and crops the bottle from the background — rejects if no `bottle`/`vase` class detected
3. CLIP (`clip_local/`) generates a normalized L2 embedding of the cropped bottle
4. The embedding is compared via dot-product cosine similarity against pre-computed reference embeddings loaded in `embeddings_cache` (in-process RAM dict, keyed by barcode)
5. **Top-2 scoring**: average of the top 2 similarity scores across all reference images for that barcode. Threshold: `75.0` (configured as `SIMILARITY_THRESHOLD` in `main.py`)
6. Returns `accepted` or `rejected` with scores

**Models (all loaded at startup, held in module globals):**
- `yolo_model` — YOLOv8n via Ultralytics
- `clip_model` / `clip_processor` — CLIP loaded from local files at `clip_local/`
- `embeddings_cache` — dict of `{barcode: [np.ndarray, ...]}` loaded from `embeddings/`

**Key design decision:** No disk I/O during inference. Images arrive as bytes in RAM, embeddings are pre-loaded at startup. The `harcoded_main.py` is the legacy version (Windows paths, disk-based, two-gate flow) — `main.py` is the optimized production server.

**Simulator client:** `rvm_withoutcam.py` is an interactive CLI that simulates the physical RVM hardware — it calls `POST /verify-barcode` then `POST /verify-bottle` against `localhost:8000`. Note: `main.py` exposes only `POST /detect` (combined single-gate), while the legacy `harcoded_main.py` had the two-gate split. The simulator still targets the old endpoint names.

## File Layout

| Path | Purpose |
|---|---|
| `main.py` | Production FastAPI app |
| `harcoded_main.py` | Legacy two-gate version (reference only) |
| `crop_reference.py` | Step 1 of data prep: YOLO-crop raw reference images |
| `generate_embeddings.py` | Step 2 of data prep: generate CLIP `.npy` embeddings |
| `rvm_withoutcam.py` | Hardware simulator CLI client |
| `clip_local/` | Locally cached CLIP model weights |
| `reference_images/<barcode>/` | Raw input photos per product |
| `cropped_reference_images/<barcode>/` | YOLO-cropped photos (intermediate) |
| `embeddings/<barcode>/` | `.npy` CLIP embeddings loaded at runtime |
| `yolov8n.pt` | YOLOv8 nano weights |
