import os
import cv2
import numpy as np
import torch
import logging
from datetime import datetime, timezone
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from auth import require_machine_key, require_admin_key
from log_shipper import emit_event, start_log_shipper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIG ---
CLIP_PATH = "/home/ubuntu/zephor-ml/clip_local"
YOLO_PATH = "yolov8n.pt"
EMBEDDINGS_DIR = "/home/ubuntu/zephor-ml/embeddings"
REFERENCE_DIR = "/home/ubuntu/zephor-ml/reference_images"
CROPPED_DIR = "/home/ubuntu/zephor-ml/cropped_reference_images"
SIMILARITY_THRESHOLD = 75.0

# --- GLOBAL MODELS ---
yolo_model = None
clip_model = None
clip_processor = None
embeddings_cache = {}

def load_local_embeddings():
    cache = {}
    if not os.path.exists(EMBEDDINGS_DIR):
        logger.warning(f"No embeddings found at {EMBEDDINGS_DIR}")
        return cache
    for barcode in os.listdir(EMBEDDINGS_DIR):
        b_dir = os.path.join(EMBEDDINGS_DIR, barcode)
        if not os.path.isdir(b_dir): continue
        cache[barcode] = []
        for npy_file in os.listdir(b_dir):
            if npy_file.endswith(".npy"):
                arr = np.load(os.path.join(b_dir, npy_file))
                cache[barcode].append(arr)
    return cache

def _get_clip_embedding(pil_img: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output"):
            emb = outputs.pooler_output
        elif hasattr(outputs, "image_embeds"):
            emb = outputs.image_embeds
        else:
            emb = outputs
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

# --- STARTUP MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model, clip_model, clip_processor, embeddings_cache

    logger.info("Loading YOLO...")
    yolo_model = YOLO(YOLO_PATH)

    logger.info("Loading CLIP from local files...")
    clip_model = CLIPModel.from_pretrained(CLIP_PATH)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_PATH)
    clip_model.eval()

    logger.info("Loading Embeddings from disk into RAM...")
    embeddings_cache = load_local_embeddings()
    logger.info(f"Loaded {len(embeddings_cache)} barcodes into memory.")

    await start_log_shipper()

    yield
    logger.info("Shutting down AI Server...")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}

def top2_score(query_emb: np.ndarray, ref_embs: list) -> float:
    scores = [float(np.dot(query_emb, r.T).flatten()[0]) * 100 for r in ref_embs]
    scores.sort(reverse=True)
    if len(scores) == 1:
        return round(scores[0], 2)
    return round((scores[0] + scores[1]) / 2, 2)

# --- DETECTION ENDPOINT ---
@app.post("/detect")
async def detect_bottle(
    barcode: str = Form(...),
    file: UploadFile = File(...),
    machine: dict = Depends(require_machine_key),
):
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_id": machine.get("machine_id"),
        "machine_name": machine.get("machine_name"),
        "barcode": barcode,
        "status": "rejected",
        "score": 0.0,
        "yolo_confidence": 0.0,
        "reason": None,
    }
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        results = yolo_model(image)
        best_box = None
        best_conf = 0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if yolo_model.names[cls_id] in ["bottle", "vase"] and conf > best_conf:
                best_conf = conf
                best_box = box

        if best_box is None:
            event["reason"] = "No plastic bottle detected"
            emit_event(event)
            return {"status": "rejected", "reason": "No plastic bottle detected", "confidence": 0}

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)

        query_emb = _get_clip_embedding(pil_img)

        ref_embs = embeddings_cache.get(barcode)
        if not ref_embs:
            event["reason"] = f"No embeddings for barcode {barcode}"
            emit_event(event)
            return {"status": "rejected", "reason": f"No embeddings for barcode {barcode}"}

        score = top2_score(query_emb, ref_embs)
        decision = "accepted" if score >= SIMILARITY_THRESHOLD else "rejected"

        event["status"] = decision
        event["score"] = score
        event["yolo_confidence"] = round(best_conf, 2)
        emit_event(event)

        return {
            "status": decision,
            "score": score,
            "threshold": SIMILARITY_THRESHOLD,
            "barcode": barcode,
            "yolo_confidence": round(best_conf, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# --- ADD BOTTLE ENDPOINT (admin only) ---
@app.post("/add-bottle")
async def add_bottle(
    barcode: str = Form(...),
    files: List[UploadFile] = File(...),
    _: None = Depends(require_admin_key),
):
    if not files:
        raise HTTPException(status_code=422, detail="At least one image file is required")

    raw_dir = os.path.join(REFERENCE_DIR, barcode)
    crop_dir = os.path.join(CROPPED_DIR, barcode)
    emb_dir = os.path.join(EMBEDDINGS_DIR, barcode)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    images_saved = 0
    images_cropped = 0
    embeddings_generated = 0

    for upload in files:
        image_bytes = await upload.read()
        filename = upload.filename or f"img_{images_saved}.jpg"
        raw_path = os.path.join(raw_dir, filename)

        with open(raw_path, "wb") as f:
            f.write(image_bytes)
        images_saved += 1

        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning(f"add-bottle: could not decode {filename}, skipping")
            continue

        results = yolo_model(image)
        best_box = None
        best_conf = 0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if yolo_model.names[cls_id] in ["bottle", "vase"] and conf > best_conf:
                best_conf = conf
                best_box = box

        if best_box is None:
            logger.warning(f"add-bottle: no bottle detected in {filename}, skipping")
            continue

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        crop_path = os.path.join(crop_dir, filename)
        cv2.imwrite(crop_path, cropped)
        images_cropped += 1

        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)
        emb = _get_clip_embedding(pil_img)
        emb_path = os.path.join(emb_dir, filename + ".npy")
        np.save(emb_path, emb)
        embeddings_generated += 1

    # Hot-reload this barcode's embeddings into the in-memory cache
    new_embs = []
    for npy_file in os.listdir(emb_dir):
        if npy_file.endswith(".npy"):
            new_embs.append(np.load(os.path.join(emb_dir, npy_file)))
    embeddings_cache[barcode] = new_embs
    logger.info(f"add-bottle: barcode={barcode} saved={images_saved} cropped={images_cropped} embeddings={embeddings_generated}")

    return {
        "barcode": barcode,
        "images_saved": images_saved,
        "images_cropped": images_cropped,
        "embeddings_generated": embeddings_generated,
    }
