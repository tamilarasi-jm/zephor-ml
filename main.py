cat > /home/ubuntu/zephor/main.py << 'EOF'
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from ultralytics import YOLO
import shutil
import os
import io
import cv2
import numpy as np
import torch
import boto3
import uuid
import logging
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Config
# --------------------------------------------------
S3_BUCKET_EMBEDDINGS = os.getenv("S3_BUCKET_EMBEDDINGS", "zephor-embeddings")
S3_BUCKET_ARCHIVE    = os.getenv("S3_BUCKET_ARCHIVE", "zephor-archive")
AWS_REGION           = os.getenv("AWS_REGION", "ap-south-1")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "75"))
CLIP_MODEL_NAME      = "openai/clip-vit-b-32"

# --------------------------------------------------
# Machine API Keys (loaded from env)
# --------------------------------------------------
MACHINE_KEYS = {
    os.getenv("MACHINE_1_ID", "ZEP-SCH001-M01"): os.getenv("MACHINE_1_KEY", "changeme1"),
    os.getenv("MACHINE_2_ID", "ZEP-SCH001-M02"): os.getenv("MACHINE_2_KEY", "changeme2"),
}

# --------------------------------------------------
# Products
# --------------------------------------------------
PRODUCTS = {
    "8902080504060": {"brand": "Aquafina"},
    "8850389100684": {"brand": "Aqua Nutrine"},
    "8901491100528": {"brand": "Coca Cola"},
}

# --------------------------------------------------
# Global model holders
# --------------------------------------------------
yolo_model    = None
clip_model    = None
clip_processor = None
embeddings_cache = {}  # { barcode: [np.ndarray, ...] }
s3_client     = None

# --------------------------------------------------
# Startup / Shutdown
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model, clip_model, clip_processor, embeddings_cache, s3_client

    logger.info("Loading YOLO...")
    yolo_model = YOLO("yolov8n.pt")

    logger.info("Loading CLIP...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model.eval()

    logger.info("Connecting to S3...")
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    logger.info("Loading embeddings from S3 into RAM...")
    embeddings_cache = load_embeddings_from_s3()
    logger.info(f"Loaded {len(embeddings_cache)} barcodes into cache")

    yield
    logger.info("Shutting down")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# S3 Embedding Loader
# --------------------------------------------------
def load_embeddings_from_s3() -> dict:
    cache = {}
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET_EMBEDDINGS, Prefix="embeddings/"):
            for obj in page.get("Contents", []):
                parts = obj["Key"].split("/")
                if len(parts) != 3:
                    continue
                barcode  = parts[1]
                body     = s3_client.get_object(Bucket=S3_BUCKET_EMBEDDINGS, Key=obj["Key"])["Body"].read()
                arr      = np.load(io.BytesIO(body))
                cache.setdefault(barcode, []).append(arr)
    except Exception as e:
        logger.error(f"S3 embedding load failed: {e}")
    return cache

# --------------------------------------------------
# Auth
# --------------------------------------------------
def authenticate_machine(request: Request):
    machine_id = request.headers.get("X-Machine-ID")
    api_key    = request.headers.get("X-API-Key")
    if not machine_id or not api_key:
        raise HTTPException(status_code=401, detail="Missing machine credentials")
    expected = MACHINE_KEYS.get(machine_id)
    if not expected or not (api_key == expected):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return machine_id

# --------------------------------------------------
# ML Helpers
# --------------------------------------------------
def crop_bottle_from_bytes(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    results  = yolo_model(image)
    best_box = None
    best_conf = 0
    for box in results[0].boxes:
        cls_id     = int(box.cls[0])
        confidence = float(box.conf[0])
        if yolo_model.names[cls_id] in ["bottle", "vase"] and confidence > best_conf:
            best_conf = confidence
            best_box  = box
    if best_box is None:
        return None
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    cropped = image[y1:y2, x1:x2]
    _, buffer = cv2.imencode(".jpg", cropped)
    return buffer.tobytes()

def get_clip_embedding(image_bytes: bytes) -> np.ndarray:
    image  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
        embedding = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()

def top2_score(query_emb: np.ndarray, ref_embs: list) -> float:
    scores = [float(np.dot(query_emb, r.T).flatten()[0]) * 100 for r in ref_embs]
    scores.sort(reverse=True)
    if len(scores) == 1:
        return round(scores[0], 2)
    return round((scores[0] + scores[1]) / 2, 2)

# --------------------------------------------------
# S3 Archival (runs in background)
# --------------------------------------------------
def archive_to_s3(
    original_bytes: bytes,
    cropped_bytes: bytes,
    barcode: str,
    decision: str,
    machine_id: str,
    request_id: str
):
    try:
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        base_key    = f"{decision}/{date_prefix}/{machine_id}/{barcode}/{request_id}"
        s3_client.put_object(
            Bucket=S3_BUCKET_ARCHIVE,
            Key=f"{base_key}/original.jpg",
            Body=original_bytes,
            ContentType="image/jpeg"
        )
        s3_client.put_object(
            Bucket=S3_BUCKET_ARCHIVE,
            Key=f"{base_key}/cropped.jpg",
            Body=cropped_bytes,
            ContentType="image/jpeg"
        )
        logger.info(f"Archived {request_id} to S3 under {base_key}")
    except Exception as e:
        logger.error(f"S3 archive failed for {request_id}: {e}")

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "barcodes_loaded": len(embeddings_cache),
        "clip": clip_model is not None,
        "yolo": yolo_model is not None,
    }

@app.post("/verify-barcode")
async def verify_barcode(request: Request, barcode: str = Form(...)):
    machine_id = authenticate_machine(request)
    product    = PRODUCTS.get(barcode)
    if not product:
        raise HTTPException(status_code=404, detail="Barcode not found")
    return {"status": "valid", "brand": product["brand"], "barcode": barcode}

@app.post("/verify-bottle")
async def verify_bottle(
    request: Request,
    background_tasks: BackgroundTasks,
    barcode: str = Form(...),
    file: UploadFile = File(...),
):
    machine_id = authenticate_machine(request)
    request_id = str(uuid.uuid4())

    product = PRODUCTS.get(barcode)
    if not product:
        raise HTTPException(status_code=404, detail="Invalid barcode")

    ref_embs = embeddings_cache.get(barcode)
    if not ref_embs:
        return {"status": "rejected", "reason": "No reference embeddings found", "request_id": request_id}

    # Read image into memory only — no disk writes during inference
    if file.size and file.size > 2_000_000:
        raise HTTPException(status_code=413, detail="Image too large (max 2MB)")

    original_bytes = await file.read()

    # Crop
    cropped_bytes = crop_bottle_from_bytes(original_bytes)
    if cropped_bytes is None:
        background_tasks.add_task(
            archive_to_s3, original_bytes, original_bytes,
            barcode, "rejected", machine_id, request_id
        )
        return {"status": "rejected", "reason": "No bottle detected", "request_id": request_id}

    # Embed + compare
    query_emb = get_clip_embedding(cropped_bytes)
    score     = top2_score(query_emb, ref_embs)
    decision  = "accepted" if score >= SIMILARITY_THRESHOLD else "rejected"

    # Archive in background — does not block response
    background_tasks.add_task(
        archive_to_s3, original_bytes, cropped_bytes,
        barcode, decision, machine_id, request_id
    )

    logger.info(f"[{request_id}] machine={machine_id} barcode={barcode} score={score} decision={decision}")

    return {
        "status": decision,
        "barcode": barcode,
        "brand": product["brand"],
        "top2_score": score,
        "threshold": SIMILARITY_THRESHOLD,
        "request_id": request_id,
    }
EOF