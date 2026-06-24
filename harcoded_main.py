from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from mangum import Mangum
import shutil
import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor
from transformers import CLIPModel

app = FastAPI()

# CORS

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

# YOLO Model

model = YOLO("yolov8n.pt")

clip_model = CLIPModel.from_pretrained(
    "D:\\Zephor\\clip_local"
)

clip_processor = CLIPProcessor.from_pretrained(
    "D:\\Zephor\\clip_local"
)

# Temporary barcode database

PRODUCTS = {
"8902080504060": {
"brand": "aquafina",
"folder": "8902080504060"
},
"8850389100684": {
"brand": "aqua nutrine",
"folder": "8850389100684"
},
"8901491100528": {
"brand": "Coca Cola",
"folder": "8901491100528"
}
}

emb_path = r"embeddings/"

def crop_bottle(image_path, output_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read image: {image_path}")
        return False

    results = model(image_path)

    best_box = None
    best_conf = 0

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[cls_id]

        if class_name in ["bottle", "vase"]:
            if confidence > best_conf:
                best_conf = confidence
                best_box = box

    if best_box is None:
        print("No bottle found")
        return False

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

    cropped = image[y1:y2, x1:x2]

    os.makedirs("cropped_uploads", exist_ok=True)

    cv2.imwrite(output_path, cropped)

    print(f"Cropped image saved: {output_path}")

    return True


def get_clip_embedding(image_path):
    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = clip_model.get_image_features(
            **inputs
        )

        # Extract the raw tensor from the output object safely
        if hasattr(outputs, "pooler_output"):
            embedding = outputs.pooler_output
        elif hasattr(outputs, "image_embeds"):
            embedding = outputs.image_embeds
        else:
            embedding = outputs

    # Now .norm() will work perfectly on the PyTorch tensor
    embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True
    )

    return embedding.cpu().numpy()

def compare_clip_embeddings(
    uploaded_embedding,
    reference_embedding
):

    score = np.dot(
        uploaded_embedding,
        reference_embedding.T
    )[0][0]

    return round(
        float(score) * 100,
        2
    )
# --------------------------------------------------

# GATE 1 : VERIFY BARCODE

# --------------------------------------------------

@app.post("/verify-barcode")
async def verify_barcode(barcode: str = Form(...)):
    product = PRODUCTS.get(barcode)

    if not product:
        raise HTTPException(
            status_code=404,
            detail="Barcode not found in database"
        )

    return {
        "status": "valid",
        "message": f"Barcode verified for {product['brand']}",
        "barcode": barcode,
        "product": product
    }

# --------------------------------------------------

# GATE 2 : VERIFY BOTTLE

# --------------------------------------------------

@app.post("/verify-bottle")
async def verify_bottle(
        barcode: str = Form(...),
        file: UploadFile = File(...)
):
    product = PRODUCTS.get(barcode)

    if not product:
        raise HTTPException(
            status_code=404,
            detail="Invalid barcode"
        )

    # --------------------------------------------------
    # Barcode -> Reference Folder Verification
    # --------------------------------------------------
    reference_folder = os.path.join(
        "cropped_reference_images",
        barcode
    )

    print(f"Looking for folder: {reference_folder}")

    if not os.path.exists(reference_folder):
        return {
            "status": "rejected",
            "message": f"No reference folder found for barcode {barcode}",
            "score": 0
        }

    print("Reference folder found!")
    # --------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------

    os.makedirs("uploads", exist_ok=True)

    filename = file.filename

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must include a filename"
        )

    image_path = os.path.join(
        "uploads",
        filename
    )

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    print(f"Image saved: {image_path}")

    # --------------------------------------------------
    # Crop uploaded bottle
    # --------------------------------------------------

    cropped_path = os.path.join(
        "cropped_uploads",
        filename
    )

    crop_success = crop_bottle(
        image_path,
        cropped_path
    )

    if not crop_success:
        return {
            "status": "rejected",
            "message": "No bottle detected in uploaded image",
            "score": 0,
            "barcode": barcode,
            "brand": product["brand"]
        }

    print("\n===================================")
    print("STEP 4 : CLIP COMPARISON")
    print("===================================")

    embedding_folder = os.path.join(
        "embeddings",
        barcode
    )

    if not os.path.exists(embedding_folder):
        return {
            "status": "rejected",
            "message": "Embedding folder not found"
        }

    uploaded_embedding = get_clip_embedding(
        cropped_path
    )

    scores = []
    comparison_results = []

    for emb_file in os.listdir(
        embedding_folder
    ):

        emb_path = os.path.join(
            embedding_folder,
            emb_file
        )

        reference_embedding = np.load(
            emb_path
        )

        score = compare_clip_embeddings(
            uploaded_embedding,
            reference_embedding
        )

        scores.append(score)

        comparison_results.append({
            "image": emb_file.replace(
                ".npy",
                ""
            ),
            "score": score
        })

        print(
            f"{emb_file} -> {score}%"
        )

    best_score = max(scores)

    top2_score = round(
        sum(
            sorted(
                scores,
                reverse=True
            )[:2]
        ) / 2,
        2
    )

    aggregate_score = round(
        sum(scores) / len(scores),
        2
    )

    threshold = 75

    decision = (
        "accepted"
        if top2_score >= threshold
        else "rejected"
    )

    return {
        "status": decision,
        "barcode": barcode,
        "brand": product["brand"],

        "similarity_engine": "CLIP",

        "best_score": best_score,
        "top2_score": top2_score,
        "aggregate_score": aggregate_score,

        "threshold": threshold,

        "crop_path": cropped_path,

        "comparisons": comparison_results
    }

handler = Mangum(app)