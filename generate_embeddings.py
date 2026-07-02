import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

CLIP_PATH = "/home/ubuntu/zephor-ml/clip_local"

print("Loading CLIP model from local files...")
clip_model = CLIPModel.from_pretrained(CLIP_PATH)
clip_processor = CLIPProcessor.from_pretrained(CLIP_PATH)
clip_model.eval()
print("CLIP loaded!")

REFERENCE_ROOT = "cropped_reference_images"
EMBEDDING_ROOT = "embeddings"
os.makedirs(EMBEDDING_ROOT, exist_ok=True)

def get_clip_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output"):
            embedding = outputs.pooler_output
        elif hasattr(outputs, "image_embeds"):
            embedding = outputs.image_embeds
        else:
            embedding = outputs
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()

def generate_embeddings():
    print("\n====================================")
    print("GENERATING REFERENCE EMBEDDINGS")
    print("====================================\n")

    if not os.path.exists(REFERENCE_ROOT):
        print(f"Folder not found: {REFERENCE_ROOT}")
        return

    for barcode in os.listdir(REFERENCE_ROOT):
        barcode_folder = os.path.join(REFERENCE_ROOT, barcode)
        if not os.path.isdir(barcode_folder):
            continue
        print(f"\nBarcode: {barcode}")
        embedding_folder = os.path.join(EMBEDDING_ROOT, barcode)
        os.makedirs(embedding_folder, exist_ok=True)

        for image_name in os.listdir(barcode_folder):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_path = os.path.join(barcode_folder, image_name)
            try:
                embedding = get_clip_embedding(image_path)
                save_path = os.path.join(embedding_folder, image_name + ".npy")
                np.save(save_path, embedding)
                print(f"  Saved -> {save_path}")
            except Exception as e:
                print(f"  Failed: {image_name} -> {e}")

    print("\n====================================")
    print("EMBEDDING GENERATION COMPLETE")
    print("====================================")

if __name__ == "__main__":
    generate_embeddings()
