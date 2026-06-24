from ultralytics import YOLO
import cv2
import os

# Load YOLO once
model = YOLO("yolov8n.pt")

REFERENCE_DIR = "reference_images"
OUTPUT_DIR = "cropped_reference_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def crop_bottle(image_path, output_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Could not read: {image_path}")
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
        print(f"❌ No bottle found: {image_path}")
        return False

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

    cropped = image[y1:y2, x1:x2]

    cv2.imwrite(output_path, cropped)

    print(
        f"✅ Cropped: {os.path.basename(image_path)} "
        f"(confidence={best_conf:.2f})"
    )

    return True


def process_all_folders():

    for barcode in os.listdir(REFERENCE_DIR):

        barcode_folder = os.path.join(
            REFERENCE_DIR,
            barcode
        )

        if not os.path.isdir(barcode_folder):
            continue

        output_barcode_folder = os.path.join(
            OUTPUT_DIR,
            barcode
        )

        os.makedirs(
            output_barcode_folder,
            exist_ok=True
        )

        print(f"\n📦 Processing barcode: {barcode}")

        for filename in os.listdir(barcode_folder):

            image_path = os.path.join(
                barcode_folder,
                filename
            )

            output_path = os.path.join(
                output_barcode_folder,
                filename
            )

            crop_bottle(
                image_path,
                output_path
            )


if __name__ == "__main__":
    process_all_folders()