import json
from pathlib import Path

import cv2

COCO_PATH = Path("data/raw/Endava/sample_synth_datasets/ds5.0/train__kpts_coco.json")
IMAGES_DIR = Path("data/raw/Endava/sample_synth_datasets/ds5.0")

TARGET_FILENAME = "data/v_0992_f_0000_rgba.png"  # change this to check a different image
OUTPUT_PATH = "bbox_check.png"


def main() -> None:
    with COCO_PATH.open() as f:
        coco = json.load(f)

    image_id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    target_id = next((iid for iid, fn in image_id_to_filename.items() if fn == TARGET_FILENAME), None)

    if target_id is None:
        raise ValueError(f"'{TARGET_FILENAME}' not found in COCO images list.")

    ann = next((a for a in coco["annotations"] if a["image_id"] == target_id), None)
    if ann is None:
        raise ValueError(f"No annotation found for image_id {target_id}.")

    bbox = ann["bbox"]
    x, y, w, h = bbox
    print(f"Target: {TARGET_FILENAME}")
    print(f"bbox: {bbox}")

    img_path = IMAGES_DIR / TARGET_FILENAME
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not load {img_path}")

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    pt1 = (int(x), int(y))
    pt2 = (int(x + w), int(y + h))
    cv2.rectangle(img, pt1, pt2, color=(0, 0, 255), thickness=3)

    cv2.imwrite(OUTPUT_PATH, img)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()