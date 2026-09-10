"""
Runs the frozen YOLO-pose model once over all train+val gauge crops,
caching its predicted center/tip (in ORIGINAL IMAGE pixel coordinates) to
JSON. R2's tip refiner trains on these -- YOLO's own imperfect
predictions, not ground truth -- matching the deployment distribution
(the same principle behind Experiment 20's crop-jitter fix).

Run from the repo root with:
    uv run python scripts/data_prep/precompute_yolo_predictions.py
"""

import json

from PIL import Image
from ultralytics import YOLO

from retroread.annotations import load_keypoint_training_data_raw
from retroread.config import ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_TRAIN_KPTS_COCO, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.crop_heatmap_dataset import compute_crop_box

YOLO_CHECKPOINT = "runs/pose/yolo_runs/experiment_30/weights/best.pt"
OUTPUT_PATH = "data/processed/yolo_predictions_cache.json"


def run_split(model, coco_path) -> dict:
    samples = load_keypoint_training_data_raw(coco_path)
    predictions = {}

    for sample in samples:
        image_path = ENDAVA_DS5_IMAGES_DIR / sample["file_name"]
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_box(sample["bbox"], image_width, image_height)
        cropped = image.crop((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)))

        pred = model.predict(cropped, verbose=False)[0]
        if pred.keypoints is None or len(pred.keypoints.xy) == 0:
            continue
        kpts = pred.keypoints.xy[0].tolist()

        predictions[sample["file_name"]] = {
            "center": [kpts[0][0] + crop_x0, kpts[0][1] + crop_y0],
            "tip": [kpts[1][0] + crop_x0, kpts[1][1] + crop_y0],
            "gt_tip": [sample["keypoints_px"][2], sample["keypoints_px"][3]],
        }

    return predictions


def main() -> None:
    model = YOLO(YOLO_CHECKPOINT)

    train_preds = run_split(model, ENDAVA_DS5_TRAIN_KPTS_COCO)
    val_preds = run_split(model, ENDAVA_DS5_VAL_KPTS_COCO)

    print(f"Train: {len(train_preds)}  Val: {len(val_preds)}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"train": train_preds, "val": val_preds}, f)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()