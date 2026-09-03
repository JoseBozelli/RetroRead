"""
Compares predicted vs true center/tip pixel positions for a few validation
images, and checks predicted pixel error against the true needle length --
to see whether position error is large relative to the vector that
determines the angle (the likely cause of the catastrophic reading error).

Run from the repo root with:
    uv run python scripts/exploratory/debug_dl_predictions.py
"""

import json
import math

import torch
from PIL import Image

from retroread.annotations import load_reading_annotations
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.keypoint_dataset import PREPROCESS
from retroread.keypoint_model import GaugeKeypointModel

CHECKPOINT_PATH = "best_model.pt"
N_SAMPLES = 5


def main() -> None:
    with ENDAVA_DS5_VAL_KPTS_COCO.open() as f:
        val_filenames = {img["file_name"] for img in json.load(f)["images"]}

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    by_filename = {d["file_name"]: d for d in reading_data.values()}
    candidates = [by_filename[fn] for fn in val_filenames if fn in by_filename][:N_SAMPLES]

    model = GaugeKeypointModel(freeze_backbone=True)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    with torch.no_grad():
        for data in candidates:
            image_path = ENDAVA_DS5_IMAGES_DIR / data["file_name"]
            image = Image.open(image_path).convert("RGB")
            w, h = image.size

            input_tensor = PREPROCESS(image).unsqueeze(0)
            pred = model(input_tensor)[0].numpy()

            pred_center = (pred[0] * w, pred[1] * h)
            pred_tip = (pred[2] * w, pred[3] * h)
            true_center = (data["center_x"], data["center_y"])
            true_tip = (data["tip_x"], data["tip_y"])

            center_error = math.hypot(pred_center[0] - true_center[0], pred_center[1] - true_center[1])
            tip_error = math.hypot(pred_tip[0] - true_tip[0], pred_tip[1] - true_tip[1])
            true_needle_length = math.hypot(true_tip[0] - true_center[0], true_tip[1] - true_center[1])

            print(f"{data['file_name']}")
            print(f"  true center={true_center}  pred center={tuple(round(v,1) for v in pred_center)}  error={center_error:.1f}px")
            print(f"  true tip={true_tip}  pred tip={tuple(round(v,1) for v in pred_tip)}  error={tip_error:.1f}px")
            print(f"  true needle length: {true_needle_length:.1f}px")
            print()


if __name__ == "__main__":
    main()