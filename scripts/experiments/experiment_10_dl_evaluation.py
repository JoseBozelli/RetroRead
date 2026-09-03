"""
Experiment 10 --  Deep Learning, evaluation on comparable metrics.

Converts the trained model's predicted keypoints into an actual gauge reading (same readin-conversion pipeline
as the classical baseline) and reports angle error, % error, and accuracy rate on the validation set  -- directly
comparable to Experiment 06's classical baseline numbers.

Methodology note: calibration uses ground-truth dial center (matching Experiment 06's convention); needle angle uses
the MODEL's own predicted center and tip (matching how the classical baseline used its own detected circle center,
not ground truth, for the needle angle). This keeps the comparsion consistent with the already-reported classical
baseline numbers.

Run from the repo root with:
    uv run python scripts/experiments/experiment_10_dl_evaluation.py
"""

import csv
import json
import math

import mlflow
import torch
from PIL import Image

from retroread.annotations import load_reading_annotations
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.keypoint_dataset import PREPROCESS
from retroread.keypoint_model import GaugeKeypointModel
from retroread.mlflow_setup import configure_tracking
from retroread.reading_conversion import angle_to_value, fit_scale_calibration

CHECKPOINT_PATH = "best_model.pt"
RESULTS_CSV = "experiment_10_results.csv"
TOLERANCE_PCT = 5.0 # same threshold as Experiment 06, for direct comparability

# Classical baseline numbers (Experiment 06), for side-by-side printing.
BASELINE_ACCURACY_RATE = 0.842
BASELINE_MEAN_PCT_ERROR = 4.26

def get_val_filenames() -> set:
    with ENDAVA_DS5_VAL_KPTS_COCO.open() as f:
        coco = json.load(f)
    return {img["file_name"] for img in coco["images"]}

def main() -> None:
    val_filenames = get_val_filenames()

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    by_filename = {d["file_name"]: d for d in reading_data.values()}

    candidates = [by_filename[fn] for fn in val_filenames if fn in by_filename]
    print(f"Validation images with complete reading annotations: {len(candidates)}")

    model = GaugeKeypointModel(freeze_backbone=True)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    results = []
    n_accurate = 0
    pct_errors = []

    with torch.no_grad():
        for data in candidates:
            image_path = ENDAVA_DS5_IMAGES_DIR / data["file_name"]
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size

            input_tensor = PREPROCESS(image).unsqueeze(0)
            predicton = model(input_tensor)[0].numpy()

            # prediction order: [cx, cy, tx, ty, minx, miny, maxx, maxy], normalized [0,1]
            pred_center_x = predicton[0] * image_width
            pred_center_y = predicton[1] * image_height
            pred_tip_x = predicton[2] * image_width
            pred_tip_y = predicton[3] * image_height

            predicted_angle = math.atan2(pred_tip_y - pred_center_y, pred_tip_x - pred_center_x)

            calibration = fit_scale_calibration(data["center_x"], data["center_y"], data["scale_labels"])
            scale_values = [s["value"] for s in data["scale_labels"]]
            scale_range = max(scale_values) - min(scale_values)

            predicted_reading = angle_to_value(predicted_angle, calibration)
            raw_error = abs(predicted_reading - data["true_value"])
            pct_error = (raw_error / scale_range) * 100 if scale_range > 0 else None

            row = {
                "file_name": data["file_name"],
                "true_value": data["true_value"],
                "predicted_value": round(predicted_reading, 3),
                "raw_error": round(raw_error, 3),
                "pct_error": round(pct_error, 2) if pct_error is not None else None,
                "accurate": False
            }

            if pct_error is not None:
                pct_errors.append(pct_error)
                if pct_error <= TOLERANCE_PCT:
                    row["accurate"] = True
                    n_accurate += 1

            results.append(row)

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    n = len(candidates)
    accuracy_rate = n_accurate / n
    mean_pct_error = sum(pct_errors) / len(pct_errors)

    configure_tracking()
    mlflow.set_experiment("retroread_deep_learning")
    with mlflow.start_run(run_name="exp10_dl_evaluation"):
        mlflow.log_param("checkpoint", CHECKPOINT_PATH)
        mlflow.log_param("n_val_candidates", n)
        mlflow.log_param("tolerance_pct", TOLERANCE_PCT)
        mlflow.log_metric("accuracy_rate", accuracy_rate)
        mlflow.log_metric("mean_pct_error", mean_pct_error)
        mlflow.log_artifact(RESULTS_CSV)

    print(f"\n{'':20} {'DL Model':>12} {'Classcial Baseline':>20}")
    print(f"{'Accuracy rate':20} {accuracy_rate:>11.1%} {BASELINE_ACCURACY_RATE:>19.1%}")
    print(f"{'Mean % error':20} {mean_pct_error:>11.2f}% {BASELINE_MEAN_PCT_ERROR:>18.2f}%")
    print(f"\nFull results: {RESULTS_CSV}")

if __name__ == "__main__":
    main()

