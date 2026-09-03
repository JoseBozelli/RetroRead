"""
Experiment 12 -- Deep Learning, heatmap model evaluation on comparable metrics.

Same evaluation methodology as Experiment 10 (ground-truth center for calibration, model's own
predicted center+tip for angle), applied to the heatmap-based model's decoded predictions.

Run from the repo root with:
    uv run python scripts/experiments/experiment_12_heatmap_evaluation.py
"""

import csv
import json
import math

import mlflow
import torch

from retroread.annotations import load_reading_annotations
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.heatmap_dataset import PREPROCESS
from retroread.heatmap_model import GaugeHeatmapModel, decode_heatmap
from retroread.mlflow_setup import configure_tracking
from retroread.reading_conversion import angle_to_value, fit_scale_calibration
from PIL import Image

CHECKPOINT_PATH = "best_heatmap_model.pt"
RESULTS_CSV = "experiment_12_results.csv"
TOLERANCE_PCT = 5.0

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

    model = GaugeHeatmapModel(freeze_backbone=True)
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
            heatmaps = model(input_tensor)
            coords = decode_heatmap(heatmaps)[0].numpy()    # [cx, cy, tx, ty, minx, miny, maxx, maxy] normalized

            pred_center_x = coords[0] * image_width
            pred_center_y = coords[1] * image_height
            pred_tip_x = coords[2] * image_width
            pred_tip_y = coords[3] * image_height

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
    with mlflow.start_run(run_name="exp12_heatmap_evaluation"):
        mlflow.log_param("checkpoint", CHECKPOINT_PATH)
        mlflow.log_metric("accuracy_rate", accuracy_rate)
        mlflow.log_metric("mean_pct_error", mean_pct_error)
        mlflow.log_artifact(RESULTS_CSV)

    print(f"\n{'':20} {'Heatmap DL':>12} {'Classical Baseline':>20}")
    print(f"{'Accuracy rate':20} {accuracy_rate:>11.1%} {BASELINE_ACCURACY_RATE:>19.1%}")
    print(f"{'Mean % error':20} {mean_pct_error:>11.2f}% {BASELINE_MEAN_PCT_ERROR:>18.2f}%")
    print(f"\nFull results: {RESULTS_CSV}")

if __name__ == "__main__":
    main()