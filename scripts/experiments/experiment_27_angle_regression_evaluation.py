"""
Experiment 27 -- sin/cos angle regression evaluation. Ground-truth
calibration (isolates angle-prediction quality specifically, matching the
convention used for every keypoint-angle DL evaluation in this project).

Run from the repo root with:
    uv run python scripts/experiments/experiment_27_angle_regression_evaluation.py
"""

import csv
import json
import math

import mlflow
import torch
from PIL import Image

from retroread.angle_regression_dataset import PREPROCESS
from retroread.angle_regression_model import AngleRegressionModel
from retroread.annotations import bbox_touches_edge, load_complete_annotations, load_reading_annotations
from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.crop_heatmap_dataset import compute_crop_box
from retroread.mlflow_setup import configure_tracking
from retroread.params import CIRCLE_PARAMS
from retroread.reading_conversion import angle_to_value, fit_scale_calibration

CHECKPOINT_PATH = "best_angle_regression_model.pt"
RESULTS_CSV = "experiment_27_results.csv"
TOLERANCE_PCT = 5.0

BASELINE_ACCURACY_RATE = 0.803
BASELINE_MEAN_PCT_ERROR = 4.82


def main() -> None:
    with ENDAVA_DS5_VAL_KPTS_COCO.open() as f:
        val_filenames = {img["file_name"] for img in json.load(f)["images"]}

    edge_filtered = load_complete_annotations(ENDAVA_DS5_VAL_KPTS_COCO)
    edge_filtered = [c for c in edge_filtered if not bbox_touches_edge(c)]
    fair_filenames = {c["file_name"] for c in edge_filtered} & val_filenames

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    reading_by_filename = {d["file_name"]: d for d in reading_data.values()}
    candidates = [reading_by_filename[fn] for fn in fair_filenames if fn in reading_by_filename]
    print(f"Evaluating on {len(candidates)} images.")

    model = AngleRegressionModel(freeze_backbone=True)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    results, n_accurate, pct_errors = [], 0, []

    with torch.no_grad():
        for reading in candidates:
            image_path = ENDAVA_DS5_IMAGES_DIR / reading["file_name"]
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size

            circle = find_gauge_circle(str(image_path), **CIRCLE_PARAMS)
            if not circle.found:
                continue
            circle_bbox = [circle.center_x - circle.radius, circle.center_y - circle.radius,
                            circle.radius * 2, circle.radius * 2]
            crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_box(circle_bbox, image_width, image_height)
            cropped = image.crop((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)))
            input_tensor = PREPROCESS(cropped).unsqueeze(0)

            pred = model(input_tensor)[0]
            angle = math.atan2(pred[0].item(), pred[1].item())

            calibration = fit_scale_calibration(reading["center_x"], reading["center_y"], reading["scale_labels"])
            scale_values = [s["value"] for s in reading["scale_labels"]]
            scale_range = max(scale_values) - min(scale_values)

            predicted = angle_to_value(angle, calibration)
            raw_error = abs(predicted - reading["true_value"])
            pct_error = (raw_error / scale_range) * 100 if scale_range > 0 else None

            row = {"file_name": reading["file_name"], "true_value": reading["true_value"],
                   "pct_error": round(pct_error, 2) if pct_error is not None else None, "accurate": False}
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

    n = len(results)
    accuracy_rate = n_accurate / n
    mean_pct_error = sum(pct_errors) / len(pct_errors) if pct_errors else None

    configure_tracking()
    mlflow.set_experiment("retroread_deep_learning")
    with mlflow.start_run(run_name="exp27_angle_regression_evaluation"):
        mlflow.log_metric("accuracy_rate", accuracy_rate)
        if mean_pct_error is not None:
            mlflow.log_metric("mean_pct_error", mean_pct_error)
        mlflow.log_artifact(RESULTS_CSV)

    print(f"{'':20} {'Angle Regr.':>12} {'Classical':>12}")
    print(f"{'Accuracy rate':20} {accuracy_rate:>11.1%} {BASELINE_ACCURACY_RATE:>11.1%}")
    if mean_pct_error is not None:
        print(f"{'Mean % error':20} {mean_pct_error:>11.2f}% {BASELINE_MEAN_PCT_ERROR:>10.2f}%")
    print(f"\nFull results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()