"""
Experiment 25 -- Multi-task model evaluation, FULLY end-to-end: calibration
built from the model's OWN predicted min/max landmarks (not ground-truth
scale-label text), closing the gap noted in docs/deep_learning.md.

Run from the repo root with:
    uv run python scripts/experiments/experiment_25_multitask_evaluation.py
"""

import csv
import json
import math

import mlflow
import torch
from PIL import Image

from retroread.annotations import bbox_touches_edge, load_complete_annotations, load_reading_annotations
from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.crop_heatmap_dataset import compute_crop_box
from retroread.crop_heatmap_model import hard_argmax_decode
from retroread.mlflow_setup import configure_tracking
from retroread.multitask_gauge_model import MultiTaskGaugeModel
from retroread.needle_segmentation_dataset import PREPROCESS
from retroread.needle_segmentation_geometry import decode_center, fit_needle_angle
from retroread.reading_conversion import ScaleCalibration, angle_to_value

CHECKPOINT_PATH = "best_multitask_model.pt"
RESULTS_CSV = "experiment_25_results.csv"
TOLERANCE_PCT = 5.0

BASELINE_ACCURACY_RATE = 0.803
BASELINE_MEAN_PCT_ERROR = 4.82


def build_calibration_from_predicted(min_norm, max_norm, center_norm, min_value, max_value) -> ScaleCalibration:
    """Two-point calibration from the model's own predicted min/max landmark positions."""
    min_angle = math.atan2(min_norm[1] - center_norm[1], min_norm[0] - center_norm[0])
    max_angle = math.atan2(max_norm[1] - center_norm[1], max_norm[0] - center_norm[0])
    unwrapped = [min_angle]
    k = round((min_angle - max_angle) / (2 * math.pi))
    unwrapped.append(max_angle + k * 2 * math.pi)
    values = [min_value, max_value]
    slope = (values[1] - values[0]) / (unwrapped[1] - unwrapped[0]) if unwrapped[1] != unwrapped[0] else 0.0
    intercept = values[0] - slope * unwrapped[0]
    return ScaleCalibration(slope=slope, intercept=intercept, reference_unwrapped_angles=unwrapped, reference_values=values)


def main() -> None:
    with ENDAVA_DS5_VAL_KPTS_COCO.open() as f:
        val_filenames = {img["file_name"] for img in json.load(f)["images"]}

    edge_filtered = load_complete_annotations(ENDAVA_DS5_VAL_KPTS_COCO)
    edge_filtered = [c for c in edge_filtered if not bbox_touches_edge(c)]
    fair_filenames = {c["file_name"] for c in edge_filtered} & val_filenames

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    reading_by_filename = {d["file_name"]: d for d in reading_data.values()}
    candidates = [reading_by_filename[fn] for fn in fair_filenames if fn in reading_by_filename]
    print(f"Evaluating on {len(candidates)} images (fully end-to-end: predicted calibration).")

    model = MultiTaskGaugeModel(freeze_backbone=True)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    results = []
    n_accurate = 0
    n_no_mask = 0
    pct_errors = []

    with torch.no_grad():
        for reading in candidates:
            image_path = ENDAVA_DS5_IMAGES_DIR / reading["file_name"]
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size

            from retroread.params import CIRCLE_PARAMS
            circle = find_gauge_circle(str(image_path), **CIRCLE_PARAMS)
            if not circle.found:
                continue
            circle_bbox = [circle.center_x - circle.radius, circle.center_y - circle.radius,
                            circle.radius * 2, circle.radius * 2]
            crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_box(circle_bbox, image_width, image_height)

            cropped = image.crop((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)))
            input_tensor = PREPROCESS(cropped).unsqueeze(0)

            mask_logits, center_logits, landmark_logits = model(input_tensor)
            center_norm = decode_center(center_logits)
            angle = fit_needle_angle(mask_logits, center_norm)

            row = {"file_name": reading["file_name"], "true_value": reading["true_value"], "pct_error": None, "accurate": False}
            if angle is None:
                n_no_mask += 1
                results.append(row)
                continue

            landmark_coords = hard_argmax_decode(torch.sigmoid(landmark_logits))[0].numpy()
            min_norm = (landmark_coords[0], landmark_coords[1])
            max_norm = (landmark_coords[2], landmark_coords[3])

            scale_values = [s["value"] for s in reading["scale_labels"]]
            min_value, max_value = min(scale_values), max(scale_values)
            scale_range = max_value - min_value

            calibration = build_calibration_from_predicted(min_norm, max_norm, center_norm, min_value, max_value)
            predicted = angle_to_value(angle, calibration)
            raw_error = abs(predicted - reading["true_value"])
            pct_error = (raw_error / scale_range) * 100 if scale_range > 0 else None

            row["pct_error"] = round(pct_error, 2) if pct_error is not None else None
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
    mean_pct_error = sum(pct_errors) / len(pct_errors) if pct_errors else None

    configure_tracking()
    mlflow.set_experiment("retroread_deep_learning")
    with mlflow.start_run(run_name="exp25_multitask_evaluation"):
        mlflow.log_metric("accuracy_rate", accuracy_rate)
        if mean_pct_error is not None:
            mlflow.log_metric("mean_pct_error", mean_pct_error)
        mlflow.log_artifact(RESULTS_CSV)

    print(f"No mask found: {n_no_mask}/{n}")
    print(f"{'':25} {'Multitask (E2E)':>16} {'Classical':>12}")
    print(f"{'Accuracy rate':25} {accuracy_rate:>15.1%} {BASELINE_ACCURACY_RATE:>11.1%}")
    if mean_pct_error is not None:
        print(f"{'Mean % error':25} {mean_pct_error:>15.2f}% {BASELINE_MEAN_PCT_ERROR:>10.2f}%")
    print(f"\nFull results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()