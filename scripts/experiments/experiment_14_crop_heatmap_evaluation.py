"""
Experiment 14 -- Deep Learning, crop-based heatmap model evaluation.

Same reading-conversion methodology as Experiments 06/10/12 (ground-truth center for calibration,
model's own predicted center+tip for angle), with predicted crop-relative coordinates converted 
back to full-image pixel space before computing the angle.

Run from the repo root with:
    uv run python scripts/experiments/experiment_14_crop_heatmap_evaluation.py
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
from retroread.crop_heatmap_dataset import PREPROCESS, compute_crop_box
from retroread.crop_heatmap_model import CropHeatmapModel, soft_argmax_decode
from retroread.mlflow_setup import configure_tracking
from retroread.params import CIRCLE_PARAMS
from retroread.reading_conversion import angle_to_value, fit_scale_calibration

CHECKPOINT_PATH = "best_crop_heatmap_model.pt"
RESULTS_CSV = "experiment_14_results.csv"
TOLERANCE_PCT = 5.0

BASELINE_ACCURACY_RATE = 0.803  # Experiment 06b, same val split, fair comparison
BASELINE_MEAN_PCT_ERROR = 4.82

def main() -> None:
    with ENDAVA_DS5_VAL_KPTS_COCO.open() as f:
        val_filenames = {img["file_name"] for img in json.load(f)["images"]}

    # Restrict to the SAME edge-filtered subset Experiment 06b used, for a
    # genuinely fair comparison -- otherwise DL is tested on harder images
    # the classical pipeline explicitly excludes.
    edge_filtered = load_complete_annotations(ENDAVA_DS5_VAL_KPTS_COCO)
    edge_filtered = [c for c in edge_filtered if not bbox_touches_edge(c)]
    matched_filenames = {c["file_name"] for c in edge_filtered} & val_filenames

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    reading_by_filename = {d["file_name"]: d for d in reading_data.values()}

    candidates = [reading_by_filename[fn] for fn in matched_filenames if fn in reading_by_filename]
    print(f"Validation images (same filtered subset as classical baseline): {len(candidates)}")

    n_circle_failed = 0

    model = CropHeatmapModel(freeze_backbone=True)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    results = []
    n_accurate = 0
    pct_errors = []

    with torch.no_grad():
        for reading in candidates:
            image_path = ENDAVA_DS5_IMAGES_DIR / reading["file_name"]
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size

            # Stage 1: classical circle detector locates the gauge -- no
            # ground-truth annotation needed, matching real deployment
            # (including the eventual real-world Aalborg evaluation).
            circle = find_gauge_circle(str(image_path), **CIRCLE_PARAMS)
            if not circle.found:
                n_circle_failed += 1
                continue

            # Convert the detected circle into an equivalent bbox, then
            # apply the same crop margin used everywhere else.
            circle_bbox = [
                circle.center_x - circle.radius,
                circle.center_y - circle.radius,
                circle.radius * 2,
                circle.radius * 2,
            ]
            crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_box(circle_bbox, image_width, image_height)
            crop_w, crop_h = crop_x1 - crop_x0, crop_y1 - crop_y0

            cropped = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            input_tensor = PREPROCESS(cropped).unsqueeze(0)

            pred_heatmaps = model(input_tensor)
            coords = soft_argmax_decode(pred_heatmaps)[0].numpy()   # normalized WITHIN crop

            # Convert back to full-image pixel coordinates.
            pred_center_x = crop_x0 + coords[0] * crop_w
            pred_center_y = crop_y0 + coords[1] * crop_h
            pred_tip_x = crop_x0 + coords[2] * crop_w
            pred_tip_y = crop_y0 + coords[3] * crop_h

            predicted_angle = math.atan2(pred_tip_y - pred_center_y, pred_tip_x - pred_center_x)

            calibration = fit_scale_calibration(reading["center_x"], reading["center_y"], reading["scale_labels"])
            scale_values = [s["value"] for s in reading["scale_labels"]]
            scale_range = max(scale_values) - min(scale_values)

            predicted_reading = angle_to_value(predicted_angle, calibration)
            raw_error = abs(predicted_reading - reading["true_value"])
            pct_error = (raw_error / scale_range) * 100 if scale_range > 0 else None

            row = {
                "file_name": reading["file_name"],
                "true_value": reading["true_value"],
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
    n_evaluated = n - n_circle_failed
    accuracy_rate = n_accurate / n_evaluated if n_evaluated else 0
    mean_pct_error = sum(pct_errors) / len(pct_errors) if pct_errors else None
    print(f"\nCircle detection failed on {n_circle_failed}/{n} images (excluded from accuracy).")

    configure_tracking()
    mlflow.set_experiment("retroread_deep_learning")
    with mlflow.start_run(run_name="exp14_crop_heatmap_evaluation"):
        mlflow.log_param("checkpoint", CHECKPOINT_PATH)
        mlflow.log_metric("accuracy_rate", accuracy_rate)
        mlflow.log_metric("mean_pct_error", mean_pct_error)
        mlflow.log_artifact(RESULTS_CSV)

    print(f"\n{'':20} {'Crop Heatmap DL':>16} {'Classical Baseline':>20}")
    print(f"{'Accuracy rate':20} {accuracy_rate:>15.1%} {BASELINE_ACCURACY_RATE:>19.1%}")
    print(f"{'Mean % error':20} {mean_pct_error:>15.2f} {BASELINE_MEAN_PCT_ERROR:>18.2f}%")
    print(f"\nFull results: {RESULTS_CSV}")

if __name__ == "__main__":
    main()