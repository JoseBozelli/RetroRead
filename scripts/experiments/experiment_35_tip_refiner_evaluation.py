"""
Experiment 35 -- R2 evaluation: YOLO's own center (unchanged) + refiner's
tip, full downstream reading pipeline, same 61-image fair subset.

Run from the repo root with:
    uv run python scripts/experiments/experiment_35_tip_refiner_evaluation.py
"""

import csv
import json
import math

import mlflow
import torch
from PIL import Image
from ultralytics import YOLO

from retroread.annotations import bbox_touches_edge, load_complete_annotations, load_reading_annotations
from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.crop_heatmap_dataset import compute_crop_box
from retroread.mlflow_setup import configure_tracking
from retroread.params import CIRCLE_PARAMS
from retroread.reading_conversion import angle_to_value, fit_scale_calibration
from retroread.tip_refiner_dataset import HEATMAP_SIZE, MODEL_INPUT_SIZE, ROI_SIZE, PREPROCESS
from retroread.tip_refiner_model import TipRefinerModel

YOLO_CHECKPOINT = "runs/pose/yolo_runs/experiment_30/weights/best.pt"
REFINER_CHECKPOINT = "best_tip_refiner_model.pt"
RESULTS_CSV = "experiment_35_results.csv"
TOLERANCE_PCT = 5.0

BASELINE_ACCURACY_RATE = 0.803
BASELINE_MEAN_PCT_ERROR = 4.82


def refine_tip(refiner_model, image, yolo_tip):
    half = ROI_SIZE // 2
    roi_x0 = yolo_tip[0] - half
    roi_y0 = yolo_tip[1] - half
    roi = image.crop((int(roi_x0), int(roi_y0), int(roi_x0) + ROI_SIZE, int(roi_y0) + ROI_SIZE))
    input_tensor = PREPROCESS(roi).unsqueeze(0)

    with torch.no_grad():
        heatmap = torch.sigmoid(refiner_model(input_tensor))[0, 0]
    idx = torch.argmax(heatmap.flatten()).item()
    hy, hx = idx // HEATMAP_SIZE, idx % HEATMAP_SIZE
    norm_x, norm_y = hx / (HEATMAP_SIZE - 1), hy / (HEATMAP_SIZE - 1)

    refined_x = roi_x0 + norm_x * ROI_SIZE
    refined_y = roi_y0 + norm_y * ROI_SIZE
    return refined_x, refined_y


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

    yolo_model = YOLO(YOLO_CHECKPOINT)
    refiner_model = TipRefinerModel(freeze_backbone=True)
    refiner_model.load_state_dict(torch.load(REFINER_CHECKPOINT))
    refiner_model.eval()

    results, n_accurate, pct_errors = [], 0, []

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

        pred = yolo_model.predict(cropped, verbose=False)[0]
        if pred.keypoints is None or len(pred.keypoints.xy) == 0:
            continue
        kpts = pred.keypoints.xy[0].tolist()
        yolo_center = (kpts[0][0] + crop_x0, kpts[0][1] + crop_y0)
        yolo_tip = (kpts[1][0] + crop_x0, kpts[1][1] + crop_y0)

        refined_tip = refine_tip(refiner_model, image, yolo_tip)

        calibration = fit_scale_calibration(reading["center_x"], reading["center_y"], reading["scale_labels"])
        scale_values = [s["value"] for s in reading["scale_labels"]]
        scale_range = max(scale_values) - min(scale_values)

        angle = math.atan2(refined_tip[1] - yolo_center[1], refined_tip[0] - yolo_center[0])
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
    with mlflow.start_run(run_name="exp35_tip_refiner_evaluation"):
        mlflow.log_metric("accuracy_rate", accuracy_rate)
        if mean_pct_error is not None:
            mlflow.log_metric("mean_pct_error", mean_pct_error)
        mlflow.log_artifact(RESULTS_CSV)

    print(f"{'':25} {'YOLO + Refiner':>16} {'Classical':>12}")
    print(f"{'Accuracy rate':25} {accuracy_rate:>15.1%} {BASELINE_ACCURACY_RATE:>11.1%}")
    if mean_pct_error is not None:
        print(f"{'Mean % error':25} {mean_pct_error:>15.2f}% {BASELINE_MEAN_PCT_ERROR:>10.2f}%")
    print(f"\nFull results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()