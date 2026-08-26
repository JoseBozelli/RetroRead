"""
Experiment 04 -- Classical CV Baseline, batch needle detection.

Hypothesis: needle detection (reach-based selection, fixed after Experiment 03) generalizes across the
candidate set, evaluated against real ground-truth needle-tip position (COCO dial_tip keypoint) -- not an 
approximation like the bbox-center proxy used for circle detection.

Run from the repo root with:
    uv run python scripts/experiments/experiment_04_batch_needle_detection.py
"""

import csv
import math
from pathlib import Path

import mlflow

from retroread.annotations import bbox_touches_edge, load_complete_annotations
from retroread.classical_baseline import find_gauge_circle
from retroread.needle_detection import detect_needle
from retroread.mlflow_setup import configure_tracking
from retroread.config import ENDAVA_DS5_TRAIN_KPTS_COCO as COCO_PATH, ENDAVA_DS5_IMAGES_DIR as IMAGES_DIR

RESULTS_CSV = Path("experiment_04_results.csv")

CIRCLE_PARAMS = {
    "dp": 1.0,
    "min_dist_fraction": 0.5,
    "param1": 100,
    "param2": 50,
    "min_radius_fraction": 0.1,
    "max_radius_fraction": 0.35
}

NEEDLE_PARAMS = {
    "inner_radius_fraction": 0.85,
    "pivot_distance_fraction": 0.25,
    "min_line_length_fraction": 0.3,
    "canny_low": 50,
    "canny_high": 150,
    "hough_threshold": 30,
    "max_line_gap": 10
}

# A detection counts as "accurate" if the angle error is within this many degrees.
ACCURACY_THRESHOLD_DEGREES = 5.0

def angle_diff_degrees(a: float, b: float) -> float:
    """Smallest absolute angular diffference between two angles (radians), in degrees."""
    diff = (a - b + math.pi)%(2*math.pi)- math.pi
    return abs(math.degrees(diff))

def main() -> None:
    all_candidates = load_complete_annotations(COCO_PATH)
    candidates = [c for c in all_candidates if not bbox_touches_edge(c)]
    print(f"Candidates with complete keypoints and full frame: {len(candidates)}")

    results = []
    n_needle_found = 0
    n_accurate = 0
    angle_errors = []
    tip_errors = []

    for candidate in candidates:
        image_path = IMAGES_DIR / candidate["file_name"]
        circle = find_gauge_circle(str(image_path), **CIRCLE_PARAMS)
        needle = detect_needle(str(image_path), circle, **NEEDLE_PARAMS)

        gt_angle = math.atan2(
            candidate["gt_tip_y"] - candidate["gt_center_y"],
            candidate["gt_tip_x"] - candidate["gt_center_x"]
        )

        row = {
            "file_name": candidate["file_name"],
            "circle_found": circle.found,
            "needle_found": needle.found,
            "tip_x": needle.tip_x,
            "tip_y": needle.tip_y,
            "gt_tip_x":candidate["gt_tip_x"],
            "gt_tip_y": candidate["gt_tip_y"],
            "angle_error_deg": None,
            "tip_error_px": None,
            "accurate": False
        }

        if needle.found:
            n_needle_found += 1
            angle_error = angle_diff_degrees(needle.angle_rad, gt_angle)
            tip_error = math.hypot(needle.tip_x - candidate["gt_tip_x"], needle.tip_y - candidate["gt_tip_y"])

            row["angle_error_deg"] = round(angle_error, 2)
            row["tip_error_px"] = round(tip_error, 1)
            angle_errors.append(angle_error)
            tip_errors.append(tip_error)

            if angle_error <= ACCURACY_THRESHOLD_DEGREES:
                row["accurate"] = True
                n_accurate +=1

        results.append(row)

    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    n = len(candidates)
    needle_detection_rate = n_needle_found / n
    accuracy_rate = n_accurate / n
    mean_angle_error = sum(angle_errors) / len(angle_errors) if angle_errors else None
    mean_tip_error = sum(tip_errors) / len(tip_errors) if tip_errors else None

    configure_tracking()
    mlflow.set_experiment("retroread_classical_baseline")
    with mlflow.start_run(run_name="exp04_batch_needle_detection"):
        mlflow.log_params(CIRCLE_PARAMS)
        mlflow.log_params(NEEDLE_PARAMS)
        mlflow.log_param("n_candidates", n)
        mlflow.log_param("accuracy_threshold_degrees", ACCURACY_THRESHOLD_DEGREES)

        mlflow.log_metric("needle_detection_rate", needle_detection_rate)
        mlflow.log_metric("accuracy_rate", accuracy_rate)
        if mean_angle_error is not None:
            mlflow.log_metric("mean_angle_error_deg", mean_angle_error)
            mlflow.log_metric("mean_tip_error_px", mean_tip_error)

        mlflow.log_artifact(str(RESULTS_CSV))

    print(f"\nNeedle detection rate: {n_needle_found}/{n} ({needle_detection_rate:.1%})")
    print(f"Accuracy rate (angle error <= {ACCURACY_THRESHOLD_DEGREES}deg): {n_accurate} / {n} ({accuracy_rate:.1%})")
    if mean_angle_error is not None:
        print(f"Mean angle error: {mean_angle_error:.2f} degrees")
        print(f"Mean tip position error: {mean_tip_error:.1f}px")
    print(f"Full results: {RESULTS_CSV}")

if __name__ == "__main__":
    main()