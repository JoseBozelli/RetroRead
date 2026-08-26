"""
Experiment 02 -- Classical CV Baseline, tighter max radius.

Hypothesis: Experiment 01's 15 inaccurate detections were caused by max_radius_fraction=0.6 being permissive enough
to let large background structures (roof trusses, pipe loops) outscore the actual gauge circle.
Tightening max_radius_fraction to 0.35 should resolve these cases, since the gauge occupies well under half the frame
in every sample reviewed.

One parameter changed from Experiment 01: max_radius_fraction (0.6 -> 0.35).
Everything else held constant for a clean comparsion.

Run from the repo root with:
    uv run python scripts/experiments/experiment_02_tighter_radius.py
"""

import csv
from pathlib import Path

import mlflow

from retroread.annotations import bbox_touches_edge, load_complete_annotations
from retroread.classical_baseline import find_gauge_circle
from retroread.mlflow_setup import configure_tracking
from retroread.config import ENDAVA_DS5_TRAIN_KPTS_COCO as COCO_PATH, ENDAVA_DS5_IMAGES_DIR as IMAGES_DIR

RESULTS_CSV = Path("experiment_02_results.csv")

HOUGH_PARAMS = {
    "dp": 1.0,
    "min_dist_fraction": 0.5,
    "param1": 100,
    "param2": 50,
    "min_radius_fraction": 0.1,
    "max_radius_fraction": 0.35     # changed from 0.6 in experiment 01
}

ACCURACY_THRESHOLD_FRACTION = 0.2

def main() -> None:
    all_candidates = load_complete_annotations(COCO_PATH)
    candidates = [c for c in all_candidates if not bbox_touches_edge(c)]
    print(f"Candidates with complete keypoints and full frame: {len(candidates)}")

    results = []
    n_found = 0
    n_accurate = 0
    center_errors = []

    for candidate in candidates:
        image_path = IMAGES_DIR / candidate["file_name"]
        detection = find_gauge_circle(str(image_path), **HOUGH_PARAMS)

        x, y, w, h = candidate["bbox"]
        bbox_center_x = x + w / 2
        bbox_center_y = y + h / 2

        row = {
            "file_name": candidate["file_name"],
            "found": detection.found,
            "center_x": detection.center_x,
            "center_y": detection.center_y,
            "radius": detection.radius,
            "bbox_center_x": bbox_center_x,
            "bbox_center_y": bbox_center_y,
            "center_error_px": None,
            "accurate": False
        }

        if detection.found:
            n_found += 1
            error = ((detection.center_x - bbox_center_x)**2 + (detection.center_y - bbox_center_y)**2)**0.5
            row["center_error_px"] = round(error, 1)
            center_errors.append(error)

            if error <= ACCURACY_THRESHOLD_FRACTION * detection.radius:
                row["accurate"] = True
                n_accurate += 1

        results.append(row)

    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    detection_rate = n_found/len(candidates)
    accuracy_rate = n_accurate/len(candidates)
    mean_error = sum(center_errors) / len(center_errors) if center_errors else None

    configure_tracking()
    mlflow.set_experiment("retroread_classical_baseline")
    with mlflow.start_run(run_name="exp02_tighter_max_radius"):
        mlflow.log_params(HOUGH_PARAMS)
        mlflow.log_param("n_candidates", len(candidates))
        mlflow.log_param("accuracy_threshold_fraction", ACCURACY_THRESHOLD_FRACTION)
        mlflow.log_param("changed_from_exp01", "max_radius_fraction: 0.6 -> 0.35")

        mlflow.log_metric("detection_rate", detection_rate)
        mlflow.log_metric("accuracy_rate", accuracy_rate)
        if mean_error is not None:
            mlflow.log_metric("mean_center_error_px", mean_error)

        mlflow.log_artifact(str(RESULTS_CSV))

    print(f"\nDetection rate: {n_found}/{len(candidates)} ({detection_rate:.1%})")
    print(f"Accuracy rte (center within {ACCURACY_THRESHOLD_FRACTION} of radius): "
          f"{n_accurate}/{len(candidates)} ({accuracy_rate:.1%})")
    if mean_error is not None:
        print(f"Mean center error: {mean_error:.1f}px")
    print(f"Full results: {RESULTS_CSV}")

if __name__ == "__main__":
    main()