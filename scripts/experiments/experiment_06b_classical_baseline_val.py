"""
Experiment 06b -- Classical baseline, evaluated on the VAL split.

Experiment 06 evaluated on the TRAIN split -- not a fair comparison against
DL evaluations (10, 12, 14), which correctly use the held-out val set.
This reruns the identical classical pipeline restricted to val images, for
a genuine apples-to-apples number.

Run from the repo root with:
    uv run python scripts/experiments/experiment_06b_classical_baseline_val.py
"""

import csv

import mlflow

from retroread.annotations import bbox_touches_edge, load_complete_annotations, load_reading_annotations
from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.mlflow_setup import configure_tracking
from retroread.needle_detection import detect_needle
from retroread.params import CIRCLE_PARAMS, NEEDLE_PARAMS
from retroread.reading_conversion import angle_to_value, fit_scale_calibration

RESULTS_CSV = "experiment_06b_results.csv"
TOLERANCE_PCT = 5.0


def main() -> None:
    edge_filtered = load_complete_annotations(ENDAVA_DS5_VAL_KPTS_COCO)
    edge_filtered = [c for c in edge_filtered if not bbox_touches_edge(c)]
    valid_filenames = {c["file_name"] for c in edge_filtered}

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    by_filename = {d["file_name"]: d for d in reading_data.values()}
    candidates = [by_filename[fn] for fn in valid_filenames if fn in by_filename]
    print(f"Val edge-filtered candidates with reading annotations: {len(candidates)}")

    results = []
    n_needle_found = 0
    n_accurate = 0
    pct_errors = []

    for data in candidates:
        image_path = str(ENDAVA_DS5_IMAGES_DIR / data["file_name"])
        calibration = fit_scale_calibration(data["center_x"], data["center_y"], data["scale_labels"])
        scale_values = [s["value"] for s in data["scale_labels"]]
        scale_range = max(scale_values) - min(scale_values)

        circle = find_gauge_circle(image_path, **CIRCLE_PARAMS)
        needle = detect_needle(image_path, circle, **NEEDLE_PARAMS)

        row = {"file_name": data["file_name"], "true_value": data["true_value"],
               "needle_found": needle.found, "pct_error": None, "accurate": False}

        if needle.found:
            n_needle_found += 1
            predicted = angle_to_value(needle.angle_rad, calibration)
            raw_error = abs(predicted - data["true_value"])
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
    mlflow.set_experiment("retroread_classical_baseline")
    with mlflow.start_run(run_name="exp06b_classical_baseline_val_split"):
        mlflow.log_param("n_candidates", n)
        mlflow.log_metric("needle_detection_rate", n_needle_found / n)
        mlflow.log_metric("accuracy_rate", accuracy_rate)
        if mean_pct_error is not None:
            mlflow.log_metric("mean_pct_error", mean_pct_error)
        mlflow.log_artifact(RESULTS_CSV)

    print(f"\nNeedle detection rate: {n_needle_found}/{n} ({n_needle_found/n:.1%})")
    print(f"Accuracy rate (val split): {n_accurate}/{n} ({accuracy_rate:.1%})")
    if mean_pct_error is not None:
        print(f"Mean % error (val split): {mean_pct_error:.2f}%")


if __name__ == "__main__":
    main()