"""
Experiment 06 -- Classical CV Baseline, batch reading conversion.

Full end-to-end evaluation: circle detection -> needle detection -> reading conversion, across te full validated
candidate set, compared against real ground-truth readings (dial.synth_dial_value).

Error is reported both in raw gauge units and as a percentage of each gauge's own scale range, since every synthetic
gauge has a randomized scale -- raw units are not comparable across images with different ranges.

Run from repo root with:
    uv run python scripts/experiments/experiment_06_batch_reading_conversion.py
"""

import csv
import statistics
from pathlib import Path

import mlflow

from retroread.annotations import bbox_touches_edge, load_complete_annotations, load_reading_annotations
from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_TRAIN_KPTS_COCO
from retroread.needle_detection import detect_needle
from retroread.params import CIRCLE_PARAMS, NEEDLE_PARAMS
from retroread.reading_conversion import angle_to_value, fit_scale_calibration
from retroread.mlflow_setup import configure_tracking

RESULTS_CSV = Path("experiment_06_results.csv")
TOLERANCE_PCT = 5.0     # "accurate" if within this % of the gauge's own scale range

def main() -> None:
    edge_filtered = load_complete_annotations(ENDAVA_DS5_TRAIN_KPTS_COCO)
    edge_filtered = [c for c in edge_filtered if not bbox_touches_edge(c)]
    valid_filenames = {c["file_name"] for c in edge_filtered}

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    by_filename = {d["file_name"]: d for d in reading_data.values()}

    candidates = [by_filename[fn] for fn in valid_filenames if fn in by_filename]
    print(f"Edge-filtered candidates: {len(valid_filenames)}")
    print(f"With complete reading annotations: {len(candidates)}")

    results = []
    n_needle_found = 0
    n_accurate = 0
    pct_errors = []
    raw_errors = []

    for data in candidates:
        image_path = str(ENDAVA_DS5_IMAGES_DIR / data["file_name"])

        calibration = fit_scale_calibration(data["center_x"], data["center_y"], data["scale_labels"])
        scale_values = [s["value"] for s in data["scale_labels"]]
        scale_range = max(scale_values) - min(scale_values)

        circle = find_gauge_circle(image_path, **CIRCLE_PARAMS)
        needle = detect_needle(image_path, circle, **NEEDLE_PARAMS)

        row = {
            "file_name": data["file_name"],
            "true_value": data["true_value"],
            "needle_found": needle.found,
            "predicted_value": None,
            "raw_error": None,
            "pct_error": None,
            "accurate": False
        }

        if needle.found:
            n_needle_found += 1
            predicted = angle_to_value(needle.angle_rad, calibration)
            raw_error = abs(predicted - data["true_value"])
            pct_error = (raw_error / scale_range) * 100 if scale_range > 0 else None

            row["predicted_value"] = round(predicted, 3)
            row["raw_error"] = round(raw_error, 3)
            row["pct_error"] = round(pct_error, 2) if pct_error is not None else None

            if pct_error is not None:
                raw_errors.append(raw_error)
                pct_errors.append(pct_error)
                if pct_error <= TOLERANCE_PCT:
                    row["accurate"] = True
                    n_accurate += 1


        results.append(row)

    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    n = len(candidates)
    needle_rate = n_needle_found / n
    accuracy_rate = n_accurate / n
    mean_pct_error = sum(pct_errors) / len(pct_errors) if pct_errors else None
    median_pct_error = statistics.median(pct_errors) if pct_errors else None
    mean_raw_error = sum(raw_errors) / len(raw_errors) if raw_errors else None

    configure_tracking()
    mlflow.set_experiment("retroread_classical_baseline")
    with mlflow.start_run(run_name="exp06_batch_reading_conversion"):
        mlflow.log_params(CIRCLE_PARAMS)
        mlflow.log_params(NEEDLE_PARAMS)
        mlflow.log_param("n_candidates", n)
        mlflow.log_param("tolerance_pct", TOLERANCE_PCT)

        mlflow.log_metric("needle_detection_rate", needle_rate)
        mlflow.log_metric("accuracy_rate_within_tolerance", accuracy_rate)
        if mean_pct_error is not None:
            mlflow.log_metric("mean_pct_error", mean_pct_error)
            mlflow.log_metric("median_pct_error", median_pct_error)
            mlflow.log_metric("mean_raw_error", mean_raw_error)

        mlflow.log_artifact(str(RESULTS_CSV))

    print(f"\nNeedle detection rate: {n_needle_found} / {n} ({needle_rate:.1%})")
    print(f"Accuracy rate (within {TOLERANCE_PCT}% of scale range): {n_accurate}/{n} ({accuracy_rate:.1%})")
    if mean_pct_error is not None:
        print(f"Mean % error: {mean_pct_error:.2f}%")
        print(f"Median % error: {median_pct_error:.2f}%")
        print(f"Mean raw error: {mean_raw_error:.3f} units")
    print(f"Full results: {RESULTS_CSV}")

if __name__ == "__main__":
    main()