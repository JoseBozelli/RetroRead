"""
Experiment 19 -- Confidence/abstention mechanism, coverage-vs-accuracy tradeoff.

Runs the full read_gauge() pipeline (circle -> needle -> reading -> confidence) across the same 61-image fair validation
subset used throughout, at several confidence thresholds, to show how raising the threshold trades reduced coverage (more
abstentions) for higher accuracy among accepted readings. This is the evidence base for choosing a default production
threshold.

Run from the repo root:
    uv run python scripts/experiments/experiment_19_confidence_abstention.py
"""
import csv

import mlflow

from retroread.annotations import bbox_touches_edge, load_complete_annotations, load_reading_annotations
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.mlflow_setup import configure_tracking
from retroread.predict import read_gauge
from retroread.reading_conversion import fit_scale_calibration

RESULTS_CSV = "experiment_19_results.csv"
THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]

def main() -> None:
    edge_filtered = load_complete_annotations(ENDAVA_DS5_VAL_KPTS_COCO)
    edge_filtered = [c for c in edge_filtered if not bbox_touches_edge(c)]
    valid_filenames = {c["file_name"] for c in edge_filtered}

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    by_filename = {d["file_name"]: d for d in reading_data.values()}
    candidates = [by_filename[fn] for fn in valid_filenames if fn in by_filename]
    print(f"Evaluating on {len(candidates)} images.")

    # Run the pipeline once per image at threshold=0.0 (never abstains on confidence) to get confidence + reading
    # for every image; apply different thresholds afterward without re-running detection.
    per_image = []
    for data in candidates:
        image_path = str(ENDAVA_DS5_IMAGES_DIR / data["file_name"])
        calibration = fit_scale_calibration(data["center_x"], data["center_y"], data["scale_labels"])
        scale_values = [s["value"] for s in data["scale_labels"]]
        scale_range = max(scale_values) - min(scale_values)

        result = read_gauge(image_path, calibration, confidence_threshold=0.0)

        pct_error = None
        if result.status == "ok" and scale_range > 0:
            pct_error = abs(result.reading - data["true_value"]) / scale_range *100

        per_image.append({
            "file_name": data["file_name"],
            "confidence": result.confidence,
            "status": result.status,
            "pct_error": pct_error
        })

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    n_total = len(per_image)
    configure_tracking()
    mlflow.set_experiment("retroread_classical_baseline")

    print(f"\n{'Threshold':>10} {'Coverage':>10} {'Accuracy (accepted)':>22} {'Mean err (accepted)':>22}")
    with mlflow.start_run(run_name="exp19_confidence_abstention"):
        for threshold in THRESHOLDS:
            accepted= [
                r for r in per_image
                if r["status"] == "ok" and r["confidence"] is not None and r["confidence"] >= threshold
            ]
            coverage = len(accepted) / n_total
            errors = [r["pct_error"] for r in accepted if r["pct_error"] is not None]
            n_accurate = sum(1 for e in errors if e <= 5.0)
            accuracy_among_accepted = n_accurate / len(accepted) if accepted else 0.0
            mean_error = sum(errors) / len(errors) if errors else None

            mlflow.log_metric(f"coverage_t{threshold}", coverage)
            mlflow.log_metric(f"accuracy_t{threshold}", accuracy_among_accepted)
            if mean_error is not None:
                mlflow.log_metric(f"mean_error_t{threshold}", mean_error)

            mean_err_str = f"{mean_error:.2f}%" if mean_error is not None else "n/a"
            print(f"{threshold:>10.2f} {coverage:>9.1%} {accuracy_among_accepted:>21.1%} {mean_err_str:>22}")

        mlflow.log_artifact(RESULTS_CSV)

    print(f"\nFull per-image results: {RESULTS_CSV}")

if __name__ == "__main__":
    main()