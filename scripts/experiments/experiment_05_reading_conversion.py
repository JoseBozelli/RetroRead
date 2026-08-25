"""
Experiment 05 -- Classical CV Baseline, reading conversion (single image).

Two separate checks:
    1. Math validation: does the angle-to-value calibration correctly recover the true readingwhen gien the
    GROUND-TRUTH needle angle (dial_tip keypoint)? Should be very close to true_value if the math is right.
    2. Full pipeline: what reading results from the DETECTED needle angle (circle detection + needle detection,
    both from earlier experiments)? This is the real end-to-end baseline error.

Run from the repo root with:
    uv run python scripts/experiments/experiment_05_reading_conversion.py
"""

import math
import sys
from pathlib import Path

import mlflow

def _find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found).")

sys.path.insert(0, str(_find_project_root(Path(__file__).resolve()) / "src"))

from retroread.annotations import load_reading_annotations
from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR
from retroread.needle_detection import detect_needle
from retroread.reading_conversion import angle_to_value, fit_scale_calibration
from retroread.params import CIRCLE_PARAMS, NEEDLE_PARAMS
from retroread.mlflow_setup import configure_tracking

TARGET_FILENAME = "data/v_0992_f_0000_rgba.png"

def main() -> None:
    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)

    target = next((d for d in reading_data.values() if d["file_name"] == TARGET_FILENAME), None)
    if target is None:
        raise ValueError(f"'{TARGET_FILENAME}' not found among complete reading annotations.")

    print(f"Target: {TARGET_FILENAME}")
    print(f"True reading: {target['true_value']}")
    print(f"Scale-label calibration points: {len(target['scale_labels'])}")

    calibration = fit_scale_calibration(target["center_x"], target["center_y"], target["scale_labels"])

    gt_angle = math.atan2(target["tip_y"] - target["center_y"], target["tip_x"] - target["center_x"])
    predicted_from_gt = angle_to_value(gt_angle, calibration)
    math_error = abs(predicted_from_gt - target["true_value"])

    print(f"\n[Math check] Predicted from ground-truth angle: {predicted_from_gt:.3f}")
    print(f"[Math check] True value: {target['true_value']}")
    print(f"[Math check] Error: {math_error:.3f}")

    image_path = str(ENDAVA_DS5_IMAGES_DIR / TARGET_FILENAME)
    circle = find_gauge_circle(image_path, **CIRCLE_PARAMS)
    needle = detect_needle(image_path, circle, **NEEDLE_PARAMS)

    pipeline_error = None
    predicted_from_pipeline = None
    if needle.found:
        predicted_from_pipeline = angle_to_value(needle.angle_rad, calibration)
        pipeline_error = abs(predicted_from_pipeline - target["true_value"])
        print(f"\n[Full pipeline] Circle found: {circle.found}, Needle found: {needle.found}")
        print(f"[Full pipeline] Predicted from detected angle: {predicted_from_pipeline:.3f}")
        print(f"[Full pipeline] Error: {pipeline_error:.3f}")
    else:
        print(f"\n[Full pipeline] Needle not detected -- no reading produced.")

    configure_tracking()
    mlflow.set_experiment("retroread_classical_baseline")
    with mlflow.start_run(run_name="exp05_reading_conversion_single_image"):
        mlflow.log_param("image_path", TARGET_FILENAME)
        mlflow.log_metric("true_value", target["true_value"])
        mlflow.log_metric("math_check_predicted", predicted_from_gt)
        mlflow.log_metric("math_check_error", math_error)
        if pipeline_error is not None:
            mlflow.log_metric("pipeline_predicted", predicted_from_pipeline)
            mlflow.log_metric("pipeline_error", pipeline_error)

if __name__ == "__main__":
    main()