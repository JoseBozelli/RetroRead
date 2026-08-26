"""
Experiment 00 - Classical CV Baseline, circle detection only.

Hypothesis: the Hough Circle Transform can reliably locate the gauge face in a representative synthetic image,
using default parameters expressed as fractions of image size.

Run from the repo root with:
    uv run python scripts/experiments/experiment_00_circle_detection.py
"""

import mlflow

from retroread.classical_baseline import draw_circle_overlay, find_gauge_circle
from retroread.config import ENDAVA_DS5_IMAGES_DIR 
from retroread.mlflow_setup import configure_tracking

IMAGE_PATH = str(ENDAVA_DS5_IMAGES_DIR / "data" / "v_0992_f_0000_rgba.png")
OVERLAY_OUTPUT =  "experiment_00_overlay.png"

HOUGH_PARAMS = {
    "dp": 1.0,
    "min_dist_fraction": 0.5,
    "param1": 100,
    "param2": 50,
    "min_radius_fraction": 0.1,
    "max_radius_fraction": 0.6
}

def main() -> None:
    configure_tracking()
    mlflow.set_experiment("retroread_classical_baseline")

    with mlflow.start_run(run_name="exp00_circle_detection"):
        mlflow.log_params(HOUGH_PARAMS)
        mlflow.log_param("image_path", IMAGE_PATH)

        detection = find_gauge_circle(IMAGE_PATH, **HOUGH_PARAMS)

        mlflow.log_metric("circle_found", 1 if detection.found else 0)
        if detection.found:
            mlflow.log_metric("center_x", detection.center_x)
            mlflow.log_metric("center_y", detection.center_y)
            mlflow.log_metric("radius", detection.radius)

        draw_circle_overlay(IMAGE_PATH, detection, OVERLAY_OUTPUT)
        mlflow.log_artifact(OVERLAY_OUTPUT)

        print(f"Circle found: {detection.found}")
        if detection.found:
            print(f"Center: ({detection.center_x}, {detection.center_y})  Radius: {detection.radius}")
        print(f"Overlay saved to: {OVERLAY_OUTPUT}")

if __name__ == "__main__":
    main()