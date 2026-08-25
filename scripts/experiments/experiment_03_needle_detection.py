"""
Experiment 03 -- Classical CV Baseline, needle detection (single image).

Hypothesis: given a correctly detected circle, the needle can be found as a straight line radiating from the
circle's center, using edge detection plus the Probabilistic Hough Line Transform restricted to the inner portion
of the circle.

Single-image check first, same pattern as Experiment 00 -- confirm the approach works visually before running it
across the full candidate set.

Run from the repor root with:
    uv run python scripts/experiment_03_needle_detection.py
"""

import sys
from pathlib import Path

import mlflow

def _find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found).")

sys.path.insert(0, str(_find_project_root(Path(__file__).resolve()) / "src"))
from retroread.classical_baseline import find_gauge_circle
from retroread.needle_detection import detect_needle, draw_needle_overlay

from retroread.config import ENDAVA_DS5_IMAGES_DIR 

IMAGE_PATH = str(ENDAVA_DS5_IMAGES_DIR / "data" / "v_0992_f_0000_rgba.png")

OVERLAY_OUTPUT = "experiment_03_overlay.png"

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

def main() -> None:
    mlflow.set_experiment("retroread_classical_baseline")

    with mlflow.start_run(run_name="exp03_needle_detection_single_image"):
        mlflow.log_params(CIRCLE_PARAMS)
        mlflow.log_params(NEEDLE_PARAMS)
        mlflow.log_param("image_path", IMAGE_PATH)

        circle = find_gauge_circle(IMAGE_PATH, **CIRCLE_PARAMS)
        needle = detect_needle(IMAGE_PATH, circle, **NEEDLE_PARAMS)

        mlflow.log_metric("circle_found", 1 if circle.found else 0)
        mlflow.log_metric("needle_found", 1 if needle.found else 0)
        if needle.found:
            mlflow.log_metric("needle_angle_rad", needle.angle_rad)
            mlflow.log_metric("needle_line_length", needle.line_length)

        draw_needle_overlay(IMAGE_PATH, circle, needle, OVERLAY_OUTPUT)
        mlflow.log_artifact(OVERLAY_OUTPUT)

        print(f"Circle found: {circle.found}")
        print(f"Needle found: {needle.found}")
        if needle.found:
            print(f"Needle tip: ({needle.tip_x}, {needle.tip_y})")
            print(f"Needle angle (rad): {needle.angle_rad:.4f}")
            print(f"Line length: {needle.line_length:.1f}px")
        print(f"Overlay saved to: {OVERLAY_OUTPUT}")

if __name__ == "__main__":
    main()