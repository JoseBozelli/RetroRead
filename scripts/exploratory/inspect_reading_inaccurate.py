"""
Generates overlay images (circle + needle) for a few reading_inaccurate cases from failure_analysis.py,
alongside their predicted vs true values, for visual inspection.

Run from the repo root with:
    uv run python script/exploratory/inspect_reading_inaccurate.py
"""

import csv
from pathlib import Path

from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_IMAGES_DIR
from retroread.needle_detection import detect_needle, draw_needle_overlay
from retroread.params import CIRCLE_PARAMS, NEEDLE_PARAMS

EXP06_CSV = Path("experiment_06_results.csv")
OUTPUT_DIR = Path("reading_inaccurate_samples")
TARGET_FILES = [
    "data/v_0117_f_0000_rgba.png",
    "data/v_0184_f_0000_rgba.png",
    "data/v_0034_f_0000_rgba.png",
]

def main() -> None:
    with EXP06_CSV.open() as f:
        exp06 = {row["file_name"]: row for row in csv.DictReader(f)}

    OUTPUT_DIR.mkdir(exist_ok=True)

    for file_name in TARGET_FILES:
        row = exp06[file_name]
        image_path = str(ENDAVA_DS5_IMAGES_DIR / file_name)

        circle = find_gauge_circle(image_path, **CIRCLE_PARAMS)
        needle = detect_needle(image_path, circle, **NEEDLE_PARAMS)

        out_name = Path(file_name).name.replace(".png", "_check.png")
        out_path = OUTPUT_DIR / out_name
        draw_needle_overlay(image_path, circle, needle, str(out_path))

        print(f"{file_name}")
        print(f" true_value={row['true_value']} predicted={row['predicted_value']} pct_error={row['pct_error']}%")
        print(f" -> {out_path}")

if __name__ == "__main__":
    main()