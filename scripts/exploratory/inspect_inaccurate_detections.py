"""
Quick visual check of a few inaccurate circle detections from experiment 01.
Not a full error analysis (that comes later, once the full baseline exists) - just a sanity check
that nothing systematic is wrong before adding needle detection on top.

Run from the repo root with:
    uv run python scripts/exploratory/inspect_innacurate_detections.py
"""
import csv
from pathlib import Path

from retroread.classical_baseline import CircleDetection, draw_circle_overlay

RESULTS_CSV = Path("experiment_01_results.csv")
IMAGES_DIR = Path("data/raw/Endava/sample_synth_datasets/ds5.0")
OUTPUT_DIR = Path("inaccurate_detection_samples")
N_SAMPLES = 5

def main() -> None:
    with RESULTS_CSV.open() as f:
        rows = list(csv.DictReader(f))

    inaccurate = [r for r in rows if r["accurate"] == "False" and r["found"] == "True"]
    print(f"Total inaccurate detections: {len(inaccurate)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    sample = inaccurate[:N_SAMPLES]

    for row in sample:
        image_path = IMAGES_DIR / row["file_name"]
        detection = CircleDetection(
            found=True,
            center_x=int(float(row["center_x"])),
            center_y=int(float(row["center_y"])),
            radius=int(float(row["radius"]))
        )
        out_name = Path(row["file_name"]).name.replace(".png", "_check.png")
        out_path = OUTPUT_DIR / out_name
        draw_circle_overlay(str(image_path), detection, str(out_path))
        print(f" {row['file_name']}: error={row['center_error_px']}px -> {out_path}")

if __name__ == "__main__":
    main()