"""
Prints ready-to-paste calibration_points JSON (and the true reading, for
comparison) for a chosen image -- useful for testing the /predict endpoint
by hand via the /docs UI, with real, correct numbers.

Run from the repo root with:
    uv run python scripts/exploratory/get_calibration_for_demo.py <filename>
Example:
    uv run python scripts/exploratory/get_calibration_for_demo.py data/v_0992_f_0000_rgba.png
"""

import json
import sys

from retroread.annotations import load_reading_annotations
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR


def main() -> None:
    filename = sys.argv[1] if len(sys.argv) > 1 else "data/v_0992_f_0000_rgba.png"

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    by_filename = {d["file_name"]: d for d in reading_data.values()}
    data = by_filename.get(filename)

    if data is None:
        print(f"No reading annotation found for '{filename}'.")
        return

    points = [{"x": round(s["x"]), "y": round(s["y"]), "value": s["value"]} for s in data["scale_labels"]]

    print(f"Full image path to upload: {ENDAVA_DS5_IMAGES_DIR / filename}")
    print(f"\ncalibration_points (paste into the /docs UI field):")
    print(json.dumps(points))
    print(f"\nTrue reading (for comparison against the API's response): {data['true_value']:.3f}")


if __name__ == "__main__":
    main()