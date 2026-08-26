"""
Finds the single worst reading_inaccurate case (120% error) and prints its full raw inputs --
calibration points, angles, unwrapping choice -- to identify whether this is a real bug or a genuinely
degenerate case.

Run from the repo root with:
    uv run python scripts/exploratory/debug_worst_reading_error.py
"""

import math

from retroread.annotations import bbox_touches_edge, load_complete_annotations, load_reading_annotations
from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_TRAIN_KPTS_COCO
from retroread.needle_detection import detect_needle
from retroread.params import CIRCLE_PARAMS, NEEDLE_PARAMS
from retroread.reading_conversion import angle_to_value, fit_scale_calibration

def main() -> None:
    edge_filtered = load_complete_annotations(ENDAVA_DS5_TRAIN_KPTS_COCO)
    edge_filtered = [c for c in edge_filtered if not bbox_touches_edge(c)]
    valid_filenames = {c["file_name"] for c in edge_filtered}

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    by_filename = {d["file_name"]: d for d in reading_data.values()}
    candidates = [by_filename[fn] for fn in valid_filenames if fn in by_filename]

    worst = None
    worst_pct_error = -1

    for data in candidates:
        image_path = str(ENDAVA_DS5_IMAGES_DIR / data["file_name"])
        calibration = fit_scale_calibration(data["center_x"], data["center_y"], data["scale_labels"])
        scale_values = [s["value"] for s in data["scale_labels"]]
        scale_range = max(scale_values) - min(scale_values)

        circle = find_gauge_circle(image_path, **CIRCLE_PARAMS)
        needle = detect_needle(image_path, circle, **NEEDLE_PARAMS)

        if not needle.found or scale_range <= 0:
            continue

        predicted = angle_to_value(needle.angle_rad, calibration)
        raw_error = abs(predicted - data["true_value"])
        pct_error = (raw_error / scale_range) * 100

        if pct_error > worst_pct_error:
            worst_pct_error = pct_error
            worst = {
                "file_name": data["file_name"],
                "true_value": data["true_value"],
                "predicted": predicted,
                "pct_error": pct_error,
                "center_x": data["center_x"],
                "center_y": data["center_y"],
                "scale_labels": data["scale_labels"],
                "needle_angle": needle.angle_rad,
                "calibration": calibration
            }

    print(f"Worst case: {worst['file_name']} pct_error={worst['pct_error']:.2f}%\n")
    print(f"True value: {worst['true_value']}")
    print(f"Predicted:{worst['predicted']:.3f}")
    print(f"Dial center: ({worst['center_x']}, {worst['center_y']})")
    print(f"\nScale-label calibration points (x, y, value, angle_deg):")
    for s in worst["scale_labels"]:
        angle = math.degrees(math.atan2(s["y"] - worst["center_y"], s["x"] - worst["center_x"]))
        print(f" ({s['x']:.0f}, {s['y']:.0f}) value={s['value']} angle={angle:.1f} dg")

    print(f"\nCalibration slope: {worst['calibration'].slope:.4f}")
    print(f"Calibration intercept: {worst['calibration'].intercept:.4f}")
    print(f"Reference unwrapped angles (deg): "
          f"{[round(math.degrees(a), 1) for a in worst['calibration'].reference_unwrapped_angles]}")
    print(f"Reference values: {worst['calibration'].reference_values}")

    print(f"\nDetected needle angle (deg): {math.degrees(worst['needle_angle']):.1f}")

if __name__ == "__main__":
    main()