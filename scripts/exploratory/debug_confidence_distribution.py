"""
Diagnostic: prints raw length_fraction and sharpness values per image
(before combining into a confidence score), to see whether they actually
vary enough to be useful, and where real thresholds should sit.

Run from the repo root with:
    uv run python scripts/exploratory/debug_confidence_distribution.py
"""

from retroread.annotations import bbox_touches_edge, load_complete_annotations, load_reading_annotations
from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.needle_detection import detect_needle
from retroread.params import CIRCLE_PARAMS, NEEDLE_PARAMS
from retroread.predict import image_sharpness
from retroread.reading_conversion import angle_to_value, fit_scale_calibration


def main() -> None:
    edge_filtered = load_complete_annotations(ENDAVA_DS5_VAL_KPTS_COCO)
    edge_filtered = [c for c in edge_filtered if not bbox_touches_edge(c)]
    valid_filenames = {c["file_name"] for c in edge_filtered}

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    by_filename = {d["file_name"]: d for d in reading_data.values()}
    candidates = [by_filename[fn] for fn in valid_filenames if fn in by_filename]

    rows = []
    for data in candidates:
        image_path = str(ENDAVA_DS5_IMAGES_DIR / data["file_name"])
        circle = find_gauge_circle(image_path, **CIRCLE_PARAMS)
        if not circle.found:
            continue
        needle = detect_needle(image_path, circle, **NEEDLE_PARAMS)
        if not needle.found:
            continue

        sharpness = image_sharpness(image_path)
        length_fraction = needle.line_length / circle.radius

        calibration = fit_scale_calibration(data["center_x"], data["center_y"], data["scale_labels"])
        scale_values = [s["value"] for s in data["scale_labels"]]
        scale_range = max(scale_values) - min(scale_values)
        predicted = angle_to_value(needle.angle_rad, calibration)
        pct_error = abs(predicted - data["true_value"]) / scale_range * 100 if scale_range > 0 else None
        accurate = pct_error is not None and pct_error <= 5.0

        rows.append((data["file_name"], length_fraction, sharpness, pct_error, accurate))

    rows.sort(key=lambda r: r[1])  # sort by length_fraction
    print(f"{'file':30} {'length_frac':>12} {'sharpness':>10} {'pct_err':>9} {'accurate':>9}")
    for fn, lf, sh, pe, acc in rows:
        pe_str = f"{pe:.2f}" if pe is not None else "n/a"
        print(f"{fn:30} {lf:>12.3f} {sh:>10.1f} {pe_str:>9} {str(acc):>9}")

    lengths = [r[1] for r in rows]
    sharpnesses = [r[2] for r in rows]
    print(f"\nlength_fraction: min={min(lengths):.3f} max={max(lengths):.3f}")
    print(f"sharpness: min={min(sharpnesses):.1f} max={max(sharpnesses):.1f}")


if __name__ == "__main__":
    main()