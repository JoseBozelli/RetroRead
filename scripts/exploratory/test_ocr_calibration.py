"""
Single-image test of OCR-based auto-calibration against a known ground
truth, before wiring it into the API.

Run from the repo root with:
    uv run python scripts/exploratory/test_ocr_calibration.py
"""

from retroread.annotations import load_reading_annotations
from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR
from retroread.ocr_calibration import detect_scale_labels
from retroread.params import CIRCLE_PARAMS
from retroread.predict import read_gauge

TARGET_FILENAME = "data/v_0992_f_0000_rgba.png"


def main() -> None:
    image_path = str(ENDAVA_DS5_IMAGES_DIR / TARGET_FILENAME)

    circle = find_gauge_circle(image_path, **CIRCLE_PARAMS)
    print(f"Circle found: {circle.found}")

    ocr_points = detect_scale_labels(image_path, circle)
    print(f"\nOCR-detected calibration points ({len(ocr_points)}):")
    for p in ocr_points:
        print(f"  x={p['x']:.0f} y={p['y']:.0f} value={p['value']}")

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    by_filename = {d["file_name"]: d for d in reading_data.values()}
    true_data = by_filename.get(TARGET_FILENAME)

    print(f"\nGround-truth scale labels ({len(true_data['scale_labels'])}), for comparison:")
    for s in true_data["scale_labels"]:
        print(f"  x={s['x']:.0f} y={s['y']:.0f} value={s['value']}")

    print(f"\nTrue reading: {true_data['true_value']:.3f}")

    result = read_gauge(image_path, calibration=None)  # None triggers OCR auto-calibration
    print(f"\nread_gauge() result:")
    print(f"  status: {result.status}")
    print(f"  reading: {result.reading}")
    print(f"  confidence: {result.confidence}")
    print(f"  reason: {result.reason}")


if __name__ == "__main__":
    main()