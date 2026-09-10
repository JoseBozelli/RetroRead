"""
Dumps EasyOCR's completely raw, unfiltered output for one image -- to see
exactly what text string was recognized before our number-parsing and
distance filtering touch it.

Run from the repo root with:
    uv run python scripts/exploratory/debug_raw_ocr.py
"""

from retroread.config import ENDAVA_DS5_IMAGES_DIR
from retroread.ocr_calibration import _get_reader

TARGET_FILENAME = "data/v_0992_f_0000_rgba.png"


def main() -> None:
    image_path = str(ENDAVA_DS5_IMAGES_DIR / TARGET_FILENAME)
    reader = _get_reader()
    results = reader.readtext(image_path)

    print(f"{len(results)} raw detections:\n")
    for bbox, text, confidence in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        print(f"  text={text!r:20} confidence={confidence:.2f}  center=({cx:.0f},{cy:.0f})  bbox={bbox}")


if __name__ == "__main__":
    main()