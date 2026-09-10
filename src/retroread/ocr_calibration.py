"""
Automatic scale calibration via OCR: detects printed numeric labels on the
gauge face and their positions, producing the same {x, y, value}
calibration points fit_scale_calibration already expects -- replacing
manual entry for gauges whose scale is legible in the photo.

Uses EasyOCR, a pretrained deep-learning text detection/recognition model
(not trained by this project) -- a deliberate use of an existing ML tool
for the sub-task it's well-suited to (digit recognition), while the
classical Hough-based pipeline remains the production choice for gauge and
needle geometry, where it was empirically shown to outperform a
from-scratch DL model (see docs/deep_learning.md). Learned components
where learning wins; classical geometry where classical geometry wins.
"""

import re

from retroread.classical_baseline import CircleDetection

_reader = None


def _get_reader():
    """Lazily loads EasyOCR's pretrained model (downloaded on first use)."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _parse_number(text: str) -> float | None:
    """Extracts a numeric value from an OCR text token, or None if it isn't a clean number."""
    cleaned = text.strip().replace(",", ".")
    match = re.fullmatch(r"-?\d+(\.\d+)?", cleaned)
    return float(cleaned) if match else None


def detect_scale_labels(image_path: str, circle: CircleDetection, margin_fraction: float = 1.15) -> list[dict]:
    """
    Returns a list of {"x", "y", "value"} calibration points found via OCR,
    restricted to text near the gauge's detected circle (radius *
    margin_fraction) -- avoids picking up unrelated background numbers or
    the gauge's own model/unit text (which OCR reads but fails to parse as
    a number, and is filtered out by _parse_number).
    """
    if not circle.found:
        return []

    reader = _get_reader()
    results = reader.readtext(image_path)

    max_dist = circle.radius * margin_fraction
    points = []

    for bbox, text, ocr_confidence in results:
        value = _parse_number(text)
        if value is None:
            continue

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)

        dist = ((cx - circle.center_x) ** 2 + (cy - circle.center_y) ** 2) ** 0.5
        if dist <= max_dist:
            points.append({"x": cx, "y": cy, "value": value})

    return points