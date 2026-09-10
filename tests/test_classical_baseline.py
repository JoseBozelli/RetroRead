"""
Regression tests for two real bugs found during real-world testing:
1. find_gauge_circle crashed (NameError) instead of gracefully returning
   "not found" when no circle was detected.
2. fit_scale_calibration produced wrong results when calibration points
   were supplied out of value order.

Run with: uv run pytest tests/test_classical_baseline.py
"""

from retroread.classical_baseline import find_gauge_circle
from retroread.reading_conversion import fit_scale_calibration


def test_find_gauge_circle_returns_not_found_without_crashing(tmp_path):
    """A blank image has no circle to find -- should return found=False, not raise."""
    from PIL import Image

    blank_image_path = tmp_path / "blank.png"
    Image.new("RGB", (200, 200), color="white").save(blank_image_path)

    result = find_gauge_circle(str(blank_image_path))

    assert result.found is False


def test_fit_scale_calibration_is_order_independent():
    """Calibration must produce the same result regardless of input point order."""
    center_x, center_y = 500.0, 500.0
    points_in_order = [
        {"x": 400.0, "y": 600.0, "value": 0},
        {"x": 500.0, "y": 400.0, "value": 5},
        {"x": 600.0, "y": 600.0, "value": 10},
    ]
    points_scrambled = [points_in_order[2], points_in_order[0], points_in_order[1]]

    calibration_ordered = fit_scale_calibration(center_x, center_y, points_in_order)
    calibration_scrambled = fit_scale_calibration(center_x, center_y, points_scrambled)

    assert calibration_ordered.slope == calibration_scrambled.slope
    assert calibration_ordered.intercept == calibration_scrambled.intercept