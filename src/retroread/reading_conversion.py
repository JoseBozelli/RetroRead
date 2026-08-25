"""
Reading conversion for the classical CV baseline (final stage): converts a needle angle into a numeric gauge reading.

Each synthetic gauge has a randomized scale (different min/max, different arc length), so there is no fixed angle-to-
value formula. Instead, each image's own scale-label annotations (pixel position + known numeric value) are used as 
calibration points, and a straight-line mapping is fit between angle and value.
"""

import math
from dataclasses import dataclass

import numpy as np

@dataclass
class ScaleCalibration:
    slope: float
    intercept: float
    reference_unwrapped_angles: list[float]
    reference_values: list[float]

def fit_scale_calibration(center_x: float, center_y: float, scale_labels: list[dict]) -> ScaleCalibration:
    """
    scale_labels: list of dicts with 'x', 'y' (pixel position) and 'value' (known numeric value), e.g. from
    scale-label annotations.
    
    Retruns a linear angle-to-value mapping. Angles are "unwrapped" before fitting -- represented as a continuous
    increasing / decreasing sequence rather than one that resets at +/- 180 degrees -- so a false jump at that boundary
    does not distort the fit.
    """
    if len(scale_labels) < 2:
        raise ValueError("At least 2 scale-label calibration points are required.")

    sorted_labels = sorted(scale_labels, key=lambda s: s["value"])
    raw_angles = [math.atan2(s["y"] - center_y, s["x"] - center_x) for s in scale_labels]
    values = [s["value"] for s in sorted_labels]

    unwrapped_angles = np.unwrap(raw_angles).tolist()

    slope, intercept = np.polyfit(unwrapped_angles, values, deg=1)

    return ScaleCalibration(
        slope=float(slope),
        intercept=float(intercept),
        reference_unwrapped_angles=unwrapped_angles,
        reference_values=values
    )

def angle_to_value(angle_rad: float, calibration: ScaleCalibration) -> float:
    """
    Convert a raw angle (radians, atan2 relative to dial center -- same convention used for calibration points)
    into a numeric reading.

    The input angle is matched to the same continuous (unwrapped) angle representation used during calibration, 
    by finding whichever angle_rad + k*2*pi (k an intenger) lands closest to the nearest calibration point. This
    avoids a false jump in the predicted value when the needle happens to sit near the +/- 180 deree boundary.
    """
    best_unwrapped = None
    best_diff = float("inf")

    for ref in calibration.reference_unwrapped_angles:
        k = round((ref - angle_rad) / (2 * math.pi))
        candidate = angle_rad + k * 2 * math.pi
        diff = abs(candidate - ref)
        if diff < best_diff:
            best_diff = diff
            best_unwrapped = candidate

    return calibration.slope * best_unwrapped + calibration.intercept