"""
End-to-end classical gauge-reading pipeline: circle detection -> needle detection -> reading conversion ->
confidence-based accept/abstain decision.

Calibration (the angle-to-value mapping) must be supplied by the caller -- this system does not read printed scale
numbers off the gauge face itself. In practice, calibration would be configured once per gauge type/model and reused
for every photo of that same gauge; see docs/PRODUCT_HYPOTHESIS.md for this scope boundary.
"""

from dataclasses import dataclass

import cv2

from retroread.classical_baseline import CircleDetection, find_gauge_circle
from retroread.needle_detection import NeedleDetection, detect_needle
from retroread.params import CIRCLE_PARAMS, NEEDLE_PARAMS
from retroread.reading_conversion import ScaleCalibration, angle_to_value

@dataclass
class GaugeReading:
    status: str     # "ok" or "unable_to_read"
    reading: float  | None = None
    confidence: float   | None = None
    reason: str     | None = None
    circle: CircleDetection | None = None
    needle: NeedleDetection | None = None

def image_sharpness(image_path: str) -> float:
    """Variance of Laplacian -- higher means sharper. See pick_sample_image.py for the same metric."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()

def compute_confidence(
    circle: CircleDetection,
    needle: NeedleDetection,
    sharpness: float,
    min_length_fraction: float = 0.45,
) -> float:
    """
    Rule-based confidence score in [0, 1]. Not a calibrated probability.

    Weighted almost entirely on needle length relative to gauge radius:
    empirically, this was found (Experiment 19 diagnostic) to cleanly
    separate the dataset's two catastrophic-error cases (length_fraction
    0.350, 0.373; errors 67% and 25%) from all others (length_fraction
    >= 0.535) -- a natural gap in the data, not a guessed threshold.

    Sharpness is intentionally given minimal weight: the same diagnostic
    found no meaningful correlation between this dataset's (incidental,
    not deliberately controlled) sharpness variation and reading accuracy.
    This is a known limitation, not a validation that sharpness doesn't
    matter -- the project's planned deliberate-blur robustness test
    (spec Section 14) was not performed; that test would be the correct
    way to establish a real sharpness-based signal. Kept at low weight
    here as a placeholder for that future work, not removed entirely,
    since severe real-world blur plausibly still matters even though this
    dataset's mild natural variation didn't show it.
    """
    if not circle.found or not needle.found:
        return 0.0

    length_fraction = (needle.line_length / circle.radius) if circle.radius else 0.0
    length_score = max(0.0, min(1.0, length_fraction / min_length_fraction)) if min_length_fraction > 0 else 1.0

    sharpness_score = max(0.0, min(1.0, sharpness / 200.0))  # loose floor, minimal influence by design

    return 0.9 * length_score + 0.1 * sharpness_score

def read_gauge(
        image_path: str,
        calibration: ScaleCalibration,
        confidence_threshold: float = 0.85,     # empirically chosen; see Experiment 19
        circle_params: dict | None = None,
        needle_params: dict | None = None,
) -> GaugeReading:
    circle_params = circle_params or CIRCLE_PARAMS
    needle_params = needle_params or NEEDLE_PARAMS

    circle = find_gauge_circle(image_path, **circle_params)
    if not circle.found:
        return GaugeReading(status="unable_to_read", reason="gauge_not_detected", circle=circle)

    needle = detect_needle(image_path, circle, **needle_params)
    if not needle.found:
        return GaugeReading(status="unable_to_read", reason="needle_not_detected", circle=circle, needle=needle)

    sharpness = image_sharpness(image_path)
    confidence = compute_confidence(circle, needle, sharpness)

    if confidence < confidence_threshold:
        return GaugeReading(
            status="unable_to_read", reason="low_confidence", confidence=confidence, circle=circle, needle=needle
        )

    reading = angle_to_value(needle.angle_rad, calibration)
    return GaugeReading(status="ok", reading=reading, confidence=confidence, circle=circle, needle=needle)