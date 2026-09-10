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
from retroread.reading_conversion import ScaleCalibration, angle_to_value, fit_scale_calibration


@dataclass
class GaugeReading:
    status: str     # "ok" or "unable_to_read"
    reading: float  | None = None
    confidence: float   | None = None
    reason: str     | None = None
    circle: CircleDetection | None = None
    needle: NeedleDetection | None = None

def _calibration_is_plausible(calibration, points: list[dict], center_x: float, center_y: float) -> bool:
    """
    Sanity check for OCR-derived calibration: for each calibration point,
    recompute its predicted value from its own angle through the fitted
    line, and reject the calibration if that self-consistency error is
    large. A garbled OCR digit tends to distort the fit badly enough to
    fail this check even though the fit always mathematically "succeeds."
    """
    import math
    from retroread.reading_conversion import angle_to_value

    values = [p["value"] for p in points]
    scale_range = max(values) - min(values)
    if scale_range <= 0:
        return False

    for p in points:
        angle = math.atan2(p["y"] - center_y, p["x"] - center_x)
        predicted = angle_to_value(angle, calibration)
        if abs(predicted - p["value"]) / scale_range > 0.05:  # more than 5% of scale range off from itself
            return False
    return True

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
    calibration: ScaleCalibration | None = None,
    confidence_threshold: float = 0.85,  # empirically chosen; see Experiment 19
    circle_params: dict | None = None,
    needle_params: dict | None = None,
) -> GaugeReading:
    circle_params = circle_params or CIRCLE_PARAMS
    needle_params = needle_params or NEEDLE_PARAMS

    circle = find_gauge_circle(image_path, **circle_params)
    if not circle.found:
        return GaugeReading(status="unable_to_read", reason="gauge_not_detected", circle=circle)

    if calibration is None:
        from retroread.ocr_calibration import detect_scale_labels
        ocr_points = detect_scale_labels(image_path, circle)
        if len(ocr_points) < 2:
            return GaugeReading(
                status="unable_to_read", reason="auto_calibration_failed", circle=circle
            )
        calibration = fit_scale_calibration(circle.center_x, circle.center_y, ocr_points)
        if not _calibration_is_plausible(calibration, ocr_points, circle.center_x, circle.center_y):
            return GaugeReading(
                status="unable_to_read", reason="auto_calibration_implausible", circle=circle
            )

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

@dataclass
class EnsembleGaugeReading(GaugeReading):
    source: str | None = None            # "agreement", "classical_confident", "yolo_refiner_confident"
    classical_reading: float | None = None
    yolo_refiner_reading: float | None = None


def read_gauge_ensemble(
    image_path: str,
    calibration: ScaleCalibration,
    yolo_model,
    refiner_model,
    agreement_threshold_pct: float = 5.0,
    confidence_threshold: float = 0.85,
    circle_params: dict | None = None,
    needle_params: dict | None = None,
) -> EnsembleGaugeReading:
    """
    Production ensemble: runs both the classical pipeline and YOLO+refiner,
    and combines them via an agreement/confidence gate -- validated (paired
    error analysis) to beat either system alone. yolo_model and
    refiner_model are passed in already-loaded, since both are expensive
    to construct and should be reused across calls, not reloaded per image.
    """
    from retroread.yolo_refiner_pipeline import get_yolo_refiner_angle

    circle_params = circle_params or CIRCLE_PARAMS
    needle_params = needle_params or NEEDLE_PARAMS

    circle = find_gauge_circle(image_path, **circle_params)
    if not circle.found:
        return EnsembleGaugeReading(status="unable_to_read", reason="gauge_not_detected")

    # --- Classical ---
    needle = detect_needle(image_path, circle, **needle_params)
    classical_reading = None
    classical_confidence = 0.0
    if needle.found:
        sharpness = image_sharpness(image_path)
        classical_confidence = compute_confidence(circle, needle, sharpness)
        classical_reading = angle_to_value(needle.angle_rad, calibration)

    # --- YOLO + refiner ---
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    circle_bbox = [circle.center_x - circle.radius, circle.center_y - circle.radius,
                    circle.radius * 2, circle.radius * 2]
    from retroread.crop_heatmap_dataset import compute_crop_box
    crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_box(circle_bbox, image.width, image.height)
    cropped = image.crop((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)))

    yolo_angle = get_yolo_refiner_angle(image_path, cropped, (crop_x0, crop_y0), yolo_model, refiner_model)
    yolo_reading = angle_to_value(yolo_angle, calibration) if yolo_angle is not None else None

    # --- Combine ---
    if classical_reading is None and yolo_reading is None:
        return EnsembleGaugeReading(status="unable_to_read", reason="both_systems_failed", circle=circle, needle=needle)
    if classical_reading is None:
        return EnsembleGaugeReading(status="ok", reading=yolo_reading, source="yolo_refiner_only",
                                     yolo_refiner_reading=yolo_reading, circle=circle, needle=needle)
    if yolo_reading is None:
        return EnsembleGaugeReading(status="ok", reading=classical_reading, source="classical_only",
                                     confidence=classical_confidence, classical_reading=classical_reading,
                                     circle=circle, needle=needle)

    values = [s for s in [calibration.reference_values[0], calibration.reference_values[-1]]]
    scale_range = abs(values[-1] - values[0]) or 1.0
    disagreement_pct = abs(classical_reading - yolo_reading) / scale_range * 100

    if disagreement_pct <= agreement_threshold_pct:
        chosen, source = classical_reading, "agreement"
    elif classical_confidence >= confidence_threshold:
        chosen, source = classical_reading, "classical_confident"
    else:
        chosen, source = yolo_reading, "yolo_refiner_confident"

    return EnsembleGaugeReading(
        status="ok", reading=chosen, source=source, confidence=classical_confidence,
        classical_reading=classical_reading, yolo_refiner_reading=yolo_reading,
        circle=circle, needle=needle,
    )