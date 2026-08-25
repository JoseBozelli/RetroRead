"""
Classical CV baseline for gauge detection (Section 6 of the project spec).

No learned components. Uses the Hough Circle Transform to locate the round gauge face in an image.
This module handles circle detection only; needle detection and reading conversion are added in later stages.
"""

import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class CircleDetection:
    """Result of attempting to find the gauge face in an image."""

    found: bool
    center_x: int | None = None
    center_y: int | None = None
    radius: int | None = None

def find_gauge_circle(
        image_path: str,
        dp: float = 1.0,
        min_dist_fraction: float = 0.5,
        param1: float = 100, 
        param2: float = 50,
        min_radius_fraction: float = 0.1,
        max_radius_fraction: float = 0.35,  # tuned in Experiment 02; see decision_log.md
) -> CircleDetection:
    """
    Attempt to find a single circular gauge face in an image using the Hough Circle Transform.
    Radius and minimum-distance parameters are expressed as fractions of the image's smaller dimension, 
    so the same defaults work across different image sizes without hand-tuning pixel values per image.
    Returns a CircleDetection with found=false if no circle was detected. If multiple circles are detected,
    the strongest (first-returned) one is used - OpenCV orders results by accumulator strength.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image as '{image_path}'")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    smaller_dim = min(img.shape[0], img.shape[1])
    min_dist = smaller_dim * min_dist_fraction
    min_radius = int(smaller_dim*min_radius_fraction)
    max_radius = int(smaller_dim*max_radius_fraction)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return Cir(found=False)

    circles = np.round(circles[0, :]).astype(int)
    x, y, r = circles[0]
    return CircleDetection(found=True, center_x=int(x), center_y=int(y), radius=int(r))

def draw_circle_overlay(image_path: str, detection: CircleDetection, output_path: str) -> None:
    """Save a copy of he image with the detected circle drawn on it, for visual inspection."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at '{image_path}'")

    if detection.found:
        cv2.circle(img, (detection.center_x, detection.center_y), detection.radius, (0, 255, 0), 3)
        cv2.circle(img, (detection.center_x, detection.center_y), 5, (0, 0, 255), -1)

    cv2.imwrite(output_path, img)