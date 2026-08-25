"""
Needle detection for the classical CV baseline (Section 6).

Given a detected gauge circle, finds the needle as a straight line radiating from the circle's center,
using edge detection plus the Probabilistic Hough Line Transform, restricted to the inner portion of the circle
to avoid tick marks and text near the dial's edge.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from retroread.classical_baseline import CircleDetection

@dataclass
class NeedleDetection:
    """Result of attempting to find the needle within a detected circle."""

    found: bool
    tip_x: int | None = None
    tip_y: int | None = None
    angle_rad: float | None = None
    line_length: float | None = None

def detect_needle(
        image_path: str,
        circle: CircleDetection,
        inner_radius_fraction: float = 0.85,
        pivot_distance_fraction: float = 0.25,
        min_line_length_fraction: float = 0.3,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 30,
        max_line_gap: int = 10
) -> NeedleDetection:
    """
    Find he needle within a detected gauge circle.

    inner_radius_fraction: only search within this fraction of the circle's radius, to exclude 
        tick marks and labels near the dial edge.
    pivot_distance_fraction: a line qualifies as a needle candidate only if at least one endpoint 
        falls within this fraction of the radius from the center (since the needle pivots at the center).
    min_line_lenght_fraction: candidate lines shorter than this fraction of the radius are discarded as noise.
    
    Angle convetion: radians, measured with cv2.HoughLinesP's native image coordinate system (y increases downward),
    via atan2(tip_y - center_y, tip_x - center_x). Not yet converted to a "clock position" or gauge scale angle --
    that conversion happens in a later stage.
    """
    if not circle.found:
        return NeedleDetection(found=False)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at '{image_path}'")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mask to the inner portion of the circle only.
    mask = np.zeros_like(gray)
    inner_radius= int(circle.radius * inner_radius_fraction)
    cv2.circle(mask, (circle.center_x, circle.center_y), inner_radius, 255, -1)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    edges = cv2.Canny(masked_gray, canny_low, canny_high)

    min_line_length = circle.radius * min_line_length_fraction
    pivot_distance = circle.radius * pivot_distance_fraction

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None:
        return NeedleDetection(found=False)

    best_line = None
    best_reach = 0.0
    best_length = 0.0

    for line in lines:
        x1, y1, x2, y2 = line.flatten()

        dist1 = math.hypot(x1 - circle.center_x, y1 - circle.center_y)
        dist2 = math.hypot(x2 - circle.center_x, y2 - circle.center_y)

        # At least one endpoint must be near the pivot.
        if dist1 > pivot_distance and dist2 > pivot_distance:
            continue

        # Select by reachh (how far the line extends from the pivot), not raw segment length --
        # this favors thelong pointing arm over a short counterweight on the opposite side, even if 
        # edge detectoin only picks up a fragment of the arm.
        reach = max(dist1, dist2)
        if reach > best_reach:
            best_reach = reach
            best_length = math.hypot(x2 - x1, y2 -y1)
            if dist1 > dist2:
                best_line = (x1, y1)
            else:
                best_line = (x2, y2)

    if best_line is None:
        return NeedleDetection(found=False)

    tip_x, tip_y = best_line
    angle = math.atan2(tip_y - circle.center_y, tip_x - circle.center_x)

    return NeedleDetection(found=True, tip_x=tip_x, tip_y=tip_y, angle_rad=angle, line_length=best_length)

def draw_needle_overlay(
        image_path: str, circle: CircleDetection, needle: NeedleDetection, output_path: str
) -> None:
    """Save a copy of the image with the detected circle and needle line drawn on it."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at '{image_path}'")

    if circle.found:
        cv2.circle(img, (circle.center_x, circle.center_y), circle.radius, (0, 255, 0), 2)
        cv2.circle(img, (circle.center_x, circle.center_y), 5, (0, 0, 255), -1)

    if needle.found:
        cv2.line(img, (circle.center_x, circle.center_y), (needle.tip_x, needle.tip_y), (255, 0, 255), 3)
        cv2.circle(img, (needle.tip_x, needle.tip_y), 6, (255, 0, 255), -1)

    cv2.imwrite(output_path, img)