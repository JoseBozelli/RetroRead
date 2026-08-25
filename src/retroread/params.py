"""
Current best-known detection parameters, established via experiments 02 (circle) and 04 (needle).
See docs/decision_log.md for the tuning history.

Used by downstream scripts that need working detection as a known-good input to test something else (e.g.,
reading conversion), not by the tuning experiments themselves -- those intentionally keep their exact tested
values inline, so each historical experiment run stays self-contained and reproducible independent of later
changes here.
"""

CIRCLE_PARAMS = {
    "dp": 1.0,
    "min_dist_fraction": 0.5,
    "param1": 100,
    "param2": 50,
    "min_radius_fraction": 0.1,
    "max_radius_fraction": 0.35,
}

NEEDLE_PARAMS = {
    "inner_radius_fraction": 0.85,
    "pivot_distance_fraction": 0.25,
    "min_line_length_fraction": 0.3,
    "canny_low": 50,
    "canny_high": 150,
    "hough_threshold": 30,
    "max_line_gap": 10,
}