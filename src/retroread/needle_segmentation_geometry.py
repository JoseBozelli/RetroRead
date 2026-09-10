"""
Geometric reading extraction from a predicted needle mask + center
heatmap: the tip is the mask foreground pixel FARTHEST from the predicted
center -- the same "reach from pivot" principle the classical needle
detector already uses (see needle_detection.py) -- rather than a single
predicted coordinate. Aggregating over potentially hundreds of mask pixels
is the core hypothesis being tested: more robust than one point.
"""

import numpy as np
import torch


def decode_center(center_logits: torch.Tensor) -> tuple[float, float]:
    """Hard-argmax peak of a single-image center heatmap. Returns normalized (x, y) in [0,1]."""
    probs = torch.sigmoid(center_logits)[0, 0]
    h, w = probs.shape
    idx = torch.argmax(probs.flatten()).item()
    y, x = idx // w, idx % w
    return x / (w - 1), y / (h - 1)


def fit_needle_angle(mask_logits: torch.Tensor, center_norm: tuple[float, float], threshold: float = 0.5) -> float | None:
    """
    mask_logits: (1, 1, H, W) raw logits for one image.
    Returns the needle angle in radians (atan2 convention matching the
    rest of the project), or None if too few foreground pixels to trust.
    """
    probs = torch.sigmoid(mask_logits)[0, 0].detach().numpy()
    h, w = probs.shape
    ys, xs = np.where(probs >= threshold)

    if len(xs) < 5:
        return None

    points_x = xs / (w - 1)
    points_y = ys / (h - 1)

    cx, cy = center_norm
    dists = np.hypot(points_x - cx, points_y - cy)
    tip_idx = np.argmax(dists)
    tip_x, tip_y = points_x[tip_idx], points_y[tip_idx]

    return float(np.arctan2(tip_y - cy, tip_x - cx))