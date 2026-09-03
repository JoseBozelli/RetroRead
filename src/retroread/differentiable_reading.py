"""
Differentiable angle -> reading conversion, used as an auxiliary training
loss so the model is directly supervised on the metric that actually
matters (the final reading), not just raw keypoint pixel distance.

Simplified vs. the classical pipeline's calibration (reading_conversion.py):
uses only the two scale-endpoint keypoints (min, max) and their known
numeric values, since the model predicts exactly those two points, not a
multi-label scale like the classical baseline's calibration uses.

Calibration anchor (center, min, max) is ground truth, not predicted --
isolates this experiment to "does explicit reading supervision improve the
model's center/tip prediction," matching the classical baseline's own
practice of using a ground-truth center as a clean calibration anchor.
"""

import math

import torch


def angular_reading_loss(
    pred_center: torch.Tensor,
    pred_tip: torch.Tensor,
    gt_center: torch.Tensor,
    gt_min: torch.Tensor,
    gt_max: torch.Tensor,
    min_value: torch.Tensor,
    max_value: torch.Tensor,
    true_reading: torch.Tensor,
) -> torch.Tensor:
    """
    All tensors are (batch, 2) except min_value/max_value/true_reading,
    which are (batch,). Returns the mean absolute reading error, expressed
    as a fraction of each sample's own scale range -- keeping this loss
    term on a comparable, bounded scale to the coordinate loss, and
    matching the project's existing "% of scale range" accuracy metric.
    """
    batch_size = pred_center.shape[0]
    two_pi = 2 * math.pi
    fraction_errors = []

    for i in range(batch_size):
        cx, cy = gt_center[i, 0], gt_center[i, 1]

        pred_angle = torch.atan2(pred_tip[i, 1] - pred_center[i, 1], pred_tip[i, 0] - pred_center[i, 0])
        min_angle = torch.atan2(gt_min[i, 1] - cy, gt_min[i, 0] - cx)
        max_angle = torch.atan2(gt_max[i, 1] - cy, gt_max[i, 0] - cx)

        # Unwrap: choose the 2*pi shift of pred_angle/max_angle that lands
        # closest to min_angle. The selection (round) has no gradient, but
        # everything else in the expression stays differentiable -- same
        # practical approach as reading_conversion.py's numpy version.
        k_pred = torch.round((min_angle - pred_angle) / two_pi)
        unwrapped_pred = pred_angle + k_pred * two_pi

        k_max = torch.round((min_angle - max_angle) / two_pi)
        unwrapped_max = max_angle + k_max * two_pi

        span = unwrapped_max - min_angle
        span = torch.where(span.abs() < 1e-6, torch.full_like(span, 1e-6), span)

        fraction = (unwrapped_pred - min_angle) / span
        predicted_reading = min_value[i] + fraction * (max_value[i] - min_value[i])

        scale_range = (max_value[i] - min_value[i]).abs()
        scale_range = torch.where(scale_range < 1e-6, torch.full_like(scale_range, 1e-6), scale_range)
        fraction_error = (predicted_reading - true_reading[i]).abs() / scale_range
        fraction_errors.append(fraction_error)

    return torch.stack(fraction_errors).mean()