"""
Training dataset for Experiment 20: applies random crop jitter (position +
scale perturbation) to approximate the classical circle detector's
imperfect crops at inference, rather than training only on idealized
ground-truth crops (see docs/deep_learning.md's crop-domain-shift finding).

Also provides a helper to build REAL detector-cropped samples (not
synthetic jitter) for validation, so checkpoint selection reflects actual
deployment conditions rather than an idealized proxy -- directly informed
by Experiment 15/16, where clean-crop validation loss improved while real
downstream accuracy got worse.
"""

import random

import numpy as np
import torch
from PIL import Image

from retroread.crop_heatmap_dataset import GAUSSIAN_SIGMA, PREPROCESS, make_gaussian_heatmap
from retroread.crop_heatmap_model import HEATMAP_SIZE, N_KEYPOINTS

POSITION_JITTER = 0.10  # fraction of bbox width/height
SCALE_JITTER = 0.15     # fraction of margin size


def compute_jittered_crop_box(bbox, image_width, image_height, margin_fraction=0.15, rng=None):
    rng = rng or random
    x, y, w, h = bbox

    scale_factor = 1.0 + rng.uniform(-SCALE_JITTER, SCALE_JITTER)
    margin_x = w * margin_fraction * scale_factor
    margin_y = h * margin_fraction * scale_factor

    shift_x = rng.uniform(-POSITION_JITTER, POSITION_JITTER) * w
    shift_y = rng.uniform(-POSITION_JITTER, POSITION_JITTER) * h

    x0 = max(0, x - margin_x + shift_x)
    y0 = max(0, y - margin_y + shift_y)
    x1 = min(image_width, x + w + margin_x + shift_x)
    y1 = min(image_height, y + h + margin_y + shift_y)
    return x0, y0, x1, y1


class CropGaugeHeatmapJitterDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[dict], images_dir):
        self.samples = samples
        self.images_dir = images_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(self.images_dir / sample["file_name"]).convert("RGB")

        crop_x0, crop_y0, crop_x1, crop_y1 = compute_jittered_crop_box(
            sample["bbox"], sample["image_width"], sample["image_height"]
        )
        crop_w, crop_h = crop_x1 - crop_x0, crop_y1 - crop_y0

        cropped = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        image_tensor = PREPROCESS(cropped)

        keypoints_px = sample["keypoints_px"]
        heatmaps = np.zeros((N_KEYPOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        norm_targets = []
        for k in range(N_KEYPOINTS):
            px, py = keypoints_px[k * 2], keypoints_px[k * 2 + 1]
            norm_x = float(np.clip((px - crop_x0) / crop_w, 0.0, 1.0))
            norm_y = float(np.clip((py - crop_y0) / crop_h, 0.0, 1.0))
            norm_targets.extend([norm_x, norm_y])
            grid_x, grid_y = norm_x * HEATMAP_SIZE, norm_y * HEATMAP_SIZE
            heatmaps[k] = make_gaussian_heatmap(grid_x, grid_y, HEATMAP_SIZE, GAUSSIAN_SIGMA)

        return image_tensor, torch.from_numpy(heatmaps), torch.tensor(norm_targets, dtype=torch.float32)


def build_detector_crop_samples(raw_samples: list[dict], images_dir) -> list[dict]:
    """
    Replaces each sample's ground-truth bbox with the classical circle
    detector's own detected region, precomputed once -- used to build a
    validation set that reflects real deployment crops, not idealized ones.
    Images where detection fails are skipped.
    """
    from retroread.classical_baseline import find_gauge_circle
    from retroread.params import CIRCLE_PARAMS

    result = []
    for sample in raw_samples:
        image_path = str(images_dir / sample["file_name"])
        circle = find_gauge_circle(image_path, **CIRCLE_PARAMS)
        if not circle.found:
            continue
        detector_bbox = [
            circle.center_x - circle.radius,
            circle.center_y - circle.radius,
            circle.radius * 2,
            circle.radius * 2,
        ]
        new_sample = dict(sample)
        new_sample["bbox"] = detector_bbox
        result.append(new_sample)
    return result