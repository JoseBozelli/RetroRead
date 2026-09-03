"""
Extends CropGaugeHeatmapDataset with per-sample reading-calibration data
(scale min/max VALUES and the true final reading) needed for the
reading-aware auxiliary loss. Reuses the crop/heatmap logic unchanged.
"""

import torch

from retroread.crop_heatmap_dataset import CropGaugeHeatmapDataset, compute_crop_box, make_gaussian_heatmap
from retroread.crop_heatmap_model import HEATMAP_SIZE, N_KEYPOINTS
import numpy as np
from PIL import Image


def build_reading_samples(raw_samples: list[dict], reading_by_filename: dict) -> list[dict]:
    """Merges raw keypoint samples with reading-conversion data (scale min/max values, true reading)."""
    combined = []
    for sample in raw_samples:
        reading = reading_by_filename.get(sample["file_name"])
        if reading is None or len(reading["scale_labels"]) < 1:
            continue
        scale_values = [s["value"] for s in reading["scale_labels"]]
        merged = dict(sample)
        merged["min_value"] = min(scale_values)
        merged["max_value"] = max(scale_values)
        merged["true_reading"] = reading["true_value"]
        combined.append(merged)
    return combined


class CropGaugeReadingDataset(CropGaugeHeatmapDataset):
    def __getitem__(self, idx: int):
        image_tensor, heatmaps, coord_targets = super().__getitem__(idx)
        sample = self.samples[idx]
        extras = torch.tensor(
            [sample["min_value"], sample["max_value"], sample["true_reading"]], dtype=torch.float32
        )
        return image_tensor, heatmaps, coord_targets, extras