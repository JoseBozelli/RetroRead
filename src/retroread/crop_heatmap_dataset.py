"""
PyTorch Dataset for the crop-based, two-stage heatmap model: crops each image to the gauge region (ground-truth
bbox + margin, matching the project's known ~15% bezel-margin finding from the classical baseline) before generating
heatmap targets -- so the model localizes keypoints within a tightly-framed gauge, not a full cluttered scene.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from retroread.crop_heatmap_model import HEATMAP_SIZE, N_KEYPOINTS

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

GAUSSIAN_SIGMA = 1.5
MARGIN_FRACTION = 0.15  # matches the classical baseline's known bezel-margin finding

def make_gaussian_heatmap(cx: float, cy: float, size: int, sigma: float) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    exponent = -((x - cx)**2 + (y - cy)**2) / (2 * sigma ** 2)
    exponent = np.clip(exponent, -50, 0)    # avoid misleading float32 cast warning on deep underflow
    return np.exp(exponent).astype(np.float32)

def compute_crop_box(bbox, image_width, image_height, margin_fraction=MARGIN_FRACTION):
    x, y, w, h = bbox
    margin_x = w * margin_fraction
    margin_y = h * margin_fraction
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(image_width, x + w + margin_x)
    y1 = min(image_height, y + h + margin_y)
    return x0, y0, x1, y1

class CropGaugeHeatmapDataset(Dataset):
    def __init__(self, samples: list[dict], images_dir):
        self.samples = samples
        self.images_dir = images_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(self.images_dir / sample["file_name"]).convert("RGB")

        crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_box(
            sample["bbox"], sample["image_width"], sample["image_height"]
        )
        crop_w = crop_x1 - crop_x0
        crop_h = crop_y1 - crop_y0

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

        return (
            image_tensor,
            torch.from_numpy(heatmaps),
            torch.tensor(norm_targets, dtype=torch.float32)
        )