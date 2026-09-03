"""
PyTorch Dataset for heatmap-based keypoint training: generates a 2D Gaussian "target heatmap" per keypoint,
instead of a raw coordinate vector.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from retroread.heatmap_model import HEATMAP_SIZE, N_KEYPOINTS

# Resize only, no crop -- see keypoint_dataset.py for why this matters: target coordinates are normalized
# against the FULL original image, and any crop would invalidate that mapping.
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

GAUSSIAN_SIGMA = 1.5    # in output-grid pixels (of a 28x28 grid), not original image pixels

def make_gaussian_heatmap(cx: float, cy: float, size: int, sigma: float) -> np.ndarray:
    """cx, cy: center position in grid pixel units (0 to size-1)."""
    y, x = np.mgrid[0:size, 0:size]
    exponent = -((x - cx)**2 + (y - cy)**2) / (2 * sigma **2)
    # Clip before exponentiating: without this, points far from the center produce values so close
    # to zero that casting to float32 triggers a misleading label "overflow" warning (it's actually underflow).
    exponent = np.clip(exponent, -50, 0)
    heatmap = np.exp(exponent)
    return heatmap.astype(np.float32)

class GaugeHeatmapDataset(Dataset):
    def __init__(self, samples: list[dict], images_dir):
        self.samples = samples
        self.images_dir = images_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = self.images_dir / sample["file_name"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = PREPROCESS(image)

        # sample["target"] = [cx, cy, tx, ty, minx, miny, maxx, maxy], normalized [0,1]
        target = sample["target"]
        heatmaps = np.zeros((N_KEYPOINTS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        for k in range(N_KEYPOINTS):
            norm_x, norm_y = target[k * 2], target[k * 2 + 1]
            grid_x = norm_x * HEATMAP_SIZE
            grid_y = norm_y * HEATMAP_SIZE
            heatmaps[k] = make_gaussian_heatmap(grid_x, grid_y, HEATMAP_SIZE, GAUSSIAN_SIGMA)

        return image_tensor, torch.from_numpy(heatmaps)