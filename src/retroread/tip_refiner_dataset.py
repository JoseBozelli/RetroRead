"""
Dataset for the R2 tip refiner: crops a small, high-resolution ROI from
the ORIGINAL full-resolution image, centered on YOLO's own predicted tip
(jittered during training, per the 32-56px empirical error range from R1),
with a Gaussian heatmap target for the true tip's position within that ROI.
"""

import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from retroread.crop_heatmap_dataset import make_gaussian_heatmap

ROI_SIZE = 128           # pixels, cropped from the original full-resolution image
MODEL_INPUT_SIZE = 224   # resized up to match the pretrained backbone's expected input
HEATMAP_SIZE = 56
GAUSSIAN_SIGMA = 1.5
JITTER_RANGE = (32, 56)  # pixels, matches R1's useful radius range

PREPROCESS = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class TipRefinerDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[dict], images_dir, jitter: bool):
        """samples: list of {"file_name", "yolo_tip": [x,y], "gt_tip": [x,y]} in original image coords."""
        self.samples = samples
        self.images_dir = images_dir
        self.jitter = jitter

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(self.images_dir / sample["file_name"]).convert("RGB")

        yolo_x, yolo_y = sample["yolo_tip"]

        if self.jitter:
            angle = random.uniform(0, 2 * 3.14159265)
            magnitude = random.uniform(*JITTER_RANGE)
            roi_center_x = yolo_x + magnitude * np.cos(angle)
            roi_center_y = yolo_y + magnitude * np.sin(angle)
        else:
            roi_center_x, roi_center_y = yolo_x, yolo_y

        half = ROI_SIZE // 2
        roi_x0 = roi_center_x - half
        roi_y0 = roi_center_y - half

        roi = image.crop((int(roi_x0), int(roi_y0), int(roi_x0) + ROI_SIZE, int(roi_y0) + ROI_SIZE))
        input_tensor = PREPROCESS(roi)

        gt_x, gt_y = sample["gt_tip"]
        norm_x = float(np.clip((gt_x - roi_x0) / ROI_SIZE, 0.0, 1.0))
        norm_y = float(np.clip((gt_y - roi_y0) / ROI_SIZE, 0.0, 1.0))
        heatmap = make_gaussian_heatmap(norm_x * HEATMAP_SIZE, norm_y * HEATMAP_SIZE, HEATMAP_SIZE, GAUSSIAN_SIGMA)
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)

        return input_tensor, heatmap_tensor