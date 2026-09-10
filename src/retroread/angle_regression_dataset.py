"""
Dataset for the sin/cos angle regression model.
"""

import math

import torch
from PIL import Image

from retroread.crop_heatmap_dataset import compute_crop_box
from retroread.crop_heatmap_jitter_dataset import compute_jittered_crop_box
from torchvision.models import MobileNet_V3_Small_Weights

PREPROCESS = MobileNet_V3_Small_Weights.DEFAULT.transforms()

class AngleRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[dict], images_dir, jitter: bool):
        self.samples = samples
        self.images_dir = images_dir
        self.jitter = jitter

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(self.images_dir / sample["file_name"]).convert("RGB")
        image_width, image_height = image.size

        if self.jitter:
            crop_x0, crop_y0, crop_x1, crop_y1 = compute_jittered_crop_box(
                sample["bbox"], image_width, image_height
            )
        else:
            crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_box(sample["bbox"], image_width, image_height)

        cropped = image.crop((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)))
        image_tensor = PREPROCESS(cropped)

        cx, cy, tx, ty = sample["keypoints_px"][0], sample["keypoints_px"][1], sample["keypoints_px"][2], sample["keypoints_px"][3]
        angle = math.atan2(ty - cy, tx - cx)
        target = torch.tensor([math.sin(angle), math.cos(angle)], dtype=torch.float32)
                              
        return image_tensor, target