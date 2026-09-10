"""Dataset for direct end-to-end regression (Experiment 28)."""

import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights

from retroread.crop_heatmap_dataset import compute_crop_box
from retroread.crop_heatmap_jitter_dataset import compute_jittered_crop_box

PREPROCESS = MobileNet_V3_Small_Weights.DEFAULT.transforms()


class DirectRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[dict], images_dir, jitter: bool):
        self.samples = samples  # each needs bbox, image_width, image_height, min_value, max_value, true_value
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

        scale_range = sample["max_value"] - sample["min_value"]
        norm_position = (sample["true_value"] - sample["min_value"]) / scale_range if scale_range > 0 else 0.0
        norm_position = max(0.0, min(1.0, norm_position))

        return image_tensor, torch.tensor([norm_position], dtype=torch.float32)