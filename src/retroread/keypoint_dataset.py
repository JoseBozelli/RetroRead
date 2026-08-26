"""
PyTorch Dataset wrapper for gauge keypoint training data.

A Dataset is PyTorch's standard interface for "a collection of (input, target) pairs a training loop can pull from" --
implementing__len__ (how many examples) and __getitem__ (fetch example i) is all PyTorch requires.
"""

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V3_Small_Weights

# Same preprocessing the pretrained backbone expects -- resize, crop, normalize pixel values to match what it was
# orignally trained on.

PREPROCESS = MobileNet_V3_Small_Weights.DEFAULT.transforms()

class GaugeKeypointDataset(Dataset):
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
        target_tensor = torch.tensor(sample["target"], dtype=torch.float32)

        return image_tensor, target_tensor