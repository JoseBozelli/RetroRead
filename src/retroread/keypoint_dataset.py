"""
PyTorch Dataset wrapper for gauge keypoint training data.

A Dataset is PyTorch's standard interface for "a collection of (input, target) pairs a training loop can pull from" --
implementing__len__ (how many examples) and __getitem__ (fetch example i) is all PyTorch requires.
"""

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Custom preprocessing: resize only, NO crop. The pretrained weight's default transform crops to 224x224, which
# can cut off keypoints that sit off-center -- since our target coordinates are normalized against the FULL original
# image, any crop invalidates that mapping. A pure resize preserves it: a point at normalized (x,y) in the original
# stays at (x,y) after uniform resizing, since nothing is cut away.

PREPROCESS = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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