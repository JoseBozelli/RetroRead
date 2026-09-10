"""
Dataset for the needle-segmentation model (Experiment 22): crops image and
needle mask together (jittered for training, matching Experiment 20's
crop-domain-shift fix), producing a fine-resolution mask target and a
gauge-center heatmap target.
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from retroread.crop_heatmap_dataset import compute_crop_box, make_gaussian_heatmap
from retroread.crop_heatmap_jitter_dataset import compute_jittered_crop_box

MASK_SIZE = 112
CENTER_HEATMAP_SIZE = 56
GAUSSIAN_SIGMA = 1.5

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def decode_rle_mask(rle) -> np.ndarray:
    from pycocotools import mask as mask_utils
    seg = dict(rle)
    if isinstance(seg["counts"], str):
        seg["counts"] = seg["counts"].encode("utf-8")
    return mask_utils.decode(seg)  # (H, W) uint8, values 0/1


class NeedleSegmentationDataset(torch.utils.data.Dataset):
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

        full_mask = decode_rle_mask(sample["needle_segmentation_rle"])

        if self.jitter:
            crop_x0, crop_y0, crop_x1, crop_y1 = compute_jittered_crop_box(
                sample["bbox"], image_width, image_height
            )
        else:
            crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_box(sample["bbox"], image_width, image_height)

        crop_x0i, crop_y0i, crop_x1i, crop_y1i = int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)
        crop_w, crop_h = crop_x1 - crop_x0, crop_y1 - crop_y0

        cropped_image = image.crop((crop_x0i, crop_y0i, crop_x1i, crop_y1i))
        image_tensor = PREPROCESS(cropped_image)

        cropped_mask = full_mask[crop_y0i:crop_y1i, crop_x0i:crop_x1i]
        mask_img = Image.fromarray((cropped_mask * 255).astype(np.uint8))
        mask_resized = mask_img.resize((MASK_SIZE, MASK_SIZE), Image.BILINEAR)
        mask_tensor = torch.from_numpy(np.array(mask_resized, dtype=np.float32) / 255.0).unsqueeze(0)

        center_norm_x = float(np.clip((sample["center_x"] - crop_x0) / crop_w, 0.0, 1.0))
        center_norm_y = float(np.clip((sample["center_y"] - crop_y0) / crop_h, 0.0, 1.0))
        center_heatmap = make_gaussian_heatmap(
            center_norm_x * CENTER_HEATMAP_SIZE, center_norm_y * CENTER_HEATMAP_SIZE,
            CENTER_HEATMAP_SIZE, GAUSSIAN_SIGMA,
        )
        center_heatmap_tensor = torch.from_numpy(center_heatmap).unsqueeze(0)

        return image_tensor, mask_tensor, center_heatmap_tensor