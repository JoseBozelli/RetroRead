"""
Dataset for the multi-task model (Experiment 24): needle mask + center
heatmap + min/max landmark heatmaps, all cropped consistently (jittered
for training).
"""

import numpy as np
import torch
from PIL import Image

from retroread.crop_heatmap_dataset import compute_crop_box, make_gaussian_heatmap
from retroread.crop_heatmap_jitter_dataset import compute_jittered_crop_box
from retroread.needle_segmentation_dataset import PREPROCESS, decode_rle_mask

MASK_SIZE = 112
HEATMAP_SIZE = 56
GAUSSIAN_SIGMA = 1.5


def build_multitask_samples(seg_samples: list[dict], keypoint_raw_samples: list[dict]) -> list[dict]:
    """Merges needle-segmentation samples with raw min/max keypoint pixel positions, by filename."""
    kp_by_filename = {s["file_name"]: s for s in keypoint_raw_samples}
    combined = []
    for sample in seg_samples:
        kp = kp_by_filename.get(sample["file_name"])
        if kp is None:
            continue
        merged = dict(sample)
        # keypoints_px order: [cx,cy, tx,ty, minx,miny, maxx,maxy]
        merged["min_x"], merged["min_y"] = kp["keypoints_px"][4], kp["keypoints_px"][5]
        merged["max_x"], merged["max_y"] = kp["keypoints_px"][6], kp["keypoints_px"][7]
        combined.append(merged)
    return combined


class MultiTaskDataset(torch.utils.data.Dataset):
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

        def norm_heatmap(px, py):
            nx = float(np.clip((px - crop_x0) / crop_w, 0.0, 1.0))
            ny = float(np.clip((py - crop_y0) / crop_h, 0.0, 1.0))
            return make_gaussian_heatmap(nx * HEATMAP_SIZE, ny * HEATMAP_SIZE, HEATMAP_SIZE, GAUSSIAN_SIGMA)

        center_heatmap = norm_heatmap(sample["center_x"], sample["center_y"])
        min_heatmap = norm_heatmap(sample["min_x"], sample["min_y"])
        max_heatmap = norm_heatmap(sample["max_x"], sample["max_y"])

        center_tensor = torch.from_numpy(center_heatmap).unsqueeze(0)
        landmark_tensor = torch.from_numpy(np.stack([min_heatmap, max_heatmap]))  # (2, H, W)

        return image_tensor, mask_tensor, center_tensor, landmark_tensor