"""
Builds a YOLO-pose-format dataset (cropped images + label .txt files) from
the existing gauge keypoint annotations, for Experiment 30 (YOLOv8-pose
fine-tuning -- a genuinely different detection
paradigm from everything else tried in this project).

YOLO pose label format per line:
  class_id x_center y_center width height kpt1_x kpt1_y kpt1_v kpt2_x ... 
all normalized [0,1] relative to the image. Since each image is already
cropped to one gauge, the "object" bounding box is the whole crop.

Run from the repo root with:
    uv run python scripts/data_prep/build_yolo_pose_dataset.py
"""

import json
from pathlib import Path

from retroread.annotations import load_keypoint_training_data_raw
from retroread.config import ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_TRAIN_KPTS_COCO, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.crop_heatmap_dataset import compute_crop_box
from PIL import Image

OUTPUT_ROOT = Path("data/processed/yolo_pose")


def build_split(coco_path, split_name: str) -> None:
    samples = load_keypoint_training_data_raw(coco_path)
    images_dir = OUTPUT_ROOT / split_name / "images"
    labels_dir = OUTPUT_ROOT / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        image = Image.open(ENDAVA_DS5_IMAGES_DIR / sample["file_name"]).convert("RGB")
        crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_box(
            sample["bbox"], sample["image_width"], sample["image_height"]
        )
        crop_w, crop_h = crop_x1 - crop_x0, crop_y1 - crop_y0

        cropped = image.crop((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)))
        out_name = f"{split_name}_{i:04d}.png"
        cropped.save(images_dir / out_name)

        kp = sample["keypoints_px"]  # [cx,cy, tx,ty, minx,miny, maxx,maxy]
        norm_kpts = []
        for j in range(0, 8, 2):
            nx = max(0.0, min(1.0, (kp[j] - crop_x0) / crop_w))
            ny = max(0.0, min(1.0, (kp[j + 1] - crop_y0) / crop_h))
            norm_kpts += [f"{nx:.6f}", f"{ny:.6f}", "2"]  # visibility=2 (visible)

        # single object covering the whole crop
        label_line = "0 0.5 0.5 1.0 1.0 " + " ".join(norm_kpts)
        (labels_dir / out_name.replace(".png", ".txt")).write_text(label_line)

    print(f"{split_name}: {len(samples)} images/labels written to {images_dir.parent}")


def main() -> None:
    build_split(ENDAVA_DS5_TRAIN_KPTS_COCO, "train")
    build_split(ENDAVA_DS5_VAL_KPTS_COCO, "val")

    data_yaml = OUTPUT_ROOT / "data.yaml"
    data_yaml.write_text(f"""\
path: {OUTPUT_ROOT.resolve()}
train: train/images
val: val/images
kpt_shape: [4, 3]
names:
  0: gauge
""")
    print(f"\nWrote {data_yaml}")


if __name__ == "__main__":
    main()