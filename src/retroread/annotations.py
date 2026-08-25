"""
Utilities for loading and filtering the Endava COCO-format gauge annotations.
Shared y scripts/pick_sample_image.py and scripts/experiment_01_batch_circle_detection.py to
avoid duplicating the annotation-parsing logic in two places.
"""
import json
from pathlib import Path

EXPECTED_KEYPOINT_COUNT = 4     # neeedle tip, center, scale min, and scale max

def load_complete_annotations(coco_path: Path) -> list[dict]:
    """
    Load COCO annotations, keeping only entries with all expected keypoints labeled and visible,
    plus a bbox and image dimensions attached.
    """
    with coco_path.open() as f:
        coco = json.load(f)

    image_id_to_info = {
        img["id"]: {"file_name": img["file_name"], "width": img["width"], "height": img["height"]}
        for img in coco["images"]
    }

    complete = []
    for ann in coco["annotations"]:
        keypoints = ann.get("keypoints", [])
        # COCO keypoints are stored as flat [x1, y1, v1, x2, y2, v2, ...];
        # v (visibility) == 2 means "labeled and visible".
        n_visible = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] == 2)

        bbox = ann.get("bbox")  # [x, y, width, height]; covers dial face only, not full bezel
        image_info = image_id_to_info.get(ann["image_id"])

        if n_visible >= EXPECTED_KEYPOINT_COUNT and bbox is not None and image_info is not None:
            # Keypoint order per COCO categories definition (face_plate):
            # index 0: dial_max, index 1: dial_min, index 2: dial_center, index 3: dial_tip.
            # Each keypoint is (x, y, visibility) -- 3 values per point.
            gt_center_x, gt_center_y = keypoints[6], keypoints[7]
            gt_tip_x, gt_tip_y = keypoints[9], keypoints[10]

            complete.append(
                {
                    "image_id": ann["image_id"],
                    "file_name": image_info["file_name"],
                    "bbox": bbox,
                    "image_width": image_info["width"],
                    "image_height": image_info["height"],
                    "gt_center_x": gt_center_x,
                    "gt_center_y": gt_center_y,
                    "gt_tip_x": gt_tip_x,
                    "gt_tip_y": gt_tip_y
                }
            )
            
    return complete

def bbox_touches_edge(candidate: dict, margin_fraction: float = 0.15) -> bool:
    """
    True if the gauge's bounding box comes within margin_fraction of an image boarder. Margin is
    generous (default 15%) because the bbox covers only the dial face, not the full bezel, so its
    true extent past the box is unknown.
    """
    x, y, w, h = candidate["bbox"]
    img_w, img_h = candidate["image_width"], candidate["image_height"]
    margin_x = margin_fraction * img_w
    margin_y = margin_fraction * img_h

    too_close_left = x < margin_x
    too_close_top = y < margin_y
    too_close_right = (x + w) > (img_w - margin_x)
    too_close_bottom = (y + h) > (img_h - margin_y)

    return too_close_left or too_close_top or too_close_right or too_close_bottom