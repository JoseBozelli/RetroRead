"""
Pick one representative sample image from Endava DS5.0 for classical CV
baseline development — chosen programmatically, not hand-picked, to avoid
unconsciously selecting an "easy" case.

Selection method:
  1. Load train__kpts_coco.json, keep only images with all 4 expected
     keypoints present (needle tip, gauge center, scale min, scale max).
  2. Compute a sharpness score (variance of Laplacian) for each candidate
     image using OpenCV.
  3. Pick the image whose sharpness is closest to the MEDIAN across all
     candidates — i.e. a typical image, not the sharpest or blurriest.

Run from the repo root with:
    uv run python scripts/pick_sample_image.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

COCO_PATH = Path("data/raw/Endava/sample_synth_datasets/ds5.0/train__kpts_coco.json")
IMAGES_DIR = Path("data/raw/Endava/sample_synth_datasets/ds5.0")

EXPECTED_KEYPOINT_COUNT = 4  # needle tip, center, scale min, scale max


def load_complete_annotations(coco_path: Path) -> list[dict]:
    with coco_path.open() as f:
        coco = json.load(f)

    # Look up filename AND image dimensions per image_id. This will allow to check whether a gauge's bounding box
    # touches the image boarder.
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

        bbox = ann.get("bbox")  # [x, y, width, height], per COCO convention
        image_info = image_id_to_info.get(ann["image_id"])

        if n_visible >= EXPECTED_KEYPOINT_COUNT and bbox is not None and image_info is not None:
            complete.append(
                {
                    "image_id": ann["image_id"],
                    "file_name": image_info["file_name"],
                    "bbox": bbox,
                    "image_width": image_info["width"],
                    "image_height": image_info["height"]
                }
            )
    return complete

def bbox_touches_edge(candidate: dict, margin_fraction: float = 0.15) -> bool:
    """True if the gauge's bounding box comes within margin_fraction of any image boarder (i.e., the
    gauge may be cropped/cut off)."""
    x, y, w, h = candidate["bbox"]
    img_w, img_h = candidate["image_width"], candidate["image_height"]
    margin_x = margin_fraction * img_w
    margin_y = margin_fraction * img_h

    too_close_left = x < margin_x
    too_close_top = y < margin_y
    too_close_right = (x + w) > (img_w - margin_x)
    too_close_bottom = (y + h) > (img_h - margin_y)

    return too_close_left or too_close_top or too_close_right or too_close_bottom

def sharpness_score(image_path: Path) -> float:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return float("nan")
    return cv2.Laplacian(img, cv2.CV_64F).var()


def main() -> None:
    if not COCO_PATH.exists():
        raise FileNotFoundError(f"Expected annotations at '{COCO_PATH}', not found.")

    all_candidates = load_complete_annotations(COCO_PATH)
    candidates = [c for c in all_candidates if not bbox_touches_edge(c)]
    print(f"Images with all {EXPECTED_KEYPOINT_COUNT} keypoints labeled: {len(all_candidates)}")
    print(f"Of those, with gauge fully inside frame (not edge-cropped): {len(candidates)}")

    scored = []
    for c in candidates:
        img_path = IMAGES_DIR / c["file_name"]
        if not img_path.exists():
            continue
        score = sharpness_score(img_path)
        if not np.isnan(score):
            scored.append((c["file_name"], score))

    if not scored:
        raise RuntimeError("No valid candidate images found. Check IMAGES_DIR path.")

    scores_only = np.array([s for _, s in scored])
    median_score = np.median(scores_only)

    # Find the candidate whose sharpness is closest to the median
    scored.sort(key=lambda pair: abs(pair[1] - median_score))
    chosen_file, chosen_score = scored[0]

    print(f"\nSharpness range across {len(scored)} candidates: "
          f"{scores_only.min():.1f} to {scores_only.max():.1f}")
    print(f"Median sharpness: {median_score:.1f}")
    print(f"\nChosen image (closest to median): {chosen_file}")
    print(f"Its sharpness score: {chosen_score:.1f}")
    print(f"Full path: {IMAGES_DIR / chosen_file}")


if __name__ == "__main__":
    main()