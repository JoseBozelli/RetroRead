"""
Y1 diagnostic (no retraining): per-keypoint pixel error and per-image
angle error for the trained YOLO-pose model, stratified by whether the
final reading was accurate -- identifies which keypoint actually drives
the 27.9% failures, without committing to any further architecture work.

Run from the repo root with:
    uv run python scripts/exploratory/diagnose_yolo_keypoint_errors.py
"""

import csv
import math

from PIL import Image
from ultralytics import YOLO

from retroread.annotations import bbox_touches_edge, load_complete_annotations, load_keypoint_training_data_raw
from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.crop_heatmap_dataset import compute_crop_box
from retroread.params import CIRCLE_PARAMS

CHECKPOINT_PATH = "runs/pose/yolo_runs/experiment_30/weights/best.pt"
KEYPOINT_NAMES = ["center", "tip", "min", "max"]


def main() -> None:
    edge_filtered = load_complete_annotations(ENDAVA_DS5_VAL_KPTS_COCO)
    edge_filtered = [c for c in edge_filtered if not bbox_touches_edge(c)]
    fair_filenames = {c["file_name"] for c in edge_filtered}

    raw_samples = load_keypoint_training_data_raw(ENDAVA_DS5_VAL_KPTS_COCO)
    raw_by_filename = {s["file_name"]: s for s in raw_samples if s["file_name"] in fair_filenames}

    with open("experiment_31_results.csv") as f:
        accuracy_by_filename = {row["file_name"]: row["accurate"] == "True" for row in csv.DictReader(f)}

    model = YOLO(CHECKPOINT_PATH)

    per_keypoint_errors = {name: {"accurate": [], "inaccurate": []} for name in KEYPOINT_NAMES}
    angle_errors = {"accurate": [], "inaccurate": []}

    for file_name, sample in raw_by_filename.items():
        image_path = ENDAVA_DS5_IMAGES_DIR / file_name
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        circle = find_gauge_circle(str(image_path), **CIRCLE_PARAMS)
        if not circle.found:
            continue
        circle_bbox = [circle.center_x - circle.radius, circle.center_y - circle.radius,
                        circle.radius * 2, circle.radius * 2]
        crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_box(circle_bbox, image_width, image_height)
        cropped = image.crop((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)))

        pred = model.predict(cropped, verbose=False)[0]
        if pred.keypoints is None or len(pred.keypoints.xy) == 0:
            continue
        pred_kpts = pred.keypoints.xy[0].tolist()

        kp = sample["keypoints_px"]
        gt_kpts = [
            (kp[0] - crop_x0, kp[1] - crop_y0),
            (kp[2] - crop_x0, kp[3] - crop_y0),
            (kp[4] - crop_x0, kp[5] - crop_y0),
            (kp[6] - crop_x0, kp[7] - crop_y0),
        ]

        bucket = "accurate" if accuracy_by_filename.get(file_name, False) else "inaccurate"

        for i, name in enumerate(KEYPOINT_NAMES):
            err = math.hypot(pred_kpts[i][0] - gt_kpts[i][0], pred_kpts[i][1] - gt_kpts[i][1])
            per_keypoint_errors[name][bucket].append(err)

        pred_angle = math.atan2(pred_kpts[1][1] - pred_kpts[0][1], pred_kpts[1][0] - pred_kpts[0][0])
        gt_angle = math.atan2(gt_kpts[1][1] - gt_kpts[0][1], gt_kpts[1][0] - gt_kpts[0][0])
        angle_err_deg = abs(math.degrees(pred_angle - gt_angle))
        angle_err_deg = min(angle_err_deg, 360 - angle_err_deg)
        angle_errors[bucket].append(angle_err_deg)

    def median(vals):
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n else float("nan")

    print(f"{'Keypoint':10} {'Median px err (accurate)':>26} {'Median px err (inaccurate)':>28}")
    for name in KEYPOINT_NAMES:
        acc = median(per_keypoint_errors[name]["accurate"])
        inacc = median(per_keypoint_errors[name]["inaccurate"])
        print(f"{name:10} {acc:>26.2f} {inacc:>28.2f}")

    print(f"\nAngle error (deg): accurate median={median(angle_errors['accurate']):.2f}  "
          f"inaccurate median={median(angle_errors['inaccurate']):.2f}")
    print(f"n accurate={len(angle_errors['accurate'])}  n inaccurate={len(angle_errors['inaccurate'])}")


if __name__ == "__main__":
    main()