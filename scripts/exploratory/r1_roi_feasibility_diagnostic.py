"""
R1 -- ROI feasibility diagnostic (no training): how often does the true
tip fall within various radii of YOLO's coarse predicted tip, and what
reading accuracy would a "perfect-within-ROI" refiner achieve at each
radius? Determines whether a local tip refiner is worth building before
writing any refiner code.

Run from the repo root with:
    uv run python scripts/exploratory/r1_roi_feasibility_diagnostic.py
"""

import math

from PIL import Image
from ultralytics import YOLO

from retroread.annotations import bbox_touches_edge, load_complete_annotations, load_keypoint_training_data_raw, load_reading_annotations
from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.crop_heatmap_dataset import compute_crop_box
from retroread.params import CIRCLE_PARAMS
from retroread.reading_conversion import angle_to_value, fit_scale_calibration

YOLO_CHECKPOINT = "runs/pose/yolo_runs/experiment_30/weights/best.pt"
TOLERANCE_PCT = 5.0
RADII = [16, 24, 32, 48, 64]


def score(center, tip, reading_data) -> float | None:
    calibration = fit_scale_calibration(reading_data["center_x"], reading_data["center_y"], reading_data["scale_labels"])
    scale_values = [s["value"] for s in reading_data["scale_labels"]]
    scale_range = max(scale_values) - min(scale_values)
    if scale_range <= 0:
        return None
    angle = math.atan2(tip[1] - center[1], tip[0] - center[0])
    predicted = angle_to_value(angle, calibration)
    return abs(predicted - reading_data["true_value"]) / scale_range * 100


def main() -> None:
    edge_filtered = load_complete_annotations(ENDAVA_DS5_VAL_KPTS_COCO)
    edge_filtered = [c for c in edge_filtered if not bbox_touches_edge(c)]
    fair_filenames = {c["file_name"] for c in edge_filtered}

    raw_samples = load_keypoint_training_data_raw(ENDAVA_DS5_VAL_KPTS_COCO)
    raw_by_filename = {s["file_name"]: s for s in raw_samples if s["file_name"] in fair_filenames}
    reading_by_filename = {d["file_name"]: d for d in load_reading_annotations(ENDAVA_DS5_COCO).values()}

    model = YOLO(YOLO_CHECKPOINT)

    records = []  # (yolo_center, yolo_tip, gt_tip, tip_dist, reading)

    for file_name, sample in raw_by_filename.items():
        reading = reading_by_filename.get(file_name)
        if reading is None:
            continue

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
        kpts = pred.keypoints.xy[0].tolist()
        yolo_center = (kpts[0][0] + crop_x0, kpts[0][1] + crop_y0)
        yolo_tip = (kpts[1][0] + crop_x0, kpts[1][1] + crop_y0)
        gt_tip = (sample["keypoints_px"][2], sample["keypoints_px"][3])

        tip_dist = math.hypot(yolo_tip[0] - gt_tip[0], yolo_tip[1] - gt_tip[1])
        records.append((yolo_center, yolo_tip, gt_tip, tip_dist, reading))

    n = len(records)
    print(f"n = {n}\n")

    baseline_errors = [score(c, t, r) for c, t, gt, d, r in records]
    baseline_errors = [e for e in baseline_errors if e is not None]
    n_acc = sum(1 for e in baseline_errors if e <= TOLERANCE_PCT)
    print(f"YOLO baseline: accuracy={n_acc/len(baseline_errors):.1%}  mean_err={sum(baseline_errors)/len(baseline_errors):.2f}%\n")

    print(f"{'ROI radius':>10} {'GT tip contained':>18} {'Oracle-within-ROI accuracy':>28} {'Oracle mean err':>18}")
    for radius in RADII:
        errors = []
        n_contained = 0
        for center, yolo_tip, gt_tip, dist, reading in records:
            if dist <= radius:
                n_contained += 1
                use_tip = gt_tip
            else:
                use_tip = yolo_tip
            e = score(center, use_tip, reading)
            if e is not None:
                errors.append(e)
        n_acc = sum(1 for e in errors if e <= TOLERANCE_PCT)
        mean_err = sum(errors) / len(errors)
        print(f"{radius:>10}px {n_contained/n:>17.1%} {n_acc/len(errors):>27.1%} {mean_err:>17.2f}%")


if __name__ == "__main__":
    main()