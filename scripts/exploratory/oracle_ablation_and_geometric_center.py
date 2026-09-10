"""
Two cheap, no-retraining diagnostics on the ORIGINAL (non-P2) YOLO-pose
checkpoint:

1. Oracle landmark ablation: substitute ground truth for one predicted
   landmark at a time (holding the rest as YOLO predicted), showing the
   causal upside of fixing each landmark independently.
2. Geometric center test: compare the classical circle detector's own
   center (already computed for cropping, free) against YOLO's predicted
   center, on the identical images -- if classical wins, that's an
   immediate, no-training hybrid improvement.

Run from the repo root with:
    uv run python scripts/exploratory/oracle_ablation_and_geometric_center.py
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

CHECKPOINT_PATH = "runs/pose/yolo_runs/experiment_30/weights/best.pt"
TOLERANCE_PCT = 5.0


def score(center, tip, reading_data) -> float | None:
    """Computes pct_error given a (center, tip) pair in ORIGINAL IMAGE pixel coords."""
    calibration = fit_scale_calibration(reading_data["center_x"], reading_data["center_y"], reading_data["scale_labels"])
    scale_values = [s["value"] for s in reading_data["scale_labels"]]
    scale_range = max(scale_values) - min(scale_values)
    if scale_range <= 0:
        return None
    angle = math.atan2(tip[1] - center[1], tip[0] - center[0])
    predicted = angle_to_value(angle, calibration)
    return abs(predicted - reading_data["true_value"]) / scale_range * 100


def summarize(name: str, errors: list[float]) -> None:
    n_accurate = sum(1 for e in errors if e <= TOLERANCE_PCT)
    mean_err = sum(errors) / len(errors) if errors else float("nan")
    print(f"{name:30} accuracy={n_accurate/len(errors):>6.1%}  mean_err={mean_err:>6.2f}%  (n={len(errors)})")


def main() -> None:
    edge_filtered = load_complete_annotations(ENDAVA_DS5_VAL_KPTS_COCO)
    edge_filtered = [c for c in edge_filtered if not bbox_touches_edge(c)]
    fair_filenames = {c["file_name"] for c in edge_filtered}

    raw_samples = load_keypoint_training_data_raw(ENDAVA_DS5_VAL_KPTS_COCO)
    raw_by_filename = {s["file_name"]: s for s in raw_samples if s["file_name"] in fair_filenames}

    reading_data_all = load_reading_annotations(ENDAVA_DS5_COCO)
    reading_by_filename = {d["file_name"]: d for d in reading_data_all.values()}

    model = YOLO(CHECKPOINT_PATH)

    baseline_errors = []
    gt_center_errors = []      # substitute GT center only
    gt_tip_errors = []          # substitute GT tip only
    geometric_center_errors = []  # substitute classical circle-detected center
    center_px_errors = {"yolo": [], "geometric": []}  # for the geometric-center comparison

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
        kpts = pred.keypoints.xy[0].tolist()  # crop-relative pixel coords: center, tip, min, max

        # Convert everything to ORIGINAL IMAGE coordinates for a consistent basis.
        pred_center = (kpts[0][0] + crop_x0, kpts[0][1] + crop_y0)
        pred_tip = (kpts[1][0] + crop_x0, kpts[1][1] + crop_y0)

        gt_center = (sample["keypoints_px"][0], sample["keypoints_px"][1])
        gt_tip = (sample["keypoints_px"][2], sample["keypoints_px"][3])

        # Classical circle detector's own center, already computed for cropping.
        geometric_center = (circle.center_x, circle.center_y)

        base = score(pred_center, pred_tip, reading)
        if base is None:
            continue
        baseline_errors.append(base)

        gt_center_errors.append(score(gt_center, pred_tip, reading))
        gt_tip_errors.append(score(pred_center, gt_tip, reading))
        geometric_center_errors.append(score(geometric_center, pred_tip, reading))

        center_px_errors["yolo"].append(math.hypot(pred_center[0] - gt_center[0], pred_center[1] - gt_center[1]))
        center_px_errors["geometric"].append(math.hypot(geometric_center[0] - gt_center[0], geometric_center[1] - gt_center[1]))

    print("=== Oracle landmark ablation (YOLO baseline vs. substituting ground truth) ===")
    summarize("YOLO baseline (both predicted)", baseline_errors)
    summarize("GT center substituted", gt_center_errors)
    summarize("GT tip substituted", gt_tip_errors)

    print("\n=== Geometric (classical) center vs. YOLO-predicted center ===")
    summarize("YOLO center + YOLO tip (baseline)", baseline_errors)
    summarize("Classical geometric center + YOLO tip", geometric_center_errors)

    def median(vals):
        s = sorted(vals)
        return s[len(s) // 2] if s else float("nan")

    print(f"\nCenter px error -- YOLO median: {median(center_px_errors['yolo']):.2f}  "
          f"Classical geometric median: {median(center_px_errors['geometric']):.2f}")


if __name__ == "__main__":
    main()