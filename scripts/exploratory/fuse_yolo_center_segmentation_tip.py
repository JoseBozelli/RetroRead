"""
Fuses YOLO's center prediction with the needle-segmentation model's
reach-based tip estimate -- both already trained, no new training. Tests
multiple pairings to avoid the "correlated error" trap just found:
independently-sourced points don't necessarily combine well.

Run from the repo root with:
    uv run python scripts/exploratory/fuse_yolo_center_segmentation_tip.py
"""

import math

import torch
from PIL import Image
from ultralytics import YOLO

from retroread.annotations import bbox_touches_edge, load_complete_annotations, load_keypoint_training_data_raw, load_reading_annotations
from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.crop_heatmap_dataset import compute_crop_box
from retroread.needle_segmentation_dataset import PREPROCESS as SEG_PREPROCESS
from retroread.needle_segmentation_geometry import decode_center as seg_decode_center
from retroread.needle_segmentation_model import NeedleSegmentationModel
from retroread.params import CIRCLE_PARAMS
from retroread.reading_conversion import angle_to_value, fit_scale_calibration

YOLO_CHECKPOINT = "runs/pose/yolo_runs/experiment_30/weights/best.pt"
SEG_CHECKPOINT = "best_needle_segmentation_model.pt"
TOLERANCE_PCT = 5.0


def score(center, tip, reading_data) -> float | None:
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
    print(f"{name:40} accuracy={n_accurate/len(errors):>6.1%}  mean_err={mean_err:>6.2f}%  (n={len(errors)})")


def mask_tip_from_center(mask_logits, center_px, image_size, threshold=0.5):
    """Reach-based tip: mask pixel farthest from a GIVEN center (crop pixel coords)."""
    import numpy as np
    probs = torch.sigmoid(mask_logits)[0, 0].detach().numpy()
    h, w = probs.shape
    ys, xs = np.where(probs >= threshold)
    if len(xs) < 5:
        return None
    scale_x, scale_y = w / image_size[0], h / image_size[1]
    px_x, px_y = xs / scale_x, ys / scale_y  # back to crop pixel space
    dists = np.hypot(px_x - center_px[0], px_y - center_px[1])
    idx = np.argmax(dists)
    return float(px_x[idx]), float(px_y[idx])


def main() -> None:
    edge_filtered = load_complete_annotations(ENDAVA_DS5_VAL_KPTS_COCO)
    edge_filtered = [c for c in edge_filtered if not bbox_touches_edge(c)]
    fair_filenames = {c["file_name"] for c in edge_filtered}

    raw_samples = load_keypoint_training_data_raw(ENDAVA_DS5_VAL_KPTS_COCO)
    raw_by_filename = {s["file_name"]: s for s in raw_samples if s["file_name"] in fair_filenames}
    reading_by_filename = {d["file_name"]: d for d in load_reading_annotations(ENDAVA_DS5_COCO).values()}

    yolo_model = YOLO(YOLO_CHECKPOINT)
    seg_model = NeedleSegmentationModel(freeze_backbone=True)
    seg_model.load_state_dict(torch.load(SEG_CHECKPOINT))
    seg_model.eval()

    baseline_errors = []          # YOLO center + YOLO tip
    yolo_center_seg_tip_errors = []  # YOLO center + segmentation-mask tip (reach from YOLO center)
    seg_only_errors = []          # segmentation's own center + segmentation tip

    with torch.no_grad():
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
            crop_w, crop_h = crop_x1 - crop_x0, crop_y1 - crop_y0
            cropped = image.crop((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)))

            # --- YOLO prediction ---
            yolo_pred = yolo_model.predict(cropped, verbose=False)[0]
            if yolo_pred.keypoints is None or len(yolo_pred.keypoints.xy) == 0:
                continue
            kpts = yolo_pred.keypoints.xy[0].tolist()
            yolo_center_crop = (kpts[0][0], kpts[0][1])
            yolo_tip_crop = (kpts[1][0], kpts[1][1])
            yolo_center_full = (yolo_center_crop[0] + crop_x0, yolo_center_crop[1] + crop_y0)
            yolo_tip_full = (yolo_tip_crop[0] + crop_x0, yolo_tip_crop[1] + crop_y0)

            # --- Segmentation prediction ---
            seg_input = SEG_PREPROCESS(cropped).unsqueeze(0)
            mask_logits, center_logits = seg_model(seg_input)
            seg_center_norm = seg_decode_center(center_logits)
            seg_center_crop = (seg_center_norm[0] * crop_w, seg_center_norm[1] * crop_h)
            seg_center_full = (seg_center_crop[0] + crop_x0, seg_center_crop[1] + crop_y0)

            # Segmentation tip, reach from YOLO's center (crop coords, then convert)
            tip_from_yolo_center = mask_tip_from_center(mask_logits, yolo_center_crop, (crop_w, crop_h))
            # Segmentation tip, reach from segmentation's own center
            tip_from_seg_center = mask_tip_from_center(mask_logits, seg_center_crop, (crop_w, crop_h))

            base = score(yolo_center_full, yolo_tip_full, reading)
            if base is not None:
                baseline_errors.append(base)

            if tip_from_yolo_center is not None:
                tip_full = (tip_from_yolo_center[0] + crop_x0, tip_from_yolo_center[1] + crop_y0)
                e = score(yolo_center_full, tip_full, reading)
                if e is not None:
                    yolo_center_seg_tip_errors.append(e)

            if tip_from_seg_center is not None:
                tip_full = (tip_from_seg_center[0] + crop_x0, tip_from_seg_center[1] + crop_y0)
                e = score(seg_center_full, tip_full, reading)
                if e is not None:
                    seg_only_errors.append(e)

    summarize("YOLO center + YOLO tip (baseline)", baseline_errors)
    summarize("YOLO center + segmentation tip", yolo_center_seg_tip_errors)
    summarize("Segmentation center + segmentation tip", seg_only_errors)


if __name__ == "__main__":
    main()