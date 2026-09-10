"""
Final analysis: paired per-image comparison of Classical CV vs. YOLO+Refiner
on the identical 61-image fair subset -- do they fail on the same images or
different ones? If complementary, simulates a simple agreement/confidence
gate to see whether a zero-additional-training hybrid beats either system
alone.

Run from the repo root with:
    uv run python scripts/exploratory/paired_error_analysis_and_gating.py
"""

import math

from PIL import Image
from ultralytics import YOLO

from retroread.annotations import bbox_touches_edge, load_complete_annotations, load_reading_annotations
from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.crop_heatmap_dataset import compute_crop_box
from retroread.needle_detection import detect_needle
from retroread.params import CIRCLE_PARAMS, NEEDLE_PARAMS
from retroread.predict import compute_confidence, image_sharpness
from retroread.reading_conversion import angle_to_value, fit_scale_calibration
from retroread.tip_refiner_dataset import HEATMAP_SIZE, ROI_SIZE, PREPROCESS as REFINER_PREPROCESS
from retroread.tip_refiner_model import TipRefinerModel
import torch

YOLO_CHECKPOINT = "runs/pose/yolo_runs/experiment_30/weights/best.pt"
REFINER_CHECKPOINT = "best_tip_refiner_model.pt"
TOLERANCE_PCT = 5.0
AGREEMENT_THRESHOLD_PCT = 5.0  # readings within this % of scale range count as "agreeing"


def refine_tip(refiner_model, image, yolo_tip):
    half = ROI_SIZE // 2
    roi_x0, roi_y0 = yolo_tip[0] - half, yolo_tip[1] - half
    roi = image.crop((int(roi_x0), int(roi_y0), int(roi_x0) + ROI_SIZE, int(roi_y0) + ROI_SIZE))
    input_tensor = REFINER_PREPROCESS(roi).unsqueeze(0)
    with torch.no_grad():
        heatmap = torch.sigmoid(refiner_model(input_tensor))[0, 0]
    idx = torch.argmax(heatmap.flatten()).item()
    hy, hx = idx // HEATMAP_SIZE, idx % HEATMAP_SIZE
    norm_x, norm_y = hx / (HEATMAP_SIZE - 1), hy / (HEATMAP_SIZE - 1)
    return roi_x0 + norm_x * ROI_SIZE, roi_y0 + norm_y * ROI_SIZE


def main() -> None:
    edge_filtered = load_complete_annotations(ENDAVA_DS5_VAL_KPTS_COCO)
    edge_filtered = [c for c in edge_filtered if not bbox_touches_edge(c)]
    fair_filenames = {c["file_name"] for c in edge_filtered}

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    reading_by_filename = {d["file_name"]: d for d in reading_data.values()}
    candidates = [reading_by_filename[fn] for fn in fair_filenames if fn in reading_by_filename]
    print(f"Comparing on {len(candidates)} images.\n")

    yolo_model = YOLO(YOLO_CHECKPOINT)
    refiner_model = TipRefinerModel(freeze_backbone=True)
    refiner_model.load_state_dict(torch.load(REFINER_CHECKPOINT))
    refiner_model.eval()

    both_accurate = only_classical = only_yolo = neither = 0
    classical_wins = yolo_wins = ties = 0
    agree_count = 0
    gate_errors = []

    for reading in candidates:
        image_path = ENDAVA_DS5_IMAGES_DIR / reading["file_name"]
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        circle = find_gauge_circle(str(image_path), **CIRCLE_PARAMS)
        if not circle.found:
            continue
        circle_bbox = [circle.center_x - circle.radius, circle.center_y - circle.radius,
                        circle.radius * 2, circle.radius * 2]
        crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_box(circle_bbox, image_width, image_height)
        cropped = image.crop((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)))

        calibration = fit_scale_calibration(reading["center_x"], reading["center_y"], reading["scale_labels"])
        scale_values = [s["value"] for s in reading["scale_labels"]]
        scale_range = max(scale_values) - min(scale_values)
        if scale_range <= 0:
            continue

        # --- Classical ---
        needle = detect_needle(str(image_path), circle, **NEEDLE_PARAMS)
        classical_val = None
        classical_confidence = 0.0
        if needle.found:
            classical_val = angle_to_value(needle.angle_rad, calibration)
            sharpness = image_sharpness(str(image_path))
            classical_confidence = compute_confidence(circle, needle, sharpness)
        classical_err = abs(classical_val - reading["true_value"]) / scale_range * 100 if classical_val is not None else None

        # --- YOLO + refiner ---
        pred = yolo_model.predict(cropped, verbose=False)[0]
        yolo_val = None
        if pred.keypoints is not None and len(pred.keypoints.xy) > 0:
            kpts = pred.keypoints.xy[0].tolist()
            yolo_center = (kpts[0][0] + crop_x0, kpts[0][1] + crop_y0)
            yolo_tip = (kpts[1][0] + crop_x0, kpts[1][1] + crop_y0)
            refined_tip = refine_tip(refiner_model, image, yolo_tip)
            angle = math.atan2(refined_tip[1] - yolo_center[1], refined_tip[0] - yolo_center[0])
            yolo_val = angle_to_value(angle, calibration)
        yolo_err = abs(yolo_val - reading["true_value"]) / scale_range * 100 if yolo_val is not None else None

        if classical_err is None or yolo_err is None:
            continue

        c_acc, y_acc = classical_err <= TOLERANCE_PCT, yolo_err <= TOLERANCE_PCT
        if c_acc and y_acc:
            both_accurate += 1
        elif c_acc:
            only_classical += 1
        elif y_acc:
            only_yolo += 1
        else:
            neither += 1

        if abs(classical_err - yolo_err) < 0.5:
            ties += 1
        elif classical_err < yolo_err:
            classical_wins += 1
        else:
            yolo_wins += 1

        # --- Simple gate: agree -> classical; disagree -> trust classical only if its own confidence is high ---
        disagreement_pct = abs(classical_val - yolo_val) / scale_range * 100
        if disagreement_pct <= AGREEMENT_THRESHOLD_PCT:
            agree_count += 1
            gate_errors.append(classical_err)
        else:
            if classical_confidence >= 0.85:
                gate_errors.append(classical_err)
            else:
                gate_errors.append(yolo_err)

    n = both_accurate + only_classical + only_yolo + neither
    print("=== Failure overlap (accuracy at 5% tolerance) ===")
    print(f"Both accurate:      {both_accurate} ({both_accurate/n:.1%})")
    print(f"Only classical:     {only_classical} ({only_classical/n:.1%})")
    print(f"Only YOLO+refiner:  {only_yolo} ({only_yolo/n:.1%})")
    print(f"Neither:            {neither} ({neither/n:.1%})")

    print(f"\n=== Per-image error comparison ===")
    print(f"Classical wins: {classical_wins}   YOLO+refiner wins: {yolo_wins}   Ties: {ties}")

    print(f"\n=== Simulated agreement/confidence gate ===")
    print(f"Agreement rate: {agree_count}/{n} ({agree_count/n:.1%})")
    n_gate_accurate = sum(1 for e in gate_errors if e <= TOLERANCE_PCT)
    print(f"Gate accuracy: {n_gate_accurate/len(gate_errors):.1%}  mean_err: {sum(gate_errors)/len(gate_errors):.2f}%")


if __name__ == "__main__":
    main()