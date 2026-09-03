"""
Diagnostic: evaluates the EXISTING Experiment 13 checkpoint (no retraining)
across multiple soft-argmax temperatures + hard argmax, and across two crop
sources (ground-truth bbox vs. classical circle detector) -- to find how
much of the 54.1% result was limited by decoding vs. crop-domain shift.

Run from the repo root with:
    uv run python scripts/exploratory/diagnose_decode_and_crop.py
"""

import json
import math

import torch
from PIL import Image

from retroread.annotations import (
    bbox_touches_edge,
    load_complete_annotations,
    load_keypoint_training_data_raw,
    load_reading_annotations,
)
from retroread.classical_baseline import find_gauge_circle
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.crop_heatmap_dataset import PREPROCESS, compute_crop_box
from retroread.crop_heatmap_model import CropHeatmapModel, hard_argmax_decode, soft_argmax_decode
from retroread.params import CIRCLE_PARAMS
from retroread.reading_conversion import angle_to_value, fit_scale_calibration

CHECKPOINT_PATH = "best_crop_heatmap_model.pt"
TOLERANCE_PCT = 5.0
TEMPERATURES = [1, 5, 10, 20, 50]


def evaluate(model, candidates, crop_source: str):
    """crop_source: 'gt' or 'detected'."""
    decode_results = {f"soft_t{t}": [] for t in TEMPERATURES}
    decode_results["hard"] = []

    with torch.no_grad():
        for bbox_data, reading in candidates:
            image_path = ENDAVA_DS5_IMAGES_DIR / bbox_data["file_name"]
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size

            if crop_source == "gt":
                crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_box(
                    bbox_data["bbox"], image_width, image_height
                )
            else:
                circle = find_gauge_circle(str(image_path), **CIRCLE_PARAMS)
                if not circle.found:
                    continue
                circle_bbox = [circle.center_x - circle.radius, circle.center_y - circle.radius,
                                circle.radius * 2, circle.radius * 2]
                crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_box(circle_bbox, image_width, image_height)

            crop_w, crop_h = crop_x1 - crop_x0, crop_y1 - crop_y0
            cropped = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            input_tensor = PREPROCESS(cropped).unsqueeze(0)

            pred_heatmaps = model(input_tensor)

            calibration = fit_scale_calibration(reading["center_x"], reading["center_y"], reading["scale_labels"])
            scale_values = [s["value"] for s in reading["scale_labels"]]
            scale_range = max(scale_values) - min(scale_values)

            def score(coords_tensor, key):
                coords = coords_tensor[0].numpy()
                pred_center_x = crop_x0 + coords[0] * crop_w
                pred_center_y = crop_y0 + coords[1] * crop_h
                pred_tip_x = crop_x0 + coords[2] * crop_w
                pred_tip_y = crop_y0 + coords[3] * crop_h
                angle = math.atan2(pred_tip_y - pred_center_y, pred_tip_x - pred_center_x)
                predicted = angle_to_value(angle, calibration)
                pct_error = abs(predicted - reading["true_value"]) / scale_range * 100 if scale_range > 0 else None
                if pct_error is not None:
                    decode_results[key].append(pct_error)

            score(hard_argmax_decode(pred_heatmaps), "hard")
            for t in TEMPERATURES:
                score(soft_argmax_decode(pred_heatmaps, temperature=t), f"soft_t{t}")

    print(f"\n--- Crop source: {crop_source} ---")
    print(f"{'Decode method':15} {'Accuracy':>10} {'Mean % err':>12}")
    for key, errors in decode_results.items():
        if not errors:
            continue
        n_accurate = sum(1 for e in errors if e <= TOLERANCE_PCT)
        acc = n_accurate / len(errors)
        mean_err = sum(errors) / len(errors)
        print(f"{key:15} {acc:>9.1%} {mean_err:>11.2f}%")


def main() -> None:
    with ENDAVA_DS5_VAL_KPTS_COCO.open() as f:
        val_filenames = {img["file_name"] for img in json.load(f)["images"]}

    edge_filtered = load_complete_annotations(ENDAVA_DS5_VAL_KPTS_COCO)
    edge_filtered = [c for c in edge_filtered if not bbox_touches_edge(c)]
    fair_filenames = {c["file_name"] for c in edge_filtered} & val_filenames

    raw_val_samples = load_keypoint_training_data_raw(ENDAVA_DS5_VAL_KPTS_COCO)
    bbox_by_filename = {s["file_name"]: s for s in raw_val_samples}

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    reading_by_filename = {d["file_name"]: d for d in reading_data.values()}

    candidates = [
        (bbox_by_filename[fn], reading_by_filename[fn])
        for fn in fair_filenames
        if fn in bbox_by_filename and fn in reading_by_filename
    ]
    print(f"Evaluating on {len(candidates)} images (same fair subset used throughout).")

    model = CropHeatmapModel(freeze_backbone=True)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    evaluate(model, candidates, "gt")
    evaluate(model, candidates, "detected")


if __name__ == "__main__":
    main()