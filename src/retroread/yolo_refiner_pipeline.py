"""
YOLO pose + local tip refiner pipeline, packaged for reuse (previously inline in the Experiment 35 evaluation
script) -- now also used by the production ensemble in predict.py
"""

import math

import torch
from PIL import Image

from retroread.tip_refiner_dataset import HEATMAP_SIZE, ROI_SIZE, PREPROCESS as REFINER_PREPROCESS

def refine_tip(refiner_model, image: Image.Image, yolo_tip: tuple[float, float]) -> tuple[float, float]:
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

def get_yolo_refiner_angle(image_path: str, cropped: Image.Image, crop_origin: tuple[float, float],
                           yolo_model, refiner_model) -> float | None:
    """Returns th needle angle (radians) from YOLO center + refined tip, or None if YOLO found nothing."""
    full_image = Image.open(image_path).convert("RGB")
    pred = yolo_model.predict(cropped, verbose=False)[0]
    if pred.keypoints is None or len(pred.keypoints.xy) == 0:
        return None

    kpts = pred.keypoints.xy[0].tolist()
    crop_x0, crop_y0 = crop_origin
    yolo_center = (kpts[0][0] + crop_x0, kpts[0][1] + crop_y0)
    yolo_tip = (kpts[1][0] + crop_x0, kpts[1][1] + crop_y0)

    refined_tip = refine_tip(refiner_model, full_image, yolo_tip)
    return math.atan2(refined_tip[1] - yolo_center[1], refined_tip[0] - yolo_center[0])