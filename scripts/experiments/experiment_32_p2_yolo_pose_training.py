"""
Experiment 32 -- P2-YOLO-Pose: adds a stride-4 (P2) detection/pose scale
to the standard YOLOv8n-pose architecture, following the pattern used in
recent gauge-reading literature (P2-YOLO-Pose, 2026) -- standard YOLO
feature pyramids (P3/P4/P5, strides 8/16/32) discard fine spatial detail
that matters for precise small-object landmark localization.

Layers 0-15 of the custom architecture are structurally identical to
stock yolov8n-pose.yaml, so pretrained weights transfer for those; layers
16+ (the new P2 branch) are new and train from scratch.

Run from the repo root with:
    uv run python scripts/experiments/experiment_32_p2_yolo_pose_training.py
"""

from pathlib import Path

from ultralytics import YOLO

CONFIG_YAML = Path("configs/yolov8-p2-pose.yaml").resolve()
DATA_YAML = Path("data/processed/yolo_pose/data.yaml").resolve()
N_EPOCHS = 25


def main() -> None:
    model = YOLO(str(CONFIG_YAML))
    model.load("yolov8n-pose.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=N_EPOCHS,
        imgsz=224,
        batch=16,
        device="cpu",
        project="yolo_runs",
        name="experiment_32",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()