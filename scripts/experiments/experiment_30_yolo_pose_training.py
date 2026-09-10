"""
Experiment 30 -- YOLOv8-pose fine-tuning.

A genuinely different detection paradigm from every other DL variant in
this project: single-shot object detection + pose estimation, using
ultralytics' pretrained yolov8n-pose checkpoint, fine-tuned on our
cropped-gauge dataset (built by build_yolo_pose_dataset.py).

Run from the repo root with:
    uv run python scripts/experiments/experiment_30_yolo_pose_training.py
"""

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path("data/processed/yolo_pose/data.yaml").resolve()
N_EPOCHS = 25


def main() -> None:
    model = YOLO("yolov8n-pose.pt")  # pretrained, downloaded on first use
    model.train(
        data=str(DATA_YAML),
        epochs=N_EPOCHS,
        imgsz=224,
        batch=16,
        device="cpu",
        project="yolo_runs",
        name="experiment_30",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()