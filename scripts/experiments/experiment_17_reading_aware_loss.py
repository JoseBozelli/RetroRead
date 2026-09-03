"""
Experiment 17 -- Deep Learning, final experiment: reading-aware auxiliary
loss.

Total loss = L_heatmap + lambda1 * L_coord + lambda2 * L_reading.

L_reading explicitly supervises the model's center/tip predictions on the
metric that actually matters -- the final gauge reading -- not just raw
pixel distance. Resumes from Experiment 13's best checkpoint (54.1%
within-tolerance, 7.28% mean error), same architecture, same crop/heatmap
setup, same single learning rate as Experiment 13. Only the loss function
changes, isolating this as a clean, attributable final experiment.

This is the closing DL experiment for this project, per agreed scope --
result accepted either way, no further architecture changes after this.

Run from the repo root with:
    uv run python scripts/experiments/experiment_17_reading_aware_loss.py
"""

import time
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from retroread.annotations import load_keypoint_training_data_raw, load_reading_annotations
from retroread.config import (
    ENDAVA_DS5_COCO,
    ENDAVA_DS5_IMAGES_DIR,
    ENDAVA_DS5_TRAIN_KPTS_COCO,
    ENDAVA_DS5_VAL_KPTS_COCO,
)
from retroread.crop_heatmap_model import CropHeatmapModel, soft_argmax_decode
from retroread.crop_heatmap_reading_dataset import CropGaugeReadingDataset, build_reading_samples
from retroread.differentiable_reading import angular_reading_loss
from retroread.mlflow_setup import configure_tracking

N_EPOCHS = 25
LEARNING_RATE = 5e-4  # matches Experiment 13 exactly -- only the loss changes
BATCH_SIZE = 16
COORD_LOSS_WEIGHT = 5.0
READING_LOSS_WEIGHT = 2.0
SOURCE_CHECKPOINT = "best_crop_heatmap_model.pt"  # Experiment 13's result: 54.1%, 7.28% error
CHECKPOINT_PATH = "best_reading_aware_model.pt"


def compute_loss(predictions, heatmap_targets, coord_targets, extras, heatmap_loss_fn):
    heatmap_loss = heatmap_loss_fn(predictions, heatmap_targets)
    decoded = soft_argmax_decode(predictions)
    coord_loss = F.l1_loss(decoded, coord_targets)

    pred_center = decoded[:, 0:2]
    pred_tip = decoded[:, 2:4]
    gt_center = coord_targets[:, 0:2]
    gt_min = coord_targets[:, 4:6]
    gt_max = coord_targets[:, 6:8]
    min_value, max_value, true_reading = extras[:, 0], extras[:, 1], extras[:, 2]

    reading_loss = angular_reading_loss(
        pred_center, pred_tip, gt_center, gt_min, gt_max, min_value, max_value, true_reading
    )

    total = heatmap_loss + COORD_LOSS_WEIGHT * coord_loss + READING_LOSS_WEIGHT * reading_loss
    return total, heatmap_loss, coord_loss, reading_loss


def run_validation(model, loader, heatmap_loss_fn):
    model.eval()
    total_loss = total_reading = 0.0
    n = 0
    with torch.no_grad():
        for images, heatmap_targets, coord_targets, extras in loader:
            predictions = model(images)
            loss, _, _, r_loss = compute_loss(predictions, heatmap_targets, coord_targets, extras, heatmap_loss_fn)
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_reading += r_loss.item() * bs
            n += bs
    return total_loss / n, total_reading / n


def main() -> None:
    train_raw = load_keypoint_training_data_raw(ENDAVA_DS5_TRAIN_KPTS_COCO)
    val_raw = load_keypoint_training_data_raw(ENDAVA_DS5_VAL_KPTS_COCO)

    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    reading_by_filename = {d["file_name"]: d for d in reading_data.values()}

    train_samples = build_reading_samples(train_raw, reading_by_filename)
    val_samples = build_reading_samples(val_raw, reading_by_filename)
    print(f"Train samples: {len(train_samples)}  Val samples: {len(val_samples)}")

    train_dataset = CropGaugeReadingDataset(train_samples, ENDAVA_DS5_IMAGES_DIR)
    val_dataset = CropGaugeReadingDataset(val_samples, ENDAVA_DS5_IMAGES_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CropHeatmapModel(freeze_backbone=True, n_unfrozen_blocks=1)
    if Path(SOURCE_CHECKPOINT).exists():
        model.load_state_dict(torch.load(SOURCE_CHECKPOINT))
        print(f"Resumed from: {SOURCE_CHECKPOINT}")

    heatmap_loss_fn = torch.nn.MSELoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    configure_tracking()
    mlflow.set_experiment("retroread_deep_learning")

    with mlflow.start_run(run_name="exp17_reading_aware_loss"):
        mlflow.log_param("n_epochs", N_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("coord_loss_weight", COORD_LOSS_WEIGHT)
        mlflow.log_param("reading_loss_weight", READING_LOSS_WEIGHT)
        mlflow.log_param("resumed_from", SOURCE_CHECKPOINT)

        best_val_loss = float("inf")

        for epoch in range(N_EPOCHS):
            epoch_start = time.time()
            model.train()
            train_loss_total = 0.0
            for images, heatmap_targets, coord_targets, extras in train_loader:
                optimizer.zero_grad()
                predictions = model(images)
                loss, h_loss, c_loss, r_loss = compute_loss(
                    predictions, heatmap_targets, coord_targets, extras, heatmap_loss_fn
                )
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item() * images.size(0)
            train_loss = train_loss_total / len(train_dataset)

            val_loss, val_reading = run_validation(model, val_loader, heatmap_loss_fn)
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{N_EPOCHS}  train_loss={train_loss:.5f}  "
                  f"val_loss={val_loss:.5f}  val_reading_frac_err={val_reading:.4f}  ({epoch_time:.1f}s)")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_reading_frac_error", val_reading, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"  -> new best val_loss, checkpoint saved.")

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact(CHECKPOINT_PATH)
        print(f"\nBest validation loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    main()