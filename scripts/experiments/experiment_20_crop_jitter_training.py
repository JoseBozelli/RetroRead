"""
Experiment 20 -- Deep Learning, crop-jitter augmentation training.

Hypothesis: training with crop perturbations approximating the classical
detector's imperfect crops will reduce train/inference domain shift and
improve end-to-end reading accuracy on real (detector-cropped) input.

Success criteria (pre-registered, evaluated on the same 61-image fair
subset as v3, using the real classical detector's crop -- not oracle):
  - Clear win:   within-tolerance >= 65% AND mean error <= 6.0%
  - Partial:     improves on v3 (54.1% / 7.28%) but doesn't clear the bar above
  - No improvement: at or below v3

Whatever the outcome, this is the final DL training experiment for this
project -- result accepted and documented either way, no further
architecture iteration after this.

Same architecture as v3 (Experiment 13): single unfrozen backbone block,
same learning rate, same heatmap+coordinate loss. Only the crop strategy
changes (jittered training crops; validated against real detector crops,
not idealized ones).

Run from the repo root with:
    uv run python scripts/experiments/experiment_20_crop_jitter_training.py
"""

import time
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from retroread.annotations import load_keypoint_training_data_raw
from retroread.config import ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_TRAIN_KPTS_COCO, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.crop_heatmap_dataset import CropGaugeHeatmapDataset
from retroread.crop_heatmap_jitter_dataset import CropGaugeHeatmapJitterDataset, build_detector_crop_samples
from retroread.crop_heatmap_model import CropHeatmapModel, soft_argmax_decode
from retroread.mlflow_setup import configure_tracking

N_EPOCHS = 25
LEARNING_RATE = 5e-4  # matches Experiment 13 exactly -- only crop strategy changes
BATCH_SIZE = 16
COORD_LOSS_WEIGHT = 5.0
SOURCE_CHECKPOINT = "best_crop_heatmap_model.pt"  # v3's result: 54.1%, 7.28% error
CHECKPOINT_PATH = "best_crop_jitter_model.pt"


def compute_loss(predictions, heatmap_targets, coord_targets, heatmap_loss_fn):
    heatmap_loss = heatmap_loss_fn(predictions, heatmap_targets)
    decoded = soft_argmax_decode(predictions)
    coord_loss = F.l1_loss(decoded, coord_targets)
    return heatmap_loss + COORD_LOSS_WEIGHT * coord_loss


def run_validation(model, loader, heatmap_loss_fn):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for images, heatmap_targets, coord_targets in loader:
            predictions = model(images)
            loss = compute_loss(predictions, heatmap_targets, coord_targets, heatmap_loss_fn)
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
    return total_loss / n


def main() -> None:
    train_raw = load_keypoint_training_data_raw(ENDAVA_DS5_TRAIN_KPTS_COCO)
    val_raw = load_keypoint_training_data_raw(ENDAVA_DS5_VAL_KPTS_COCO)

    print("Building real detector-crop validation set (one-time circle detection per image)...")
    val_detector_samples = build_detector_crop_samples(val_raw, ENDAVA_DS5_IMAGES_DIR)
    print(f"Train samples: {len(train_raw)}  Val samples (detector-cropped): {len(val_detector_samples)}")

    train_dataset = CropGaugeHeatmapJitterDataset(train_raw, ENDAVA_DS5_IMAGES_DIR)
    val_dataset = CropGaugeHeatmapDataset(val_detector_samples, ENDAVA_DS5_IMAGES_DIR)
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

    with mlflow.start_run(run_name="exp20_crop_jitter_training"):
        mlflow.log_param("n_epochs", N_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("coord_loss_weight", COORD_LOSS_WEIGHT)
        mlflow.log_param("position_jitter", 0.10)
        mlflow.log_param("scale_jitter", 0.15)
        mlflow.log_param("validation_crop_source", "real_classical_detector")
        mlflow.log_param("resumed_from", SOURCE_CHECKPOINT)

        best_val_loss = float("inf")

        for epoch in range(N_EPOCHS):
            epoch_start = time.time()
            model.train()
            train_loss_total = 0.0
            for images, heatmap_targets, coord_targets in train_loader:
                optimizer.zero_grad()
                predictions = model(images)
                loss = compute_loss(predictions, heatmap_targets, coord_targets, heatmap_loss_fn)
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item() * images.size(0)
            train_loss = train_loss_total / len(train_dataset)

            val_loss = run_validation(model, val_loader, heatmap_loss_fn)
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{N_EPOCHS}  train_loss={train_loss:.5f}  "
                  f"val_loss(detector-crop)={val_loss:.5f}  ({epoch_time:.1f}s)")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss_detector_crop", val_loss, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"  -> new best val_loss, checkpoint saved.")

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact(CHECKPOINT_PATH)
        print(f"\nBest validation loss (detector-crop): {best_val_loss:.5f}")


if __name__ == "__main__":
    main()