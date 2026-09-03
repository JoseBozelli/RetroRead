"""
Experiment 13 -- Deep Learning, crop-based two-stage heatmap model.

Third DL architecture: ground-truth crop (Stage 1) + higher-resolution decoder with a skip connection + 
combined heatmap/coordinate loss (Stage 2). Tests whether full-scene localization -- not landmark learning
itself -- was the bottleneck in Expeiments 09-12.

Run from the repo root with:
    uv run python scripts/experiments/experiment_13_crop_heatmap_training.py
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
from retroread.crop_heatmap_model import CropHeatmapModel, soft_argmax_decode
from retroread.mlflow_setup import configure_tracking

N_EPOCHS = 25
LEARNING_RATE = 5e-4
BATCH_SIZE = 16
COORD_LOSS_WEIGHT = 5.0     # coordinate loss is on [0,1] scale, heatmap loss on pixel-intensity scale -- balances the two
CHECKPOINT_PATH = "best_crop_heatmap_model.pt"

def compute_loss(predictions, heatmap_targets, coord_targets, heatmap_loss_fn):
    heatmap_loss = heatmap_loss_fn(predictions, heatmap_targets)
    decoded = soft_argmax_decode(predictions)
    coord_loss = F.l1_loss(decoded, coord_targets)
    total = heatmap_loss + COORD_LOSS_WEIGHT * coord_loss
    return total, heatmap_loss, coord_loss

def run_validation(model, loader, heatmap_loss_fn):
    model.eval()
    total_loss = total_heatmap = total_coord = 0.0
    n = 0
    with torch.no_grad():
        for images, heatmap_targets, coord_targets in loader:
            predictions = model(images)
            loss, h_loss, c_loss = compute_loss(predictions, heatmap_targets, coord_targets, heatmap_loss_fn)
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_heatmap += h_loss.item() * bs
            total_coord += c_loss.item() * bs
            n += bs
    return total_loss / n, total_heatmap / n, total_coord / n

def main() -> None:
    train_samples = load_keypoint_training_data_raw(ENDAVA_DS5_TRAIN_KPTS_COCO)
    val_samples = load_keypoint_training_data_raw(ENDAVA_DS5_VAL_KPTS_COCO)
    print(f"Train samples: {len(train_samples)}  Val samples: {len(val_samples)}")

    train_dataset = CropGaugeHeatmapDataset(train_samples, ENDAVA_DS5_IMAGES_DIR)
    val_dataset = CropGaugeHeatmapDataset(val_samples, ENDAVA_DS5_IMAGES_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CropHeatmapModel(freeze_backbone=True)
    if Path(CHECKPOINT_PATH).exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
        print(f"Resumed from existing checkpoint: {CHECKPOINT_PATH}")

    heatmap_loss_fn = torch.nn.MSELoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    configure_tracking()
    mlflow.set_experiment("retroread_deep_learning")

    with mlflow.start_run(run_name="exp13_crop_heatmap_training"):
        mlflow.log_param("n_train", len(train_samples))
        mlflow.log_param("n_val", len(val_samples))
        mlflow.log_param("n_epochs", N_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("coord_loss_weight", COORD_LOSS_WEIGHT)
        mlflow.log_param("architecture", "crop_heatmap_skip_softargmax")

        best_val_loss = float("inf")

        for epoch in range(N_EPOCHS):
            epoch_start = time.time()
            model.train()
            train_loss_total = 0.0
            for images, heatmap_targets, coord_targets in train_loader:
                optimizer.zero_grad()
                predictions = model(images)
                loss, h_loss, c_loss = compute_loss(predictions, heatmap_targets, coord_targets, heatmap_loss_fn)
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item() * images.size(0)
            train_loss = train_loss_total / len(train_dataset)

            val_loss, val_heatmap, val_coord = run_validation(model, val_loader, heatmap_loss_fn)

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{N_EPOCHS} train_loss={train_loss:.5f} "
                  f"val_loss={val_loss:.5f} (heatmap={val_heatmap:.5f} coord={val_coord:.5f}) ({epoch_time:.1f}s)")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_coord_loss", val_coord, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f" -> new best val_loss, checkpoint saved.")

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact(CHECKPOINT_PATH)
        print(f"\nBest validation loss: {best_val_loss:.5f}")

if __name__ == "__main__":
    main()