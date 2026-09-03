"""
Experiment 11 -- Deep Learning, heatmap-based model, full training run.

Second DL architecture attempt: preserves spatial structure via heatmap prediction instead of flattened
coordinate regression (see heatmap_model.py for the reasoning). Same train/eval split, smae checkpoint
pattern as Experiment 09.

Run from the repo root with:
    uv run python scripts/experiments/experiment_11_heatmap_training.py
"""

import time

import mlflow
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from retroread.annotations import load_keypoint_training_data
from retroread.config import ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_TRAIN_KPTS_COCO, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.heatmap_dataset import GaugeHeatmapDataset
from retroread.heatmap_model import GaugeHeatmapModel
from retroread.mlflow_setup import configure_tracking

N_EPOCHS = 20
LEARNING_RATE = 5e-4
BATCH_SIZE = 16
CHECKPOINT_PATH = "best_heatmap_model.pt"

def run_validation(model, loader, loss_fn) -> float:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for images, targets in loader:
            predictions = model(images)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item() * images.size(0)
            n_samples += images.size(0)
    return total_loss / n_samples

def main() -> None:
    train_samples = load_keypoint_training_data(ENDAVA_DS5_TRAIN_KPTS_COCO)
    val_samples = load_keypoint_training_data(ENDAVA_DS5_VAL_KPTS_COCO)
    print(f"Train samples: {len(train_samples)} Val samples: {len(val_samples)}")

    train_dataset = GaugeHeatmapDataset(train_samples, ENDAVA_DS5_IMAGES_DIR)
    val_dataset = GaugeHeatmapDataset(val_samples, ENDAVA_DS5_IMAGES_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = GaugeHeatmapModel(freeze_backbone=True)
    if Path(CHECKPOINT_PATH).exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
        print(f"Resumed from existing checkpoint: {CHECKPOINT_PATH}")

    loss_fn = torch.nn.MSELoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    configure_tracking()
    mlflow.set_experiment("retroread_deep_learning")

    with mlflow.start_run(run_name="exp11_heatmap_training"):
        mlflow.log_param("n_train", len(train_samples))
        mlflow.log_param("n_val", len(val_samples))
        mlflow.log_param("n_epochs", N_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("architecture", "heatmap_upsample")

        best_val_loss = float("inf")

        for epoch in range(N_EPOCHS):
            epoch_start = time.time()

            model.train()
            train_loss_total = 0.0
            for images, targets in train_loader:
                optimizer.zero_grad()
                predictions = model(images)
                loss = loss_fn(predictions, targets)
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item() * images.size(0)
            train_loss = train_loss_total / len(train_dataset)

            val_loss = run_validation(model, val_loader, loss_fn)

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{N_EPOCHS} train_loss={train_loss:.6f} "
                  f"val_loss={val_loss:.6f} ({epoch_time:.1f}s)")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f" -> new best val_loss, checkpoint saved.")

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact(CHECKPOINT_PATH)
        print(f"\nBest validation loss: {best_val_loss:.6f}")
        print(f"Checkpoint saved to: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()