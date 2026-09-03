"""
Experiment 09 -- Deep Learning, full training run.

Trains on the full 800-image training set, validates each epoch against the held-out 200-image validation set
(confirmed leak-free earlier), and saves the checkpoint with the best validation loss.

Starting with a modest epoch count to gauge CPU timing and loss trend before comitting to a longer run.

Run from the repo root with:
    uv run python scripts/experiments/experiment_09_full_training.py
"""

import time

import mlflow
import torch
from torch.utils.data import DataLoader

from retroread.annotations import load_keypoint_training_data
from retroread.config import ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_TRAIN_KPTS_COCO, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.keypoint_dataset import GaugeKeypointDataset
from retroread.keypoint_model import GaugeKeypointModel
from retroread.mlflow_setup import configure_tracking

N_EPOCHS = 10
LEARNING_RATE = 5e-4    # lower, since pretrained backbone weights are now partially trainable
BATCH_SIZE = 16
CHECKPOINT_PATH = "best_model.pt"

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

    train_dataset = GaugeKeypointDataset(train_samples, ENDAVA_DS5_IMAGES_DIR)
    val_dataset = GaugeKeypointDataset(val_samples, ENDAVA_DS5_IMAGES_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = GaugeKeypointModel(freeze_backbone=True)

    loss_fn = torch.nn.MSELoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    configure_tracking()
    mlflow.set_experiment("retroread_deep_learning")

    with mlflow.start_run(run_name="exp09_full_training"):
        mlflow.log_param("n_train", len(train_samples))
        mlflow.log_param("n_val", len(val_samples))
        mlflow.log_param("n_epochs", N_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)

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
            print(f"Epoch {epoch + 1}/{N_EPOCHS} train_loss={train_loss:.5f} "
                  f"val_loss = {val_loss:.5f} ({epoch_time:.1f}s)")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f" -> new best val_loss, checkpoint saved")

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact(CHECKPOINT_PATH)

        print(f"\nBest validation loss: {best_val_loss:.5f}")
        print(f"Checkpoint saved to: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()