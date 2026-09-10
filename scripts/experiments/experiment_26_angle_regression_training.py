"""
Experiment 26 -- sin/cos angle regression.

Run from the repo root with:
    uv run python scripts/experiments/experiment_26_angle_regression_training.py
"""

import time

import mlflow
import torch
from torch.utils.data import DataLoader

from retroread.angle_regression_dataset import AngleRegressionDataset
from retroread.angle_regression_model import AngleRegressionModel
from retroread.annotations import load_keypoint_training_data_raw
from retroread.config import ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_TRAIN_KPTS_COCO, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.mlflow_setup import configure_tracking

N_EPOCHS = 25
LEARNING_RATE = 5e-4
BATCH_SIZE = 16
CHECKPOINT_PATH = "best_angle_regression_model.pt"


def run_validation(model, loader, loss_fn):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            preds = model(images)
            loss = loss_fn(preds, targets)
            total += loss.item() * images.size(0)
            n += images.size(0)
    return total / n


def main() -> None:
    train_samples = load_keypoint_training_data_raw(ENDAVA_DS5_TRAIN_KPTS_COCO)
    val_samples = load_keypoint_training_data_raw(ENDAVA_DS5_VAL_KPTS_COCO)
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    train_dataset = AngleRegressionDataset(train_samples, ENDAVA_DS5_IMAGES_DIR, jitter=True)
    val_dataset = AngleRegressionDataset(val_samples, ENDAVA_DS5_IMAGES_DIR, jitter=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AngleRegressionModel(freeze_backbone=True, n_unfrozen_blocks=1)
    loss_fn = torch.nn.MSELoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    configure_tracking()
    mlflow.set_experiment("retroread_deep_learning")

    with mlflow.start_run(run_name="exp26_angle_regression_training"):
        mlflow.log_param("n_epochs", N_EPOCHS)
        mlflow.log_param("architecture", "sincos_angle_regression")

        best_val_loss = float("inf")
        for epoch in range(N_EPOCHS):
            epoch_start = time.time()
            model.train()
            train_loss_total = 0.0
            for images, targets in train_loader:
                optimizer.zero_grad()
                preds = model(images)
                loss = loss_fn(preds, targets)
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item() * images.size(0)
            train_loss = train_loss_total / len(train_dataset)

            val_loss = run_validation(model, val_loader, loss_fn)
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{N_EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  ({epoch_time:.1f}s)")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print("  -> new best val_loss, checkpoint saved.")

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact(CHECKPOINT_PATH)
        print(f"\nBest validation loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    main()