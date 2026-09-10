"""
Experiment 28 -- Direct end-to-end numeric regression. Expected to perform poorly by design 
(see model docstring) -- included to demonstrate WHY geometry-aware approaches are preferred, not
because it's expected to compete.

Run from the repo root with:
    uv run python scripts/experiments/experiment_28_direct_regression_training.py
"""

import time

import mlflow
import torch
from torch.utils.data import DataLoader

from retroread.annotations import load_keypoint_training_data_raw, load_reading_annotations
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_TRAIN_KPTS_COCO, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.direct_regression_dataset import DirectRegressionDataset
from retroread.direct_regression_model import DirectRegressionModel
from retroread.mlflow_setup import configure_tracking

N_EPOCHS = 25
LEARNING_RATE = 5e-4    # smaller output head appears sensitive to the shared 5e-4 rate; see docs
BATCH_SIZE = 16
CHECKPOINT_PATH = "best_direct_regression_model.pt"


def build_samples(raw_samples, reading_by_filename):
    combined = []
    for s in raw_samples:
        reading = reading_by_filename.get(s["file_name"])
        if reading is None or len(reading["scale_labels"]) < 1:
            continue
        scale_values = [sl["value"] for sl in reading["scale_labels"]]
        merged = dict(s)
        merged["min_value"] = min(scale_values)
        merged["max_value"] = max(scale_values)
        merged["true_value"] = reading["true_value"]
        combined.append(merged)
    return combined


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
    reading_data = load_reading_annotations(ENDAVA_DS5_COCO)
    reading_by_filename = {d["file_name"]: d for d in reading_data.values()}

    train_raw = load_keypoint_training_data_raw(ENDAVA_DS5_TRAIN_KPTS_COCO)
    val_raw = load_keypoint_training_data_raw(ENDAVA_DS5_VAL_KPTS_COCO)
    train_samples = build_samples(train_raw, reading_by_filename)
    val_samples = build_samples(val_raw, reading_by_filename)
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    train_dataset = DirectRegressionDataset(train_samples, ENDAVA_DS5_IMAGES_DIR, jitter=True)
    val_dataset = DirectRegressionDataset(val_samples, ENDAVA_DS5_IMAGES_DIR, jitter=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DirectRegressionModel(freeze_backbone=True, n_unfrozen_blocks=1)
    loss_fn = torch.nn.MSELoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    configure_tracking()
    mlflow.set_experiment("retroread_deep_learning")

    with mlflow.start_run(run_name="exp28_direct_regression_training"):
        mlflow.log_param("n_epochs", N_EPOCHS)
        mlflow.log_param("architecture", "direct_normalized_position_regression")

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