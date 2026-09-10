"""
Experiment 34 -- R2: local high-resolution tip refiner.

Trains on YOLO's own predicted tips (precomputed by
precompute_yolo_predictions.py), not ground-truth-centered crops --
matching the deployment distribution, per the lesson from Experiment 20's
crop-domain-shift fix. Center stays YOLO's own prediction throughout
(the oracle ablation showed touching center independently hurts).

Pre-registered success criteria (evaluated in Experiment 35):
  >=77% within-tolerance AND <4.82% mean error: genuinely competitive
  ~78-79% and ~4.3-4.5%: captures most of R1's oracle upside
  ~72-74%: refiner isn't extracting the oracle opportunity -- stop

Run from the repo root with:
    uv run python scripts/experiments/experiment_34_tip_refiner_training.py
"""

import json
import time

import mlflow
import torch
from torch.utils.data import DataLoader

from retroread.config import ENDAVA_DS5_IMAGES_DIR
from retroread.mlflow_setup import configure_tracking
from retroread.tip_refiner_dataset import TipRefinerDataset
from retroread.tip_refiner_model import TipRefinerModel

PREDICTIONS_CACHE = "data/processed/yolo_predictions_cache.json"
N_EPOCHS = 25
LEARNING_RATE = 5e-4
BATCH_SIZE = 16
CHECKPOINT_PATH = "best_tip_refiner_model.pt"


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
    with open(PREDICTIONS_CACHE) as f:
        cache = json.load(f)

    train_samples = [
        {"file_name": fn, "yolo_tip": d["tip"], "gt_tip": d["gt_tip"]}
        for fn, d in cache["train"].items()
    ]
    val_samples = [
        {"file_name": fn, "yolo_tip": d["tip"], "gt_tip": d["gt_tip"]}
        for fn, d in cache["val"].items()
    ]
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    train_dataset = TipRefinerDataset(train_samples, ENDAVA_DS5_IMAGES_DIR, jitter=True)
    val_dataset = TipRefinerDataset(val_samples, ENDAVA_DS5_IMAGES_DIR, jitter=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TipRefinerModel(freeze_backbone=True, n_unfrozen_blocks=1)
    loss_fn = torch.nn.MSELoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    configure_tracking()
    mlflow.set_experiment("retroread_deep_learning")

    with mlflow.start_run(run_name="exp34_tip_refiner_training"):
        mlflow.log_param("n_epochs", N_EPOCHS)
        mlflow.log_param("roi_size", 128)
        mlflow.log_param("jitter_range", "32-56px")

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
            print(f"Epoch {epoch + 1}/{N_EPOCHS}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  ({epoch_time:.1f}s)")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print("  -> new best val_loss, checkpoint saved.")

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact(CHECKPOINT_PATH)
        print(f"\nBest validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()