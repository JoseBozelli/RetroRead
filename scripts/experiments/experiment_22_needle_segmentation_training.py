"""
Experiment 22 -- Deep Learning, needle segmentation model.

Hypothesis: predicting the needle's full pixel mask -- aggregating evidence
across many pixels -- produces a more robust angle estimate than
predicting a single tip coordinate (Experiments 09-20), particularly under
production-realistic (detector-cropped, glare-affected) conditions.

Two output heads (needle mask, center heatmap); the reading is computed
geometrically (reach-based tip selection from mask + center), matching the
classical pipeline's own convention, not predicted directly.

Trained with crop jitter (Experiment 20's finding); validated on
non-jittered ground-truth crops for stable checkpoint selection during
this already-large architecture change.

Run from the repo root with:
    uv run python scripts/experiments/experiment_22_needle_segmentation_training.py
"""

import json
import time

import mlflow
import torch
from torch.utils.data import DataLoader

from retroread.annotations import load_needle_segmentation_data
from retroread.config import ENDAVA_DS5_COCO, ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_TRAIN_KPTS_COCO, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.mlflow_setup import configure_tracking
from retroread.needle_segmentation_dataset import NeedleSegmentationDataset
from retroread.needle_segmentation_model import NeedleSegmentationModel

N_EPOCHS = 25
LEARNING_RATE = 5e-4
BATCH_SIZE = 16
MASK_LOSS_WEIGHT = 1.0
CENTER_LOSS_WEIGHT = 1.0
CHECKPOINT_PATH = "best_needle_segmentation_model.pt"


def compute_loss(mask_logits, center_logits, mask_targets, center_targets, mask_loss_fn, center_loss_fn):
    mask_loss = mask_loss_fn(mask_logits, mask_targets)
    center_loss = center_loss_fn(torch.sigmoid(center_logits), center_targets)
    return MASK_LOSS_WEIGHT * mask_loss + CENTER_LOSS_WEIGHT * center_loss


def run_validation(model, loader, mask_loss_fn, center_loss_fn):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for images, mask_targets, center_targets in loader:
            mask_logits, center_logits = model(images)
            loss = compute_loss(mask_logits, center_logits, mask_targets, center_targets, mask_loss_fn, center_loss_fn)
            total += loss.item() * images.size(0)
            n += images.size(0)
    return total / n


def main() -> None:
    all_data = load_needle_segmentation_data(ENDAVA_DS5_COCO)
    print(f"Total samples with needle mask + center + bbox: {len(all_data)}")

    with ENDAVA_DS5_TRAIN_KPTS_COCO.open() as f:
        train_filenames = {img["file_name"] for img in json.load(f)["images"]}
    with ENDAVA_DS5_VAL_KPTS_COCO.open() as f:
        val_filenames = {img["file_name"] for img in json.load(f)["images"]}

    train_samples = [d for d in all_data if d["file_name"] in train_filenames]
    val_samples = [d for d in all_data if d["file_name"] in val_filenames]
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    train_dataset = NeedleSegmentationDataset(train_samples, ENDAVA_DS5_IMAGES_DIR, jitter=True)
    val_dataset = NeedleSegmentationDataset(val_samples, ENDAVA_DS5_IMAGES_DIR, jitter=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = NeedleSegmentationModel(freeze_backbone=True, n_unfrozen_blocks=1)

    mask_loss_fn = torch.nn.BCEWithLogitsLoss()
    center_loss_fn = torch.nn.MSELoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    configure_tracking()
    mlflow.set_experiment("retroread_deep_learning")

    with mlflow.start_run(run_name="exp22_needle_segmentation_training"):
        mlflow.log_param("n_epochs", N_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("architecture", "needle_segmentation_plus_center_heatmap")

        best_val_loss = float("inf")
        for epoch in range(N_EPOCHS):
            epoch_start = time.time()
            model.train()
            train_loss_total = 0.0
            for images, mask_targets, center_targets in train_loader:
                optimizer.zero_grad()
                mask_logits, center_logits = model(images)
                loss = compute_loss(mask_logits, center_logits, mask_targets, center_targets, mask_loss_fn, center_loss_fn)
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item() * images.size(0)
            train_loss = train_loss_total / len(train_dataset)

            val_loss = run_validation(model, val_loader, mask_loss_fn, center_loss_fn)
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