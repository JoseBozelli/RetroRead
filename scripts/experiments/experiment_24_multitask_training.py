"""
Experiment 24 -- Deep Learning, multi-task model (mask + center + scale landmarks).

Run from the repo root with:
    uv run python scripts/experiments/experiment_24_multitask_training.py
"""
import json
import time

import mlflow
import torch
from torch.utils.data import DataLoader

from retroread.annotations import load_keypoint_training_data_raw, load_needle_segmentation_data
from retroread.config import ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_TRAIN_KPTS_COCO, ENDAVA_DS5_VAL_KPTS_COCO
from retroread.mlflow_setup import configure_tracking
from retroread.multitask_dataset import MultiTaskDataset, build_multitask_samples
from retroread.multitask_gauge_model import MultiTaskGaugeModel

N_EPOCHS = 25
LEARNING_RATE = 5e-4
BATCH_SIZE = 16
CHECKPOINT_PATH = "best_multitask_model.pt"

def compute_loss(mask_logits, center_logits, landmark_logits, mask_t, center_t, landmark_t, bce, mse):
    return(
        bce(mask_logits, mask_t)
        + mse(torch.sigmoid(center_logits), center_t)
        + mse(torch.sigmoid(landmark_logits), landmark_t)
    )

def run_validation(model, loader, bce, mse):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for images, mask_t, center_t, landmark_t in loader:
            mask_logits, center_logits, landmark_logits = model(images)
            loss = compute_loss(mask_logits, center_logits, landmark_logits, mask_t, center_t, landmark_t, bce, mse)
            total += loss.item() * images.size(0)
            n += images.size(0)
    return total / n

def main() -> None:
    seg_data = load_needle_segmentation_data
    from retroread.config import ENDAVA_DS5_COCO
    seg_samples = load_needle_segmentation_data(ENDAVA_DS5_COCO)
    kp_train = load_keypoint_training_data_raw(ENDAVA_DS5_TRAIN_KPTS_COCO)
    kp_val = load_keypoint_training_data_raw(ENDAVA_DS5_VAL_KPTS_COCO)

    with ENDAVA_DS5_TRAIN_KPTS_COCO.open() as f:
        train_filenames = {img["file_name"] for img in json.load(f)["images"]}
    with ENDAVA_DS5_VAL_KPTS_COCO.open() as f:
        val_filenames = {img["file_name"] for img in json.load(f)["images"]}

    seg_train = [s for s in seg_samples if s["file_name"] in train_filenames]
    seg_val = [s for s in seg_samples if s["file_name"] in val_filenames]

    train_samples = build_multitask_samples(seg_train, kp_train)
    val_samples = build_multitask_samples(seg_val, kp_val)
    print(f"Train: {len(train_samples)} Val: {len(val_samples)}")

    train_dataset = MultiTaskDataset(train_samples, ENDAVA_DS5_IMAGES_DIR, jitter=True)
    val_dataset = MultiTaskDataset(val_samples, ENDAVA_DS5_IMAGES_DIR, jitter=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultiTaskGaugeModel(freeze_backbone=True, n_unfrozen_blocks=1)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    configure_tracking()
    mlflow.set_experiment("retroread_deep_learning")

    with mlflow.start_run(run_name="exp24_multitask_training"):
        mlflow.log_param("n_epochs", N_EPOCHS)
        mlflow.log_param("architecture", "multitask_mask_center_landmarks")

        best_val_loss = float("inf")
        for epoch in range(N_EPOCHS):
            epoch_start = time.time()
            model.train()
            train_loss_total = 0.0
            for images, mask_t, center_t, landmark_t in train_loader:
                optimizer.zero_grad()
                mask_logits, center_logits, landmark_logits = model(images)
                loss = compute_loss(mask_logits, center_logits, landmark_logits, mask_t, center_t, landmark_t, bce, mse)
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item() * images.size(0)
            train_loss = train_loss_total / len(train_dataset)

            val_loss = run_validation(model, val_loader, bce, mse)
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{N_EPOCHS} train_loss={train_loss:.5f} val_loss={val_loss:.5} ({epoch_time:.1f}s)")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(" -> new best val_loss, checkpoint saved.")

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact(CHECKPOINT_PATH)
        print(f"\nBest validation loss: {best_val_loss:.5f}")

if __name__ == "__main__":
    main()