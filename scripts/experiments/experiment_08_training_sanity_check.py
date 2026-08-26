"""
Experiment 08 -- Deep Learning, training loop sanity check.

Trains on a small subset (30 images) for a few epochs, to confirm the
training loop is mechanically correct -- loss should decrease -- before
committing to a longer run on the full dataset.

Run from the repo root with:
    uv run python scripts/experiments/experiment_08_training_sanity_check.py
"""

import mlflow
import torch
from torch.utils.data import DataLoader

from retroread.annotations import load_keypoint_training_data
from retroread.config import ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_TRAIN_KPTS_COCO
from retroread.keypoint_dataset import GaugeKeypointDataset
from retroread.keypoint_model import GaugeKeypointModel
from retroread.mlflow_setup import configure_tracking

N_SUBSET = 30
N_EPOCHS = 5
LEARNING_RATE = 1e-3
BATCH_SIZE = 6


def main() -> None:
    all_samples = load_keypoint_training_data(ENDAVA_DS5_TRAIN_KPTS_COCO)
    subset = all_samples[:N_SUBSET]
    print(f"Training on {len(subset)} images for {N_EPOCHS} epochs (sanity check only).")

    dataset = GaugeKeypointDataset(subset, ENDAVA_DS5_IMAGES_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GaugeKeypointModel(freeze_backbone=True)
    model.train()

    loss_fn = torch.nn.MSELoss()
    # Only the head's parameters need an optimizer -- backbone is frozen.
    optimizer = torch.optim.Adam(model.head.parameters(), lr=LEARNING_RATE)

    configure_tracking()
    mlflow.set_experiment("retroread_deep_learning")

    with mlflow.start_run(run_name="exp08_training_sanity_check"):
        mlflow.log_param("n_subset", N_SUBSET)
        mlflow.log_param("n_epochs", N_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)

        epoch_losses = []
        for epoch in range(N_EPOCHS):
            total_loss = 0.0
            for images, targets in loader:
                optimizer.zero_grad()          # clear gradients from the previous step
                predictions = model(images)     # forward pass
                loss = loss_fn(predictions, targets)
                loss.backward()                 # compute gradients
                optimizer.step()                # update the head's weights
                total_loss += loss.item() * images.size(0)

            avg_loss = total_loss / len(dataset)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{N_EPOCHS}  avg MSE loss: {avg_loss:.5f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        print(f"\nLoss went from {epoch_losses[0]:.5f} to {epoch_losses[-1]:.5f}")
        if epoch_losses[-1] < epoch_losses[0]:
            print("Loss decreased -- training loop is working correctly.")
        else:
            print("Loss did NOT decrease -- something is wrong, worth investigating before scaling up.")


if __name__ == "__main__":
    main()