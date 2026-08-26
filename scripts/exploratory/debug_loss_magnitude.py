"""
Prints raw predictions vs targets for one batch, to see directly where the high (~1.0) MSE loss is coming from.

Run from the repo root with:
    uv run python scripts/exploratory/debug_loss_magnitude.py
"""
import torch
from torch.utils.data import DataLoader

from retroread.annotations import load_keypoint_training_data
from retroread.config import ENDAVA_DS5_IMAGES_DIR, ENDAVA_DS5_TRAIN_KPTS_COCO
from retroread.keypoint_dataset import GaugeKeypointDataset
from retroread.keypoint_model import GaugeKeypointModel

N_SUBSET = 30
BATCH_SIZE = 6

LABELS = ["center_x", "center_y", "tip_x", "tip_y", "min_x", "min_y", "max_x", "max_y"]

def main() -> None:
    all_samples = load_keypoint_training_data(ENDAVA_DS5_TRAIN_KPTS_COCO)
    subset = all_samples[:N_SUBSET]

    dataset = GaugeKeypointDataset(subset, ENDAVA_DS5_IMAGES_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = GaugeKeypointModel(freeze_backbone=True)
    model.eval()

    images, targets = next(iter(loader))
    print(f"Image tensor shape: {images.shape}")
    print(f"Target tensor shape: {targets.shape}")

    with torch.no_grad():
        predictions = model(images)
    print(f"Prediction tensor shape: {predictions.shape}\n")

    loss_fn = torch.nn.MSELoss()
    total_loss = loss_fn(predictions, targets)
    print(f"Batch MSE loss: {total_loss.item():.5f}\n")

    print(f"{'label':>10} {'pred':>8} {'target':>8} {'sq_diff':>8}")
    for row in range(images.shape[0]):
        print(f"--- sample {row} ---")
        for col, label in enumerate(LABELS):
            p = predictions[row, col].item()
            t = targets[row, col].item()
            sq_diff = (p - t)**2
            print(f"{label:>10} {p:>8.3f} {t:>8.3f} {sq_diff:>8.3f}")

if __name__ == "__main__":
    main()