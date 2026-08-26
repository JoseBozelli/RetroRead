"""
Experiment 07 -- Deep Learning, model sanity check (no training).

Confirms the model builds correctly and produces the expected output shape on one real image, before writing any
training loop.

Run from the repo root with:
    uv run python scripts/experiments/experiment_07_model_sanity_check.py
"""

from PIL import Image

from retroread.config import ENDAVA_DS5_IMAGES_DIR
from retroread.keypoint_model import GaugeKeypointModel

TARGET_FILENAME = "data/v_0992_f_0000_rgba.png"

def main() -> None:
    model = GaugeKeypointModel(freeze_backbone=True)
    model.eval()

    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_total_params:,}")
    print(f"Trainable parameters (head only): {n_trainable_params:,}")

    image_path = ENDAVA_DS5_IMAGES_DIR / TARGET_FILENAME
    image = Image.open(image_path).convert("RGB")

    input_tensor = model.preprocess(image).unsqueeze(0) # add batch dimension
    print(f"\nInput tensor shape: {tuple(input_tensor.shape)}")

    output = model(input_tensor)
    print(f"Output tensor shape: {tuple(output.shape)}")
    print(f"Output values (normalized [0, 1]): {output.detach().numpy().round(3)}")

    if output.shape == (1, 8):
        print("\nShape check passed: output is (1, 8) as expected.")
    else:
        print(f"\nUNEXPECTED SHAPE -- expected (1, 8), got {tuple(output.shape)}")

if __name__ == "__main__":
    main()