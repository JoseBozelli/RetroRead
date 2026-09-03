"""
Checks every training sample's target coordinates for NaN, infinity, or
values far outside the expected [0,1] normalized range -- to find the
actual cause of the overflow warning / NaN loss in heatmap training.

Run from the repo root with:
    uv run python scripts/exploratory/debug_target_values.py
"""

import math

from retroread.annotations import load_keypoint_training_data
from retroread.config import ENDAVA_DS5_TRAIN_KPTS_COCO


def main() -> None:
    samples = load_keypoint_training_data(ENDAVA_DS5_TRAIN_KPTS_COCO)
    print(f"Checking {len(samples)} samples...")

    n_bad = 0
    for sample in samples:
        target = sample["target"]
        for i, v in enumerate(target):
            if math.isnan(v) or math.isinf(v) or v < -1 or v > 2:
                print(f"BAD VALUE: {sample['file_name']}  index={i}  value={v}")
                print(f"  full target: {target}")
                n_bad += 1
                break

    print(f"\n{n_bad}/{len(samples)} samples have a suspicious target value.")


if __name__ == "__main__":
    main()