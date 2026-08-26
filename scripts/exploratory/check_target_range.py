"""
Checks whether any normalized keypoint targets fall outside [0,1] -- would explain the high (~1.0) training loss
in Experiment 08.

Run from the repo root with:
    uv run python scripts/exploratory/check_target_range.py
"""

from retroread.annotations import load_keypoint_training_data
from retroread.config import ENDAVA_DS5_TRAIN_KPTS_COCO

N_SUBSET = 30   # match Experiment 08's subset size

def main() -> None:
    all_samples = load_keypoint_training_data(ENDAVA_DS5_TRAIN_KPTS_COCO)
    subset = all_samples[:N_SUBSET]

    out_of_range_count = 0
    for sample in subset:
        target = sample["target"]
        if any(v < 0 or v > 1 for v in target):
            out_of_range_count += 1
            print(f"{sample['file_name']}: target={[round(v, 3) for v in target]}")

    print(f"\n{out_of_range_count}/{len(subset)} samples have at least one out-of-[0,1] target value.")

if __name__ == "__main__":
    main()