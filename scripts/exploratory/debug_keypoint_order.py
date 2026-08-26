"""
Diagnostic: find the keypoint ordering (which index in the flat keypoints array corresponds to which
named point) from the COCO categories definition.

Run from the repo root with:
    uv run python scripts/exploratory/debug_keypoing_order.py
"""

import json
from pathlib import Path

from retroread.config import ENDAVA_DS5_TRAIN_KPTS_COCO as COCO_PATH

def main() -> None:
    with COCO_PATH.open() as f:
        coco = json.load(f)

    print("Top-level keys:", list(coco.keys()))
    print()

    if "categories" in coco:
        for cat in coco["categories"]:
            print("Category:", cat.get("name"))
            print(" keys:", list(cat.keys()))
            if "keypoints" in cat:
                print(" keypoint order:")
                for i, kp_name in enumerate(cat["keypoints"]):
                    print(f" index{i}: {kp_name}")
            if "skeleton" in cat:
                print(" skeleton:", cat["skeleton"])
    else:
        print("No 'categories' key found in this file.")

    # Also show one full raw annotation for cross-reference
    print("\nSample annotation (first one):")
    ann = coco["annotations"][0]
    for k, v in ann.items():
        if k == "keypoints":
            print(f" {k}: {v}")
        elif isinstance(v, list) and len(v) > 10:
            print(f" {k}: [list of {len(v)} items, omitted]")
        else:
            print(f" {k}: {v}")

if __name__ == "__main__":
    main()