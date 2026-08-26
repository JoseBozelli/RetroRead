"""
Diagnostic: check whether the full coco.json (as opposed to the kpts-only subset) carries additional fields --
specifically, an actual numeric scale value -- not present in train__kpts_coco.json

Run from the repo root with:
    uv run python scripts/exploratory/debug_full_coco.py
"""

import json
from pathlib import Path

from retroread.config import ENDAVA_DS5_COCO as COCO_PATH

def main() -> None:
    with COCO_PATH.open() as f:
        coco = json.load(f)

    print("Categories in full coco.json:")
    for cat in coco["categories"]:
        print(f" id={cat['id']} name={cat['name']} keys={list(cat.keys())}")

    print(f"\nTotal annotations: {len(coco['annotations'])}")

    ann_for_image_1 = [a for a in coco["annotations"] if a["image_id"] == 1]
    print(f"\nAll annotations for image_id=1 ({len(ann_for_image_1)} total):\n")

    for ann in ann_for_image_1:
        print("---")
        print(f" category: {ann.get('category_name')}")
        for k, v in ann.items():
            if isinstance(v, list) and len(v) > 10:
                print(f" {k}: [list of {len(v)} items, omitted]")
            elif isinstance(v, dict):
                print(f" {k}: {{dict, omitted}}")
            else:
                print(f" {k}: {v}")

if __name__ == "__main__":
    main()