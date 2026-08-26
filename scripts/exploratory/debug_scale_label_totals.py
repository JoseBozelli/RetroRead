import json
from pathlib import Path

from retroread.config import ENDAVA_DS5_TRAIN_KPTS_COCO as COCO_PATH

with COCO_PATH.open() as f:
    coco = json.load(f)

scale_label_cat = next((c for c in coco["categories"] if c["name"] == "scale-label"), None)
cat_id = scale_label_cat["id"]

all_matches = [a for a in coco["annotations"] if a["category_id"] == cat_id]
print(f"Total scale-label annotations across entire file: {len(all_matches)}")

if all_matches:
    print("\nFirst match, full contents:")
    for k, v in all_matches[0].items():
        if isinstance(v, list) and len(v) > 10:
            print(f" {k}: [list of {len(v)} items, omitted]")
        elif isinstance(v, dict):
            print(f" {k}: {{dict, omitted}}")
        else:
            print(f" {k}: {v}")