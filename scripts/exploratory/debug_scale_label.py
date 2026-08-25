"""
Diagnostic: inspect the 'scale-label' category's annotation to find where the gauge's actual numeric
scale range (min-max values) is stored.

Run from the repo root with:
    uv run python scripts/debug_scale_label.py
"""
import sys
import json
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found).")

sys.path.insert(0, str(_find_project_root(Path(__file__).resolve()) / "src"))

from retroread.config import ENDAVA_DS5_TRAIN_KPTS_COCO as COCO_PATH

def main() -> None:
    with COCO_PATH.open() as f:
        coco = json.load(f)

    # Find the scale-label category id
    scale_label_cat = next((c for c in coco["categories"] if c["name"] == "scale-label"), None)
    if scale_label_cat is None:
        print("No 'scale-label' category found.")
        return

    print("scale-label category:", scale_label_cat)
    cat_id = scale_label_cat["id"]

    # Find annotations belonging to image_id=1 (same image used in earlier debug scripts)
    matches = [a for a in coco["annotations"] if a["category_id"] == cat_id and a["image_id"] == 1]
    print(f"\n{len(matches)} scale-label annotations for image_id=1:\n")

    for ann in matches:
        print("---")
        for k, v in ann.items():
            if isinstance(v, list) and len(v) > 10:
                print(f" {k}: [list of {len(v)} items, omitted]")
            elif isinstance(v, dict):
                print(f" {k}: {{dict, omitted}}")
            else:
                print(f" {k}: {v}")

if __name__ == "__main__":
    main()