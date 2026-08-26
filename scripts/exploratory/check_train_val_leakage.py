"""
Checks for exact-duplicate images between train__kpts_coco.json and val__kpts_coco.json, using file content hashes.

Run from the repo root with:
    uv run python scripts/exploratory/check_train_val_leakage.py
"""

import hashlib
import json
from pathlib import Path

from retroread.config import ENDAVA_DS5_DIR, ENDAVA_DS5_IMAGES_DIR

TRAIN_COCO = ENDAVA_DS5_DIR / "train__kpts_coco.json"
VAL_COCO = ENDAVA_DS5_DIR / "val__kpts_coco.json"

def get_filenames(coco_path: Path) -> list[str]:
    with coco_path.open() as f:
        coco = json.load(f)
    return [img["file_name"] for img in coco["images"]]

def hash_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def main() -> None:
    train_files = get_filenames(TRAIN_COCO)
    val_files = get_filenames(VAL_COCO)

    print(f"Train images: {len(train_files)}")
    print(f"Val images: {len(val_files)}")

    overlap_by_name = set(train_files) & set(val_files)
    print(f"\nOverlap by filename: {len(overlap_by_name)}")

    print("\nHashing all images to check for duplicate CONTENT (different filenames, same image)...")
    train_hashes = {hash_file(ENDAVA_DS5_IMAGES_DIR / fn): fn for fn in train_files}
    val_hashes = {hash_file(ENDAVA_DS5_IMAGES_DIR / fn): fn for fn in val_files}

    overlap_by_content = set(train_hashes.keys()) & set(val_hashes.keys())
    print(f"Overlap by content hash: {len(overlap_by_content)}")

    if overlap_by_content:
        print("\nExample overlapping pairs:")
        for h in list(overlap_by_content)[:5]:
            print(f" train: {train_hashes[h]} <-> val: {val_hashes[h]}")
    else:
        print("\nNo content-level leakage found -- train/val split appears clean.")

if __name__ == "__main__":
    main()