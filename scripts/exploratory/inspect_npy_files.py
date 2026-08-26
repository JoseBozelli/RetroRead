"""
Inspect the .npy files in data/raw/Aalborg/5 Data from run on raw videos/.

Purpose: figure out what these files actually contain (shape, dtype, sample, values)
before deciding whether they can serve as ground-truth readings for the real-world case-study evaluation.

Run from the repo root with uv run python scripts/exploratory/inspect_npy_files.py
"""

from pathlib import Path
import numpy as np

from retroread.config import AALBORG_ANGLE_DATA_DIR as RAW_DIR

def inspect_file(npy_path: Path) -> None:
    try:
        arr = np.load(npy_path, allow_pickle=True)
    except Exception as e:
        print(f"  [FAILED TO LOAD] {e}")
        return

    print(f" shape: {arr.shape}")
    print(f" dtype: {arr.dtype}")

    flat = arr.flatten() if hasattr(arr, "flatten") else arr
    n_preview = min(10, len(flat)) if hasattr(flat, "__len__") else 0

    if n_preview > 0:
        print(f" first {n_preview} values: {flat[:n_preview]}")
        try:
            print(f" min: {np.nanmin(arr):.4f} max: {np.nanmax(arr):.4f} mean {np.nanmean(arr):.4f}")
        except (TypeError, ValueError):
            print(" (non-numerical contents, skiping min/max/mean)")
    else:
        print(f" value: {arr}")

def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Expected raw data at '{RAW_DIR}', not found.")

    gauge_dirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir() and p.name.startswith("data man"))

    for gauge_dir in gauge_dirs:
        print(f"\n{'=' * 60}")
        print(f"{gauge_dir.name}")
        print("=" * 60)

        npy_files = sorted(gauge_dir.glob("*.npy"))
        for npy_path in npy_files:
            print(f"\n{npy_path.name}:")
            inspect_file(npy_path)

        txt_files = sorted(gauge_dir.glob("*.txt"))
        for txt_path in txt_files:
            print(f"\n{txt_path.name} (first 5 lines)")
            with txt_path.open() as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    print(f" {line.rstrip()}")

if __name__ == "__main__":
    main()