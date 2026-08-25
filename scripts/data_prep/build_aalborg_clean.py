"""
Build the cleaned Aalborg real-world gauge dataset from data/raw into data/processed.

What this does:
- Reads all PNG frames under data/raw/Aalborg/4 Test of videos/man*/
- Drops empty (zero-byte) files
- Copies the remaining valid frames into data/processed/aalborg_clean/man*/
- Writes a manifest CSV recording every kept frame, for reproducibility

Run from the repo root with:
    uv run python scrilpts/build_aalborg_clean.py
"""

import csv
import sys
import re
import shutil
import numpy as np
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found).")

sys.path.insert(0, str(_find_project_root(Path(__file__).resolve()) / "src"))

from retroread.config import AALBORG_TEST_FRAMES_DIR as RAW_DIR, AALBORG_CLEAN_DIR as OUT_DIR, AALBORG_CLEAN_MANIFEST as MANIFEST_PATH, AALBORG_ANGLE_DATA_DIR as ANGLE_DIR

FRAME_PATTERN = re.compile(r"_(\d+)\.png$")

def load_angle_array(gauge_name: str) -> np.ndarray | None:
    """Load angle.npy for a gauge, if it exists. Returns None if missing."""
    angle_path = ANGLE_DIR / f"data {gauge_name}" / "angle.npy"
    if not angle_path.exists():
        return None
    return np.load(angle_path)

def build_clean_dataset() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Expected raw data at '{RAW_DIR}', but it was not found. "
            "Run this script from the repo root."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    total_seen = 0
    total_empty = 0
    total_kept = 0

    gauge_dirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir() and p.name.startswith("man"))

    for gauge_dir in gauge_dirs:
        gauge_name = gauge_dir.name
        out_gauge_dir = OUT_DIR / gauge_name
        out_gauge_dir.mkdir(parents=True, exist_ok=True)

        angle_array = load_angle_array(gauge_name)

        png_files = sorted(gauge_dir.glob("*.png"))

        for png_path in png_files:
            total_seen += 1
            size_bytes = png_path.stat().st_size

            if size_bytes == 0:
                total_empty += 1
                continue

            dest_path = out_gauge_dir / png_path.name
            shutil.copy2(png_path, dest_path)
            total_kept += 1

            # Look up a reference angle from the original paper's algorithm output, if available.
            # 0.0 is a "no reading" sentinel in this data, not a real angle - treat it as missing, not as a value.
            reference_angle = ""
            match = FRAME_PATTERN.search(png_path.name)
            if match and angle_array is not None:
                frame_idx = int(match.group(1))
                if frame_idx < len(angle_array) and angle_array[frame_idx] != 0.0:
                    reference_angle = float(angle_array[frame_idx])

            manifest_rows.append(
                {
                    "gauge": gauge_name,
                    "filename": png_path.name,
                    "source_path": str(png_path),
                    "processed_path": str(dest_path),
                    "size_bytes": size_bytes,
                    "reference_angle": reference_angle,
                }
            )

    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["gauge", "filename", "source_path", "processed_path", "size_bytes", "reference_angle"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Scanned: {total_seen} PNG files across {len(gauge_dirs)} gauges.")
    print(f"Dropped: {total_empty} empty files")
    print(f"Kept: {total_kept} valid frames -> {OUT_DIR}")
    print(f"Manifest: {MANIFEST_PATH} ({len(manifest_rows)} rows.)")

    print("\nPer-gauge breakdown:")
    per_gauge_counts: dict[str, int] = {}
    for row in manifest_rows:
        per_gauge_counts[row["gauge"]] = per_gauge_counts.get(row["gauge"], 0) + 1
    for gauge_name in sorted(per_gauge_counts):
        print(f"  {gauge_name}: {per_gauge_counts[gauge_name]}")

if __name__ == "__main__":
    build_clean_dataset()