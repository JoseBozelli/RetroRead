"""
Check alignment between:
  - PNG frame filenames in data/raw/Aalborg/4 Test of videos/man*/  (e.g. man2_145.png -> frame index 145)
  - errorfile.txt frame indices in data/raw/Aalborg/5 Data from run on raw videos/data man*/
  - angle.npy values at those same indices

Goal: decide whether folder 5's angle.npy is usable as a reference value for
the frames we actually have, and whether errorfile.txt explains the empty
PNG frames found in folder 4.

Run from the repo root with:
    uv run python scripts/check_aalborg_alignment.py
"""

import re
import sys
from pathlib import Path

import numpy as np

def _find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found).")

sys.path.insert(0, str(_find_project_root(Path(__file__).resolve()) / "src"))

from retroread.config import AALBORG_TEST_FRAMES_DIR as FRAMES_DIR, AALBORG_ANGLE_DATA_DIR as DATA_DIR

FRAME_PATTERN = re.compile(r"_(\d+)\.png$")


def get_frame_indices(gauge_dir: Path) -> tuple[set[int], set[int]]:
    """Return (all_indices, empty_indices) for a gauge's PNG files."""
    all_idx = set()
    empty_idx = set()
    for png_path in gauge_dir.glob("*.png"):
        match = FRAME_PATTERN.search(png_path.name)
        if not match:
            continue
        idx = int(match.group(1))
        all_idx.add(idx)
        if png_path.stat().st_size == 0:
            empty_idx.add(idx)
    return all_idx, empty_idx


def get_error_indices(errorfile_path: Path) -> set[int]:
    """Parse errorfile.txt: first line is the video filename, rest are frame numbers."""
    indices = set()
    with errorfile_path.open() as f:
        lines = f.readlines()
    for line in lines[1:]:  # skip first line (video filename)
        line = line.strip()
        if line.isdigit():
            indices.add(int(line))
    return indices


def main() -> None:
    gauge_names = sorted(p.name for p in FRAMES_DIR.iterdir() if p.is_dir() and p.name.startswith("man"))

    for gauge_name in gauge_names:
        print(f"\n{'=' * 60}")
        print(f"{gauge_name}")
        print("=" * 60)

        frames_gauge_dir = FRAMES_DIR / gauge_name
        data_gauge_dir = DATA_DIR / f"data {gauge_name}"

        all_idx, empty_idx = get_frame_indices(frames_gauge_dir)
        usable_idx = all_idx - empty_idx

        # find errorfile (name varies: "errorfile.txt" or "errorfile manX.txt")
        errorfile_candidates = list(data_gauge_dir.glob("errorfile*.txt"))
        angle_path = data_gauge_dir / "angle.npy"

        if not errorfile_candidates or not angle_path.exists():
            print("  Missing errorfile or angle.npy, skipping.")
            continue

        error_idx = get_error_indices(errorfile_candidates[0])
        angle = np.load(angle_path)

        print(f"  Total frames in '4 Test of videos': {len(all_idx)} (empty: {len(empty_idx)}, usable: {len(usable_idx)})")
        print(f"  angle.npy length: {len(angle)}")
        print(f"  errorfile.txt flagged frames: {len(error_idx)}")

        # Overlap: do empty PNG frames match errorfile-flagged frames?
        overlap_empty_error = empty_idx & error_idx
        if empty_idx:
            pct = 100 * len(overlap_empty_error) / len(empty_idx)
            print(f"  Empty PNGs also flagged in errorfile.txt: {len(overlap_empty_error)}/{len(empty_idx)} ({pct:.0f}%)")

        # For usable frames, check angle.npy values: how many are exactly 0.0 (suspicious) vs non-zero
        in_range_usable = [i for i in usable_idx if i < len(angle)]
        zero_angle_count = sum(1 for i in in_range_usable if angle[i] == 0.0)
        out_of_range = [i for i in usable_idx if i >= len(angle)]

        print(f"  Usable frames with valid angle.npy index: {len(in_range_usable)}/{len(usable_idx)}")
        if out_of_range:
            print(f"  Usable frames with OUT-OF-RANGE index (angle.npy too short): {len(out_of_range)}")
        if in_range_usable:
            pct_zero = 100 * zero_angle_count / len(in_range_usable)
            print(f"  Of those, angle.npy == 0.0 exactly: {zero_angle_count} ({pct_zero:.0f}%)")

        # Show a handful of sample (frame_index, angle_value) pairs for manual sanity check
        sample_idx = sorted(in_range_usable)[:5]
        print(f"  Sample (frame_index -> angle value):")
        for i in sample_idx:
            flagged = " [in errorfile]" if i in error_idx else ""
            print(f"    {i} -> {angle[i]:.4f}{flagged}")


if __name__ == "__main__":
    main()