"""
Central location for all data paths used across RetroRead scripts.

Single source of truth: every script imports paths from here instead of redefining them locally.
If the repo's data folder structure ever changes, it changes in exactly one place.

All paths are relative to the project root and assume scripts are run from the project root (
e.g., 'uv run python scripts/whatever.py')
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# --- Endava synthetic dataset ---
ENDAVA_DIR = DATA_RAW_DIR / "Endava"
ENDAVA_DS5_DIR = ENDAVA_DIR / "sample_synth_datasets" / "ds5.0"
ENDAVA_DS5_IMAGES_DIR = ENDAVA_DS5_DIR  # file_name in COCO already includes "data/" prefix
ENDAVA_DS5_TRAIN_KPTS_COCO = ENDAVA_DS5_DIR / "train__kpts_coco.json"
ENDAVA_DS5_COCO = ENDAVA_DS5_DIR / "coco.json"

ENDAVA_DS6_DIR = ENDAVA_DIR / "sample_synth_datasets" / "ds6.0"
ENDAVA_DS6_TRAIN_KPTS_COCO = ENDAVA_DS6_DIR / "train__kpts_coco.json"

# --- Aalborg real-world dataset ---
AALBORG_DIR = DATA_RAW_DIR / "Aalborg"
AALBORG_TEST_FRAMES_DIR = AALBORG_DIR / "4 Test of videos"
AALBORG_ANGLE_DATA_DIR = AALBORG_DIR / "5 Data from run on raw videos"

AALBORG_CLEAN_DIR = DATA_PROCESSED_DIR / "aalborg_clean"
AALBORG_CLEAN_MANIFEST = DATA_PROCESSED_DIR / "aalborg_clean_manifest.csv"