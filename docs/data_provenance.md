# Data Provenance

This document records exactly where each dataset came from, when it was
downloaded, under what license, and how to verify a copy is authentic.
See `docs/decision_log.md` for the reasoning behind dataset selection and
any cleaning/deduplication steps applied.

---

## Endava Synthetic Gauge Dataset (DS5.0 + DS6.0)

- **Source URL:** https://www.kaggle.com/datasets/endava/synthetic-data-for-precision-gauge-reading
- **Downloaded:** August 21, 2026
- **License:** CC BY-NC-SA 4.0 (non-commercial, share-alike)
- **Contents used:** `sample_synth_datasets/` (ds5.0: 1000 images, ds6.0: 500 images), `inference_man_test_videos/`
- **SHA256 (original .zip):** `e3451b832b8cbbb7889c8ec763eb5e405a8cac0a40fab20c48f283c68054e042`
- **Notes:** Originally planned to use Cambridge SyntheticGauges/RealGauges
  (Howells, Charles & Cipolla, CVPR 2021) as the primary synthetic dataset;
  its hosting link (jjcvision.com) was found dead at time of access. See
  `docs/decision_log.md` for full reasoning behind the substitution.

---

## Aalborg Pressure Gauge Reader Data

- **Source URL:** https://www.kaggle.com/datasets/juliusgrassme/pressure-gauge-reader-data
- **Downloaded:** August 21, 2026
- **License:** CC BY-SA 4.0 (share-alike, commercially permissive)
- **Contents used:** `4 Test of videos/` (real-world gauge photo frames, 7 gauges)
- **SHA256 (original .zip):** `21bee2a626e3130ba643215b0556d9d93b5b89669ca3ae36bef297d20a2478f6`
- **Notes:** Archive contained a fully duplicated `Data/` subtree (removed)
  and 175 empty/corrupted PNG frames out of 532 (filtered out during
  processing, not deleted from `data/raw`). See `docs/decision_log.md` for
  full cleaning details and per-gauge usable frame counts.
- **Unused folders and why:**
  - `1 Training videos/`, `2 Test videos/`, `3 Misc handheld videos/` —
    raw and edited video files (.mov/.mp4), not extracted image frames.
    RetroRead's MVP scope is static-image gauge reading, so no frame
    extraction was performed from these; the "4 Test of videos" folder
    already provides pre-extracted frames, which is a closer match to our
    actual use case (photo, not video, input).
  - `5 Data from run on raw videos/` — investigated via
    `scripts/inspect_npy_files.py`. Contains, per gauge: `alpha.npy` and
    `beta.npy` (uniformly zero across all 7 gauges — purpose unknown,
    likely unused/deprecated output channels; **not used**), `angle.npy`
    (real-valued angle readings in radians, one per raw video frame — this
    is the original paper's own classical image-processing algorithm
    output, **not independently verified ground truth**), and
    `errorfile.txt` (frame indices where the original algorithm failed to
    produce a reading).
  - **Decision:** `angle.npy` will be used only as a *reference/comparison*
    value ("does RetroRead's output agree with a published prior method's
    output on the same frame?"), never presented as ground truth. Whether
    frame indices in `angle.npy` and `errorfile.txt` actually align with
    the specific frames sampled in "4 Test of videos" has not yet been
    verified — **open question**, to be checked before this data is used
    in any evaluation table. If alignment can't be confirmed, this folder
    will be excluded from evaluation entirely and the real-world case
    study will rely on manual reading of the "4 Test of videos" frames
    instead.