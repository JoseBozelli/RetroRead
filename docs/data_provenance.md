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
- **Unused in this project:** `1 Training videos/`, `2 Test videos/`,
  `3 Misc handheld videos/` (raw/edited video, not extracted frames),
  `5 Data from run on raw videos/` (contains `angle.npy`/`alpha.npy`/
  `beta.npy` per gauge — likely ground-truth or algorithm output from the
  original paper; **to be investigated** as a possible ground-truth source
  for the real-world case-study table before evaluation).