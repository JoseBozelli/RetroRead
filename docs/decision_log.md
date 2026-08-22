### Aalborg dataset cleanup and validation

**Observation:** The Aalborg real-world gauge dataset (12.4GB zip) contained
a fully duplicated copy of its entire contents nested under an extra `Data/`
subfolder, and 175 of 532 PNG frames (33%) under "4 Test of videos" were
zero-byte empty files, concentrated in the man2 (138/245) and man6 (37/65)
gauge subfolders.

**Decision:** Removed the redundant `Aalborg/Data/` subtree. Excluded empty
files from the usable dataset. Verified all remaining 357 PNGs are valid,
uncorrupted image files via `file` command inspection.

**Alternatives considered:** Using the raw/edited video files directly
instead of pre-extracted frames, to recover the missing man2/man6 frames.
Deferred — out of scope for MVP; usable frame counts (23–107 per gauge)
are sufficient for a real-world case-study table.

**Consequence:** Real-world case-study set covers 7 physical gauges with
usable frame counts ranging from 23 (man1) to 107 (man2), rather than a
uniform count across all 7. This asymmetry will be reported transparently
in the case-study table rather than averaged into a single aggregate metric.

**Note on "raw":** `data/raw/Aalborg` is not strictly byte-identical to the
original Kaggle download — the verified duplicate `Data/` subtree was
removed. This was a lossless deduplication (every removed file had an
identical hash to one that was kept), not a data-altering edit, and is
fully reconstructable by re-downloading the original archive. The empty
frame files were left in place in `data/raw` and are filtered out only in
`data/processed`.

---
### Aalborg angle.npy reference values — resolved

**Observation:** `alpha.npy`/`beta.npy` were uniformly zero across all 7
gauges (purpose still unconfirmed). Alignment check (`scripts/check_aalborg_alignment.py`)
showed `angle.npy` frame indices align correctly with "4 Test of videos"
filenames, but `0.0` values are heavily concentrated (man2: 99%, man5: 70%,
man6: 18%) and correlate with `errorfile.txt`-flagged frames — confirming
`0.0` is a "detection failed" sentinel from the original algorithm, not a
real angle reading.

**Decision:** `angle.npy` values are used as an optional `reference_angle`
column in the processed manifest, populated only when the value is
non-zero. Explicitly documented as the original paper's own classical
algorithm output — a comparison point, not verified ground truth.
`alpha.npy`/`beta.npy` remain unused. man2 and man5 will have substantially
reduced reference-angle coverage as a result; this is reported transparently
per-gauge rather than papered over.

**Consequence:** Real-world case-study table will show a `reference_angle`
column that is blank for a meaningful fraction of man2/man5 frames — by
design, not by bug.

---
### Endava bbox annotation scope

**Observation:** The `bbox` field in Endava's COCO annotations covers only
the dial face (glass + markings), not the full physical gauge including its
outer bezel. Confirmed by drawing bbox overlays on sample images — the
bezel visibly extends beyond the annotated box on some images.

**Decision:** Any edge-proximity or framing checks based on `bbox` use a
generous margin (15%) as a buffer for the unlabeled bezel, since the bezel's
true extent isn't directly annotated. Development sample image selected
using this corrected filter: `data/v_0992_f_0000_rgba.png` (verified
visually to have the full bezel inside the frame).