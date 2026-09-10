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

---
### Classical baseline — max_radius_fraction tuning (Experiments 01–02)

**Observation:** Experiment 01 (default max_radius_fraction=0.6) achieved
100% detection rate but only 94.3% accuracy (250/265) against bbox-center
comparison, mean error 11.8px. Visual inspection of 5 inaccurate cases
showed a consistent pattern: the detected circle was centered on large
background structures (warehouse roof trusses, industrial piping) rather
than the gauge, likely because the permissive max radius allowed these
larger background patterns to outscore the actual gauge circle.

**Decision:** Tightened max_radius_fraction to 0.35 (Experiment 02).
Detection rate unchanged (100%), accuracy improved to 97.7% (259/265),
mean error dropped to 6.6px. Adopted as the new default in
`find_gauge_circle`.

**Consequence:** 6 remaining inaccurate cases not further investigated at
this stage — deferred to the full Section 15 error analysis once needle
detection and reading conversion exist, since categorizing a circle-only
failure in isolation would need to be redone once the complete pipeline
is in view.

---
### Classical baseline — needle detection (Experiments 03–04)

**Observation:** Single-image test (Exp03) initially misidentified the
needle's counterweight as the pointer; fixed by selecting the Hough line
candidate by maximum reach from center rather than raw segment length.
Batch validation (Exp04) against real ground truth (COCO `dial_tip`
keypoint, not an approximation) achieved 95.8% needle detection rate,
90.9% angle accuracy within 5°, mean angle error 4.44°.

**Decision:** Accepted as the classical baseline's needle-detection
component. Remaining failures (11 non-detections, borderline-accuracy
cases) deferred to the full Section 15 error analysis rather than
investigated individually now.

---
## Deep learning investigation

The DL architecture search (nine models, YOLO pose, the tip refiner, and
the final production ensemble) is documented in full in
`docs/deep_learning.md`, with the complete scannable experiment list in
`docs/experiment_master_table.md`. Not duplicated here to avoid the two
documents drifting out of sync as that investigation progressed well
beyond an early four-model comparison.