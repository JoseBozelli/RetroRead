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