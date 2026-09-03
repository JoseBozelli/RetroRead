# Deep Learning: Keypoint Model — Full Experimentation Record

This document covers the complete deep learning effort for RetroRead's gauge-reading
pipeline: hypotheses, architectural decisions, quantitative results, failed
experiments, a post-hoc audit that changed the final conclusion, limitations,
model selection, and deferred future work. It is written to stand alongside
`docs/decision_log.md` and `docs/error_analysis.md` as the authoritative record
of this project's DL work.

---

## 1. Goal and hypothesis

Predict the same four keypoints the classical baseline relies on — dial center,
needle tip, scale minimum, scale maximum — using a small pretrained CNN
(MobileNetV3-Small), transfer-learned given the project's CPU-only, one-week
constraints. The question: can a learned approach match or exceed the classical
Hough-Transform baseline's reading accuracy?

---

## 2. Experiment progression

### v1 — Flattened coordinate regression (Experiments 07–10)

Frozen backbone → global average pool → flatten → dense layers → 8 raw
coordinates, trained with MSE loss.

**Result: 9.0% within-tolerance, 38.9% mean error** (full 200-image val set).
Diagnosis: `Flatten()` after global pooling discards all spatial information —
the head has no signal for *where* in the image a feature came from, only
*that* it occurred somewhere. Confirmed by inspecting predictions directly:
outputs clustered near one location regardless of the input image's actual
content.

### v2 — Heatmap-based prediction (Experiments 11–12)

Replaced flatten+regression with a spatial decoder: the backbone's feature map
is upsampled back through transposed convolutions, producing one small
heatmap per keypoint (a 2D likelihood map), decoded via soft-argmax (a
differentiable weighted-average position).

**Result: 46.5% within-tolerance, 12.09% mean error** (full 200-image val set) —
after fixing an implementation bug along the way (see §5). A real, substantial
improvement, confirming the flatten diagnosis.

### v3 — Crop-staged, higher-resolution decoder, skip connection, combined loss (Experiments 13–14)

Three changes together, motivated by: (a) the model was asked to localize
within a full, cluttered 1024×1024 scene rather than a tightly-framed gauge —
unlike the classical baseline, which only reasons about the region it already
detected; (b) the decoder's coarsest layer (7×7) limits achievable precision;
(c) heatmap-shape loss alone doesn't directly penalize the coordinate error
that actually matters.

Changes: (1) crop to the gauge region (ground-truth bbox + 15% margin,
matching the classical baseline's known bezel-margin finding) before the
model ever sees the image; (2) a third upsampling stage (7→14→28→56) with one
skip connection from an earlier, higher-resolution backbone layer; (3) a
combined loss (heatmap MSE + weighted coordinate L1 via soft-argmax).

**Result: 54.1% within-tolerance, 7.28% mean error** (61-image fair subset,
matching the classical baseline's own edge-filtered evaluation range — see
§6 on comparison methodology). Best DL result achieved.

### v4 — Backbone fine-tuning experiments (Experiments 15–16)

Two controlled sub-experiments testing whether adapting more of the
pretrained backbone would close the remaining gap to classical:

- Differential LR on the single already-unfrozen block: **no improvement.**
- Unfreezing 3 backbone blocks with a three-tier LR structure (very low for
  newly-unfrozen layers, low for the existing block, unchanged for the
  head): **54.1% → 47.5%, mean error 7.28% → 8.88% — regressed.**

Notably, validation *loss* improved throughout this run while downstream
*reading accuracy* got worse — the clearest instance in this project of loss
and the true target metric diverging. Rejected; reverted to the v3 checkpoint.

### v5 — Reading-aware auxiliary loss (Experiments 17–18)

Added a third loss term explicitly computing a differentiable approximation
of the final reading and penalizing its distance from the true value,
motivated by: the classical baseline's needle-detection logic is implicitly
reading-aware (it selects lines by *reach from the pivot*, not raw length,
because that's what the angle calculation needs) — the DL model had no
equivalent signal.

**Result: 14.8% within-tolerance, 25.63% mean error — severely regressed**,
despite training-time reading error looking good (~6%).

**Root cause (post-hoc audit): objective mismatch, not a training failure.**
The training-time reading-loss used a *simplified two-point calibration*
(model's predicted center/tip, ground-truth min/max keypoints, linear
interpolation) — mathematically different from the evaluation pipeline's
*actual* calibration (`fit_scale_calibration` against ground-truth center and
**all** scale-label text positions, a richer multi-point fit). Decreasing the
training-time approximation did not guarantee improving the real evaluation
metric, because they were not the same function. This is a legitimate,
informative negative result: it demonstrates that a reading-aware loss must
use the *exact* production reading calculation, not a simplified stand-in, or
risk optimizing the wrong target entirely. Not pursued further.

---

## 3. Post-hoc audit: was 54.1% actually the ceiling?

Before finalizing v3 as the best result, two cheap, no-retraining diagnostics
were run directly against the existing checkpoint, prompted by a review of
the evaluation methodology:

**Soft-argmax temperature sweep.** Theoretical concern: with heatmap values
near `[0,1]` and default temperature, spatial softmax over a 56×56 grid could
be too diffuse to concentrate probability at the true peak, pulling
predictions toward the grid center. Tested temperatures 1–50 and hard argmax
against the same checkpoint. **Result: temperature=1 (the training-time
default) was optimal; every sharper setting degraded accuracy**, in one case
nearly tripling the error. The theoretical concern was legitimate, but the
model — trained with `soft_argmax_decode` at temperature=1 embedded directly
in its coordinate loss — had already adapted its heatmap amplitudes to be
well-calibrated specifically for that decode setting. Post-hoc sharpening
fought against that learned calibration rather than improving it. Hypothesis
tested and rejected by evidence, not by reasoning alone.

**Crop-domain shift.** The model trains exclusively on ground-truth bbox
crops, but Experiments 14/16/18 evaluate using the classical circle
detector's crop instead (the realistic deployment path — no ground-truth
annotation exists for real photos). This conflates two different questions:
"how good is the model's landmark prediction" vs. "how good is the full
detector+model pipeline." Evaluated both, holding decoding fixed at t=1:

| Evaluation condition | Within ±5% | Mean % scale error |
|---|---|---|
| Classical CV baseline | 80.3% | 4.82% |
| DL + oracle/GT crop | 65.6% | 4.36% |
| Hybrid deployment pipeline: detected crop → DL | 54.1% | 7.28% |

**Important caveat on the middle row:** "DL + oracle/GT crop" measures pointer-reading
performance when gauge localization is idealized via the annotated crop —
it does **not** mean the full DL landmark system (including its own predicted
scale-min/max) achieves 4.36%. Calibration in every DL evaluation row still
uses ground-truth scale-label positions, not the model's own min/max
predictions (see §4). The gap between the oracle-crop and deployment rows
(65.6%→54.1%, 4.36%→7.28%) isolates **crop-domain shift** as a real,
specific, and — importantly — fixable source of error.

---

## 4. Evaluation scope, stated explicitly

For controlled comparison of pointer-reading performance, scale calibration
was held fixed using ground-truth annotations (center position and all
scale-label positions/values) in every DL evaluation. The model's predicted
scale-min and scale-max keypoints were trained as auxiliary geometry targets
(supervised via the heatmap and coordinate losses, alongside center/tip) but
were **not** used in the headline reading metric. This isolates the
comparison to pointer-angle prediction quality specifically, matching how the
classical baseline's own evaluation (Experiments 05/06) also used a
ground-truth center as its calibration anchor. A fully end-to-end DL
evaluation — using the model's own predicted min/max for calibration too —
was not performed and is noted as future work.

---

## 5. Known implementation issue (not retroactively fixed)

`crop_heatmap_dataset.py`'s `make_gaussian_heatmap` target generation uses
`grid_x = norm_x * HEATMAP_SIZE` rather than the technically correct
`norm_x * (HEATMAP_SIZE - 1)` (valid grid indices run 0 to `HEATMAP_SIZE-1`,
not `HEATMAP_SIZE`). This was discovered during the post-hoc decode audit.
**Deliberately not changed in the codebase**, and no checkpoint was
retrained against a corrected version: `best_crop_heatmap_model.pt` (and
downstream checkpoints derived from it) were trained against the *existing*
implementation, and silently editing the encoding now would make the current
code no longer reproduce the reported results. This is intentional — an MLOps
principle worth stating plainly: don't rewrite history after discovering an
implementation defect in an already-reported result. A corrected encoding is
listed as future work (§7), to be implemented as an explicitly separate,
versioned path if pursued.

A separate, smaller bug (an MLflow run mislabeled `exp14_...` instead of
`exp18_...` in the reading-aware evaluation script) was metadata-only and
was corrected, since it does not affect model reproducibility.

---

## 6. Final comparison and conclusion

| Method | Within ±5% tolerance | Mean % scale error |
|---|---|---|
| **Classical CV (Hough Transform)** | **80.3%** | **4.82%** |
| DL v1 — flattened coordinate regression | 9.0%* | 38.9%* |
| DL v2 — heatmap decoder | 46.5%* | 12.09%* |
| DL v3 (final) — crop-staged, skip connection, combined loss | 54.1% | 7.28% |
| DL v3 + oracle/GT crop (diagnostic, not deployable) | 65.6% | 4.36% |
| DL v4 — backbone fine-tuning variants | 47.5% (worse) | 8.88% (worse) |
| DL v5 — reading-aware loss | 14.8% (worse) | 25.63% (worse) |

*v1/v2 evaluated on the full 200-image validation set, prior to restricting
evaluation to the classical baseline's 61-image fair-comparison subset;
directionally comparable, not precisely so. v3 onward uses the fair subset.

**Conclusion.** The initial coordinate-regression architecture struggled with
spatial localization; replacing it with heatmap-based prediction improved
learning substantially. Cropping the input around the gauge, increasing
heatmap resolution, adding a skip connection, and combining heatmap and
coordinate supervision produced the strongest DL model. A post-hoc audit
showed the model's 54.1% deployment result was partly limited by a
train/inference crop mismatch — idealized ground-truth crops raised
within-tolerance performance to 65.6% and lowered mean error to 4.36%.
Additional backbone fine-tuning and a reading-aware auxiliary loss were each
tested as targeted, well-justified final experiments; neither improved
downstream performance, and both were rejected with documented root causes
rather than abandoned without explanation.

The classical CV baseline remains substantially more reliable at the ±5%
acceptance threshold (80.3% vs. 54.1%) and is retained as RetroRead's
production candidate. The DL architecture is a promising, well-diagnosed
research direction — not deployed, because it is not yet the better system,
not because deep learning was abandoned prematurely.

This is, itself, a legitimate product decision: RetroRead's business goal is
reliable gauge reading, not deployment of the more sophisticated algorithm
for its own sake.

---

## 7. Deferred future work

- **Crop-jitter augmentation.** Train on deliberately perturbed crops
  (position/scale jitter mimicking the classical detector's actual error
  distribution) instead of only perfect ground-truth crops, so the model
  learns robustness to the crop distribution it actually receives at
  inference — directly targets the 65.6%→54.1% crop-domain-shift gap
  identified in §3.
- **Corrected heatmap target encoding** (`HEATMAP_SIZE - 1`), implemented as
  an explicitly separate, versioned path rather than silently replacing the
  current implementation (§5).
- **Fully end-to-end DL evaluation** using the model's own predicted
  scale-min/max for calibration, rather than ground-truth (§4) — a stricter,
  more realistic measure of complete system performance.
- **Reading-aware loss, corrected** to use the exact production calibration
  function (multi-point `fit_scale_calibration`), not the simplified
  two-point approximation that caused v5's regression.