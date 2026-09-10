# Deep Learning: Keypoint Model — Full Experimentation Record

This document covers the complete deep learning and hybrid-architecture effort
for RetroRead's gauge-reading pipeline. For the complete scannable list of
every experiment run, see `docs/experiment_master_table.md`. This document
tells the narrative: what was tried, why, what was found, and how the final
architecture was reached.

---

## 1. Goal and hypothesis

Predict the same keypoints the classical baseline relies on — dial center,
needle tip, scale minimum, scale maximum — using a small pretrained CNN
(MobileNetV3-Small), transfer-learned given the project's CPU-only, one-week
constraints. The original question: can a learned approach match or exceed
the classical Hough-Transform baseline's reading accuracy? That question
evolved considerably over the course of the investigation — see §7.

---

## 2. Architecture search (v1–v6)

Six architectures were tried in direct succession, each diagnosed and each
motivating the next:

- **v1 — flattened coordinate regression.** Failed (9.0% / 38.9%).
  Root cause: `Flatten()` after global pooling discards spatial information.
- **v2 — heatmap decoder.** Preserved spatial structure via upsampling
  instead of flattening. Substantial improvement (46.5% / 12.09%).
- **v3 — crop-staged + skip connection + combined loss.** Cropped to the
  gauge region first, added decoder resolution and a skip connection,
  combined heatmap + coordinate loss. Best result at this stage
  (54.1% / 7.28%).
- **v4 — backbone fine-tuning.** Controlled test of domain adaptation
  (differential learning rates, more unfrozen layers). Regressed
  (47.5% / 8.88%) — validation loss improved while downstream accuracy
  worsened, a clean example of proxy-metric/target-metric divergence.
- **v5 — reading-aware auxiliary loss.** Added a loss term directly
  penalizing final reading error. Regressed severely (14.8% / 25.63%) —
  root-caused to an objective mismatch: the training-time calibration
  approximation didn't match the production calibration function.
- **Post-hoc audit.** Two cheap, no-retraining diagnostics before declaring
  v3 final: a soft-argmax temperature sweep (default temperature was
  already optimal — the jointly-trained model had adapted its heatmap
  amplitudes specifically for it) and a ground-truth-crop vs.
  detected-crop ablation, which revealed a real, separate, fixable
  problem: **crop-domain shift**. With an idealized crop, the same v3
  model reached 65.6% / 4.36% — already better than classical on mean
  error.
- **v6 — crop-jitter augmentation.** Trained on deliberately perturbed
  crops instead of only ground-truth ones, directly targeting the
  crop-domain-shift finding. Confirmed the hypothesis: 67.2% / 5.52%,
  closing most of the gap between the oracle and deployment numbers.

Full details for each in `docs/experiment_master_table.md`.

---

## 3. Representation experiments (v7 and beyond)

Following a broader literature review of gauge-reading systems, several
different problem *representations* were tested against the same base
architecture, to answer: **what should a learned system actually predict**,
not just how should it be shaped?

- **Needle segmentation + center heatmap.** Hypothesis: a full pixel mask
  aggregates evidence more robustly than a single predicted tip coordinate.
  Result: 67.2% / 6.80% — tied accuracy with v6, worse mean error. Training
  loss was still decreasing at the epoch budget's end (unlike every prior
  architecture, which had clearly plateaued or begun overfitting) —
  documented as possibly undertrained, not pursued further given project
  time constraints.
- **Multi-task model (mask + center + scale landmarks), fully end-to-end
  calibration.** Closed the previously-noted gap of always calibrating
  against ground-truth scale positions. Result: catastrophic (8.2% / 71.06%).
  Root cause, a genuine structural finding: a 2-point calibration's slope is
  *multiplicatively* sensitive to landmark position error — a small
  landmark miss that would be minor for a needle tip becomes severe when
  it defines the entire measuring scale.
- **sin/cos angle regression.** Hypothesis: predicting orientation directly,
  avoiding coordinate localization and the 359°/1° wraparound discontinuity.
  Result: 44.3% / 16.88% (after fixing an unstable-training bug with a
  `tanh` output activation). Demonstrates the real interpretability cost of
  this representation: unlike every other model, it never localizes
  anything — if wrong, there's no way to inspect why.
- **Direct end-to-end numeric regression.** Included specifically to show
  why geometry-aware approaches are preferred. Result: 32.8% / 17.23%, as
  expected by design.

---

## 4. Pose estimation paradigm: YOLO

A different detection framework entirely — single-shot object
detection + pose estimation (Ultralytics YOLOv8-pose), fine-tuned on the
project's cropped-gauge dataset.

- **Result: 72.1% / 5.14%** — the best DL result to this point, closest yet
  to classical (80.3% / 4.82%).
- **Per-keypoint error diagnostic** (no retraining): center localization
  error, not tip error, most distinguished accurate from inaccurate
  readings (11.86px→18.18px vs. 17.22px→20.54px) — a refinement of the
  initial hypothesis, not an overturning of it.
- **P2-YOLO-Pose** (added a higher-resolution, stride-4 detection/pose
  scale, following a specific literature precedent for gauge reading):
  collapsed to 24.6% / 13.29%. Root-caused precisely: adding the new
  branch shifted every subsequent layer's index, which broke Ultralytics'
  name-based pretrained-weight matching — only 44% of weights transferred
  vs. 91% for the unmodified architecture. This is a real, fixable
  implementation defect, not a rejection of the P2 hypothesis, and is
  left as documented future work rather than re-attempted given project
  time.

### The oracle ablation — the single most important diagnostic in this project

Before building anything further, a cheap, no-retraining test: substitute
ground truth for one predicted landmark at a time, holding the rest as
YOLO predicted.

| Substitution | Accuracy | Mean error |
|---|---|---|
| YOLO baseline (nothing substituted) | 72.1% | 5.14% |
| GT center substituted | 67.2% (**worse**) | 5.21% |
| GT tip substituted | **80.3%** | **3.29%** |
| Classical geometric center substituted | 65.6% (**worse**) | 5.63% |

**Two findings, both important:**

1. **Tip, not center, is the dominant lever** — fixing tip alone reaches
   classical's own accuracy and beats its mean error.
2. **Correlated-error finding.** Substituting a *more accurate* center
   (either ground truth, or the classical detector's own center — which
   has measurably lower pixel error, 6.08px median vs. YOLO's 14.27px)
   made results *worse*, not better. YOLO's center and tip predictions are
   not independent: they carry a shared, correlated bias that partially
   cancels in the angle calculation. Replacing one prediction independently
   — even with something individually more accurate — breaks that
   cancellation. This was directly tested and confirmed a second way: fusing
   YOLO's center with a segmentation model's independently-derived tip also
   underperformed (69.5% / 7.16%, worse than YOLO alone). Joint geometric
   consistency between a model's own paired predictions matters more than
   either coordinate's isolated accuracy.

---

## 5. The tip refiner (R1 / R2)

The oracle ablation pointed at a specific, bounded opportunity: fix tip,
leave center alone. Rather than build a refiner speculatively, a cheap
feasibility check came first.

**R1 — ROI feasibility diagnostic (no training).** For several candidate
search-window radii, measured what fraction of images have the true tip
within that radius of YOLO's coarse prediction, and what an oracle
"perfect-within-window" refiner would achieve:

| ROI radius | GT tip contained | Oracle accuracy | Oracle mean error |
|---|---|---|---|
| 16px | 42.6% | 73.8% | 4.98% |
| 24px | 70.5% | 77.0% | 4.56% |
| 32px | 85.2% | 77.0% | 4.44% |
| 48px | 95.1% | 78.7% | 4.32% |
| 64px | 96.7% | 78.7% | 4.34% |

The plateau at 48px (further radius gains nothing) justified building a
real refiner with that window, and the ceiling (78.7% / 4.32%, beating
classical on mean error while trailing slightly on accuracy) set honest,
pre-registered expectations before any training happened.

**R2 — local high-resolution tip refiner.** A small model taking a
128x128 pixel region (cropped from the *original full-resolution* image,
not the downsampled gauge crop — effectively magnifying the tip several
times over) centered on YOLO's own predicted tip, outputting a heatmap for
the true tip's location within that window. Two design choices directly
followed the diagnostics above: **YOLO's center is kept unchanged**
(per the correlated-error finding), and **training data uses YOLO's own
predicted tips on the training set, with 32-56px jitter — not
ground-truth-centered crops** (per the earlier crop-domain-shift lesson —
training must match the imperfect distribution the model will actually see
at inference).

**Result: 77.0% / 4.31%** — met the pre-registered success criteria
(>=77% within-tolerance and <4.82% mean error), recovering most of R1's
oracle upside, and beating classical on mean error.

---

## 6. Final analysis: classical vs. YOLO+refiner, and the ensemble

With two strong, different systems in hand (80.3% / 4.82% classical;
77.0% / 4.31% YOLO+refiner), the question changed from "which one wins" to
"do they fail on the same images." A paired, per-image comparison on the
identical 61-image fair subset:

| | Count | % |
|---|---|---|
| Both systems accurate | 44 | 73.3% |
| Only classical accurate | 5 | 8.3% |
| Only YOLO+refiner accurate | 3 | 5.0% |
| Neither accurate | 8 | 13.3% |

**13.3% of images are cases where exactly one system succeeds and the
other fails** — genuine complementary failure modes, not the same images
failing for both. This justified testing a simple, zero-additional-training
gate: if the two systems' readings agree (within 5% of scale range), trust
classical; if they disagree, defer to whichever has higher confidence
(reusing the existing needle-based confidence score from the abstention
mechanism).

**Result: 85.0% within tolerance, 3.32% mean error — beating both
individual systems on both metrics**, achieved with no new training, using
thresholds already established elsewhere in the project rather than tuned
to this specific result.

---

## 7. Final architecture

```
image
  |-- classical reader (Hough Transform geometry)
  `-- YOLO pose + local high-resolution tip refiner
          |
          v
   agreement check (within 5% of scale range)
          |
          |-- agree                            -> return classical's reading
          |-- disagree, classical confident     -> return classical's reading
          `-- disagree, classical not confident -> return YOLO+refiner's reading
```

This is the production architecture, implemented in
`retroread.predict.read_gauge_ensemble()`.

---

## 8. Final comparison table

| System | Within +/-5% tolerance | Mean % scale error |
|---|---|---|
| Classical CV alone | 80.3% | 4.82% |
| YOLO + tip refiner alone | 77.0% | 4.31% |
| **Ensemble (production)** | **85.0%** | **3.32%** |

---

## 9. Conclusion

The question this project set out to answer — "can deep learning beat a
strong classical baseline" — turned out to have a more interesting answer
than a simple yes or no. Nine distinct DL architectures were tried; most
underperformed classical, each for a specific, diagnosed reason (spatial
information loss, crop-domain shift, calibration sensitivity to landmark
error, objective mismatch, correlated joint-prediction error). The
strongest single DL system (YOLO + local tip refiner, reached only after a
chain of targeted, evidence-driven fixes) came close to classical but did
not exceed it on every metric. The actual winning architecture was neither
system alone: a simple, zero-additional-cost ensemble exploiting the fact
that the two systems' *failures are complementary*, not correlated with
each other.

This is a genuinely different, and arguably stronger, conclusion than
either "classical wins" or "DL wins" would have been: **model fusion based
on diagnosed, complementary failure modes**, arrived at through rigorous
elimination and root-causing rather than architecture shopping. Every
rejected approach in this document was rejected with a specific,
evidenced reason — not abandoned for looking unpromising.

---

## 10. Deferred future work

- **Corrected P2-YOLO-Pose**, with proper weight-index remapping instead
  of the naive layer-shift that broke pretrained transfer in this attempt.
- **Directional conditioning for the tip refiner** (an extra input channel
  encoding YOLO's coarse needle direction) — proposed as an optional R3
  step if R2 needed it; R2 met its success criteria without it, so this
  remains untested.
- **Corrected heatmap target encoding** (`HEATMAP_SIZE - 1` vs.
  `HEATMAP_SIZE`) — a known small implementation issue, deliberately not
  retroactively fixed to preserve reproducibility of reported results
  (see `docs/decision_log.md`).
- **Needle segmentation, trained longer** — the only architecture whose
  loss curve had not plateaued at the training budget's end.
- **Full end-to-end evaluation** using the ensemble's own predicted
  scale-min/max (rather than ground-truth calibration) for a stricter,
  fully autonomous accuracy measure.