# RetroRead: Master Experiment Table

Every experiment run in this project, in order, with why it was run, what
happened, and what was decided. `decision_log.md` and `deep_learning.md`
narrate the reasoning in prose; this table is the curated, scannable
reference. For the raw, unfiltered MLflow tracking data behind these
numbers, see `docs/experiment_results.md`.

## Classical CV Baseline

| # | Experiment | Why | Result | Decision |
|---|---|---|---|---|
| 00 | Single-image circle detection | Confirm Hough circle detection works at all before batch testing | Found circle correctly on first try | Proceed to batch |
| 01 | Batch circle detection (default params) | Does it generalize beyond one image? | 100% detection, 94.3% accuracy vs. bbox center | Visual inspection of failures next |
| — | Visual inspection of failures | 5 failures looked suspicious | Detected circle was background structure (roof trusses), not gauge — `max_radius_fraction` too permissive | Tune radius |
| 02 | Tighter `max_radius_fraction` (0.6→0.35) | Test the radius hypothesis | 100% detection, 97.7% accuracy, mean error 11.8px→6.6px | Adopted as new default |
| 03 | Single-image needle detection | Confirm needle detection works before batch | Initially picked the counterweight, not the pointer | Fix selection logic |
| — | Reach-based tip selection fix | Length-based selection favored short, clean counterweight over fragmented real arm | Fixed by selecting by *reach from pivot*, not raw segment length | Adopted |
| 04 | Batch needle detection | Does the fix generalize? | 95.8% detection, 90.9% angle accuracy within 5° | Proceed to reading conversion |
| 05 | Single-image reading conversion | Validate angle→value calibration math | Small (~0.3 unit) discrepancy, plausibly text-label vs. tick-position offset | Needs batch check |
| 06 | Batch reading conversion (train split) | Full pipeline accuracy | 84.2% within tolerance, 4.26% mean error | **Later found to be an unfair, in-sample number** |
| 06b | Classical baseline, VAL split (fair) | Correct the train/val comparison bug | **80.3% within tolerance, 4.82% mean error** | **True classical baseline number, used throughout** |
| — | Section 15 error analysis | Characterize the "inaccurate" category | 64.5% were near-threshold near-misses (5–8% error); worst outlier (120% error) traced to a needle-detection wrong-line pick | Confirms needle detection is the main lever, not calibration math |
| 19 | Confidence/abstention mechanism | Reduce catastrophic failures via a confidence gate | Threshold 0.85 (needle length-ratio based) removes both catastrophic cases: 84.5% accuracy, 3.41% mean error at 95.1% coverage | Adopted as production abstention rule |

## Deep Learning — Architecture Search

*All DL evaluations from Experiment 13 onward use the same 61-image
fair subset (edge-filtered, matching classical's own operating range).
v1/v2 used the full 200-image val set before this correction.*

| # | Experiment | Why | Result | Decision |
|---|---|---|---|---|
| 07–08 | Model/training sanity checks | Confirm plumbing before real training | Shapes correct; found & fixed a genuine freeze bug (backbone wasn't actually frozen) | Proceed |
| 09–10 | v1: flattened coordinate regression | First DL keypoint model | 9.0% / 38.9% | Root-caused: `Flatten()` after pooling discards spatial information |
| 11–12 | v2: heatmap decoder | Fix v1's spatial-info loss | Initially NaN loss (Gaussian target overflow bug, fixed); then 46.5% / 12.09% | Real improvement confirms diagnosis |
| 13–14 | v3: crop-staged + skip connection + combined loss | Full-scene search was still a confound; low decoder resolution; heatmap-only loss doesn't penalize final coordinate error | **54.1% / 7.28%** | Best result at this point |
| 15–16 | v4: backbone fine-tuning (differential LR, 3 unfrozen blocks) | Reviewer-recommended control experiment for domain adaptation | 47.5% / 8.88% (worse); val loss improved while accuracy worsened | Rejected — proxy/target metric divergence |
| 17–18 | v5: reading-aware auxiliary loss | Make the loss reflect the real business metric | 14.8% / 25.63% (much worse) | Rejected — training-time calibration approximation didn't match production calibration (objective mismatch) |
| — | Post-hoc audit: soft-argmax temperature sweep | Theoretical concern that default temperature was too diffuse | t=1 (default) was optimal; sharper settings all worse | Hypothesis tested and rejected — model had already adapted to t=1 |
| — | Post-hoc audit: GT-crop vs. detected-crop ablation | Isolate landmark quality from crop-domain shift | GT crop: 65.6% / 4.36% (beats classical on error); detected crop: 54.1% / 7.28% | Crop-domain shift confirmed as a real, separate, fixable source of error |
| 20–21 | v6: crop-jitter augmentation | Directly targets the crop-domain-shift finding | **67.2% / 5.52%** | Confirmed hypothesis; adopted as best "pure CNN keypoint" result |
| 22–23 | Needle segmentation + center heatmap | Test whether dense pixel evidence beats a single tip coordinate | 67.2% / 6.80% (tied accuracy, worse error); loss still improving at epoch 25 | Accepted as-is; noted as possibly undertrained (not pursued further) |
| 24–25 | Multi-task (mask + center + scale landmarks), fully end-to-end calibration | Close the "predicted calibration" gap | 8.2% / 71.06% (catastrophic) | Root-caused: 2-point calibration slope is multiplicatively sensitive to landmark position error — a structural, not architectural, finding |
| 26–27 | sin/cos angle regression | Test a wraparound-safe, non-spatial representation | 32.8% / 13.53%; fixed unstable training with `tanh` output → **44.3% / 16.88%** | Demonstrates the interpretability cost of black-box regression |
| 28–29 | Direct end-to-end regression | Show why geometry-aware approaches are preferred | 32.8% / 17.23%; attempted LR fix made it worse (0.0% / 37.35%), reverted | Confirms hypothesis: direct regression underperforms by design |

## Deep Learning — Pose Estimation Paradigm

| # | Experiment | Why | Result | Decision |
|---|---|---|---|---|
| 30–31 | YOLOv8-pose fine-tune | Different framework/paradigm (single-shot detection+pose) | **72.1% / 5.14%** | Best DL result to this point, closest to classical |
| — | Per-keypoint error diagnostic | Which landmark drives the 27.9% failures? | Center error most discriminates accurate/inaccurate (11.86→18.18px); max shows no signal | Refines (doesn't overturn) the "tip matters most" hypothesis |
| 32–33 | P2-YOLO-Pose (added stride-4 head) | Literature-backed hypothesis: higher-res features improve fine landmark precision | 24.6% / 13.29% (collapsed) | Root-caused: layer-index shift broke pretrained weight transfer (91%→44% of weights transferred) — an implementation defect, not a rejection of the P2 hypothesis. Left as documented future work. |
| — | Oracle landmark ablation | What's the causal upside of fixing each landmark? | GT tip alone: 80.3% / 3.29% (ceiling); GT center alone: 67.2% / 5.21% (**worse**); classical geometric center: 65.6% / 5.63% (**worse**, despite better raw pixel accuracy) | **Correlated-error finding**: YOLO's center/tip are jointly consistent; independently "correcting" one breaks that consistency |
| — | Cross-model fusion attempt | Test the correlated-error finding across models | YOLO center + segmentation tip: 69.5% / 7.16% (worse); segmentation's own consistent pair: 62.7% / 8.32% (worse) | Confirms the correlated-error finding generalizes — naive component-mixing doesn't work |
| — | R1: ROI feasibility diagnostic | Is a local tip refiner even feasible, before building one? | Oracle-within-ROI plateaus at 78.7% / 4.32% by 48px radius (95.1% tip containment) | Justifies building R2, narrowly scoped |
| 34–35 | R2: local high-resolution tip refiner | Frozen YOLO center (per the ablation) + refined tip from a small high-res ROI, trained on YOLO's own predicted tips (not GT) | **77.0% / 4.31%** — meets pre-registered success criteria, beats classical on mean error | Adopted as best single DL system |

## Final: Ensemble

| Analysis | Why | Result | Decision |
|---|---|---|---|
| Paired error analysis | Does classical vs. YOLO+refiner fail on the same images? | 73.3% both correct; 8.3% classical-only correct; 5.0% YOLO-only correct; 13.3% neither — genuine complementary failures | Justifies a gating hybrid |
| Simulated agreement/confidence gate | Zero-additional-training hybrid using existing confidence scoring | **85.0% / 3.32%** — beats *both* individual systems on *both* metrics | **Final production architecture: classical + YOLO/refiner ensemble, gated by agreement and confidence** |

## Final numbers, all in one place

| System | Within ±5% tolerance | Mean % scale error |
|---|---|---|
| Classical CV alone | 80.3% | 4.82% |
| YOLO + tip refiner alone | 77.0% | 4.31% |
| **Ensemble (production)** | **85.0%** | **3.32%** |