# Error Analysis (Section 15)

Based on Experiment 06's full-pipeline results (265 candidates), cross-referenced with Experiment 04's per-stage detection results.

> **Note on the numbers below:** this analysis was performed on Experiment
> 06 (265 candidates, train split) — before the train/val evaluation bug
> was found and corrected in Experiment 06b (61 candidates, val split,
> 80.3% / 4.82%, used everywhere else in this project). The qualitative
> findings here (failure categories, the bimodal error distribution, the
> wrong-line root cause) are independent of that correction and still
> hold. For the authoritative accuracy figures, see
> `docs/experiment_master_table.md`.

## Failure categories

| Category | Count | % |
|---|---|---|
| Circle detection failed | 0 | 0.0% |
| Needle detection failed (no line found) | 11 | 4.2% |
| Reading inaccurate (>5% of scale range) | 31 | 11.7% |
| Accurate | 223 | 84.2% |

## Reading-inaccurate breakdown

Of the 31 "reading inaccurate" cases, error magnitude is bimodal, not uniform:
- 20 (64.5%) fall in the 5-8% range -- near-threshold near-misses, not meaningful failures. The true reliable-accuracy rate is closer to 92% at a slightly looser tolerance.
- 6 cases show large errors (15%-121%). Root-cause investigation of the worst case (121% error) traced the fault to needle detection finding an incorrect line -- the same wrong-line failure mode documented in Experiment 03/04 -- not a defect in the calibration or angle-conversion math. Large reading errors are downstream amplifications of already-documented needle-angle inaccuracy, not a new failure category.

## What would need to improve for real-world use

Needle detection is the primary bottleneck: both the non-detection rate (4.2%) and the angle-accuracy tail (driving the large reading errors) trace back to the same Hough-line-selection step occasionally picking a non-needle line. This is the most direct target for the deep learning model to improve on.