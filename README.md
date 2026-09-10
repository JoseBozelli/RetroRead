# RetroRead

**An end-to-end computer-vision / ML engineering case study for automated
analog gauge reading.**

RetroRead converts photographs of legacy analog gauges into numeric
readings. Two fundamentally different approaches were developed and
rigorously evaluated: a geometric computer-vision pipeline (Hough
Transform) and a deep-learning keypoint/pose-estimation system. After nine
tracked DL architectures, a chain of root-caused failures and fixes, and a
series of production-domain diagnostics, the project's actual production
model is neither approach alone — it's a small, evidence-driven ensemble
of both.

## Production model selection

| Approach | Within ±5% tolerance | Mean % scale error | Decision |
|---|---|---|---|
| Classical CV (Hough Transform) | 80.3% | 4.82% | Deployed component |
| DL v1 (flattened coordinate regression) | 9.0% | 38.9% | Rejected |
| DL v2 (heatmap decoder) | 46.5% | 12.09% | Iterate |
| DL v3 (crop-staged, skip connection) | 54.1% | 7.28% | Diagnose |
| DL v3 + oracle crop (diagnostic) | 65.6% | 4.36% | Reveals crop-domain shift |
| DL v6 (+ crop-jitter augmentation) | 67.2% | 5.52% | Confirms fix |
| YOLO-pose fine-tune | 72.1% | 5.14% | Closest single DL result |
| YOLO + local tip refiner (R2) | 77.0% | 4.31% | Best single DL result |
| **Ensemble (classical + YOLO/refiner, gated)** | **85.0%** | **3.32%** | **Deployed — production model** |

**Why isn't a single neural network deployed?** No individual DL
architecture reliably beat the classical baseline on every metric. But a
paired error analysis showed classical and YOLO+refiner fail on genuinely
*different* images (13.3% of cases are one-succeeds-one-fails, not both
failing together) — so a simple agreement/confidence gate between the two,
built with zero additional training, beats both individual systems. Full
reasoning, every rejected architecture, and why each failed:
[`docs/deep_learning.md`](docs/deep_learning.md). Every experiment,
scannable: [`docs/experiment_master_table.md`](docs/experiment_master_table.md).

This progression — strong baseline → DL architecture search → iterative
diagnosis → production-domain stress-testing → evidence-based model
selection → ensemble — is the actual subject of this project, more than
any single number in the table above.

---

## Architecture

```
image
  ├── classical reader (Hough Transform: circle → needle → angle → calibration)
  └── YOLO pose + local high-resolution tip refiner
          │
          ▼
   agreement check (within 5% of scale range)
          │
          ├── agree                            → return classical's reading
          ├── disagree, classical confident     → return classical's reading
          └── disagree, classical not confident → return YOLO+refiner's reading
```

Both branches include a confidence/abstention mechanism — the system
reports `"unable_to_read"` with a specific reason rather than forcing a
guess when detection quality is low.

---

## Quickstart

```bash
git clone <repo-url>
cd RetroRead
uv sync
```

Run the classical baseline against the held-out validation set:

```bash
uv run python scripts/experiments/experiment_06b_classical_baseline_val.py
```
Expected output: 80.3% within-tolerance, 4.82% mean error (61 candidates).

Start the API:

```bash
uv run uvicorn retroread.api:app --reload
```
Then open `http://127.0.0.1:8000/docs` for an interactive interface, or
POST an image + calibration points to `/predict`. See
[`src/retroread/api.py`](src/retroread/api.py) for the request format, and
`scripts/exploratory/get_calibration_for_demo.py` for a helper that prints
ready-to-use calibration points for any image in the dataset.

Run the Streamlit UI:

```bash
uv run streamlit run app.py
```
Upload a gauge photo, calibrate by reading the numbers OCR marks on the
image (or click to add points it missed), and get a reading from the
production ensemble.

Run tests:

```bash
uv run pytest tests/ -v
```

---

## Repository structure

```
src/retroread/                  # importable package: models, datasets, pipelines, API
scripts/
  data_prep/                    # dataset acquisition and preparation
  experiments/                  # every numbered experiment (00-35), reproducible
  exploratory/                  # diagnostics, debugging, one-off analyses
  reporting/                    # MLflow summary export
docs/
  PRODUCT_HYPOTHESIS.md         # original problem framing and falsifiable hypotheses
  decision_log.md               # running log of every significant decision, with rationale
  data_provenance.md            # dataset sources, licenses, checksums
  error_analysis.md             # classical baseline failure-mode breakdown (Section 15)
  deep_learning.md              # full DL narrative: every architecture, why, what happened
  experiment_results.md         # raw MLflow export -- unfiltered audit trail
  experiment_master_table.md    # curated, scannable table -- why/result/decision per experiment
  glossary.md                   # terminology reference
tests/                          # API tests
configs/                        # custom model architecture definitions (P2-YOLO-Pose)
```

---

## Data

- **Synthetic training data:** Endava's synthetic gauge dataset (CC BY-NC-SA
  4.0), 1,000 rendered images with full COCO-format annotations (keypoints,
  segmentation masks, scale-label calibration points).
- **Real-world validation:** a small case study using genuine photographed
  gauges — see `docs/real_world_case_study.md` for methodology and findings.

Full provenance, checksums, and licensing: [`docs/data_provenance.md`](docs/data_provenance.md).

---

## Key engineering practices demonstrated

- **MLOps:** every experiment tracked in MLflow (params, metrics, artifacts),
  with a version-controlled, human-readable export
  (`docs/experiment_results.md`) since the tracking store itself is
  gitignored.
- **Reproducibility discipline:** known implementation issues are documented,
  not silently fixed after the fact, when doing so would make already-reported
  results unreproducible from the current code (see `docs/deep_learning.md` §10).
- **Evidence-based model selection:** every architecture change was
  motivated by a specific diagnosis, tested against pre-registered success
  criteria, and accepted or rejected on that basis — not iterated
  indefinitely chasing marginal gains.
- **Honest negative results:** rejected approaches are documented with root
  causes, not hidden. Several (v4, v5, the multi-task model, P2-YOLO-Pose)
  are as informative as the approaches that worked.
- **Confidence and abstention:** the system is designed to say "I don't
  know" rather than return an unreliable reading, with thresholds chosen
  from measured coverage/accuracy tradeoffs, not guessed.

---

## Limitations and scope

- Calibration (the angle-to-value mapping) must be supplied per gauge —
  the system does not read printed scale numbers autonomously by default.
  An experimental OCR-based auto-calibration path exists
  (`src/retroread/ocr_calibration.py`) but is not the default, due to a
  documented glare-related reliability issue.
- The ensemble's fully autonomous evaluation (using its own predicted
  scale landmarks rather than reference calibration) is deferred future
  work — see `docs/deep_learning.md` §10.
- Real-world validation is a small, targeted case study (see
  `docs/real_world_case_study.md`), not exhaustive coverage of all gauge types
  and conditions.
- **Dual-scale gauges** (two concentric scales in different units on one
  dial) are not represented in training data and cause both classical and
  DL systems to independently misread — a specific, real-world-validated
  finding, see `docs/real_world_case_study.md`.

## Deferred future work

See `docs/deep_learning.md` §10 for the full list, including a corrected
P2-YOLO-Pose implementation, directional conditioning for the tip refiner,
and crop-jitter-trained needle segmentation.

## License

Code is MIT licensed (see `LICENSE`). Datasets have their own separate
licenses — see `docs/data_provenance.md`.