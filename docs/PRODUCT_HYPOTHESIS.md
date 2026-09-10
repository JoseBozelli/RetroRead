# Product Hypothesis: RetroRead

> **Note on framing:** this is a *product* hypothesis exercised through a *portfolio* project, not a claim of real market validation. No customer interviews, no field deployment, no commercial data. Where this document says "hypothesis" or "evidence," that evidence is limited to what a technical prototype and public research/benchmark data can show — not what real users or a real market would confirm. That distinction is intentional and stated explicitly rather than implied.

---

## User

A technician, maintenance worker, or field operator responsible for periodically recording readings from analog gauges — pressure, temperature, level — in a facility, plant, or piece of equipment that has not been retrofitted with digital sensors.

## Pain

Readings are currently recorded manually: walk to the gauge, read it, write it down (clipboard, spreadsheet, or a basic computerized maintenance management system - CMMS - entry). This is:
- time-consuming, especially across many gauges or hard-to-access locations
- error-prone (transcription mistakes, illegible handwriting, misread scales)
- undocumented in the moment — no photographic record of what was actually seen
- not continuously loggable without either replacing the gauge or installing new instrumentation

## Current alternative

Three paths exist today, none of them satisfying for this specific gap:
1. **Keep doing it manually** — cheap, but slow and error-prone, with no digital trail.
2. **Replace the analog gauge with a digital/connected sensor** — solves the problem but is often expensive, disruptive, or impractical for equipment that still functions fine mechanically.
3. **Full industrial Internet of Things (IoT) / Supervisory Control and Data Acquisition (SCADA) retrofit** — addresses this and much more, but is a large capital and integration project, disproportionate if the actual need is "digitize this one reading."

RetroRead's hypothesis sits in the gap between (1) and (2): **can a photo substitute for a sensor?**

## Hypothesis

> Computer vision can convert a photograph of a legacy analog gauge into a reliable digital measurement — reliable enough to reduce manual transcription and produce a trustworthy digital log — without replacing the underlying equipment, **provided the system can recognize and communicate when it cannot make a confident reading.**

This is deliberately narrow: one instrument type (round analog gauge), one task (image → numeric reading), one outcome (reduce manual transcription while creating a digital record).

### Three falsifiable sub-hypotheses

**H1 — Measurement.** A vision pipeline (classical CV baseline, then a keypoint-based deep learning model) can convert detected gauge geometry into a numeric reading within an operationally useful tolerance band on held-out data.
*Fails if:* neither approach gets meaningfully close to ground truth on the synthetic test set, or the confidence/tolerance tradeoff never reaches a coverage level that would be useful in practice.

**H2 — Robustness.** The deep learning model degrades less severely than the classical Hough-based baseline under controlled image perturbations (brightness, contrast, blur, rotation).
*Fails if:* the classical baseline is equally or more robust than the DL model — a legitimate and useful negative result, not a failure of the project (see the churn project's Random Forest precedent).

**H3 — Domain transfer.** A model trained primarily on synthetic gauge imagery (SyntheticGauges) produces usable — not necessarily excellent — readings on real photographs (RealGauges, plus Roboflow "Analog Gauge Meter" CC BY 4.0 real-image set).
*Fails if:* performance collapses so completely on real images that no confidence threshold yields a usable coverage/accuracy tradeoff.

## Success criteria (stated before seeing results)

- A **numeric tolerance band** (to be pinned down once we know the gauge unit range in the training data — e.g., a % of full-scale range) defines what counts as an "accurate" reading.
- Success is **not** "beat Howells et al.'s reported <1° figure." That figure appears to be an in-domain synthetic-test result; a follow-up paper (Leon-Alcazar et al., WACV 2024) re-ran the same method on Howells' own real-world test set and reported per-gauge angle errors averaging roughly 9.5° (range ~2°–26° across the six real gauges). RetroRead's real-world benchmark is that ~9.5° cross-domain figure, not the sub-1° in-domain one — and even matching it isn't the bar. The actual deliverable is a **documented, honest accuracy/coverage tradeoff curve**, not a single headline number.
- Success **is**: a working end-to-end pipeline, a defensible baseline-vs-DL comparison (in either direction), a real confidence/abstention mechanism with a coverage-vs-error table, and transparent real-world case analysis on the 6 RealGauges gauges plus the independent CC BY set — regardless of which direction the numbers land.

## MVP

Static image upload → detected gauge geometry (visual overlay) → numeric reading + confidence → accept/abstain decision → optional log entry. Deployed as a FastAPI service behind a small Streamlit interface. No live video, no edge hardware, no multi-instrument support (see `ROADMAP.md` for deferred scope).

## Evidence

**Note on dataset substitution:** the Cambridge SyntheticGauges/RealGauges
dataset (Howells, Charles & Cipolla) referenced in the original framing
below was unavailable at download time (dead hosting link). It was
substituted with Endava's synthetic gauge dataset — see
`docs/data_provenance.md` for full reasoning. All results below use the
Endava dataset for synthetic evaluation.

- **Classical baseline performance:** 80.3% within ±5% tolerance, 4.82%
  mean error (Experiment 06b, 61-image held-out validation set).
- **DL model performance (in-domain synthetic):** nine architectures
  tried; best single DL system (YOLO pose + local tip refiner) reached
  77.0% / 4.31%. Full architecture search and root-causing:
  `docs/deep_learning.md`.
- **DL model performance (RealGauges case-by-case):** not performed —
  RealGauges was unavailable (see dataset substitution note above).
- **DL model performance (independent CC BY real-image perception
  check):** the originally planned Roboflow CC BY real-image set was not
  used. Real-world validation was instead performed on five genuine
  photographed gauges sourced independently — see
  `docs/real_world_case_study.md`.
- **Confidence-threshold vs. coverage vs. accuracy table:** Experiment 19
  — classical alone reaches 84.5% accuracy at 95.1% coverage (threshold
  0.85). The production ensemble (classical + DL, agreement/confidence
  gated) reaches 85.0% / 3.32% — see `docs/deep_learning.md` §6-8.
- **Robustness matrix (perturbation type vs. degradation):** not
  executed — an explicitly deferred, out-of-scope item. The real-world
  case study surfaced a related, unplanned finding instead: dual-scale
  gauges (a structural feature absent from training data) cause
  predictable, explainable failures in both systems — see
  `docs/real_world_case_study.md`.

**On H3 (domain transfer) specifically:** the Aalborg dataset, originally
intended as the real-world domain-transfer test, was found on inspection
to be clean line-drawings rather than photographs (see
`docs/data_provenance.md`) — it does not test H3 as originally framed.
H3 was instead tested via the real-world case study above. Result: the
ensemble generalizes well to real photographs resembling the training
distribution, and fails in a specific, diagnosed way (dual-scale gauges)
outside it — a partial, honestly-scoped confirmation of H3, not the
clean "usable across the board" result originally hoped for.

## Unknowns

Things this project cannot answer and would require real customer contact to learn:
- Whether technicians would trust and adopt a system that sometimes refuses to answer, versus preferring an always-on (even if less reliable) reading.
- What tolerance band actually matters operationally for a given gauge type (a pressure gauge tolerance is not a temperature gauge tolerance).
- Real-world image capture conditions in an actual facility (lighting, gauge condition, camera used) — RealGauges' 6 gauges and the CC BY set are proxies, not a substitute for this.
- Whether a photo-based workflow fits into existing operator routines/CMMS tools, or would be seen as extra friction.
- Willingness to pay / build-vs-buy calculus for a facility already weighing full sensor replacement.

## Possible niche

Small-to-mid facilities (labs, workshops, older manufacturing sites) with a handful of legacy analog gauges where full IoT retrofit is disproportionate to the value of the reading, but manual logging is a recurring operational annoyance. Underserved relative to large industrial sites, which are the typical target of SCADA/IoT vendors.

## Data provenance note (for commercialization framing)

This portfolio prototype validates the technical architecture using non-commercial research data (SyntheticGauges/RealGauges, CC BY-NC 4.0, Howells/Charles/Cipolla, University of Cambridge). A commercial version would require retraining and validation on permissively licensed, internally generated, or customer-owned imagery — the CC BY 4.0 real-image sets used here for perception/domain-shift checks demonstrate that such sources exist, but are not sufficient alone for the full numeric-reading task.
