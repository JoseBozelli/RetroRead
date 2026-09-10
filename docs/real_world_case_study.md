# Real-World Case Study

## Methodology

RetroRead was trained and evaluated entirely on Endava's synthetic gauge
renders. This case study tests it against **genuine photographed gauges**
— a meaningfully different question than anything in the rest of this
project's evaluation.

**Why not the Aalborg dataset?** Aalborg's "test videos" were investigated
directly and turned out to be clean white line-drawings on a black
background (see `docs/decision_log.md`) — not photographs. They may be
useful as a style-transfer robustness check, but they don't test what
"real-world" is meant to mean here, so they were set aside for this case
study in favor of genuine photographs.

**Five real gauge photographs** were sourced online, saved to
`data/real_world_samples/`, and each was read two ways:
1. **Independently, by a human**, before running anything through
   RetroRead — this is the ground truth for comparison.
2. **Through RetroRead's production ensemble** (`app.py`), using
   human-provided calibration (reading the gauge's own printed numbers,
   clicking their positions — the same interface a real user would use).

This is a small, honest sample (n=5) — not a claim of comprehensive
real-world coverage. See `docs/deep_learning.md` for the project's broader
scope limitations.

## Results

| Gauge | Human reading | RetroRead reading | Absolute error | Source |
|---|---|---|---|---|
| 1 (dual-scale: psi outer, kPa inner) | 0 | 311.961 | 311.961 | Classical (100% confidence; systems disagreed) |
| 2 | 1.5 | 1.736 | 0.236 | DL ensemble only (classical detection failed) |
| 3 (dual-scale: psi outer, kPa inner) | 1450 | 1128.154 | 321.846 | Classical (100% confidence; systems disagreed) |
| 4 | 46.1 | 46.268 | 0.168 | Classical and DL agreed |
| 5 | 5.2 | 5.375 | 0.175 | Classical and DL agreed |

## Key finding: agreement is a meaningful signal; classical's own confidence score, alone, is not

**Every case where both systems agreed produced an excellent reading**
(gauges 4, 5 — errors of 0.168 and 0.175, well within the ±5% tolerance
used throughout this project). **Every case where classical was used alone
at 100% self-reported confidence produced a badly wrong reading** (gauges
1, 3 — errors of 312 and 322).

This is a direct, real-world confirmation of the reasoning behind building
the ensemble in the first place (see `docs/deep_learning.md` §6): a single
system's own confidence score, based only on its own internal geometry, is
not sufficient on its own. Cross-system *agreement* — two independently-built
systems reaching similar answers — is the more trustworthy signal, exactly
as the synthetic-data paired error analysis found.

## The dual-scale failure mode

Gauges 1 and 3 share a specific structural feature that does not exist
anywhere in the training data: **two concentric scales on the same dial**
(an outer black scale in one unit, an inner red scale in another). This is
a genuinely common real-world gauge design — and RetroRead's synthetic
training set, built entirely from single-scale gauge renders, has no
representation of it at all.

To isolate the cause, both systems' individual (pre-gate) readings were
inspected directly:

| Gauge | Classical reading | DL ensemble reading | True value |
|---|---|---|---|
| 1 | 311.961 | 206.149 | 0 |
| 3 | 1128.154 | 1787.676 | 1450 |

**Both systems are substantially wrong, and — critically — in
inconsistent relative directions**: DL is closer on gauge 1, classical is
closer on gauge 3. This pattern rules out the simpler explanation (a
single shared calibration input mistake, which would bias both systems'
outputs the same way) and points instead to **independent confusion in
each system's own needle/angle detection** — most plausibly, each
occasionally locking onto the wrong scale's tick marks or the secondary
needle, rather than a calibration-entry error.

**This is a genuine, structural limitation, not a tunable one.** The
current confidence mechanism is built entirely from classical's own
needle geometry (length relative to gauge radius, image sharpness) — it
has no way to detect "this may be the wrong scale entirely." No amount of
threshold tuning fixes that; it requires either training data that
includes dual-scale gauges, or an explicit scale-selection step neither
system currently has.

## Conclusion

On a small but genuinely diverse real-world sample, RetroRead's ensemble
performed excellently on gauges resembling its training distribution
(single-scale, clear geometry) — 3 of 5 cases with errors under 0.25 units
— and failed predictably and explainably on gauges outside that
distribution (dual-scale designs). The failure mode is specific,
understood, and honestly attributable to a known, undertested gap in the
training data, not a vague "real-world is harder" hand-wave.

## Deferred future work

- **Dual-scale gauge representation** in training data (synthetic
  generation or additional real photographs) — the single most
  actionable finding from this case study.
- **A richer confidence signal** that can flag scale-ambiguity, not just
  needle-detection quality — e.g., checking whether detected calibration
  points cluster into two visually distinct groups (an outer and inner
  ring) as a scale-confusion heuristic.
- **A larger real-world sample** — five gauges is enough to surface a
  real pattern, not enough to be a comprehensive real-world benchmark.