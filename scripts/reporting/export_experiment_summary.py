"""
Exports all MLflow runs from the retroread_classical_baseline experiment into a committed, human-readable
markdown file -- since the MLflow tracking store itself (mlruns/, mlflow.db) is gitignored, this is the durable,
version-controlled record of experiment history for anyone browsing the repo without running MLflow locally.

Re-run this any time after new experiments are added; it fully regenerates the output file from current tracking data,
not an incremental update.

Run from the repo root with:
    uv run python scripts/reporting/export_experiment_summary.py
"""

import sys
from pathlib import Path

import mlflow

def _find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found).")

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT / "src"))

EXPERIMENT_NAME = "retroread_classical_baseline"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "experiment_results.md"

def main() -> None:
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], output_format="list")
    runs = sorted(runs, key=lambda r: r.info.start_time)

    lines = [
        "# Classical CV Baseline - Experiment Results",
        "",
        f"Auto-generated from MLflow tracking data ({len(runs)} runs)."
        "Regenerate with 'uv run python scripts/reporting/export_experiment_summary.py' "
        "after any new experiment. Full interactive tracking (artifacts, overlay "
        "images, per-run comparison) is available locally via 'uv run mlflow ui' "
        "after reproducing the data setup -- see docs/data_provenance.md",
        ""
    ]

    for run in runs:
        name = run.data.tags.get("mlflow.runName", run.info.run_id)
        lines.append(f"## {name}")
        lines.append("")
        lines.append("**Parameters:**")
        lines.append("")
        for k, v in sorted(run.data.params.items()):
            lines.append(f"- '{k}': {v}")
        lines.append("")
        lines.append("**Metrics:**")
        lines.append("")
        for k, v in sorted(run.data.metrics.items()):
            lines.append(f"- '{k}': {v:.4f}" if isinstance(v, float) else f"- '{k}': {v}")
        lines.append("")
        lines.append("---")
        lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {len(runs)} runs to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()