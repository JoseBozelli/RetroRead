"""
Exports all MLflow runs from both retroread_classical_baseline and
retroread_deep_learning experiments into a committed, human-readable
markdown file -- since the MLflow tracking store itself (mlruns/,
mlflow.db) is gitignored, this is the durable, version-controlled record
of experiment history for anyone browsing the repo without running MLflow
locally.

Re-run this any time after new experiments are added; it fully
regenerates the output file from current tracking data.

Run from the repo root with:
    uv run python scripts/reporting/export_experiment_summary.py
"""

from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_NAMES = ["retroread_classical_baseline", "retroread_deep_learning"]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "experiment_results.md"


def main() -> None:
    lines = [
        "# All Experiments — MLflow Export",
        "",
        "Auto-generated from MLflow tracking data. Regenerate with "
        "`uv run python scripts/reporting/export_experiment_summary.py` after any new "
        "experiment. Full interactive tracking (artifacts, overlay images, per-run "
        "comparison) is available locally via `uv run mlflow ui` after reproducing "
        "the data setup -- see docs/data_provenance.md",
        "",
    ]

    total_runs = 0
    for experiment_name in EXPERIMENT_NAMES:
        runs = mlflow.search_runs(experiment_names=[experiment_name], output_format="list")
        runs = sorted(runs, key=lambda r: r.info.start_time)
        total_runs += len(runs)

        lines.append(f"## {experiment_name} ({len(runs)} runs)")
        lines.append("")

        for run in runs:
            name = run.data.tags.get("mlflow.runName", run.info.run_id)
            lines.append(f"### {name}")
            lines.append("")
            lines.append("**Parameters:**")
            lines.append("")
            for k, v in sorted(run.data.params.items()):
                lines.append(f"- `{k}`: {v}")
            lines.append("")
            lines.append("**Metrics:**")
            lines.append("")
            for k, v in sorted(run.data.metrics.items()):
                lines.append(f"- `{k}`: {v:.4f}" if isinstance(v, float) else f"- `{k}`: {v}")
            lines.append("")
            lines.append("---")
            lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {total_runs} runs across {len(EXPERIMENT_NAMES)} experiments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()