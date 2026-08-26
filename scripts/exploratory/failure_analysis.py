"""
Error analysis: categorizes every candidate image into a failure stage, by joining experiment 04 (circle/needle detection)
with experiment_06 (reading conversion accuracy) results by filename. No new detection runs -- reuses already-
computed results.

Run from the repo root with:
    uv run python scripts/exploratory/failure_analysis.py
"""

import csv
from pathlib import Path

EXP04_CSV = Path("experiment_04_results.csv")
EXP06_CSV = Path("experiment_06_results.csv")

def load_csv(path: Path) -> dict:
    with path.open() as f:
        return {row["file_name"]: row for row in csv.DictReader(f)}

def main()-> None:
    exp04 = load_csv(EXP04_CSV)
    exp06 = load_csv(EXP06_CSV)

    categories = {
        "circle_failed": [],
        "needle_failed": [],
        "reading_inaccurate": [],
        "accurate": []
    }

    for file_name, row06 in exp06.items():
        row04 = exp04.get(file_name)
        if row04 is None:
            continue

        circle_found = row04["circle_found"] == "True"
        needle_found = row04["needle_found"] == "True"
        accurate = row06["accurate"] == "True"

        if not circle_found:
            categories["circle_failed"].append(file_name)
        elif not needle_found:
            categories["needle_failed"].append(file_name)
        elif not accurate:
            categories["reading_inaccurate"].append(file_name)
        else:
            categories["accurate"].append(file_name)

    total = sum(len(v) for v in categories.values())
    print(f"Total candidates: {total}\n")
    for name, files in categories.items():
        pct = len(files) / total * 100
        print(f"{name} {len(files)} ({pct:.1f}%)")

    print(f"\nSample filenames for each failure category (up to 3 each):")
    for name in ["circle_failed", "needle_failed", "reading_inaccurate"]:
        print(f"\n{name}:")
        for fn in categories[name][:3]:
            print(f" {fn}")

if __name__ == "__main__":
    main()