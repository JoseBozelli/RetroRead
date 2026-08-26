"""
Checks whether the reading_inaccurate category (Section 15) is mostly near-threshold near-misses, or genuinely
large errors -- before drawing any conclusion about the reading conversion's reliability.

Run from the repo root with:
    uv run python scripts/exploratory/check_inaccurate_distribution.py
"""

import csv
from pathlib import Path

EXP04_CSV = Path("experiment_04_results.csv")
EXP06_CSV = Path("experiment_06_results.csv")

def main() -> None:
    with EXP04_CSV.open() as f:
        exp04 = {row["file_name"]: row for row in csv.DictReader(f)}
    with EXP06_CSV.open() as f:
        exp06 = list(csv.DictReader(f))

    inaccurate_errors = []
    for row in exp06:
        row04 = exp04.get(row["file_name"])
        if row04 is None:
            continue
        circle_found = row04["circle_found"] == "True"
        needle_found = row04["needle_found"] == "True"
        accurate = row["accurate"] == "True"
        if circle_found and needle_found and not accurate and row["pct_error"]:
            inaccurate_errors.append(float(row["pct_error"]))

    inaccurate_errors.sort()
    n = len(inaccurate_errors)
    print(f"reading_inaccurate count: {n}")
    print(f"min: {inaccurate_errors[0]:.2f}% max: {inaccurate_errors[-1]:.2f}%")
    print(f"median: {inaccurate_errors[n // 2]:.2f}%")

    near_threshold = sum(1 for e in inaccurate_errors if e <= 7.0)
    print(f"\nWithin 5-7% (near-threshold near misses): {near_threshold} / {n} ({near_threshold/n:.1%})")
    print(f"Above 10% (genuinely large errors): {sum(1 for e in inaccurate_errors if e > 10)}/{n}")

    print(f"\nFull sorted list: {inaccurate_errors}")

if __name__ == "__main__":
    main()