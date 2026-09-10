"""
Lists one candidate frame per physical Aalborg gauge (man1-man7).

Originally written to pick a real-world case-study set from Aalborg.
Running this and inspecting a sample frame revealed the "test videos" are
clean white line-drawings on black, not photographs -- see
docs/real_world_case_study.md for why the case study uses genuine
photographs instead. Kept as the record of that investigation.

Run from the repo root with:
    uv run python scripts/exploratory/list_aalborg_candidates.py
"""

import csv

from retroread.config import AALBORG_CLEAN_MANIFEST


def main() -> None:
    seen_gauges = set()
    with open(AALBORG_CLEAN_MANIFEST) as f:
        for row in csv.DictReader(f):
            gauge = row["gauge"]
            if gauge not in seen_gauges:
                seen_gauges.add(gauge)
                print(f"{gauge}: {row['processed_path']}  (reference_angle={row['reference_angle']})")


if __name__ == "__main__":
    main()