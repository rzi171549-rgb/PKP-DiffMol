"""Report public main-table results for the PKP-DiffMol release.

Final task checkpoints are not included in this public release. This script
reports the main table results from results/main_table_results.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_results(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def print_result(row: dict[str, str]) -> None:
    print(f"dataset={row['dataset']}")
    print(f"seed={row['seed']}")
    print(f"test_auc={row['test_auc']}")
    print(f"test_auc_percent={row['test_auc_percent']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report PKP-DiffMol main-table results.")
    parser.add_argument("--dataset", default=None, help="Optional dataset name to report.")
    args = parser.parse_args()

    results_path = repo_root() / "results" / "main_table_results.csv"
    rows = load_results(results_path)
    if args.dataset:
        dataset = args.dataset.lower()
        rows = [row for row in rows if row["dataset"].lower() == dataset]
        if not rows:
            raise SystemExit(f"Dataset not found in {results_path}: {args.dataset}")

    print("Final task checkpoints are not included in this public release.")
    print("This script reports the main table results from results/main_table_results.csv.")
    for index, row in enumerate(rows):
        if index:
            print()
        print_result(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
