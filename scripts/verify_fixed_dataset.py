from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdhgnn.data import (
    compare_fixed_datasets,
    load_fixed_hypergraph_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two fixed hypergraph datasets element by element."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = load_fixed_hypergraph_dataset(args.reference)
    candidate = load_fixed_hypergraph_dataset(args.candidate)
    report = compare_fixed_datasets(reference, candidate)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    if not report["exact_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
