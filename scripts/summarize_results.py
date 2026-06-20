#!/usr/bin/env python3
"""Summarize flat JSON metric files into a CSV table."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="JSON metric files")
    parser.add_argument("--output", default="outputs/summary.csv")
    args = parser.parse_args()
    rows = []
    for result_path in args.results:
        path = Path(result_path)
        with path.open(encoding="utf-8") as handle:
            values = json.load(handle)
        row = {"result_file": str(path)}
        row.update({key: value for key, value in values.items() if isinstance(value, (int, float))})
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} result rows to {output}")


if __name__ == "__main__":
    main()
