#!/usr/bin/env python3
"""Validate a dataset config and report expected manual input locations."""

import argparse
from pathlib import Path

from ecg_shift_bench.datasets.registry import create_dataset
from ecg_shift_bench.utils.config import load_yaml, require_keys
from ecg_shift_bench.utils.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["ptbxl", "chapman", "sph", "code15"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", help="Override the portable root in the dataset config")
    parser.add_argument("--check-record", help="Load one record to validate waveform access")
    args = parser.parse_args()

    config = load_yaml(args.config)
    require_keys(config, ["root", "metadata_file", "records_root"], "dataset config")
    root = (
        Path(args.root).expanduser().resolve()
        if args.root
        else resolve_project_path(config["root"])
    )
    dataset = create_dataset(args.dataset, root, config)
    metadata_path = root / config["metadata_file"]
    records_path = root / config["records_root"]
    print(f"Dataset: {dataset.name}")
    print(f"Metadata expected at: {metadata_path}")
    print(f"Signals expected under: {records_path}")
    if not metadata_path.exists():
        print("Status: data not found; download it manually and follow its license.")
        return
    metadata = dataset.load_metadata()
    print(f"Status: metadata validated ({len(metadata)} records).")
    if args.check_record:
        signal = dataset.load_signal(args.check_record)
        print(f"Waveform validated: shape={signal.shape}, dtype={signal.dtype}")
    else:
        print("Pass --check-record <record_id> to validate one waveform without preprocessing.")


if __name__ == "__main__":
    main()
