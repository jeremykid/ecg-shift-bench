#!/usr/bin/env python3
"""Create a leakage-safe leave-one-domain-out split manifest."""

import argparse
from pathlib import Path

import pandas as pd

from ecg_shift_bench.splits.leave_one_domain_out import leave_one_domain_out
from ecg_shift_bench.utils.config import load_yaml, require_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata", help="Combined CSV with record_id, patient_id, and domain")
    parser.add_argument("--output-dir", default="data/splits")
    args = parser.parse_args()

    config = load_yaml(args.config)
    require_keys(config, ["source_datasets", "target_datasets"], "experiment config")
    if not args.metadata:
        print("Split protocol validated.")
        print(f"Sources: {config['source_datasets']}")
        print(f"Targets: {config['target_datasets']}")
        print("Pass --metadata after dataset indexing to write record-level manifests.")
        return

    metadata = pd.read_csv(args.metadata)
    held_out = config.get("held_out_domain")
    if held_out is None:
        targets = config["target_datasets"]
        if len(targets) != 1:
            raise ValueError(
                "Specify held_out_domain when target_datasets does not contain one item"
            )
        held_out = targets[0]
    splits = leave_one_domain_out(metadata, held_out)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        output = output_dir / f"{config.get('experiment', 'experiment')}_{name}.csv"
        frame.to_csv(output, index=False)
        print(f"Wrote {len(frame)} records to {output}")


if __name__ == "__main__":
    main()
