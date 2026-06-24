#!/usr/bin/env python3
"""Write dataset audit and split artifacts."""

import argparse
from pathlib import Path

from ecg_shift_bench.datasets.audit import audit_dataset, write_alignment_audit_outputs
from ecg_shift_bench.datasets.registry import create_dataset
from ecg_shift_bench.utils.config import load_yaml, require_keys
from ecg_shift_bench.utils.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["ptbxl", "chapman", "sph", "code15"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", help="Override the root in the dataset config")
    parser.add_argument("--output-dir", default="outputs/alignment")
    parser.add_argument(
        "--waveform-check-limit",
        type=int,
        default=0,
        help="Number of records to load for waveform usability checks; 0 means metadata only",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    require_keys(config, ["root", "metadata_file", "records_root"], "dataset config")
    root = (
        Path(args.root).expanduser().resolve()
        if args.root
        else resolve_project_path(config["root"])
    )
    dataset = create_dataset(args.dataset, root, config)
    result = audit_dataset(dataset, waveform_check_limit=args.waveform_check_limit)
    paths = write_alignment_audit_outputs(result, args.output_dir)
    print(f"Dataset: {dataset.name}")
    print(f"Records: {result.audit['records_total']}")
    print(f"Usable records: {result.audit['records_usable']}")
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
