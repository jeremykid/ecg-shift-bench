#!/usr/bin/env python3
"""Run the dataset discriminator study on the aligned ECG contract."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ecg_shift_bench.labels.dataset_ids import DATASET_ID_ORDER
from ecg_shift_bench.training.discriminator import DatasetSpec, run_dataset_discriminator
from ecg_shift_bench.utils.config import load_yaml, require_keys
from ecg_shift_bench.utils.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Experiment YAML for the discriminator study"
    )
    parser.add_argument("--ptbxl-root", required=True, help="PTB-XL release root")
    parser.add_argument("--code15-root", required=True, help="CODE-15 release root")
    parser.add_argument("--chapman-root", required=True, help="Chapman release root")
    parser.add_argument("--sph-root", required=True, help="SPH release root")
    parser.add_argument("--output-dir", required=True, help="Artifact directory for the study run")
    parser.add_argument("--device", default="cuda:0", help="Torch CUDA device, for example cuda:0")
    parser.add_argument("--mode", choices=["multiclass", "pairwise"], default="multiclass")
    parser.add_argument(
        "--subset",
        choices=["uncontrolled", "label_balanced", "normal_only", "random_label"],
        default="uncontrolled",
    )
    parser.add_argument(
        "--pair", nargs=2, metavar=("LEFT", "RIGHT"), help="Two datasets for pairwise mode"
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate one real batch and stop before training",
    )
    args = parser.parse_args()
    if args.mode == "pairwise" and not args.pair:
        parser.error("--pair is required when --mode pairwise")

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    require_keys(
        config, ["experiment", "model", "training", "dataset_configs"], "experiment config"
    )

    dataset_roots = {
        "PTBXL": Path(args.ptbxl_root).expanduser().resolve(),
        "CODE15": Path(args.code15_root).expanduser().resolve(),
        "CHAPMAN": Path(args.chapman_root).expanduser().resolve(),
        "SPH": Path(args.sph_root).expanduser().resolve(),
    }
    dataset_specs: list[DatasetSpec] = []
    for dataset_name in DATASET_ID_ORDER:
        dataset_config_path = resolve_project_path(config["dataset_configs"][dataset_name])
        dataset_specs.append(
            DatasetSpec(
                name=dataset_name,
                root=dataset_roots[dataset_name],
                config=load_yaml(dataset_config_path),
                config_path=dataset_config_path,
            )
        )

    command = shlex.join([sys.executable, *sys.argv])
    status = run_dataset_discriminator(
        experiment_config=config,
        experiment_config_path=config_path,
        dataset_specs=dataset_specs,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        requested_device=args.device,
        command=command,
        mode=args.mode,
        subset=args.subset,
        pair=tuple(args.pair) if args.pair else None,
        preflight_only=args.preflight_only,
    )
    print(f"Run status: {status['status']}")


if __name__ == "__main__":
    main()
