#!/usr/bin/env python3
"""Run ECG-Adapt-inspired multi-label UDA for every directed dataset pair."""

from __future__ import annotations

import argparse
import copy
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAMES = ("CHAPMAN", "PTBXL", "SPH", "CODE15")


def _parse_root_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Root overrides must use DATASET=PATH, got {value!r}")
        name, path = value.split("=", 1)
        overrides[name.strip().upper()] = path.strip()
    return overrides


def _pair_slug(source: str, target: str) -> str:
    return f"{source.lower()}_to_{target.lower()}"


def _dataset_config_path(dataset_name: str) -> str:
    return f"configs/datasets/{dataset_name.lower()}.yaml"


def _pair_experiment_config(
    template: dict[str, Any],
    *,
    source: str,
    target: str,
) -> dict[str, Any]:
    updated = copy.deepcopy(template)
    updated["experiment"] = f"uda_ecg_adapt_{_pair_slug(source, target)}"
    updated["source_datasets"] = [source]
    updated["target_datasets"] = [target]
    updated["dataset_configs"] = {
        "source": _dataset_config_path(source),
        "target": _dataset_config_path(target),
    }
    return updated


def _write_pair_config(
    *,
    pair_output_dir: Path,
    template: dict[str, Any],
    source: str,
    target: str,
) -> Path:
    pair_output_dir.mkdir(parents=True, exist_ok=True)
    pair_config_path = pair_output_dir / "launch_config.yaml"
    pair_config = _pair_experiment_config(template, source=source, target=target)
    pair_config_path.write_text(yaml.safe_dump(pair_config, sort_keys=False), encoding="utf-8")
    return pair_config_path


def _build_command(
    *,
    pair_config_path: Path,
    source: str,
    target: str,
    pair_output_dir: Path,
    device: str,
    root_overrides: dict[str, str],
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("train.py")),
        "--config",
        str(pair_config_path),
        "--source-dataset",
        source,
        "--target-dataset",
        target,
        "--output-dir",
        str(pair_output_dir),
        "--device",
        device,
    ]
    if source in root_overrides:
        command.extend(["--source-root", root_overrides[source]])
    if target in root_overrides:
        command.extend(["--target-root", root_overrides[target]])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/uda_ecg_adapt.yaml",
        help="Pair-shaped ECG-Adapt-inspired multi-label experiment template",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/uda/ecg_adapt_all_pairs",
        help="Directory that receives one full-run subdirectory per directed pair",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device, for example cuda:0")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Optional dataset root override for any dataset name",
    )
    args = parser.parse_args()

    config = Path(args.config).expanduser().resolve()
    template = yaml.safe_load(config.read_text(encoding="utf-8"))
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    root_overrides = _parse_root_overrides(list(args.root))
    env = dict(os.environ)
    failures = 0

    for source in DATASET_NAMES:
        for target in DATASET_NAMES:
            if source == target:
                continue
            pair_output_dir = output_root / _pair_slug(source, target)
            pair_config_path = _write_pair_config(
                pair_output_dir=pair_output_dir,
                template=template,
                source=source,
                target=target,
            )
            command = _build_command(
                pair_config_path=pair_config_path,
                source=source,
                target=target,
                pair_output_dir=pair_output_dir,
                device=args.device,
                root_overrides=root_overrides,
            )
            print(f"==> train {source} -> {target}")
            result = subprocess.run(command, check=False, env=env)
            if result.returncode != 0:
                failures += 1
                print(f"FAILED: {source} -> {target}")
                print(f"command: {shlex.join(command)}")
            else:
                print(f"OK: {source} -> {target}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
