#!/usr/bin/env python3
"""Run source-only preflight checks for every directed dataset pair."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from itertools import permutations
from pathlib import Path

DATASET_NAMES = ("PTBXL", "CHAPMAN", "SPH", "CODE15")


def _parse_root_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Root overrides must use DATASET=PATH, got {value!r}")
        name, path = value.split("=", 1)
        overrides[name.strip().upper()] = path.strip()
    return overrides


def _build_command(
    *,
    config: Path,
    source: str,
    target: str,
    output_root: Path,
    device: str,
    root_overrides: dict[str, str],
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("train.py")),
        "--config",
        str(config),
        "--source-dataset",
        source,
        "--target-dataset",
        target,
        "--output-dir",
        str(output_root / f"{source.lower()}_to_{target.lower()}"),
        "--device",
        device,
        "--preflight-only",
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
        default="configs/experiments/source_only.yaml",
        help="Pair-shaped source-only experiment template",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/source-only-cross-domain/_preflight",
        help="Directory that receives one preflight subdirectory per pair",
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
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    root_overrides = _parse_root_overrides(list(args.root))
    env = dict(os.environ)
    failures = 0

    for source, target in permutations(DATASET_NAMES, 2):
        command = _build_command(
            config=config,
            source=source,
            target=target,
            output_root=output_root,
            device=args.device,
            root_overrides=root_overrides,
        )
        print(f"==> preflight {source} -> {target}")
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
