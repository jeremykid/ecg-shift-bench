#!/usr/bin/env python3
"""Export dataset statistics, split manifests, and summary reports."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _bootstrap_project_python() -> None:
    """Prefer the repository virtualenv when the current interpreter lacks wfdb."""
    try:
        import wfdb  # noqa: F401

        return
    except ModuleNotFoundError:
        candidates = [PROJECT_ROOT, Path.cwd()]
        try:
            git_common_dir = subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--git-common-dir"],
                text=True,
            ).strip()
        except Exception:
            git_common_dir = ""
        if git_common_dir:
            candidates.append(Path(git_common_dir).resolve().parent)
        for base in candidates:
            venv_python = base / ".venv" / "bin" / "python"
            if venv_python.exists():
                os.execv(str(venv_python), [str(venv_python), *sys.argv])
        raise


_bootstrap_project_python()

from ecg_shift_bench.datasets.audit import audit_dataset
from ecg_shift_bench.datasets.statistics import (
    write_batch_dataset_statistics_report,
    write_dataset_statistics_outputs,
)
from ecg_shift_bench.utils.config import load_yaml, require_keys
from ecg_shift_bench.utils.paths import resolve_project_path

DATASET_CONFIGS = {
    "ptbxl": "configs/datasets/ptbxl.yaml",
    "code15": "configs/datasets/code15.yaml",
    "chapman": "configs/datasets/chapman.yaml",
    "sph": "configs/datasets/sph.yaml",
}

create_dataset = None


def _create_dataset(name: str, root: Path, config: dict[str, object]):
    """Resolve the dataset constructor lazily so optional backends stay optional."""
    global create_dataset
    if create_dataset is None:
        from ecg_shift_bench.datasets import create_dataset as dataset_factory

        create_dataset = dataset_factory
    return create_dataset(name, root, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/dataset_statistics")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--waveform-check-limit",
        type=int,
        default=None,
        help="Limit waveform validation to this many records; omit for a full audit.",
    )
    mode.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip waveform loading and only audit metadata and split manifests.",
    )
    parser.add_argument("--ptbxl-root")
    parser.add_argument("--code15-root")
    parser.add_argument("--chapman-root")
    parser.add_argument("--sph-root")
    args = parser.parse_args()

    root_overrides = {
        "ptbxl": args.ptbxl_root,
        "code15": args.code15_root,
        "chapman": args.chapman_root,
        "sph": args.sph_root,
    }

    results = []
    output_dir = Path(args.output_dir)
    for dataset_name, config_path in DATASET_CONFIGS.items():
        config = load_yaml(config_path)
        require_keys(config, ["root", "metadata_file", "records_root"], "dataset config")
        root = (
            Path(root_overrides[dataset_name]).expanduser().resolve()
            if root_overrides[dataset_name]
            else resolve_project_path(config["root"])
        )
        dataset = _create_dataset(dataset_name, root, config)
        waveform_check_limit = 0 if args.metadata_only else args.waveform_check_limit
        result = audit_dataset(dataset, waveform_check_limit=waveform_check_limit)
        write_dataset_statistics_outputs(result, output_dir)
        results.append(result)
        print(f"Exported statistics for {dataset.name} -> {output_dir / result.dataset}")

    report_path = write_batch_dataset_statistics_report(results, output_dir)
    print(f"Wrote batch report: {report_path}")


if __name__ == "__main__":
    main()
