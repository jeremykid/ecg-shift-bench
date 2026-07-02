#!/usr/bin/env python3
"""Evaluate saved NumPy predictions or validate an evaluation config."""

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ecg_shift_bench.evaluation.metrics import multilabel_metrics
from ecg_shift_bench.training.source_only_cross_domain import (
    rebuild_source_only_cross_domain_results,
)
from ecg_shift_bench.utils.config import load_yaml, require_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--predictions", help="NPZ containing y_true and y_score arrays")
    parser.add_argument("--run-dir", help="Rebuild standard tables from a completed run")
    args = parser.parse_args()
    if args.run_dir:
        status = rebuild_source_only_cross_domain_results(
            run_dir=Path(args.run_dir).expanduser().resolve(),
            command=shlex.join([sys.executable, *sys.argv]),
        )
        print(f"Rebuilt run status: {status['status']['status']}")
        return
    if not args.config:
        parser.error("--config is required unless --run-dir is supplied")
    config = load_yaml(args.config)
    require_keys(config, ["canonical_labels", "evaluation"], "experiment config")
    if not args.predictions:
        print(f"Evaluation config validated: {config['evaluation']['metrics']}")
        print("Pass --predictions <file.npz> after model inference.")
        return
    arrays = np.load(args.predictions)
    metrics = multilabel_metrics(arrays["y_true"], arrays["y_score"], config["canonical_labels"])
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
