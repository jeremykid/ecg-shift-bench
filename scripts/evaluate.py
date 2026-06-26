#!/usr/bin/env python3
"""Evaluate saved NumPy predictions or validate an evaluation config."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ecg_shift_bench.evaluation.metrics import multilabel_metrics
from ecg_shift_bench.utils.config import load_yaml, require_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", help="NPZ containing y_true and y_score arrays")
    args = parser.parse_args()
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
