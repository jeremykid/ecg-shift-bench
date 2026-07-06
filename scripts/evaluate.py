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
from ecg_shift_bench.evaluation.metrics import source_script_multilabel_report
from ecg_shift_bench.utils.config import load_yaml, require_keys


def _load_thresholds(path: Path) -> dict[str, float] | list[float] | tuple[float, ...] | np.ndarray:
    if path.suffix.lower() == ".npz":
        arrays = np.load(path, allow_pickle=True)
        if "thresholds" not in arrays:
            raise ValueError(f"{path} does not contain thresholds")
        if "label_names" in arrays:
            label_names = [str(name) for name in np.asarray(arrays["label_names"], dtype=object).tolist()]
            thresholds_array = np.asarray(arrays["thresholds"], dtype=float)
            if thresholds_array.shape != (len(label_names),):
                raise ValueError("thresholds array must match label_names length")
            return {label: float(thresholds_array[index]) for index, label in enumerate(label_names)}
        return np.asarray(arrays["thresholds"], dtype=float)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "thresholds" in payload:
            thresholds = payload["thresholds"]
            if isinstance(thresholds, dict):
                return {str(key): float(value) for key, value in thresholds.items()}
            return thresholds
        return {str(key): float(value) for key, value in payload.items()}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", help="NPZ containing y_true and y_score arrays")
    parser.add_argument(
        "--thresholds",
        help="Optional JSON/NPZ thresholds aligned with the prediction labels",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    require_keys(config, ["canonical_labels", "evaluation"], "experiment config")
    if not args.predictions:
        print(f"Evaluation config validated: {config['evaluation']['metrics']}")
        print("Pass --predictions <file.npz> after model inference.")
        return
    arrays = np.load(args.predictions, allow_pickle=True)
    if args.thresholds or "thresholds" in arrays:
        thresholds = _load_thresholds(Path(args.thresholds)) if args.thresholds else arrays["thresholds"]
        if "label_names" in arrays:
            label_names = [str(name) for name in np.asarray(arrays["label_names"], dtype=object).tolist()]
        else:
            label_names = list(config["canonical_labels"])
        metrics = source_script_multilabel_report(
            arrays["y_true"],
            arrays["y_score"],
            label_names,
            thresholds,
        )
    else:
        metrics = multilabel_metrics(arrays["y_true"], arrays["y_score"], config["canonical_labels"])
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
