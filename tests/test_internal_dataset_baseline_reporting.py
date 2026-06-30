from pathlib import Path

import numpy as np
import pandas as pd

from ecg_shift_bench.labels.canonical import CANONICAL_LABELS
from ecg_shift_bench.internal_dataset_baseline_reporting import (
    write_internal_dataset_baseline_result_figures,
)


def _write_summaries(output_root: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "dataset": "ptbxl",
                "dataset_name": "PTBXL",
                "output_dir": str(output_root / "ptbxl"),
                "train_records": 10,
                "validation_records": 2,
                "test_records": 2,
                "best_epoch": 1,
                "best_validation_macro_auprc": 0.81,
                "validation_accuracy": 0.78,
                "validation_auroc": 0.82,
                "validation_auprc": 0.84,
                "validation_f1_score": 0.80,
                "validation_prec": 0.79,
                "validation_rec": 0.76,
                "validation_sensitivity": 0.76,
                "validation_spec": 0.74,
                "validation_aprec": 0.85,
                "validation_br_score": 0.12,
                "validation_tn": 100,
                "validation_fp": 12,
                "validation_fn": 8,
                "validation_tp": 44,
                "test_accuracy": 0.78,
                "test_auroc": 0.72,
                "test_auprc": 0.74,
                "test_f1_score": 0.76,
                "test_prec": 0.77,
                "test_rec": 0.80,
                "test_sensitivity": 0.80,
                "test_spec": 0.82,
                "test_aprec": 0.75,
                "test_br_score": 0.13,
                "test_tn": 90,
                "test_fp": 14,
                "test_fn": 7,
                "test_tp": 55,
            },
            {
                "dataset": "code15",
                "dataset_name": "CODE15",
                "output_dir": str(output_root / "code15"),
                "train_records": 10,
                "validation_records": 2,
                "test_records": 2,
                "best_epoch": 1,
                "best_validation_macro_auprc": 0.71,
                "validation_accuracy": 0.68,
                "validation_auroc": 0.62,
                "validation_auprc": 0.64,
                "validation_f1_score": 0.70,
                "validation_prec": 0.69,
                "validation_rec": 0.66,
                "validation_sensitivity": 0.66,
                "validation_spec": 0.64,
                "validation_aprec": 0.65,
                "validation_br_score": 0.22,
                "validation_tn": 110,
                "validation_fp": 22,
                "validation_fn": 18,
                "validation_tp": 64,
                "test_accuracy": 0.58,
                "test_auroc": 0.52,
                "test_auprc": 0.54,
                "test_f1_score": 0.56,
                "test_prec": 0.57,
                "test_rec": 0.60,
                "test_sensitivity": 0.60,
                "test_spec": 0.62,
                "test_aprec": 0.55,
                "test_br_score": 0.23,
                "test_tn": 120,
                "test_fp": 24,
                "test_fn": 20,
                "test_tp": 70,
            },
            {
                "dataset": "chapman",
                "dataset_name": "CHAPMAN",
                "output_dir": str(output_root / "chapman"),
                "train_records": 10,
                "validation_records": 2,
                "test_records": 2,
                "best_epoch": 1,
                "best_validation_macro_auprc": 0.61,
                "validation_accuracy": 0.58,
                "validation_auroc": 0.52,
                "validation_auprc": 0.54,
                "validation_f1_score": 0.60,
                "validation_prec": 0.59,
                "validation_rec": 0.56,
                "validation_sensitivity": 0.56,
                "validation_spec": 0.54,
                "validation_aprec": 0.55,
                "validation_br_score": 0.32,
                "validation_tn": 130,
                "validation_fp": 32,
                "validation_fn": 28,
                "validation_tp": 84,
                "test_accuracy": 0.48,
                "test_auroc": 0.42,
                "test_auprc": 0.44,
                "test_f1_score": 0.46,
                "test_prec": 0.47,
                "test_rec": 0.50,
                "test_sensitivity": 0.50,
                "test_spec": 0.52,
                "test_aprec": 0.45,
                "test_br_score": 0.33,
                "test_tn": 140,
                "test_fp": 34,
                "test_fn": 30,
                "test_tp": 90,
            },
            {
                "dataset": "sph",
                "dataset_name": "SPH",
                "output_dir": str(output_root / "sph"),
                "train_records": 10,
                "validation_records": 2,
                "test_records": 2,
                "best_epoch": 1,
                "best_validation_macro_auprc": 0.91,
                "validation_accuracy": 0.88,
                "validation_auroc": 0.92,
                "validation_auprc": 0.94,
                "validation_f1_score": 0.90,
                "validation_prec": 0.89,
                "validation_rec": 0.86,
                "validation_sensitivity": 0.86,
                "validation_spec": 0.84,
                "validation_aprec": 0.95,
                "validation_br_score": 0.42,
                "validation_tn": 150,
                "validation_fp": 42,
                "validation_fn": 38,
                "validation_tp": 94,
                "test_accuracy": 0.88,
                "test_auroc": 0.82,
                "test_auprc": 0.84,
                "test_f1_score": 0.86,
                "test_prec": 0.87,
                "test_rec": 0.90,
                "test_sensitivity": 0.90,
                "test_spec": 0.92,
                "test_aprec": 0.85,
                "test_br_score": 0.43,
                "test_tn": 160,
                "test_fp": 44,
                "test_fn": 40,
                "test_tp": 104,
            },
        ]
    )
    summary.to_csv(output_root / "results_summary.csv", index=False)

    per_class_rows = []
    for dataset in ("ptbxl", "code15", "chapman", "sph"):
        for label_index, label in enumerate(CANONICAL_LABELS):
            per_class_rows.append(
                {
                    "dataset": dataset,
                    "split": "test",
                    "label": label,
                    "threshold": 0.5,
                    "accuracy": 0.55 + 0.01 * label_index,
                    "auroc": 0.6 + 0.01 * label_index,
                    "auprc": 0.5 + 0.01 * label_index,
                    "f1_score": 0.59 + 0.01 * label_index,
                    "prec": 0.56 + 0.01 * label_index,
                    "rec": 0.57 + 0.01 * label_index,
                    "sensitivity": 0.57 + 0.01 * label_index,
                    "spec": 0.58 + 0.01 * label_index,
                    "aprec": 0.51 + 0.01 * label_index,
                    "br_score": 0.11 + 0.01 * label_index,
                    "tn": 20 + label_index,
                    "fp": 2 + label_index,
                    "fn": 3 + label_index,
                    "tp": 4 + label_index,
                    "support": 10 + label_index,
                }
            )
    pd.DataFrame(per_class_rows).to_csv(output_root / "per_class_summary.csv", index=False)


def _write_predictions(output_root: Path) -> None:
    train_y_true = np.array(
        [
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
        ],
        dtype=int,
    )
    train_y_score = np.array(
        [
            [0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
            [0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
            [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
        ],
        dtype=float,
    )
    validation_y_true = np.array(
        [
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
        ],
        dtype=int,
    )
    validation_y_score = np.array(
        [
            [0.25, 0.75, 0.35, 0.65, 0.45, 0.55],
            [0.75, 0.25, 0.65, 0.35, 0.55, 0.45],
            [0.15, 0.85, 0.25, 0.75, 0.35, 0.65],
            [0.85, 0.15, 0.75, 0.25, 0.65, 0.35],
        ],
        dtype=float,
    )
    test_y_true = validation_y_true
    test_y_score = np.array(
        [
            [0.3, 0.7, 0.4, 0.6, 0.5, 0.5],
            [0.7, 0.3, 0.6, 0.4, 0.5, 0.5],
            [0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
            [0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
        ],
        dtype=float,
    )
    payload = {
        "label_names": np.asarray(CANONICAL_LABELS, dtype="U32"),
        "train_y_true": train_y_true,
        "train_y_score": train_y_score,
        "validation_y_true": validation_y_true,
        "validation_y_score": validation_y_score,
        "test_y_true": test_y_true,
        "test_y_score": test_y_score,
    }
    for dataset in ("ptbxl", "code15", "chapman", "sph"):
        dataset_dir = output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dataset_dir / "issue11_predictions.npz", **payload)


def test_result_figures_are_written_from_saved_summaries(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / "resnet1d_internal_dataset_baseline_results"
    output_root.mkdir(parents=True)
    _write_summaries(output_root)

    figure_paths = write_internal_dataset_baseline_result_figures(output_root)

    expected_keys = {
        "overall_metrics_png",
        "overall_metrics_svg",
        "per_label_metrics_png",
        "per_label_metrics_svg",
        "readme",
    }
    assert set(figure_paths) == expected_keys
    for path_text in figure_paths.values():
        path = Path(path_text)
        assert path.is_file()
        assert path.stat().st_size > 0

    readme = (output_root / "README.md").read_text(encoding="utf-8")
    assert "generate_report.py" in readme


def test_result_figures_include_source_script_reports_when_predictions_exist(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs" / "resnet1d_internal_dataset_baseline_results"
    output_root.mkdir(parents=True)
    _write_summaries(output_root)
    _write_predictions(output_root)

    figure_paths = write_internal_dataset_baseline_result_figures(output_root)

    assert "ptbxl_validation_report_json" in figure_paths
    assert "ptbxl_validation_report_curves_png" in figure_paths
    assert "ptbxl_validation_report_aux_png" in figure_paths
    assert Path(figure_paths["ptbxl_validation_report_json"]).is_file()
    assert Path(figure_paths["ptbxl_validation_report_curves_png"]).is_file()
    assert Path(figure_paths["ptbxl_validation_report_aux_png"]).is_file()
    readme = (output_root / "README.md").read_text(encoding="utf-8")
    assert "Source-script parity reports" in readme
