from pathlib import Path

import pandas as pd

from ecg_shift_bench.labels.canonical import CANONICAL_LABELS
from ecg_shift_bench.issue11_reporting import write_issue11_result_figures


def _write_issue11_summaries(output_root: Path) -> None:
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
                "validation_macro_auroc": 0.82,
                "validation_micro_auroc": 0.83,
                "validation_macro_auprc": 0.84,
                "validation_micro_auprc": 0.85,
                "test_macro_auroc": 0.72,
                "test_micro_auroc": 0.73,
                "test_macro_auprc": 0.74,
                "test_micro_auprc": 0.75,
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
                "validation_macro_auroc": 0.62,
                "validation_micro_auroc": 0.63,
                "validation_macro_auprc": 0.64,
                "validation_micro_auprc": 0.65,
                "test_macro_auroc": 0.52,
                "test_micro_auroc": 0.53,
                "test_macro_auprc": 0.54,
                "test_micro_auprc": 0.55,
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
                "validation_macro_auroc": 0.52,
                "validation_micro_auroc": 0.53,
                "validation_macro_auprc": 0.54,
                "validation_micro_auprc": 0.55,
                "test_macro_auroc": 0.42,
                "test_micro_auroc": 0.43,
                "test_macro_auprc": 0.44,
                "test_micro_auprc": 0.45,
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
                "validation_macro_auroc": 0.92,
                "validation_micro_auroc": 0.93,
                "validation_macro_auprc": 0.94,
                "validation_micro_auprc": 0.95,
                "test_macro_auroc": 0.82,
                "test_micro_auroc": 0.83,
                "test_macro_auprc": 0.84,
                "test_micro_auprc": 0.85,
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
                    "auroc": 0.6 + 0.01 * label_index,
                    "auprc": 0.5 + 0.01 * label_index,
                    "support": 10 + label_index,
                }
            )
    pd.DataFrame(per_class_rows).to_csv(output_root / "per_class_summary.csv", index=False)


def test_issue11_result_figures_are_written_from_saved_summaries(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / "issue11_internal_baseline_results"
    output_root.mkdir(parents=True)
    _write_issue11_summaries(output_root)

    figure_paths = write_issue11_result_figures(output_root)

    expected_keys = {
        "issue11_overall_metrics_png",
        "issue11_overall_metrics_svg",
        "issue11_per_label_metrics_png",
        "issue11_per_label_metrics_svg",
        "issue11_readme",
    }
    assert set(figure_paths) == expected_keys
    for path_text in figure_paths.values():
        path = Path(path_text)
        assert path.is_file()
        assert path.stat().st_size > 0
