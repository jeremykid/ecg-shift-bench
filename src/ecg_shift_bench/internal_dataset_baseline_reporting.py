"""Result-page helpers for the internal supervised baseline."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hmean

from ecg_shift_bench.labels.canonical import CANONICAL_LABELS
from ecg_shift_bench.evaluation.metrics import optimal_multilabel_thresholds, source_script_multilabel_report

TEST_SPLIT = "test"


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def _save_figure(fig, output_dir: Path, stem: str) -> dict[str, str]:
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return {"png": str(png_path), "svg": str(svg_path)}


def _format_value(value: object, *, precision: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "n/a"
        return f"{float(value):.{precision}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    rows = frame.loc[:, columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in rows.iterrows():
        body.append("| " + " | ".join(_format_value(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *body])


def _write_summary_readme(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    figure_paths: dict[str, str],
    report_paths: dict[str, dict[str, str]] | None = None,
) -> Path:
    readme_path = output_dir / "README.md"
    summary_columns = [
        "dataset",
        "dataset_name",
        "output_dir",
        "train_records",
        "validation_records",
        "test_records",
        "best_epoch",
        "best_validation_macro_auprc",
        "validation_accuracy",
        "validation_auroc",
        "validation_auprc",
        "validation_f1_score",
        "validation_prec",
        "validation_rec",
        "validation_sensitivity",
        "validation_spec",
        "validation_aprec",
        "validation_br_score",
        "test_accuracy",
        "test_auroc",
        "test_auprc",
        "test_f1_score",
        "test_prec",
        "test_rec",
        "test_sensitivity",
        "test_spec",
        "test_aprec",
        "test_br_score",
    ]
    per_class_columns = [
        "dataset",
        "split",
        "label",
        "threshold",
        "accuracy",
        "auroc",
        "auprc",
        "f1_score",
        "prec",
        "rec",
        "sensitivity",
        "spec",
        "aprec",
        "br_score",
        "tn",
        "fp",
        "fn",
        "tp",
        "support",
    ]
    readme_lines = [
        "# ResNet1D Internal Dataset Baseline Results",
        "",
        "This folder contains the saved evaluation tables and figures for the internal supervised baseline.",
        "Metrics follow `jeremykid/Statistical_tool/genearte_reports/generate_report.py`: thresholds are selected on the training split with ROC / Youden's J, predictions use strict `pred_proba > threshold`, and the per-label report keeps both `auprc` and `aprec` as separate fields.",
        "Per-label undefined values remain `n/a` in the rendered table and `NaN` in the CSV outputs.",
        "",
        "## Overall summary",
        "",
        _markdown_table(summary, summary_columns),
        "",
        "## Per-label metrics",
        "",
        _markdown_table(per_class, per_class_columns),
        "",
        "## Figures",
        "",
        f"![Overall metrics]({Path(figure_paths['overall_metrics_png']).name})",
        "",
        f"![Per-label metrics]({Path(figure_paths['per_label_metrics_png']).name})",
        "",
        "## Files",
        "",
        "- `results_summary.csv`",
        "- `per_class_summary.csv`",
        "- `resnet1d_internal_dataset_baseline_overall_metrics.png`",
        "- `resnet1d_internal_dataset_baseline_overall_metrics.svg`",
        "- `resnet1d_internal_dataset_baseline_per_label_metrics.png`",
        "- `resnet1d_internal_dataset_baseline_per_label_metrics.svg`",
    ]
    if report_paths:
        readme_lines.extend(
            [
                "",
                "## Source-script parity reports",
                "",
                "Detailed report artifacts are written alongside each dataset output directory and are generated from the saved train/validation/test predictions.",
            ]
        )
        for dataset_key, split_paths in report_paths.items():
            readme_lines.extend(["", f"### {dataset_key}"])
            for split_name, paths in split_paths.items():
                readme_lines.append(f"- `{split_name}`")
                for key in (
                    "report_json",
                    "report_curves_png",
                    "report_curves_svg",
                    "report_aux_png",
                    "report_aux_svg",
                ):
                    path = paths.get(key)
                    if path:
                        readme_lines.append(f"  - `{Path(path).relative_to(output_dir)}`")
    readme_path.write_text("\n".join(readme_lines).rstrip() + "\n", encoding="utf-8")
    return readme_path


def _overall_summary_figure(summary: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    metric_specs = [
        ("test_auroc", "Test AUROC"),
        ("test_auprc", "Test AUPRC"),
        ("test_aprec", "Test APREC"),
        ("test_f1_score", "Test F1"),
        ("test_accuracy", "Test accuracy"),
        ("test_spec", "Test specificity"),
    ]
    dataset_labels = summary["dataset"].astype(str).tolist()
    colors = plt.get_cmap("tab10")(np.linspace(0.1, 0.9, len(dataset_labels)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.ravel()
    for axis, (column, title) in zip(axes, metric_specs, strict=True):
        values = summary[column].astype(float).to_numpy()
        axis.bar(dataset_labels, values, color=colors, edgecolor="black", linewidth=0.5)
        axis.set_title(title)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
        for index, value in enumerate(values):
            if np.isfinite(value):
                axis.text(index, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Internal baseline test-split summary metrics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    paths = _save_figure(fig, output_dir, "resnet1d_internal_dataset_baseline_overall_metrics")
    plt.close(fig)
    return {
        "overall_metrics_png": paths["png"],
        "overall_metrics_svg": paths["svg"],
    }


def _per_label_heatmap(
    table: pd.DataFrame,
    *,
    metric: str,
    title: str,
    dataset_order: list[str],
    ax,
) -> object:
    pivot = (
        table.pivot(index="label", columns="dataset", values=metric)
        .reindex(index=CANONICAL_LABELS, columns=dataset_order)
    )
    data = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    image = ax.imshow(masked, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(column) for column in pivot.columns], rotation=20)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Label")
    for row_index, label in enumerate(pivot.index):
        for col_index, dataset in enumerate(pivot.columns):
            value = pivot.loc[label, dataset]
            if pd.notna(value):
                text = f"{float(value):.2f}"
            else:
                text = "n/a"
            ax.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    return image


def _per_label_summary_figure(per_class: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    test_rows = per_class.loc[per_class["split"].astype(str) == TEST_SPLIT].copy()
    if test_rows.empty:
        raise ValueError("Per-class summary does not contain test split rows")

    dataset_order = list(dict.fromkeys(test_rows["dataset"].astype(str).tolist()))
    test_rows["dataset"] = pd.Categorical(test_rows["dataset"].astype(str), categories=dataset_order)
    test_rows = test_rows.sort_values(["label", "dataset"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True, constrained_layout=True)
    heatmap0 = _per_label_heatmap(
        test_rows,
        metric="auroc",
        title="Test per-label AUROC",
        dataset_order=dataset_order,
        ax=axes[0],
    )
    _per_label_heatmap(
        test_rows,
        metric="auprc",
        title="Test per-label AUPRC",
        dataset_order=dataset_order,
        ax=axes[1],
    )
    fig.colorbar(heatmap0, ax=axes, shrink=0.85, label="Score")
    fig.suptitle("Internal baseline per-label test-split metrics", fontsize=14)
    paths = _save_figure(fig, output_dir, "resnet1d_internal_dataset_baseline_per_label_metrics")
    plt.close(fig)
    return {
        "per_label_metrics_png": paths["png"],
        "per_label_metrics_svg": paths["svg"],
    }


def _load_issue11_prediction_bundle(dataset_dir: Path) -> dict[str, object] | None:
    bundle_path = dataset_dir / "issue11_predictions.npz"
    if not bundle_path.is_file():
        return None
    with np.load(bundle_path, allow_pickle=True) as bundle:
        if "label_names" not in bundle.files:
            raise ValueError(f"{bundle_path} is missing label_names")
        label_names = [str(label) for label in np.asarray(bundle["label_names"], dtype=object).tolist()]
        split_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split_name in ("train", "validation", "test"):
            truth_key = f"{split_name}_y_true"
            score_key = f"{split_name}_y_score"
            if truth_key not in bundle.files or score_key not in bundle.files:
                raise ValueError(f"{bundle_path} is missing {truth_key!r} or {score_key!r}")
            split_predictions[split_name] = (
                np.asarray(bundle[truth_key]),
                np.asarray(bundle[score_key]),
            )
    return {
        "path": bundle_path,
        "label_names": label_names,
        "split_predictions": split_predictions,
    }


def _save_report_json(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _plot_source_script_label_curves(
    *,
    truth: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    report: dict[str, object],
    label_name: str,
    split_name: str,
    row_axes,
) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    precision, recall, thr = precision_recall_curve(truth, scores)
    has_both_classes = np.unique(truth).size >= 2
    pred = (scores > threshold).astype(int)
    ap = float(report.get("aprec", float("nan")))
    reported_f1 = float(report.get("f1_score", float("nan")))
    auc_value = float(report.get("auroc", float("nan")))

    pr_ax, roc_ax, thr_f1_ax, prec_thr_ax, rec_thr_ax = row_axes

    pr_ax.set_title(f"{label_name} PR")
    if has_both_classes:
        pr_ax.step(recall, precision, color="b", alpha=0.2, where="post")
        pr_ax.fill_between(recall, precision, step="post", alpha=0.2, color="b")
        pr_ax.plot(
            [float(report.get("rec", float("nan"))), 0],
            [float(report.get("prec", float("nan"))), float(report.get("prec", float("nan")))],
            color="blue",
            linestyle="--",
        )
        pr_ax.plot(
            [float(report.get("rec", float("nan"))), float(report.get("rec", float("nan")))],
            [float(report.get("prec", float("nan"))), 0],
            color="blue",
            linestyle="--",
        )
    else:
        pr_ax.text(0.5, 0.5, "n/a", ha="center", va="center")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_ylim([0.0, 1.05])
    pr_ax.set_xlim([0.0, 1.0])
    pr_ax.grid()
    pr_ax.set_title(f"{label_name} PR: AP={ap:0.2f}, F1={reported_f1:0.2f}")

    roc_ax.set_title("Receiver Operating Characteristic")
    if has_both_classes:
        fpr, tpr, _ = roc_curve(truth, scores)
        roc_ax.plot(fpr, tpr, "b", label=f"AUC = {auc_value:0.2f}")
        roc_ax.legend(loc="lower right")
        roc_ax.plot([0, 1], [0, 1], "r--")
        tn = float(report.get("tn", 0))
        fp = float(report.get("fp", 0))
        fn = float(report.get("fn", 0))
        tp = float(report.get("tp", 0))
        tprate = tp / (tp + fn) if (tp + fn) else float("nan")
        fprate = fp / (fp + tn) if (fp + tn) else float("nan")
        if np.isfinite(tprate) and np.isfinite(fprate):
            roc_ax.plot([fprate, 0], [tprate, tprate], color="blue", linestyle="--")
            roc_ax.plot([fprate, fprate], [0, tprate], color="blue", linestyle="--")
    else:
        roc_ax.text(0.5, 0.5, "n/a", ha="center", va="center")
    roc_ax.set_xlim([0, 1])
    roc_ax.set_ylim([0, 1])
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_xlabel("False Positive Rate")

    mask = recall[:-1] != 0
    thr_f1_ax.set_title("Threshold Vs F1-Score")
    if mask.any():
        f1_vec = [hmean([precision[index], recall[index]]) for index in range(int(np.sum(recall != 0)))]
        thr_f1_ax.step(thr[mask], f1_vec, color="r", alpha=0.2, where="post")
        thr_f1_ax.fill_between(thr[mask], f1_vec, step="post", alpha=0.2, color="r")
        thr_f1_ax.axvline(x=0.5, color="r")
        thr_f1_ax.set_title(
            f"Threshold Vs F1-Score: Max F1 ={float(np.max(f1_vec)):0.2f}, "
            f"Reported F1={reported_f1:0.2f}"
        )
    else:
        thr_f1_ax.text(0.5, 0.5, "n/a", ha="center", va="center")
    thr_f1_ax.set_xlabel("Threshold")
    thr_f1_ax.set_ylabel("Estimated F1-Scores")
    thr_f1_ax.set_ylim([0.0, 1.0])

    if len(thr) > 0:
        prec_thr_ax.step(precision[:-1], thr, color="b", alpha=0.2, where="post")
        prec_thr_ax.fill_between(precision[:-1], thr, alpha=0.2, color="b", step="post")
    else:
        prec_thr_ax.text(0.5, 0.5, "n/a", ha="center", va="center")
    prec_thr_ax.set_xlabel("precision")
    prec_thr_ax.set_ylabel("Threshold")
    prec_thr_ax.set_xticks(np.arange(0, 1, step=0.1))
    prec_thr_ax.set_yticks(np.arange(0, 1, step=0.1))
    prec_thr_ax.grid()

    if len(thr) > 0:
        rec_thr_ax.step(recall[:-1], thr, color="b", alpha=0.2, where="post")
        rec_thr_ax.fill_between(recall[:-1], thr, alpha=0.2, color="b", step="post")
    else:
        rec_thr_ax.text(0.5, 0.5, "n/a", ha="center", va="center")
    rec_thr_ax.set_xlabel("Recall")
    rec_thr_ax.set_ylabel("Threshold")
    rec_thr_ax.set_xticks(np.arange(0, 1, step=0.1))
    rec_thr_ax.set_yticks(np.arange(0, 1, step=0.1))
    rec_thr_ax.grid()
    pr_ax.text(
        0.99,
        0.01,
        f"thr={threshold:.3f}",
        transform=pr_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    roc_ax.text(
        0.99,
        0.01,
        f"{split_name}",
        transform=roc_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )


def _plot_source_script_label_aux(
    *,
    truth: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    report: dict[str, object],
    aux_axes,
) -> None:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    conf_ax, calib_ax = aux_axes
    pred = (scores > threshold).astype(int)

    cm = confusion_matrix(truth, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=conf_ax, colorbar=False)
    conf_ax.set_title(
        f"Confusion matrix : Acc={float(report.get('accuracy', float('nan'))):0.2f}"
    )

    if np.unique(truth).size >= 2:
        fraction_of_positives, mean_predicted_value = calibration_curve(truth, scores, n_bins=10)
        calib_ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            "o-",
            label="DL Model",
            linewidth=3,
        )
        calib_ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated Model", linewidth=1.5)
        calib_ax.legend()
    else:
        calib_ax.text(0.5, 0.5, "n/a", ha="center", va="center")
    calib_ax.set_title("Calibration Plot")
    calib_ax.set_xlabel("Mean Predicted Score")
    calib_ax.set_ylabel("Fraction of True Positives")


def _write_source_script_reports(
    *,
    dataset_dir: Path,
    dataset_name: str,
    label_names: list[str],
    split_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, str]:
    train_truth, train_scores = split_predictions["train"]
    thresholds = optimal_multilabel_thresholds(train_truth, train_scores, label_names)
    report_dir = dataset_dir / "generate_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_paths: dict[str, str] = {}
    for split_name in ("validation", "test"):
        truth, scores = split_predictions[split_name]
        report = source_script_multilabel_report(truth, scores, label_names, thresholds)
        report_json_path = report_dir / f"{dataset_name}_{split_name}_report.json"
        _save_report_json(report_json_path, report)
        report_paths[f"{dataset_name}_{split_name}_report_json"] = str(report_json_path)

        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        curve_fig, curve_axes = plt.subplots(
            len(label_names),
            5,
            figsize=(24, max(1, len(label_names)) * 4),
            constrained_layout=True,
        )
        aux_fig, aux_axes = plt.subplots(
            len(label_names),
            2,
            figsize=(12, max(1, len(label_names)) * 4),
            constrained_layout=True,
        )
        if len(label_names) == 1:
            curve_axes = np.expand_dims(curve_axes, axis=0)
            aux_axes = np.expand_dims(aux_axes, axis=0)
        for row_index, label in enumerate(label_names):
            label_report = report["per_label_reports"][label]
            threshold = float(report["thresholds"][label])
            _plot_source_script_label_curves(
                truth=truth[:, row_index],
                scores=scores[:, row_index],
                threshold=threshold,
                report=label_report,
                label_name=label,
                split_name=split_name,
                row_axes=curve_axes[row_index],
            )
            _plot_source_script_label_aux(
                truth=truth[:, row_index],
                scores=scores[:, row_index],
                threshold=threshold,
                report=label_report,
                aux_axes=aux_axes[row_index],
            )
            curve_axes[row_index, 0].set_ylabel(label)
            aux_axes[row_index, 0].set_ylabel(label)
        curve_fig.suptitle(f"{dataset_name} {split_name} source-script parity curves", fontsize=14)
        aux_fig.suptitle(f"{dataset_name} {split_name} source-script parity diagnostics", fontsize=14)
        curve_paths = _save_figure(curve_fig, report_dir, f"{dataset_name}_{split_name}_report_curves")
        aux_paths = _save_figure(aux_fig, report_dir, f"{dataset_name}_{split_name}_report_aux")
        plt.close(curve_fig)
        plt.close(aux_fig)
        report_paths[f"{dataset_name}_{split_name}_report_curves_png"] = curve_paths["png"]
        report_paths[f"{dataset_name}_{split_name}_report_curves_svg"] = curve_paths["svg"]
        report_paths[f"{dataset_name}_{split_name}_report_aux_png"] = aux_paths["png"]
        report_paths[f"{dataset_name}_{split_name}_report_aux_svg"] = aux_paths["svg"]

    return report_paths


def write_internal_dataset_baseline_result_figures(output_root: str | Path) -> dict[str, str]:
    """Create result figures and a markdown summary from saved tables."""
    output_dir = Path(output_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    summary_path = output_dir / "results_summary.csv"
    per_class_path = output_dir / "per_class_summary.csv"
    summary = pd.read_csv(summary_path)
    per_class = pd.read_csv(per_class_path)
    _require_columns(
        summary,
        {
            "dataset",
            "dataset_name",
            "output_dir",
            "train_records",
            "validation_records",
            "test_records",
            "best_epoch",
            "best_validation_macro_auprc",
            "validation_accuracy",
            "validation_auroc",
            "validation_auprc",
            "validation_f1_score",
            "validation_prec",
            "validation_rec",
            "validation_sensitivity",
            "validation_spec",
            "validation_aprec",
            "validation_br_score",
            "validation_tn",
            "validation_fp",
            "validation_fn",
            "validation_tp",
            "test_accuracy",
            "test_auroc",
            "test_auprc",
            "test_f1_score",
            "test_prec",
            "test_rec",
            "test_sensitivity",
            "test_spec",
            "test_aprec",
            "test_br_score",
            "test_tn",
            "test_fp",
            "test_fn",
            "test_tp",
        },
        summary_path,
    )
    _require_columns(
        per_class,
        {
            "dataset",
            "split",
            "label",
            "threshold",
            "accuracy",
            "auroc",
            "auprc",
            "f1_score",
            "prec",
            "rec",
            "sensitivity",
            "spec",
            "aprec",
            "br_score",
            "tn",
            "fp",
            "fn",
            "tp",
            "support",
        },
        per_class_path,
    )

    figure_paths: dict[str, str] = {}
    figure_paths.update(_overall_summary_figure(summary, output_dir))
    figure_paths.update(_per_label_summary_figure(per_class, output_dir))
    report_paths: dict[str, dict[str, dict[str, str]]] = {}
    for _, row in summary.iterrows():
        dataset_key = str(row["dataset"])
        dataset_name = str(row["dataset_name"])
        dataset_dir = Path(str(row["output_dir"]))
        bundle = _load_issue11_prediction_bundle(dataset_dir)
        if bundle is None:
            continue
        split_predictions = bundle["split_predictions"]
        label_names = list(bundle["label_names"])
        dataset_reports = _write_source_script_reports(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            label_names=label_names,
            split_predictions=split_predictions,  # type: ignore[arg-type]
        )
        report_paths[dataset_key] = {}
        for split_name in ("validation", "test"):
            report_paths[dataset_key][split_name] = {
                "report_json": dataset_reports[f"{dataset_name}_{split_name}_report_json"],
                "report_curves_png": dataset_reports[f"{dataset_name}_{split_name}_report_curves_png"],
                "report_curves_svg": dataset_reports[f"{dataset_name}_{split_name}_report_curves_svg"],
                "report_aux_png": dataset_reports[f"{dataset_name}_{split_name}_report_aux_png"],
                "report_aux_svg": dataset_reports[f"{dataset_name}_{split_name}_report_aux_svg"],
            }
            figure_paths[f"{dataset_key}_{split_name}_report_json"] = dataset_reports[
                f"{dataset_name}_{split_name}_report_json"
            ]
            figure_paths[f"{dataset_key}_{split_name}_report_curves_png"] = dataset_reports[
                f"{dataset_name}_{split_name}_report_curves_png"
            ]
            figure_paths[f"{dataset_key}_{split_name}_report_curves_svg"] = dataset_reports[
                f"{dataset_name}_{split_name}_report_curves_svg"
            ]
            figure_paths[f"{dataset_key}_{split_name}_report_aux_png"] = dataset_reports[
                f"{dataset_name}_{split_name}_report_aux_png"
            ]
            figure_paths[f"{dataset_key}_{split_name}_report_aux_svg"] = dataset_reports[
                f"{dataset_name}_{split_name}_report_aux_svg"
            ]
    figure_paths["readme"] = str(
        _write_summary_readme(
            output_dir=output_dir,
            summary=summary,
            per_class=per_class,
            figure_paths=figure_paths,
            report_paths=report_paths or None,
        )
    )
    return figure_paths
