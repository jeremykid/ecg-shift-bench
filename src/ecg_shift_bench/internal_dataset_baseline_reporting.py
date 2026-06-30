"""Result-page helpers for the internal supervised baseline."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_shift_bench.labels.canonical import CANONICAL_LABELS

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
) -> Path:
    readme_path = output_dir / "README.md"
    summary_columns = [
        "dataset",
        "dataset_name",
        "best_epoch",
        "validation_macro_auprc",
        "test_macro_auroc",
        "test_micro_auroc",
        "test_macro_auprc",
        "test_micro_auprc",
    ]
    per_class_columns = ["dataset", "split", "label", "auroc", "auprc", "support"]
    readme_lines = [
        "# ResNet1D Internal Dataset Baseline Results",
        "",
        "This folder contains the saved evaluation tables and figures for the internal supervised baseline.",
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
    readme_path.write_text("\n".join(readme_lines).rstrip() + "\n", encoding="utf-8")
    return readme_path


def _overall_summary_figure(summary: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    metric_specs = [
        ("test_macro_auroc", "Test macro AUROC"),
        ("test_micro_auroc", "Test micro AUROC"),
        ("test_macro_auprc", "Test macro AUPRC"),
        ("test_micro_auprc", "Test micro AUPRC"),
    ]
    dataset_labels = summary["dataset"].astype(str).tolist()
    colors = plt.get_cmap("tab10")(np.linspace(0.1, 0.9, len(dataset_labels)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
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
            "test_macro_auroc",
            "test_micro_auroc",
            "test_macro_auprc",
            "test_micro_auprc",
        },
        summary_path,
    )
    _require_columns(per_class, {"dataset", "split", "label", "auroc", "auprc"}, per_class_path)

    figure_paths: dict[str, str] = {}
    figure_paths.update(_overall_summary_figure(summary, output_dir))
    figure_paths.update(_per_label_summary_figure(per_class, output_dir))
    figure_paths["readme"] = str(
        _write_summary_readme(
            output_dir=output_dir,
            summary=summary,
            per_class=per_class,
            figure_paths=figure_paths,
        )
    )
    return figure_paths
