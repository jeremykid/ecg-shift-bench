#!/usr/bin/env python3
"""Visualize previously exported dataset statistics and write a concise report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

DATASET_ORDER = ("ptbxl", "code15", "chapman", "sph")
SPLIT_ORDER = ("train", "validation", "test")
LABEL_ORDER = ("AF", "RBBB", "LBBB", "1dAVB", "SB", "ST")
DISPLAY_NAMES = {
    "ptbxl": "PTB-XL",
    "code15": "CODE-15",
    "chapman": "Chapman",
    "sph": "SPH",
}
DATASET_COLORS = {
    "ptbxl": "#4C78A8",
    "code15": "#F58518",
    "chapman": "#E45756",
    "sph": "#54A24B",
}
plt.rcParams["svg.hashsalt"] = "ecg-shift-bench"


def discover_dataset_dirs(stats_dir: str | Path) -> dict[str, Path]:
    """Return dataset directories in a stable, publication-friendly order."""
    root = Path(stats_dir)
    expected_files = {
        "audit.json",
        "reproducibility.json",
        "positive_rate.csv",
        "label_distribution.csv",
        "split_summary.csv",
        "split_positive_rate.csv",
    }
    discovered = {
        path.name: path
        for path in root.iterdir()
        if path.is_dir()
        and path.name != "figures"
        and any((path / filename).is_file() for filename in expected_files)
    }
    ordered_names = [name for name in DATASET_ORDER if name in discovered]
    ordered_names.extend(sorted(set(discovered) - set(ordered_names)))
    return {name: discovered[name] for name in ordered_names}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    return frame


def load_positive_rates(dataset_dirs: dict[str, Path]) -> tuple[pd.DataFrame, list[str]]:
    """Load dataset-level positive rates into a dataset-by-label table."""
    rows: list[pd.DataFrame] = []
    warnings: list[str] = []
    for dataset, directory in dataset_dirs.items():
        path = directory / "positive_rate.csv"
        try:
            frame = _read_csv(path, {"label", "positive_rate"})
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
            warnings.append(f"{dataset}: positive-rate heatmap skipped this dataset ({exc}).")
            continue
        frame = frame[["label", "positive_rate"]].copy()
        frame["dataset"] = dataset
        rows.append(frame)
    if not rows:
        return pd.DataFrame(), warnings
    return _pivot_labels(pd.concat(rows, ignore_index=True), "positive_rate"), warnings


def load_label_distribution(
    dataset_dirs: dict[str, Path],
) -> tuple[pd.DataFrame, list[str]]:
    """Load positive counts into a dataset-by-label table."""
    rows: list[pd.DataFrame] = []
    warnings: list[str] = []
    for dataset, directory in dataset_dirs.items():
        path = directory / "label_distribution.csv"
        try:
            frame = _read_csv(path, {"label", "positive_count"})
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
            warnings.append(f"{dataset}: positive-count heatmap skipped this dataset ({exc}).")
            continue
        frame = frame[["label", "positive_count"]].copy()
        frame["dataset"] = dataset
        rows.append(frame)
    if not rows:
        return pd.DataFrame(), warnings
    return _pivot_labels(pd.concat(rows, ignore_index=True), "positive_count"), warnings


def _pivot_labels(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    pivoted = frame.pivot(index="dataset", columns="label", values=value_column)
    datasets = [name for name in DATASET_ORDER if name in pivoted.index]
    datasets.extend(sorted(set(pivoted.index) - set(datasets)))
    labels = [label for label in LABEL_ORDER if label in pivoted.columns]
    labels.extend(sorted(set(pivoted.columns) - set(labels)))
    return pivoted.reindex(index=datasets, columns=labels)


def load_split_summary(
    dataset_dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load split sizes and gaps, deriving gaps from rate files when necessary."""
    size_rows: list[pd.DataFrame] = []
    gap_rows: list[pd.DataFrame] = []
    warnings: list[str] = []
    for dataset, directory in dataset_dirs.items():
        path = directory / "split_summary.csv"
        try:
            summary = _read_csv(path, {"split", "records"})
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
            warnings.append(f"{dataset}: split figures skipped this dataset ({exc}).")
            continue

        sizes = summary[["split", "records"]].copy()
        sizes["dataset"] = dataset
        size_rows.append(sizes)

        if "max_absolute_label_gap" in summary:
            gaps = summary[["split", "max_absolute_label_gap"]].copy()
        else:
            try:
                gaps = _derive_split_gaps(directory)
            except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
                warnings.append(f"{dataset}: split-balance gap skipped this dataset ({exc}).")
                continue
        gaps["dataset"] = dataset
        gap_rows.append(gaps)

    sizes = pd.concat(size_rows, ignore_index=True) if size_rows else pd.DataFrame()
    gaps = pd.concat(gap_rows, ignore_index=True) if gap_rows else pd.DataFrame()
    return sizes, gaps, warnings


def _derive_split_gaps(directory: Path) -> pd.DataFrame:
    detailed = _derive_split_label_gaps(directory)
    return (
        detailed.groupby("split", as_index=False)["absolute_gap"]
        .max()
        .rename(columns={"absolute_gap": "max_absolute_label_gap"})
    )


def _derive_split_label_gaps(directory: Path) -> pd.DataFrame:
    global_rates = _read_csv(
        directory / "positive_rate.csv", {"label", "positive_rate"}
    ).rename(columns={"positive_rate": "global_rate"})
    split_rates = _read_csv(
        directory / "split_positive_rate.csv", {"split", "label", "positive_rate"}
    )
    merged = split_rates.merge(global_rates[["label", "global_rate"]], on="label", how="left")
    if merged["global_rate"].isna().any():
        raise ValueError("split labels do not match global positive-rate labels")
    merged["absolute_gap"] = (merged["positive_rate"] - merged["global_rate"]).abs()
    return merged[["split", "label", "absolute_gap"]]


def load_split_label_gaps(
    dataset_dirs: dict[str, Path],
) -> tuple[pd.DataFrame, list[str]]:
    """Load label-specific split gaps for the detailed balance heatmap."""
    rows: list[pd.DataFrame] = []
    warnings: list[str] = []
    for dataset, directory in dataset_dirs.items():
        try:
            frame = _derive_split_label_gaps(directory)
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
            warnings.append(f"{dataset}: split-label gap heatmap skipped this dataset ({exc}).")
            continue
        frame["dataset"] = dataset
        rows.append(frame)
    if not rows:
        return pd.DataFrame(), warnings
    return pd.concat(rows, ignore_index=True), warnings


def load_audits(
    dataset_dirs: dict[str, Path],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Load audit and reproducibility metadata."""
    metadata: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for dataset, directory in dataset_dirs.items():
        try:
            audit = _read_json(directory / "audit.json")
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            warnings.append(f"{dataset}: audit metadata unavailable ({exc}).")
            continue
        try:
            reproducibility = _read_json(directory / "reproducibility.json")
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            reproducibility = {}
            warnings.append(f"{dataset}: waveform contract unavailable ({exc}).")
        metadata[dataset] = {"audit": audit, "reproducibility": reproducibility}
    return metadata, warnings


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    metadata: dict[str, Any]
    if path.suffix == ".png":
        metadata = {"Software": "ecg-shift-bench"}
    elif path.suffix == ".pdf":
        metadata = {"Creator": "ecg-shift-bench", "CreationDate": None, "ModDate": None}
    else:
        metadata = {"Creator": "ecg-shift-bench", "Date": None}
    fig.savefig(path, dpi=300, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def plot_positive_rate_heatmap(rates: pd.DataFrame, output_path: Path) -> bool:
    """Plot dataset-level label positive rates as percentages."""
    if rates.empty:
        return False
    values = rates.to_numpy(dtype=float) * 100
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(np.ma.masked_invalid(values), cmap="YlOrRd", aspect="auto", vmin=0)
    _configure_heatmap_axes(ax, rates)
    for row, column in np.ndindex(values.shape):
        if np.isfinite(values[row, column]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color=_heatmap_text_color(image, values[row, column]),
            )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Positive rate (%)")
    ax.set_title("Label Positive Rates Across Datasets")
    _save_figure(fig, output_path)
    return True


def plot_positive_count_heatmap(counts: pd.DataFrame, output_path: Path) -> bool:
    """Plot positive counts with logarithmic color scaling and exact annotations."""
    if counts.empty:
        return False
    values = counts.to_numpy(dtype=float)
    positive_values = values[np.isfinite(values) & (values > 0)]
    norm = None
    if positive_values.size:
        norm = LogNorm(vmin=max(1, float(positive_values.min())), vmax=float(positive_values.max()))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(np.ma.masked_invalid(values), cmap="Blues", aspect="auto", norm=norm)
    _configure_heatmap_axes(ax, counts)
    for row, column in np.ndindex(values.shape):
        if np.isfinite(values[row, column]):
            ax.text(
                column,
                row,
                f"{values[row, column]:,.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color=_heatmap_text_color(image, values[row, column]),
            )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Positive count (log color scale)" if norm else "Positive count")
    ax.set_title("Label Positive Counts Across Datasets")
    _save_figure(fig, output_path)
    return True


def _configure_heatmap_axes(ax: plt.Axes, frame: pd.DataFrame) -> None:
    ax.set_xticks(np.arange(len(frame.columns)), frame.columns)
    ax.set_yticks(np.arange(len(frame.index)), [_display_name(name) for name in frame.index])
    ax.set_xlabel("Label")
    ax.set_ylabel("Dataset")
    ax.tick_params(top=False, bottom=True)


def _heatmap_text_color(image: Any, value: float) -> str:
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.45 else "black"


def plot_label_prior_range(rates: pd.DataFrame, output_path: Path) -> bool:
    """Plot each label's prevalence range and dataset-specific rates."""
    if rates.empty:
        return False
    percentages = rates * 100
    labels = list(percentages.columns)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for index, label in enumerate(labels):
        values = percentages[label].dropna()
        if not values.empty:
            ax.hlines(index, values.min(), values.max(), color="#B8B8B8", linewidth=2, zorder=1)
    for dataset in percentages.index:
        ax.scatter(
            percentages.loc[dataset],
            y,
            s=48,
            color=DATASET_COLORS.get(dataset, "#777777"),
            edgecolor="white",
            linewidth=0.6,
            label=_display_name(dataset),
            zorder=2,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Positive rate (%)")
    ax.set_ylabel("Label")
    ax.set_title("Cross-Dataset Label-Prior Range")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    _save_figure(fig, output_path)
    return True


def plot_split_label_gap_heatmap(label_gaps: pd.DataFrame, output_path: Path) -> bool:
    """Plot absolute label-rate gaps for every dataset and split."""
    if label_gaps.empty:
        return False
    table = label_gaps.pivot_table(
        index=["dataset", "split"],
        columns="label",
        values="absolute_gap",
        aggfunc="max",
    )
    row_order = [
        (dataset, split)
        for dataset in _ordered_datasets(table.index.get_level_values("dataset"))
        for split in SPLIT_ORDER
        if (dataset, split) in table.index
    ]
    labels = [label for label in LABEL_ORDER if label in table.columns]
    labels.extend(sorted(set(table.columns) - set(labels)))
    table = table.reindex(index=pd.MultiIndex.from_tuples(row_order), columns=labels)
    values = table.to_numpy(dtype=float) * 100
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    image = ax.imshow(np.ma.masked_invalid(values), cmap="Reds", aspect="auto", vmin=0)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(
        np.arange(len(row_order)),
        [f"{_display_name(dataset)} / {split}" for dataset, split in row_order],
    )
    ax.set_xlabel("Label")
    ax.set_ylabel("Dataset / split")
    ax.set_title("Split-to-Overall Label-Rate Gap")
    for row, column in np.ndindex(values.shape):
        value = values[row, column]
        if np.isfinite(value):
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold" if value > 1.0 else "normal",
                color=_heatmap_text_color(image, value),
            )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Absolute gap (percentage points)")
    _save_figure(fig, output_path)
    return True


def write_summary_report(
    output_path: Path,
    metadata: dict[str, dict[str, Any]],
    split_sizes: pd.DataFrame,
    rates: pd.DataFrame,
    counts: pd.DataFrame,
    gaps: pd.DataFrame,
    warnings: list[str],
) -> None:
    """Write a deterministic narrative summary of the loaded statistics."""
    lines = [
        "# Dataset Statistics Visualization Summary",
        "",
        "This report was generated only from the exported statistics artifacts; no raw ECG "
        "waveforms were read.",
        "",
        "## Dataset scale overview",
        "",
    ]
    for dataset, item in metadata.items():
        audit = item["audit"]
        patients = audit.get("patients_total")
        patient_text = "N/A" if patients is None else f"{int(patients):,}"
        lines.append(
            f"- **{_display_name(dataset)}:** {int(audit.get('records_total', 0)):,} records; "
            f"{patient_text} patients; {int(audit.get('records_excluded', 0)):,} excluded."
        )

    lines.extend(["", "## Missing and excluded records", ""])
    for dataset, item in metadata.items():
        audit = item["audit"]
        total = audit.get("records_total")
        usable = audit.get("records_usable")
        excluded = audit.get("records_excluded")
        missing_labels = {
            label: count
            for label, count in audit.get("missing_labels", {}).items()
            if count
        }
        missing_text = (
            ", ".join(f"{label}: {count}" for label, count in sorted(missing_labels.items()))
            if missing_labels
            else "no missing label annotations reported"
        )
        lines.append(
            f"- **{_display_name(dataset)}:** {_format_count(usable)} usable of "
            f"{_format_count(total)} total; {_format_count(excluded)} excluded; {missing_text}."
        )

    lines.extend(["", "## Split composition and policy", ""])
    for dataset, item in metadata.items():
        audit = item["audit"]
        policy = audit.get("split_policy", {})
        dataset_splits = (
            split_sizes.loc[split_sizes["dataset"] == dataset]
            if "dataset" in split_sizes
            else pd.DataFrame()
        )
        total = dataset_splits["records"].sum() if not dataset_splits.empty else 0
        percentages = []
        for split in SPLIT_ORDER:
            selected = dataset_splits.loc[dataset_splits["split"] == split, "records"]
            if not selected.empty and total:
                percentages.append(f"{split} {100 * float(selected.iloc[0]) / total:.1f}%")
        split_level = policy.get("split_level", "N/A")
        note = (
            "patient-level assignment limits patient leakage"
            if split_level == "patient"
            else "record-level assignment cannot guarantee patient independence"
        )
        lines.append(
            f"- **{_display_name(dataset)}:** source `{policy.get('split_source', 'N/A')}`, "
            f"level `{split_level}` ({', '.join(percentages) or 'percentages unavailable'}); "
            f"{note}."
        )

    lines.extend(["", "## Waveform contract", ""])
    for dataset, item in metadata.items():
        contract = item["reproducibility"]
        leads = contract.get("lead_order", [])
        conversion = (
            f"{contract.get('source_unit', 'N/A')} → {contract.get('target_unit', 'N/A')}"
            if contract.get("unit_converted")
            else f"{contract.get('target_unit', contract.get('source_unit', 'N/A'))}, no conversion"
        )
        lines.append(
            f"- **{_display_name(dataset)}:** {contract.get('target_sampling_rate', 'N/A')} Hz, "
            f"{contract.get('target_length', 'N/A')} samples, {len(leads) or 'N/A'} leads, "
            f"units {conversion}."
        )

    lines.extend(["", "## Label-prior shift observations", ""])
    if rates.empty:
        lines.append("- Positive-rate data were unavailable.")
    else:
        for label in rates.columns:
            series = rates[label].dropna()
            if series.empty:
                continue
            low_name, high_name = series.idxmin(), series.idxmax()
            lines.append(
                f"- **{label}:** {_display_name(low_name)} {100 * series.min():.2f}% to "
                f"{_display_name(high_name)} {100 * series.max():.2f}% "
                f"({100 * (series.max() - series.min()):.2f} percentage-point range)."
            )

    lines.extend(["", "## Rare-label reliability warnings", ""])
    if counts.empty:
        lines.append("- Positive-count data were unavailable.")
    else:
        for dataset, row in counts.iterrows():
            available = row.dropna()
            if available.empty:
                continue
            minimum = available.min()
            labels = ", ".join(available.index[available == minimum])
            lines.append(
                f"- **{_display_name(dataset)}:** the smallest positive count is "
                f"{int(minimum):,} for {labels}; estimates for the least represented labels "
                "should receive wider uncertainty bounds and cautious interpretation."
            )

    lines.extend(["", "## Split-balance warnings", ""])
    high_gaps = gaps.loc[gaps.get("max_absolute_label_gap", pd.Series(dtype=float)) > 0.01]
    if high_gaps.empty:
        lines.append("- No available split has a maximum absolute label gap above 0.01.")
    else:
        for row in high_gaps.itertuples(index=False):
            lines.append(
                f"- **{_display_name(row.dataset)} {row.split}:** gap "
                f"{row.max_absolute_label_gap:.4f} exceeds 0.01."
            )

    lines.extend(
        [
            "",
            "## Implications for cross-dataset benchmarking",
            "",
            "- Dataset size and label prevalence differ substantially, so pooled metrics can be "
            "dominated by larger domains and should be accompanied by per-domain results.",
            "- Positive-rate differences indicate label-prior shift; calibration and decision "
            "thresholds learned in one domain may not transfer directly.",
            "- Low positive counts increase metric variance, especially for split-level and "
            "cross-domain estimates; report confidence intervals where possible.",
            "- Chapman uses a record-level split because patient identifiers are unavailable, "
            "so its within-dataset estimates do not provide the same leakage protection as "
            "patient-level splits.",
        ]
    )

    lines.extend(["", "## Generation warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in dict.fromkeys(warnings))
    else:
        lines.append("- None.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _ordered_datasets(names: Any) -> list[str]:
    name_set = set(names)
    ordered = [name for name in DATASET_ORDER if name in name_set]
    ordered.extend(sorted(name_set - set(ordered)))
    return ordered


def _display_name(dataset: str) -> str:
    return DISPLAY_NAMES.get(dataset, dataset)


def _format_count(value: Any) -> str:
    return "N/A" if value is None else f"{int(value):,}"


def generate_visualizations(stats_dir: Path, output_dir: Path, image_format: str) -> list[Path]:
    """Generate all available figures and the summary report."""
    dataset_dirs = discover_dataset_dirs(stats_dir)
    if not dataset_dirs:
        raise ValueError(f"No dataset statistics directories found under {stats_dir}")

    metadata, audit_warnings = load_audits(dataset_dirs)
    rates, rate_warnings = load_positive_rates(dataset_dirs)
    counts, count_warnings = load_label_distribution(dataset_dirs)
    split_sizes, gaps, split_warnings = load_split_summary(dataset_dirs)
    label_gaps, label_gap_warnings = load_split_label_gaps(dataset_dirs)
    warnings = (
        audit_warnings
        + rate_warnings
        + count_warnings
        + split_warnings
        + label_gap_warnings
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    figures = [
        (
            plot_positive_rate_heatmap,
            (rates,),
            "positive_rate_heatmap",
        ),
        (
            plot_positive_count_heatmap,
            (counts,),
            "positive_count_heatmap",
        ),
        (
            plot_label_prior_range,
            (rates,),
            "label_prior_range",
        ),
        (
            plot_split_label_gap_heatmap,
            (label_gaps,),
            "split_label_gap_heatmap",
        ),
    ]
    for function, arguments, stem in figures:
        path = output_dir / f"{stem}.{image_format}"
        if function(*arguments, path):
            outputs.append(path)
        else:
            warnings.append(f"{stem}: figure not generated because no usable data were available.")

    summary_path = output_dir / "dataset_statistics_summary.md"
    write_summary_report(
        summary_path,
        metadata,
        split_sizes,
        rates,
        counts,
        gaps,
        warnings,
    )
    outputs.append(summary_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-dir", default="outputs/dataset_statistics")
    parser.add_argument("--output-dir", default="outputs/dataset_statistics/figures")
    parser.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_visualizations(
        Path(args.stats_dir), Path(args.output_dir), args.format.lower()
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
