"""Dataset statistics report generation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ecg_shift_bench.datasets.audit import DatasetAuditResult
from ecg_shift_bench.labels.canonical import CANONICAL_LABELS

SPLIT_NAMES = ("train", "validation", "test")


def write_dataset_statistics_outputs(
    result: DatasetAuditResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write the per-dataset statistics artifacts to a stable directory."""
    dataset_dir = Path(output_dir) / result.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = result.split_manifest.copy()
    split_counts = _split_counts(split_manifest)
    label_distribution = _label_distribution(result.audit)
    positive_rate = _positive_rate(result.audit)

    paths = {
        "audit": dataset_dir / "audit.json",
        "split_manifest": dataset_dir / "split_manifest.csv",
        "train": dataset_dir / "train.csv",
        "validation": dataset_dir / "validation.csv",
        "test": dataset_dir / "test.csv",
        "exclusions": dataset_dir / "exclusions.csv",
        "reproducibility": dataset_dir / "reproducibility.json",
        "label_distribution": dataset_dir / "label_distribution.csv",
        "positive_rate": dataset_dir / "positive_rate.csv",
        "split_label_distribution": dataset_dir / "split_label_distribution.csv",
        "split_positive_rate": dataset_dir / "split_positive_rate.csv",
        "split_summary": dataset_dir / "split_summary.csv",
        "summary": dataset_dir / "summary.md",
    }

    _write_json(paths["audit"], result.audit)
    split_manifest.to_csv(paths["split_manifest"], index=False)
    for split in SPLIT_NAMES:
        split_manifest.loc[split_manifest["split"] == split].to_csv(paths[split], index=False)
    result.exclusions.to_csv(paths["exclusions"], index=False)
    _write_json(paths["reproducibility"], result.reproducibility)
    label_distribution.to_csv(paths["label_distribution"], index=False)
    positive_rate.to_csv(paths["positive_rate"], index=False)
    split_label_distribution = _split_label_distribution(result.audit)
    split_positive_rate = _split_positive_rate(result.audit)
    split_label_distribution.to_csv(paths["split_label_distribution"], index=False)
    split_positive_rate.to_csv(paths["split_positive_rate"], index=False)
    split_counts.to_csv(paths["split_summary"], index=False)
    paths["summary"].write_text(render_dataset_statistics_report(result), encoding="utf-8")

    return {key: str(path) for key, path in paths.items()}


def render_dataset_statistics_report(result: DatasetAuditResult) -> str:
    """Render one dataset's statistics as a markdown report."""
    audit = result.audit
    split_summary = _split_counts(result.split_manifest)
    label_distribution = _label_distribution(audit)
    positive_rate = _positive_rate(audit)
    split_label_distribution = _split_label_distribution(audit)
    split_positive_rate = _split_positive_rate(audit)
    split_policy = audit.get("split_policy", {})
    waveform_check = audit.get("waveform_check", {})
    split_balance_warnings = _split_balance_warnings(audit)

    lines = [
        f"# {audit.get('dataset', result.dataset.upper())} Dataset Statistics",
        "",
        "## Overview",
        "",
        _markdown_table(
            [
                [
                    result.dataset,
                    audit.get("domain", "n/a"),
                    audit.get("records_total", "n/a"),
                    audit.get("patients_total", "n/a"),
                    len(audit.get("available_labels", CANONICAL_LABELS)),
                    audit.get("records_usable", "n/a"),
                    audit.get("records_excluded", "n/a"),
                ]
            ],
            [
                "Dataset",
                "Domain",
                "Records total",
                "Patients total",
                "Classes",
                "Records usable",
                "Records excluded",
            ],
        ),
        "",
        "## Split Policy",
        "",
        _markdown_table(
            [
                [
                    split_policy.get("split_source", "n/a"),
                    split_policy.get("split_level", "n/a"),
                    split_policy.get("split_algorithm", "n/a"),
                    split_policy.get("seed", "n/a"),
                    split_policy.get("train_fraction", "n/a"),
                    split_policy.get("validation_fraction", "n/a"),
                    split_policy.get("test_fraction", "n/a"),
                ]
            ],
            [
                "split_source",
                "split_level",
                "split_algorithm",
                "seed",
                "train_fraction",
                "validation_fraction",
                "test_fraction",
            ],
        ),
        "",
        "## Waveform Contract",
        "",
        _markdown_table(
            [
                [
                    waveform_check.get("mode", "n/a"),
                    waveform_check.get("checked_records", "n/a"),
                    waveform_check.get("target_sampling_rate", "n/a"),
                    waveform_check.get("target_length", "n/a"),
                    ", ".join(waveform_check.get("lead_order", [])) or "n/a",
                    waveform_check.get("source_unit", "n/a"),
                    waveform_check.get("target_unit", "n/a"),
                ]
            ],
            [
                "mode",
                "checked_records",
                "target_sampling_rate",
                "target_length",
                "lead_order",
                "source_unit",
                "target_unit",
            ],
        ),
        "",
        "## Split Sizes",
        "",
        _markdown_table(
            split_summary[["split", "records", "patients"]].to_dict(orient="records"),
            ["split", "records", "patients"],
        ),
        "",
        "## Split Balance",
        "",
        _markdown_table(
            split_balance_rows(audit),
            [
                "split",
                "records",
                "patients",
                "max_absolute_label_gap",
                "skewed_labels",
            ],
        ),
        "",
        "## Label Distribution",
        "",
        _markdown_table(
            label_distribution.to_dict(orient="records"),
            ["label", "positive_count"],
        ),
        "",
        "## Positive rate",
        "",
        _markdown_table(
            positive_rate.to_dict(orient="records"),
            ["label", "positive_rate"],
        ),
        "",
        "## Split Label Distribution",
        "",
        _markdown_table(
            split_label_distribution.to_dict(orient="records"),
            ["split", "label", "positive_count"],
        ),
        "",
        "## Split Positive rate",
        "",
        _markdown_table(
            split_positive_rate.to_dict(orient="records"),
            ["split", "label", "positive_rate"],
        ),
        "",
        "## Missing or Invalid Records",
        "",
        f"- Records excluded: {audit.get('records_excluded', 'n/a')}",
        f"- Records usable: {audit.get('records_usable', 'n/a')}",
    ]
    if split_balance_warnings:
        lines.append("- Split balance warnings:")
        for warning in split_balance_warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("- Split balance warnings: none")
    exclusions = result.exclusions
    if exclusions.empty:
        lines.append("- Exclusions: none")
    else:
        lines.append("- Exclusions:")
        for row in exclusions.to_dict(orient="records"):
            lines.append(f"  - {row.get('record_id', 'n/a')}: {row.get('reason', 'n/a')}")
    return "\n".join(lines).rstrip() + "\n"


def render_batch_dataset_statistics_report(results: Sequence[DatasetAuditResult]) -> str:
    """Render a combined report covering all dataset statistics outputs."""
    lines = ["# Dataset Statistics Report", "", "## Overview", ""]
    overview_rows = []
    for result in results:
        audit = result.audit
        split_policy = audit.get("split_policy", {})
        split_summary = _split_counts(result.split_manifest)
        overview_rows.append(
            [
                result.dataset,
                audit.get("dataset", result.dataset.upper()),
                audit.get("domain", "n/a"),
                audit.get("records_total", "n/a"),
                audit.get("patients_total", "n/a"),
                audit.get("records_excluded", "n/a"),
                split_policy.get("split_source", "n/a"),
                split_policy.get("split_level", "n/a"),
                split_summary.loc[split_summary["split"] == "train", "records"].iloc[0],
                split_summary.loc[split_summary["split"] == "validation", "records"].iloc[0],
                split_summary.loc[split_summary["split"] == "test", "records"].iloc[0],
            ]
        )
    lines.append(
        _markdown_table(
            overview_rows,
            [
                "dataset",
                "name",
                "domain",
                "records_total",
                "patients_total",
                "records_excluded",
                "split_source",
                "split_level",
                "train_records",
                "validation_records",
                "test_records",
            ],
        )
    )
    warnings: list[str] = []
    for result in results:
        for warning in _split_balance_warnings(result.audit):
            warnings.append(f"{result.dataset}: {warning}")
    if warnings:
        lines.extend(["", "## Split Balance Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])
    for result in results:
        lines.extend(["", render_dataset_statistics_report(result).rstrip()])
    return "\n".join(lines).rstrip() + "\n"


def write_batch_dataset_statistics_report(
    results: Sequence[DatasetAuditResult],
    output_dir: str | Path,
) -> str:
    """Write the combined markdown report for all dataset statistics outputs."""
    output_path = Path(output_dir) / "dataset_statistics_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_batch_dataset_statistics_report(results), encoding="utf-8")
    return str(output_path)


def _split_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        frame = manifest.loc[manifest["split"] == split]
        row: dict[str, Any] = {
            "split": split,
            "records": int(len(frame)),
            "patients": int(frame["patient_id"].nunique()) if "patient_id" in frame else "n/a",
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=["split", "records", "patients"])


def split_balance_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_rates = audit.get("positive_prevalence", {})
    split_summary = audit.get("split_label_summary", {})
    for split in SPLIT_NAMES:
        summary = split_summary.get(split, {})
        split_rates = summary.get("positive_prevalence", {})
        max_gap = 0.0
        skewed_labels: list[str] = []
        for label in audit.get("available_labels", CANONICAL_LABELS):
            global_rate = float(global_rates.get(label, 0.0))
            split_rate = float(split_rates.get(label, 0.0))
            gap = abs(split_rate - global_rate)
            max_gap = max(max_gap, gap)
            if _is_materially_skewed(global_rate, split_rate):
                skewed_labels.append(label)
        rows.append(
            {
                "split": split,
                "records": int(summary.get("records", 0)),
                "patients": summary.get("patients", "n/a"),
                "max_absolute_label_gap": round(max_gap, 4),
                "skewed_labels": ", ".join(skewed_labels) if skewed_labels else "none",
            }
        )
    return rows


def _label_distribution(audit: dict[str, Any]) -> pd.DataFrame:
    positive_counts = audit.get("positive_counts", {})
    rows = [
        {"label": label, "positive_count": int(positive_counts.get(label, 0))}
        for label in audit.get("available_labels", CANONICAL_LABELS)
    ]
    return pd.DataFrame(rows, columns=["label", "positive_count"])


def _positive_rate(audit: dict[str, Any]) -> pd.DataFrame:
    positive_rate = audit.get("positive_prevalence", {})
    rows = [
        {"label": label, "positive_rate": float(positive_rate.get(label, 0.0))}
        for label in audit.get("available_labels", CANONICAL_LABELS)
    ]
    return pd.DataFrame(rows, columns=["label", "positive_rate"])


def _split_label_distribution(audit: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_summary = audit.get("split_label_summary", {})
    for split in SPLIT_NAMES:
        summary = split_summary.get(split, {})
        counts = summary.get("positive_counts", {})
        for label in audit.get("available_labels", CANONICAL_LABELS):
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "positive_count": int(counts.get(label, 0)),
                }
            )
    return pd.DataFrame(rows, columns=["split", "label", "positive_count"])


def _split_positive_rate(audit: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_summary = audit.get("split_label_summary", {})
    for split in SPLIT_NAMES:
        summary = split_summary.get(split, {})
        rates = summary.get("positive_prevalence", {})
        for label in audit.get("available_labels", CANONICAL_LABELS):
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "positive_rate": float(rates.get(label, 0.0)),
                }
            )
    return pd.DataFrame(rows, columns=["split", "label", "positive_rate"])


def _split_balance_warnings(audit: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    split_summary = audit.get("split_label_summary", {})
    global_rates = audit.get("positive_prevalence", {})
    for split, summary in split_summary.items():
        rates = summary.get("positive_prevalence", {})
        skewed = [
            label
            for label in audit.get("available_labels", CANONICAL_LABELS)
            if _is_materially_skewed(float(global_rates.get(label, 0.0)), float(rates.get(label, 0.0)))
        ]
        if skewed:
            warnings.append(f"{split}: skewed labels {', '.join(skewed)}")
    return warnings


def _is_materially_skewed(global_rate: float, split_rate: float) -> bool:
    if global_rate == 0.0:
        return split_rate > 0.0
    return abs(split_rate - global_rate) >= 0.10


def _markdown_table(rows: Sequence[dict[str, Any] | Sequence[Any]], headers: Sequence[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        if isinstance(row, dict):
            values = [row.get(header, "n/a") for header in headers]
        else:
            values = list(row)
        lines.append("| " + " | ".join(_format_cell(value) for value in values) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
