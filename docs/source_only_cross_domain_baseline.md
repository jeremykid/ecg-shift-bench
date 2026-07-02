# Source-Only Cross-Domain Baseline

This benchmark path is the lower-bound reference for cross-domain ECG evaluation.
It trains on labeled source-domain records only and evaluates the selected model
directly on the target-domain test split, with no target labels, no unlabeled
target data, and no test-time parameter updates.

## What It Measures

The goal is to quantify how much dataset shift remains after using only the
shared ECG alignment contract. The resulting score is the reference row that
future adaptation methods should compare against.

## Shared Preprocessing

The source-only baseline uses alignment only:

- unit conversion
- resampling to the shared rate
- crop or pad to the shared length

No z-score preprocessing, standardization, or other normalization is used
anywhere in this path.

## Configuration

Each experiment config represents one source-target pair. The reusable config
stores the source and target dataset definitions, the shared label space, the
model contract, the evaluation metrics, and the explicit protocol flags.

## Reporting

Each completed run writes a standardized summary and per-label table so the
result can be compared directly across datasets and future leaderboards. The
recorded fields include:

- source dataset
- target dataset
- label space
- model architecture
- evaluation metrics
- random seed
- split version
- preprocessing version

The source-train scores define the per-label thresholds, and the source
validation split selects the best checkpoint by macro AUPRC.

## Example Run Path

One concrete output path is:

`outputs/source-only-cross-domain/ptbxl_to_chapman/`

That directory contains the saved checkpoint, `run_status.json`, prediction
bundles, and the standard summary tables:

- `results_summary.csv`
- `per_class_summary.csv`
- `validation_metrics.json`
- `test_metrics.json`

## Contract

The implementation uses the canonical six-label order:

`AF, RBBB, LBBB, 1dAVB, SB, ST`

The report contract follows the shared source-script metric format so the same
tables can be reused as the lower-bound row in UDA, SDA, and TTA comparisons.
