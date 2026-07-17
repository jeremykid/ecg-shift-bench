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

Run one configured pair with:

```bash
python scripts/train.py \
  --config configs/experiments/source_only.yaml \
  --source-dataset PTBXL \
  --target-dataset CHAPMAN \
  --output-dir outputs/source-only-cross-domain/ptbxl_to_chapman
```

The pair can be overridden without changing the shared protocol. The all-pair
preflight covers PTBXL, CHAPMAN, SPH, CODE15, and CPSC.

## Protocol

```text
Train: labeled source-domain training data
Adaptation: none
Evaluation: target-domain test set
Target labels during training: unavailable
Target unlabeled data during training: unavailable
Model updates during testing: none
```

The target dataset is constructed only after source-only training and
source-validation checkpoint selection finish. Target labels are used only to
calculate final report metrics.

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
- training log

The source-train scores define the per-label thresholds, and the source
validation split selects the best checkpoint by macro AUPRC.

## Completed Results

The completed evidence contains all 20 directed pairs among PTBXL, CHAPMAN,
SPH, CODE15, and CPSC. Compact reviewable tables are published at:

- `outputs/source_only_cross_domain_results/results_summary.csv`
- `outputs/source_only_cross_domain_results/per_class_summary.csv`

For PTBXL to CHAPMAN, the target-test metrics are:

- macro AUROC: `0.9738494210136177`
- macro AUPRC: `0.6965388649964256`
- macro F1: `0.5949690224212706`
- macro accuracy: `0.9173320807891029`

Full runtime directories additionally contain the selected checkpoint,
configuration snapshot, manifests, predictions, and JSON metrics. Those large
or machine-local artifacts remain ignored and are not part of the published
benchmark tables.

## Contract

The implementation uses the canonical six-label order:

`AF, RBBB, LBBB, 1dAVB, SB, ST`

The report contract follows the shared source-script metric format so the same
tables can be reused as the lower-bound row in UDA, SDA, and TTA comparisons.
