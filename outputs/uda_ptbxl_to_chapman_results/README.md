# PTBXL to CHAPMAN UDA results

This directory publishes compact, reviewable evidence for the UDA benchmark while
keeping checkpoints, predictions, full logs, and split manifests as local runtime
artifacts.

## Source-only baseline

- Method: `source_only`
- Source: `PTBXL`
- Target: `CHAPMAN`
- Shared label space: `AF|RBBB|LBBB|1dAVB|SB|ST`
- Seed: `42`
- Model: `resnet1d`
- Model selection: source-validation macro AUROC
- Target labels available during training: `false`
- Target-test macro AUROC: `0.9718231924973129`
- Target-test macro AUPRC: `0.7147386382917856`
- Target-test macro F1: `0.5939583058404645`

The published files are:

- `source_only_results_summary.csv`: aggregate train, source-validation,
  target-validation, and target-test metrics.
- `source_only_per_class_summary.csv`: per-label metrics for the same splits.

## Split identity

The source-only and CORAL runs used byte-identical source and target split
manifests:

- Source manifest SHA-256:
  `bb32c92dad3f98f383cbf843f8a4d6eee26ed56bf39742f3edb9f6fbb9157ff0`
- Target manifest SHA-256:
  `18e535d4fb2095e2fb7ed875c5a8cdb8f56fd458e491c2b04978162786b161fc`

Target labels were not exposed to the training step and were used only for final
metric reporting. Full runtime artifacts remain under the ignored local
`outputs/uda/` tree.

## CORAL baseline

- Method: `coral`
- Adaptation weight: `lambda=0.1`
- Batch size: `64`
- Epochs: `30`
- Seed: `42`
- Alignment feature: pooled encoder representation from `model.forward_features`
- Best epoch: `22`
- Model selection: source-validation macro AUROC
- Target labels available during training: `false`
- Target-test macro AUROC: `0.9725158667341406`
- Target-test macro AUPRC: `0.7060469105222515`
- Target-test macro F1: `0.5769366243697699`

Compared with source-only, CORAL increased target-test macro AUROC by
`0.0006926742368277`, while macro AUPRC decreased by `0.0086917277695341`,
macro F1 decreased by `0.0170216814706946`, and macro accuracy decreased by
`0.0018005323312978`. The result is reported as observed rather than presented
as a uniform improvement.

The run recorded Git commit `fc5c8f5428c54e8673153fdd757d49f4fd8fa247`
with `git_dirty: true`. CUDA also emitted a warning that deterministic mode does
not make the relevant CuBLAS operation deterministic unless
`CUBLAS_WORKSPACE_CONFIG` is set. These provenance limitations are retained in
the report rather than hidden.

The CORAL aggregate and per-label metrics are published as
`coral_results_summary.csv` and `coral_per_class_summary.csv`. The direct
four-metric comparison is in `source_only_vs_coral.csv`.
