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
