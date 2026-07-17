# Source-Only Cross-Domain Results

These compact tables publish the completed strict source-only lower-bound
experiment for ECGShiftBench.

## Protocol

- Training data: labeled source records only
- Adaptation: none
- Target inputs during training: unavailable
- Target labels during training: unavailable
- Model updates during target testing: none
- Checkpoint selection: source-validation macro AUPRC
- Per-label threshold selection: source-train labels and predictions only
- Model: ResNet1D
- Canonical labels: `AF, RBBB, LBBB, 1dAVB, SB, ST`
- Random seed: `42`
- Preprocessing version: `shared_alignment_v1`

## Coverage and provenance

- Historical result commit: `0a4ac46fdf82352fdd616e17e16e86bb2bbd64cb`
- Datasets: PTBXL, CHAPMAN, SPH, CODE15, CPSC
- Directed source-target pairs: 20
- Aggregate rows: 20
- Target-test per-label rows: 120

`results_summary.csv` contains one aggregate row per directed pair.
`per_class_summary.csv` contains the six target-test label rows for each pair.
Machine-specific `source_root` and `target_root` columns were deliberately
removed; all other retained values are copied exactly from the completed run.

Checkpoints, predictions, split manifests, full logs, and per-run JSON files
remain ignored runtime artifacts. The compact tables are sufficient for the
leaderboard reference row and per-label comparison required by Issue #20.
