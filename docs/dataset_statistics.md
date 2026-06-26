# Dataset Statistics Export

This repository includes a reproducible export for the dataset statistics work described in
issue 10.

## Command

```bash
python scripts/export_dataset_statistics.py \
  --ptbxl-root /path/to/ptb-xl \
  --code15-root /path/to/code15 \
  --chapman-root /path/to/chapman \
  --sph-root /path/to/sph \
  --output-dir outputs/dataset_statistics
```

If a root flag is omitted, the command falls back to the path configured in the corresponding
dataset YAML file.

By default, the exporter performs a full waveform audit. To skip waveform loading and only
audit metadata/splits, pass `--metadata-only`.

If the current Python interpreter does not have `wfdb` installed, the exporter automatically
re-executes itself with the repository virtual environment at `.venv/bin/python` when that
interpreter is available.

## Outputs

Each dataset is written to `outputs/dataset_statistics/<dataset>/` with:

- `audit.json`
- `split_manifest.csv`
- `train.csv`
- `validation.csv`
- `test.csv`
- `exclusions.csv`
- `reproducibility.json`
- `label_distribution.csv`
- `positive_rate.csv`
- `split_label_distribution.csv`
- `split_positive_rate.csv`
- `split_summary.csv`
- `summary.md`

The batch report is written to:

- `outputs/dataset_statistics/dataset_statistics_report.md`

## Split Policy

- PTB-XL uses the official `strat_fold` split.
- CODE-15 and SPH use label-aware patient-level `70/10/20` splits with seed `42`, falling back
  to deterministic random patient splits only if stratification is not feasible.
- Chapman uses the record-level `70/10/20` fallback because the current adapter does not expose
  a reliable patient identifier.

The generated summaries report the split strategy, sampling rate, target length, label counts,
positive rates, split-level label balance, and any excluded or invalid records.
