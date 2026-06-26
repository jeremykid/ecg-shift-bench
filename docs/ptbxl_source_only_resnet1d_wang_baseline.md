# PTB-XL six-label source-only ResNet1D-Wang baseline

## Outcome

The experiment `ptbxl-source-only-resnet1d-wang-500hz-e1-seed42` completed one
full epoch on PTB-XL 1.0.3. The run used all official patient-safe training
records, produced a reloadable checkpoint, and wrote complete validation and
test metrics for the six canonical labels.

| Split | Records | Macro AUROC | Micro AUROC | Macro AUPRC | Micro AUPRC |
|---|---:|---:|---:|---:|---:|
| Validation | 2,183 | 0.826080 | 0.862878 | 0.378406 | 0.450766 |
| Test | 2,198 | 0.836040 | 0.867781 | 0.369420 | 0.458015 |

Training covered 17,418 records in 273 batches. Mean epoch loss was `0.424533`;
all observed batch losses were finite (`0.126099` to `1.059405`).

## Dataset snapshot

- Snapshot ID: `ptbxl-1.0.3-six-label-v1`
- Source: PTB-XL 1.0.3, 500 Hz WFDB records in physical mV units
- Inclusion: all 21,799 metadata records, including records negative for all six targets
- Official folds: 1–8 train, 9 validation, 10 test
- Split counts: 17,418 / 2,183 / 2,198
- Patient overlap between every split pair: zero
- Split manifest SHA-256:
  `0ab5e4ce887fd4813a1fb95a278509cee0fc8e4b50a673371c4f08da215f7c19`
- Canonical target order: `AF, RBBB, LBBB, 1dAVB, SB, ST`

The committed snapshot manifest records the metadata, SCP ontology, `RECORDS`,
and official checksum-index identities without storing a machine-specific data
path. It is located at
`configs/datasets/snapshots/ptbxl-1.0.3-six-label-v1.yaml`.

## Model and training contract

`resnet1d_wang` is a pure-PyTorch port of the legacy benchmark architecture:

- 12-lead input with shape `(12, 5000)`
- 128-channel, kernel-7 stem
- three Wang residual stages using kernel sizes 5 and 3
- adaptive max and average pooling concatenation
- six-logit classification head
- no shared benchmark normalization; this legacy baseline consumes raw mV waveforms
  without filtering, cropping, or augmentation
- unweighted `BCEWithLogitsLoss`
- AdamW, learning rate `1e-3`, weight decay `1e-4`
- batch size 64, eight workers, seed 42, and no scheduler
- FP16 AMP on CUDA

The five-output compatibility model has exactly 440,837 parameters. The
six-output experiment model has 441,094 parameters.

## Reproduction

Install the project dependencies, provision PTB-XL 1.0.3 manually, and run:

```bash
python scripts/train.py \
  --config configs/experiments/ptbxl_source_only_resnet1d_wang_e1.yaml \
  --dataset-config configs/datasets/ptbxl.yaml \
  --root /path/to/ptb-xl/1.0.3 \
  --device cuda:0 \
  --output-dir outputs/ptbxl-source-only-resnet1d-wang-500hz-e1-seed42
```

The workflow validates snapshot hashes and split invariants before loading
waveforms, performs one real-batch forward/backward preflight, trains one full
epoch, selects by validation macro AUPRC, reloads the best checkpoint, and then
evaluates the test fold.

## Artifacts

The ignored runtime directory
`outputs/ptbxl-source-only-resnet1d-wang-500hz-e1-seed42/` contains:

- copied experiment, dataset, and snapshot configurations
- `snapshot_identity.json` and the hashed `split_manifest.csv`
- `history.json`
- `best_checkpoint.pt`
- `validation_metrics.json` and `test_metrics.json`
- `run_status.json`

The final status is `completed`. It records clean source commit
`03e4b7e56279aa6c57540f8da2073da7de6e5395`, the V100 device, FP16 AMP,
timestamps, the command, artifact paths, and recovery guidance.

## Validation

The final verification completed successfully:

```text
Ruff: all checks passed
Pytest: 23 passed
Git diff check: passed
Real-data snapshot integration: passed
Real-data FP16 forward/backward preflight: passed
Checkpoint save/reload: passed
Validation/test metric completeness: passed
```

## Known limitation

CUDA adaptive max-pool backward does not guarantee bitwise determinism. The run
uses fixed seeds and deterministic algorithms in warn-only mode, and records
this limitation in `run_status.json`. No CODE-15 data, legacy checkpoint, PTB-XL
five-superclass targets, 2.5-second crops, or domain-gap metric are part of this
experiment.
