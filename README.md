# ECGShiftBench

**ECGShiftBench: A Cross-Dataset Benchmark for Domain Adaptation under Covariate Shift in
12-lead ECG Classification**

ECGShiftBench is a research-oriented Python benchmark for measuring how multi-label ECG
classifiers transfer across datasets, hospitals, countries, and acquisition sources. It
provides a shared six-label task, patient-safe split utilities, compact neural baselines,
and evaluation code for covariate shift, domain generalization (DG), and unsupervised domain
adaptation (UDA).

This repository is an initial scaffold. It does **not** distribute or download ECG data.

## Motivation

Strong random-split performance within one ECG dataset can conceal clinically relevant
distribution shifts. ECGShiftBench makes the domain boundary explicit and evaluates models
on held-out sources under consistent label and preprocessing contracts. The benchmark is
designed to keep label harmonization assumptions visible rather than treating similarly
named diagnoses as automatically equivalent.

## Initial datasets

| Dataset | Domain interpretation | Initial adapter status |
|---|---|---|
| PTB-XL | German clinical source | Validated metadata and WFDB loader |
| Chapman-Shaoxing / Ningbo | Chinese hospital/acquisition source | Metadata skeleton |
| SPH | Chinese hospital/acquisition source | Metadata skeleton |
| CODE-15% | Brazilian telehealth source | Metadata skeleton |

PhysioNet/CinC 2020/2021 and MIMIC-IV-ECG are candidates for later adapters. Dataset identity
is the default domain; future releases may expose reliable within-dataset site/device domains.

## Canonical task

The initial task is multi-label prediction of AF (atrial fibrillation), RBBB (right bundle
branch block), LBBB (left bundle branch block), 1dAVB (first-degree atrioventricular block),
SB (sinus bradycardia), and ST (sinus tachycardia). **ST means sinus tachycardia, not an
ST-segment abnormality.** See [label harmonization](docs/label_harmonization.md) for the
mapping assumptions and known approximations.

## Benchmark settings

- **Source-only:** supervised training on labeled source domains; target labels are evaluation-only.
- **Multi-source:** supervised learning from multiple labeled source domains.
- **Leave-one-domain-out DG:** train on all but one domain and evaluate on the held-out domain.
- **UDA:** source labels and unlabeled target inputs may be used; target labels remain hidden.
- **Target-supervised/few-shot:** future, separately named protocols that explicitly permit target labels.

Primary metrics are macro/micro AUROC and AUPRC, per-label values, and the domain gap
(`in_domain_score - out_domain_score`). Undefined per-label metrics are reported as NaN rather
than silently replaced. Patient identifiers must never cross train, validation, and test sets.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For tests and local quality checks:

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

## Data preparation

Obtain each dataset manually from its official distributor, accept its terms, and place it
under `data/raw/<dataset>/` (or change `root` in the YAML file). Configs and dataset cards
document the expected initial layout. The adapters validate metadata but waveform loading is
deliberately marked TODO until release-specific units, lead order, and file conventions are
verified.

```bash
python scripts/prepare_dataset.py --dataset ptbxl --config configs/datasets/ptbxl.yaml
python scripts/create_splits.py --config configs/experiments/leave_one_domain_out.yaml
python scripts/train.py --config configs/experiments/source_only.yaml
python scripts/evaluate.py --config configs/experiments/source_only.yaml
```

The config-only commands validate and report next inputs without accessing remote data. Once
you have a combined record index, pass it to `create_splits.py --metadata ...`. A synthetic
end-to-end optimizer check is available with `train.py --smoke-test`.

The fixed PTB-XL 1.0.3 source-only baseline validates its immutable snapshot before accessing
waveforms, performs a real-batch forward/backward preflight, and then trains for one epoch:

```bash
python scripts/train.py \
  --config configs/experiments/ptbxl_source_only_resnet1d_wang_e1.yaml \
  --dataset-config configs/datasets/ptbxl.yaml \
  --root /path/to/ptb-xl/1.0.3 \
  --device cuda:0 \
  --output-dir outputs/ptbxl-source-only-resnet1d-wang-500hz-e1-seed42
```

Add `--preflight-only` to stop after the snapshot and real-batch checks.

Dataset roots remain portable in version control. Use `--root` for local storage outside the
repository and optionally load one waveform as an integration check:

```bash
python scripts/prepare_dataset.py --dataset ptbxl --config configs/datasets/ptbxl.yaml \
  --root /path/to/ptb-xl/1.0.3 --check-record 1
python scripts/prepare_dataset.py --dataset code15 --config configs/datasets/code15.yaml \
  --root /path/to/code_15 --check-record 123456
```

## Repository structure

```text
configs/                 Dataset and experiment YAML templates
data/                    Ignored raw/processed data and split manifests
docs/                    Protocol, mappings, dataset cards, reproducibility
scripts/                 Preparation, splitting, training, evaluation CLIs
src/ecg_shift_bench/
  datasets/              Common interface and source adapters
  labels/                Canonical task and harmonization
  preprocessing/         Resampling, sizing, unit alignment, quality checks
  splits/                Patient-safe and domain-held-out splits
  models/                Small ResNet1D, ResNet1DWang, XResNet-style, and InceptionTime models
  methods/               Source-only baseline and DA extension points
  training/              Loss, optimizer, and epoch helpers
  evaluation/            Discrimination, calibration, and domain-gap metrics
  utils/                 Config, paths, logging, and seed handling
tests/                    Data-free unit tests
```

## Reproducibility contract

Every reported run should retain the dataset release/checksum, indexing and exclusion logs,
label mapping version, split manifest, preprocessing parameters, configuration, seed, package
environment, checkpoint, and per-domain/per-label results. Details are in
[docs/reproducibility.md](docs/reproducibility.md).

## Citation

```bibtex
@misc{ecgshiftbench2026,
  title  = {ECGShiftBench: A Cross-Dataset Benchmark for Domain Adaptation under
            Covariate Shift in 12-lead ECG Classification},
  author = {ECGShiftBench Contributors},
  year   = {2026},
  note   = {Work in progress}
}
```

## Data and clinical-use disclaimer

Users are responsible for following every dataset's license, access conditions, privacy
requirements, and citation policy. Do not commit source ECGs or identifiable metadata. This
software is for research benchmarking and is not a medical device or clinical decision tool.
