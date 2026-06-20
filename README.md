# ECGShiftBench

**A Cross-Dataset Benchmark for Studying Covariate Shift, Domain Generalisation, and Unsupervised Domain Adaptation in 12-Lead ECG Classification.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

ECGShiftBench is a Python benchmark toolkit that provides:

- **Standardised dataset loaders** for four large 12-lead ECG collections.
- **Three benchmark protocols** covering different levels of distributional shift:
  1. *Covariate Shift* — train on one dataset, test on others.
  2. *Domain Generalisation* — Leave-One-Dataset-Out (LODO) zero-shot evaluation.
  3. *Unsupervised Domain Adaptation (UDA)* — leverage unlabelled target data.
- **A baseline 1D-ResNet model** for ECG classification.
- **Evaluation metrics**: macro-AUROC, macro-F1, and sensitivity at 90 % specificity.
- **ECG preprocessing utilities**: bandpass filtering, baseline-wander removal, normalisation,
  resampling, and padding/truncation.

---

## Supported Datasets

| Name | Country | Recordings | Sampling Rate | Labels |
|------|---------|-----------|---------------|--------|
| [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) | Germany | 21,799 | 100 / 500 Hz | 5 superclasses (NORM, MI, STTC, CD, HYP) |
| [Chapman-Shaoxing](https://physionet.org/content/ecg-arrhythmia/1.0.0/) | China | 10,646 | 500 Hz | 4 rhythm classes (AFIB, GSVT, SB, SR) |
| [CPSC 2018](http://2018.icbeb.org/Challenge.html) | China | 6,877 | 500 Hz | 9 classes (Normal, AF, I-AVB, LBBB, RBBB, …) |
| [G12EC](https://physionet.org/content/georgia-12-lead-ecg-challenge/1.0.0/) | USA | 10,344 | 500 Hz | 23 SNOMED-CT classes |

> **Note:** You must download these datasets separately.  See [Dataset Setup](#dataset-setup) below.

---

## Installation

```bash
git clone https://github.com/jeremykid/ecg-shift-bench.git
cd ecg-shift-bench
pip install -e ".[dev]"
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.13
- NumPy, SciPy, Pandas, scikit-learn
- wfdb ≥ 4.0 (for reading WFDB-format ECG files)
- h5py, tqdm, PyYAML

---

## Dataset Setup

### PTB-XL

1. Download from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/):
   ```bash
   wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
   ```
2. The directory should contain `ptbxl_database.csv`, `scp_statements.csv`,
   `records100/`, and `records500/`.

### Chapman-Shaoxing

1. Download from [PhysioNet](https://physionet.org/content/ecg-arrhythmia/1.0.0/).
2. The directory should contain `Diagnostics.csv` and `ECGData/`.

### CPSC 2018

1. Download from the [CPSC 2018 challenge website](http://2018.icbeb.org/Challenge.html).
2. The directory should contain `REFERENCE.csv` and `data/*.mat`.

### G12EC (Georgia 12-lead)

1. Download from [PhysioNet](https://physionet.org/content/georgia-12-lead-ecg-challenge/1.0.0/).
2. The directory should contain the raw `*.hea` / `*.mat` record files.

---

## Quick Start

### 1. Load a dataset

```python
from ecgshiftbench.datasets import PTBXLDataset
from ecgshiftbench.utils import bandpass_filter, normalize_signal, pad_or_truncate
import torch

def preprocess(signal):
    signal = bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=500.0)
    signal = normalize_signal(signal, method="zscore")
    signal = pad_or_truncate(signal, target_length=5000)
    return torch.from_numpy(signal)

dataset = PTBXLDataset(
    root="/data/ptbxl",
    split="train",
    transform=preprocess,
    labels=["NORM", "MI", "STTC", "CD", "HYP"],
)
print(dataset)  # PTBXLDataset(split='train', n_samples=17441, labels=[…])
signal, label = dataset[0]
print(signal.shape, label.shape)  # (12, 5000)  (5,)
```

### 2. Train a baseline model

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ecgshiftbench.models import resnet1d_18

model = resnet1d_18(num_leads=12, num_classes=5)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

for signals, labels in loader:
    logits = model(signals)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 3. Evaluate covariate shift

```python
from ecgshiftbench.benchmarks import CovariateShiftBenchmark
from ecgshiftbench.datasets import ChapmanDataset, CPSC2018Dataset

benchmark = CovariateShiftBenchmark(
    source=ptbxl_train,
    targets=[
        ChapmanDataset(root="/data/chapman", split="test", transform=preprocess),
        CPSC2018Dataset(root="/data/cpsc2018", split="test", transform=preprocess),
    ],
)
results = benchmark.evaluate(model)
for r in results:
    print(r)
# BenchmarkResult(protocol='covariate_shift', 'PTB-XL' → 'Chapman-Shaoxing', macro_auroc=0.7123, …)
```

### 4. Domain generalisation (LODO)

```python
from ecgshiftbench.benchmarks import DomainGeneralizationBenchmark

benchmark = DomainGeneralizationBenchmark(
    datasets={
        "ptbxl":    ptbxl_test,
        "chapman":  chapman_test,
        "cpsc2018": cpsc_test,
        "g12ec":    g12ec_test,
    }
)
results = benchmark.evaluate_lodo(model_for=lambda name: load_model_trained_without(name))
```

### 5. Unsupervised domain adaptation

```python
from ecgshiftbench.benchmarks import UDABenchmark

benchmark = UDABenchmark(
    source=ptbxl_train,
    target_unlabelled=chapman_train,  # used by adaptation algorithm
    target_test=chapman_test,
)
results = benchmark.evaluate(adapted_model)
```

---

## Benchmark Protocols

### Covariate Shift

Train on a single source dataset; evaluate on one or more target datasets without
any adaptation.  Measures raw cross-dataset performance gap.

### Domain Generalisation

Leave-One-Dataset-Out (LODO): train on *N − 1* datasets, test on the held-out one.
Assesses zero-shot generalisation across different recording equipment and populations.

### Unsupervised Domain Adaptation

The model may use *unlabelled* target-domain samples during training (e.g. via feature
alignment, adversarial training, or pseudo-labelling).  Evaluation uses the labelled
target test split, identical to the covariate-shift setting.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `macro_auroc` | Macro-averaged AUROC across all active labels |
| `macro_f1` | Macro-averaged F1 score (threshold = 0.5 by default) |
| `sensitivity_at_90sp` | Mean sensitivity at 90 % specificity |

---

## Preprocessing Utilities

```python
from ecgshiftbench.utils import (
    bandpass_filter,        # Butterworth band-pass (default 0.5–40 Hz)
    remove_baseline_wander, # High-pass filter to remove slow drift
    normalize_signal,       # Z-score or min-max normalisation
    resample_signal,        # Resample to a target sampling rate
    pad_or_truncate,        # Pad with zeros or truncate to a fixed length
)
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Running Examples

```bash
# Covariate shift
python examples/run_covariate_shift.py \
    --ptbxl /data/ptbxl --chapman /data/chapman --cpsc2018 /data/cpsc2018

# Domain generalisation (LODO)
python examples/run_domain_generalization.py \
    --ptbxl /data/ptbxl --chapman /data/chapman \
    --cpsc2018 /data/cpsc2018 --g12ec /data/g12ec

# Unsupervised domain adaptation
python examples/run_uda.py --ptbxl /data/ptbxl --chapman /data/chapman
```

---

## Project Structure

```
ecgshiftbench/
├── datasets/          # Dataset loaders
│   ├── base.py        # Abstract ECGDataset and DatasetInfo
│   ├── ptbxl.py       # PTB-XL
│   ├── chapman.py     # Chapman-Shaoxing
│   ├── cpsc2018.py    # CPSC 2018
│   └── g12ec.py       # Georgia 12-lead ECG
├── benchmarks/        # Benchmark protocols
│   ├── base.py        # BenchmarkProtocol and BenchmarkResult
│   ├── covariate_shift.py
│   ├── domain_generalization.py
│   └── uda.py
├── models/            # Baseline models
│   └── resnet1d.py    # 1D-ResNet (ResNet-18 / ResNet-34)
├── metrics/           # Evaluation metrics
│   └── classification.py
└── utils/             # Signal processing utilities
    └── preprocessing.py
tests/                 # Unit tests
examples/              # End-to-end example scripts
```

---

## Citation

If you use ECGShiftBench in your research, please cite:

```bibtex
@software{ecgshiftbench2024,
  title        = {{ECGShiftBench}: A Cross-Dataset Benchmark for ECG Domain Adaptation},
  author       = {ECGShiftBench Contributors},
  year         = {2024},
  url          = {https://github.com/jeremykid/ecg-shift-bench},
}
```

---

## License

This project is released under the [MIT License](LICENSE).
