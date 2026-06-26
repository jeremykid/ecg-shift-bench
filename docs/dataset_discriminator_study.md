# Dataset discriminator study

This study trains a model to predict which aligned ECG dataset a record came from.

It uses the shared aligned waveform contract:

- lead-first ECG tensors with shape `(12, L)`
- canonical 12-lead order
- unified sampling rate
- fixed signal length
- unified physical unit (`mV`)

The default discriminator backbone is config-driven and currently set to `xresnet1d`, an
XResNet-style 1D backbone.

Supported study modes:

- uncontrolled
- pairwise
- label-balanced
- normal-only
- random-label baseline

Supported evaluation outputs:

- accuracy
- balanced accuracy
- macro F1
- AUROC
- confusion matrix
- per-class support

The study uses aligned waveforms on demand and does not write a second ECG store.

It is a separability diagnostic, not a clinical prediction benchmark.

It can tell us whether the datasets are easy to tell apart after alignment.
It cannot by itself prove clinical correctness or downstream model utility.
