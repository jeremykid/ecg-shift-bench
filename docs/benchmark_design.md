# Benchmark design

## Scope

The benchmark studies cross-domain generalization for 12-lead, multi-label ECG classification.
A domain is an acquisition population/source: initially one public dataset, and later a
documented hospital, country, or device subgroup where provenance supports that distinction.

## Protocols

1. **Source-only:** train and tune on source-domain records only. Target labels are opened once
   for final evaluation.
2. **Multi-source:** combine or balance multiple labeled sources, retaining domain identity in
   the manifest and reporting each source separately.
3. **Leave-one-domain-out:** hold one complete domain out; fit and select models using remaining
   domains; rotate the held-out domain when all adapters are validated.
4. **UDA:** target signals may be observed without labels during adaptation, but target labels
   stay hidden from loss computation, early stopping, model selection, and hyperparameter
   selection. UDA runs select checkpoints from source validation only.
   For per-label F1 reporting, thresholds are derived from source-train scores and then fixed;
   target labels are never used for threshold tuning.

## CORAL UDA baseline

The CORAL baseline uses the issue 25 UDA workflow and adds a Deep CORAL-style
feature alignment objective. Source labels provide the multi-label BCE
classification loss, while unlabeled target inputs contribute only to the
CORAL covariance alignment loss.

CORAL aligns the pooled encoder representation returned by
`model.forward_features`. For the default `resnet1d` configuration this is the
feature vector immediately before the classifier head. The training objective is:

`source classification loss + lambda * coral_loss`

where `coral_loss` is the squared Frobenius distance between source and target
feature covariance matrices divided by `4 * feature_dim * feature_dim`. The
initial PTB-XL to Chapman configuration records `method_params.lambda: 0.1`.
This value must not be tuned using target labels; standard UDA checkpoint
selection remains source-validation-only.

Target-supervised and few-shot experiments must use distinct protocol names and predeclare the
number and selection procedure for labeled target examples.

## Unit of splitting

The patient is the grouping unit. All ECGs from one patient belong to one subset. Dataset-level
tests additionally hold out the full target domain. Duplicate/near-duplicate detection across
public sources is a future quality gate.

## Evaluation

Report macro/micro AUROC and AUPRC, per-label results, confidence intervals, prevalence, sample
counts, and the domain gap. Report calibration where probabilities are interpreted. A metric
with no positive or no negative examples is undefined (NaN), and the support must be shown.

## Threats to validity

Cross-dataset changes include label policy, population, devices, filtering, lead quality, and
record duration. Therefore a measured gap is not automatically causal evidence of covariate
shift alone. Label mapping decisions and exclusions are part of the benchmark definition.
