# Dataset Statistics Visualization Summary

This report was generated only from the exported statistics artifacts; no raw ECG waveforms were read.

## Dataset scale overview

- **PTB-XL:** 21,799 records; 18,869 patients; 0 excluded.
- **CODE-15:** 345,779 records; 233,770 patients; 0 excluded.
- **Chapman:** 10,646 records; N/A patients; 0 excluded.
- **SPH:** 25,770 records; 24,666 patients; 0 excluded.

## Missing and excluded records

- **PTB-XL:** 21,799 usable of 21,799 total; 0 excluded; no missing label annotations reported.
- **CODE-15:** 345,779 usable of 345,779 total; 0 excluded; no missing label annotations reported.
- **Chapman:** 10,646 usable of 10,646 total; 0 excluded; no missing label annotations reported.
- **SPH:** 25,770 usable of 25,770 total; 0 excluded; no missing label annotations reported.

## Split composition and policy

- **PTB-XL:** source `official`, level `patient` (train 79.9%, validation 10.0%, test 10.1%); patient-level assignment limits patient leakage.
- **CODE-15:** source `generated`, level `patient` (train 70.0%, validation 10.0%, test 20.0%); patient-level assignment limits patient leakage.
- **Chapman:** source `generated`, level `record` (train 70.0%, validation 10.0%, test 20.0%); record-level assignment cannot guarantee patient independence.
- **SPH:** source `generated`, level `patient` (train 70.0%, validation 10.0%, test 20.0%); patient-level assignment limits patient leakage.

## Waveform contract

- **PTB-XL:** 500 Hz, 5000 samples, 12 leads, units mV, no conversion.
- **CODE-15:** 500 Hz, 5000 samples, 12 leads, units mV, no conversion.
- **Chapman:** 500 Hz, 5000 samples, 12 leads, units uV → mV.
- **SPH:** 500 Hz, 5000 samples, 12 leads, units mV, no conversion.

## Label-prior shift observations

- **AF:** CODE-15 2.03% to Chapman 16.72% (14.69 percentage-point range).
- **RBBB:** SPH 2.76% to PTB-XL 7.61% (4.85 percentage-point range).
- **LBBB:** SPH 0.33% to PTB-XL 2.81% (2.49 percentage-point range).
- **1dAVB:** SPH 0.92% to PTB-XL 3.64% (2.71 percentage-point range).
- **SB:** CODE-15 1.62% to Chapman 36.53% (34.91 percentage-point range).
- **ST:** CODE-15 2.19% to Chapman 14.73% (12.54 percentage-point range).

## Rare-label reliability warnings

- **PTB-XL:** the smallest positive count is 613 for LBBB; estimates for the least represented labels should receive wider uncertainty bounds and cautious interpretation.
- **CODE-15:** the smallest positive count is 5,605 for SB; estimates for the least represented labels should receive wider uncertainty bounds and cautious interpretation.
- **Chapman:** the smallest positive count is 94 for LBBB; estimates for the least represented labels should receive wider uncertainty bounds and cautious interpretation.
- **SPH:** the smallest positive count is 84 for LBBB; estimates for the least represented labels should receive wider uncertainty bounds and cautious interpretation.

## Split-balance warnings

- **Chapman test:** gap 0.0105 exceeds 0.01.
- **Chapman validation:** gap 0.0139 exceeds 0.01.

## Implications for cross-dataset benchmarking

- Dataset size and label prevalence differ substantially, so pooled metrics can be dominated by larger domains and should be accompanied by per-domain results.
- Positive-rate differences indicate label-prior shift; calibration and decision thresholds learned in one domain may not transfer directly.
- Low positive counts increase metric variance, especially for split-level and cross-domain estimates; report confidence intervals where possible.
- Chapman uses a record-level split because patient identifiers are unavailable, so its within-dataset estimates do not provide the same leakage protection as patient-level splits.

## Generation warnings

- None.
