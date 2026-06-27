# CODE15 Dataset Statistics

## Overview

| Dataset | Domain | Records total | Patients total | Classes | Records usable | Records excluded |
| --- | --- | --- | --- | --- | --- | --- |
| code15 | code15_brazil | 345779 | 233770 | 6 | 345779 | 0 |

## Split Policy

| split_source | split_level | split_algorithm | seed | train_fraction | validation_fraction | test_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| generated | patient | patient_level_random | 42 | 0.7 | 0.1 | 0.2 |

## Waveform Contract

| mode | checked_records | target_sampling_rate | target_length | lead_order | source_unit | target_unit |
| --- | --- | --- | --- | --- | --- | --- |
| full | 345779 | 500 | 5000 | I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 | mV | mV |

## Split Sizes

| split | records | patients |
| --- | --- | --- |
| train | 241957 | 163639 |
| validation | 34562 | 23377 |
| test | 69260 | 46754 |

## Split Balance

| split | records | patients | max_absolute_label_gap | skewed_labels |
| --- | --- | --- | --- | --- |
| train | 241957 | 163639 | 0.0002 | none |
| validation | 34562 | 23377 | 0.0003 | none |
| test | 69260 | 46754 | 0.0008 | none |

## Label Distribution

| label | positive_count |
| --- | --- |
| AF | 7033 |
| RBBB | 9672 |
| LBBB | 6026 |
| 1dAVB | 5716 |
| SB | 5605 |
| ST | 7584 |

## Positive rate

| label | positive_rate |
| --- | --- |
| AF | 0.0203 |
| RBBB | 0.028 |
| LBBB | 0.0174 |
| 1dAVB | 0.0165 |
| SB | 0.0162 |
| ST | 0.0219 |

## Split Label Distribution

| split | label | positive_count |
| --- | --- | --- |
| train | AF | 4972 |
| train | RBBB | 6740 |
| train | LBBB | 4244 |
| train | 1dAVB | 4033 |
| train | SB | 3947 |
| train | ST | 5315 |
| validation | AF | 709 |
| validation | RBBB | 962 |
| validation | LBBB | 606 |
| validation | 1dAVB | 571 |
| validation | SB | 558 |
| validation | ST | 769 |
| test | AF | 1352 |
| test | RBBB | 1970 |
| test | LBBB | 1176 |
| test | 1dAVB | 1112 |
| test | SB | 1100 |
| test | ST | 1500 |

## Split Positive rate

| split | label | positive_rate |
| --- | --- | --- |
| train | AF | 0.0205 |
| train | RBBB | 0.0279 |
| train | LBBB | 0.0175 |
| train | 1dAVB | 0.0167 |
| train | SB | 0.0163 |
| train | ST | 0.022 |
| validation | AF | 0.0205 |
| validation | RBBB | 0.0278 |
| validation | LBBB | 0.0175 |
| validation | 1dAVB | 0.0165 |
| validation | SB | 0.0161 |
| validation | ST | 0.0222 |
| test | AF | 0.0195 |
| test | RBBB | 0.0284 |
| test | LBBB | 0.017 |
| test | 1dAVB | 0.0161 |
| test | SB | 0.0159 |
| test | ST | 0.0217 |

## Missing or Invalid Records

- Records excluded: 0
- Records usable: 345779
- Split balance warnings: none
- Exclusions: none
