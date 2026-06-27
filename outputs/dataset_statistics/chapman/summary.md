# CHAPMAN Dataset Statistics

## Overview

| Dataset | Domain | Records total | Patients total | Classes | Records usable | Records excluded |
| --- | --- | --- | --- | --- | --- | --- |
| chapman | chapman_ningbo_china | 10646 | n/a | 6 | 10646 | 0 |

## Split Policy

| split_source | split_level | split_algorithm | seed | train_fraction | validation_fraction | test_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| generated | record | record_level_random_no_patient_id | 42 | 0.7 | 0.1 | 0.2 |

## Waveform Contract

| mode | checked_records | target_sampling_rate | target_length | lead_order | source_unit | target_unit |
| --- | --- | --- | --- | --- | --- | --- |
| full | 10646 | 500 | 5000 | I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 | uV | mV |

## Split Sizes

| split | records | patients |
| --- | --- | --- |
| train | 7452 | n/a |
| validation | 1065 | n/a |
| test | 2129 | n/a |

## Split Balance

| split | records | patients | max_absolute_label_gap | skewed_labels |
| --- | --- | --- | --- | --- |
| train | 7452 | n/a | 0.0048 | none |
| validation | 1065 | n/a | 0.0139 | none |
| test | 2129 | n/a | 0.0105 | none |

## Label Distribution

| label | positive_count |
| --- | --- |
| AF | 1780 |
| RBBB | 460 |
| LBBB | 94 |
| 1dAVB | 252 |
| SB | 3889 |
| ST | 1568 |

## Positive rate

| label | positive_rate |
| --- | --- |
| AF | 0.1672 |
| RBBB | 0.0432 |
| LBBB | 0.0088 |
| 1dAVB | 0.0237 |
| SB | 0.3653 |
| ST | 0.1473 |

## Split Label Distribution

| split | label | positive_count |
| --- | --- | --- |
| train | AF | 1249 |
| train | RBBB | 326 |
| train | LBBB | 68 |
| train | 1dAVB | 161 |
| train | SB | 2731 |
| train | ST | 1062 |
| validation | AF | 192 |
| validation | RBBB | 50 |
| validation | LBBB | 12 |
| validation | 1dAVB | 40 |
| validation | SB | 381 |
| validation | ST | 170 |
| test | AF | 339 |
| test | RBBB | 84 |
| test | LBBB | 14 |
| test | 1dAVB | 51 |
| test | SB | 777 |
| test | ST | 336 |

## Split Positive rate

| split | label | positive_rate |
| --- | --- | --- |
| train | AF | 0.1676 |
| train | RBBB | 0.0437 |
| train | LBBB | 0.0091 |
| train | 1dAVB | 0.0216 |
| train | SB | 0.3665 |
| train | ST | 0.1425 |
| validation | AF | 0.1803 |
| validation | RBBB | 0.0469 |
| validation | LBBB | 0.0113 |
| validation | 1dAVB | 0.0376 |
| validation | SB | 0.3577 |
| validation | ST | 0.1596 |
| test | AF | 0.1592 |
| test | RBBB | 0.0395 |
| test | LBBB | 0.0066 |
| test | 1dAVB | 0.024 |
| test | SB | 0.365 |
| test | ST | 0.1578 |

## Missing or Invalid Records

- Records excluded: 0
- Records usable: 10646
- Split balance warnings: none
- Exclusions: none
