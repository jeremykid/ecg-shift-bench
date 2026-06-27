# PTBXL Dataset Statistics

## Overview

| Dataset | Domain | Records total | Patients total | Classes | Records usable | Records excluded |
| --- | --- | --- | --- | --- | --- | --- |
| ptbxl | ptbxl_germany | 21799 | 18869 | 6 | 21799 | 0 |

## Split Policy

| split_source | split_level | split_algorithm | seed | train_fraction | validation_fraction | test_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| official | patient | official_strat_fold | n/a | n/a | n/a | n/a |

## Waveform Contract

| mode | checked_records | target_sampling_rate | target_length | lead_order | source_unit | target_unit |
| --- | --- | --- | --- | --- | --- | --- |
| full | 21799 | 500 | 5000 | I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 | mV | mV |

## Split Sizes

| split | records | patients |
| --- | --- | --- |
| train | 17418 | 15023 |
| validation | 2183 | 1942 |
| test | 2198 | 1904 |

## Split Balance

| split | records | patients | max_absolute_label_gap | skewed_labels |
| --- | --- | --- | --- | --- |
| train | 17418 | 15023 | 0.0001 | none |
| validation | 2183 | 1942 | 0.0003 | none |
| test | 2198 | 1904 | 0.0006 | none |

## Label Distribution

| label | positive_count |
| --- | --- |
| AF | 1514 |
| RBBB | 1658 |
| LBBB | 613 |
| 1dAVB | 793 |
| SB | 637 |
| ST | 826 |

## Positive rate

| label | positive_rate |
| --- | --- |
| AF | 0.0695 |
| RBBB | 0.0761 |
| LBBB | 0.0281 |
| 1dAVB | 0.0364 |
| SB | 0.0292 |
| ST | 0.0379 |

## Split Label Distribution

| split | label | positive_count |
| --- | --- | --- |
| train | AF | 1211 |
| train | RBBB | 1326 |
| train | LBBB | 490 |
| train | 1dAVB | 634 |
| train | SB | 509 |
| train | ST | 661 |
| validation | AF | 151 |
| validation | RBBB | 166 |
| validation | LBBB | 61 |
| validation | 1dAVB | 80 |
| validation | SB | 64 |
| validation | ST | 83 |
| test | AF | 152 |
| test | RBBB | 166 |
| test | LBBB | 62 |
| test | 1dAVB | 79 |
| test | SB | 64 |
| test | ST | 82 |

## Split Positive rate

| split | label | positive_rate |
| --- | --- | --- |
| train | AF | 0.0695 |
| train | RBBB | 0.0761 |
| train | LBBB | 0.0281 |
| train | 1dAVB | 0.0364 |
| train | SB | 0.0292 |
| train | ST | 0.0379 |
| validation | AF | 0.0692 |
| validation | RBBB | 0.076 |
| validation | LBBB | 0.0279 |
| validation | 1dAVB | 0.0366 |
| validation | SB | 0.0293 |
| validation | ST | 0.038 |
| test | AF | 0.0692 |
| test | RBBB | 0.0755 |
| test | LBBB | 0.0282 |
| test | 1dAVB | 0.0359 |
| test | SB | 0.0291 |
| test | ST | 0.0373 |

## Missing or Invalid Records

- Records excluded: 0
- Records usable: 21799
- Split balance warnings: none
- Exclusions: none
