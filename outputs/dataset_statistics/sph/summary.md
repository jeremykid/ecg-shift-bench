# SPH Dataset Statistics

## Overview

| Dataset | Domain | Records total | Patients total | Classes | Records usable | Records excluded |
| --- | --- | --- | --- | --- | --- | --- |
| sph | sph_china | 25770 | 24666 | 6 | 25770 | 0 |

## Split Policy

| split_source | split_level | split_algorithm | seed | train_fraction | validation_fraction | test_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| generated | patient | patient_level_random | 42 | 0.7 | 0.1 | 0.2 |

## Waveform Contract

| mode | checked_records | target_sampling_rate | target_length | lead_order | source_unit | target_unit |
| --- | --- | --- | --- | --- | --- | --- |
| full | 25770 | 500 | 5000 | I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 | mV | mV |

## Split Sizes

| split | records | patients |
| --- | --- | --- |
| train | 18049 | 17266 |
| validation | 2575 | 2467 |
| test | 5146 | 4933 |

## Split Balance

| split | records | patients | max_absolute_label_gap | skewed_labels |
| --- | --- | --- | --- | --- |
| train | 18049 | 17266 | 0.0006 | none |
| validation | 2575 | 2467 | 0.0037 | none |
| test | 5146 | 4933 | 0.0022 | none |

## Label Distribution

| label | positive_count |
| --- | --- |
| AF | 675 |
| RBBB | 710 |
| LBBB | 84 |
| 1dAVB | 238 |
| SB | 2711 |
| ST | 725 |

## Positive rate

| label | positive_rate |
| --- | --- |
| AF | 0.0262 |
| RBBB | 0.0276 |
| LBBB | 0.0033 |
| 1dAVB | 0.0092 |
| SB | 0.1052 |
| ST | 0.0281 |

## Split Label Distribution

| split | label | positive_count |
| --- | --- | --- |
| train | AF | 462 |
| train | RBBB | 505 |
| train | LBBB | 52 |
| train | 1dAVB | 168 |
| train | SB | 1903 |
| train | ST | 506 |
| validation | AF | 76 |
| validation | RBBB | 70 |
| validation | LBBB | 10 |
| validation | 1dAVB | 20 |
| validation | SB | 264 |
| validation | ST | 63 |
| test | AF | 137 |
| test | RBBB | 135 |
| test | LBBB | 22 |
| test | 1dAVB | 50 |
| test | SB | 544 |
| test | ST | 156 |

## Split Positive rate

| split | label | positive_rate |
| --- | --- | --- |
| train | AF | 0.0256 |
| train | RBBB | 0.028 |
| train | LBBB | 0.0029 |
| train | 1dAVB | 0.0093 |
| train | SB | 0.1054 |
| train | ST | 0.028 |
| validation | AF | 0.0295 |
| validation | RBBB | 0.0272 |
| validation | LBBB | 0.0039 |
| validation | 1dAVB | 0.0078 |
| validation | SB | 0.1025 |
| validation | ST | 0.0245 |
| test | AF | 0.0266 |
| test | RBBB | 0.0262 |
| test | LBBB | 0.0043 |
| test | 1dAVB | 0.0097 |
| test | SB | 0.1057 |
| test | ST | 0.0303 |

## Missing or Invalid Records

- Records excluded: 0
- Records usable: 25770
- Split balance warnings: none
- Exclusions: none
