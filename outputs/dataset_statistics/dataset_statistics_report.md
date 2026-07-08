# Dataset Statistics Report

## Overview

| dataset | name | domain | records_total | patients_total | records_excluded | split_source | split_level | train_records | validation_records | test_records |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ptbxl | PTBXL | ptbxl_germany | 21799 | 18869 | 0 | official | patient | 17418 | 2183 | 2198 |
| code15 | CODE15 | code15_brazil | 345779 | 233770 | 0 | generated | patient | 241957 | 34562 | 69260 |
| chapman | CHAPMAN | chapman_ningbo_china | 10646 | n/a | 0 | generated | record | 7452 | 1065 | 2129 |
| sph | SPH | sph_china | 25770 | 24666 | 0 | generated | patient | 18049 | 2575 | 5146 |

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
