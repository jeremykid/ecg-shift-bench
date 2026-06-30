# ResNet1D Internal Dataset Baseline Results

This folder contains the saved evaluation tables and figures for the internal supervised baseline.

## Overall summary

| dataset | dataset_name | best_epoch | validation_macro_auprc | test_macro_auroc | test_micro_auroc | test_macro_auprc | test_micro_auprc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ptbxl | ptbxl | 5 | 0.680 | 0.964 | 0.968 | 0.668 | 0.724 |
| code15 | code15 | 5 | 0.645 | 0.985 | 0.987 | 0.634 | 0.668 |
| chapman | chapman | 4 | 0.751 | 0.944 | 0.982 | 0.768 | 0.909 |
| sph | sph | 5 | 0.813 | 0.986 | 0.987 | 0.791 | 0.817 |

## Per-label metrics

| dataset | split | label | auroc | auprc | support |
| --- | --- | --- | --- | --- | --- |
| ptbxl | validation | AF | 0.974 | 0.848 | 151 |
| ptbxl | validation | RBBB | 0.983 | 0.872 | 166 |
| ptbxl | validation | LBBB | 0.981 | 0.807 | 61 |
| ptbxl | validation | 1dAVB | 0.939 | 0.499 | 80 |
| ptbxl | validation | SB | 0.947 | 0.484 | 64 |
| ptbxl | validation | ST | 0.967 | 0.567 | 83 |
| ptbxl | test | AF | 0.961 | 0.819 | 152 |
| ptbxl | test | RBBB | 0.988 | 0.879 | 166 |
| ptbxl | test | LBBB | 0.982 | 0.881 | 62 |
| ptbxl | test | 1dAVB | 0.937 | 0.437 | 79 |
| ptbxl | test | SB | 0.938 | 0.356 | 64 |
| ptbxl | test | ST | 0.978 | 0.637 | 82 |
| code15 | validation | AF | 0.985 | 0.676 | 709 |
| code15 | validation | RBBB | 0.992 | 0.789 | 962 |
| code15 | validation | LBBB | 0.995 | 0.774 | 606 |
| code15 | validation | 1dAVB | 0.978 | 0.531 | 571 |
| code15 | validation | SB | 0.966 | 0.405 | 558 |
| code15 | validation | ST | 0.989 | 0.695 | 769 |
| code15 | test | AF | 0.985 | 0.668 | 1352 |
| code15 | test | RBBB | 0.993 | 0.792 | 1970 |
| code15 | test | LBBB | 0.995 | 0.807 | 1176 |
| code15 | test | 1dAVB | 0.980 | 0.473 | 1112 |
| code15 | test | SB | 0.971 | 0.412 | 1100 |
| code15 | test | ST | 0.987 | 0.652 | 1500 |
| chapman | validation | AF | 0.980 | 0.926 | 192 |
| chapman | validation | RBBB | 0.992 | 0.899 | 50 |
| chapman | validation | LBBB | 0.997 | 0.719 | 12 |
| chapman | validation | 1dAVB | 0.689 | 0.089 | 40 |
| chapman | validation | SB | 0.981 | 0.957 | 381 |
| chapman | validation | ST | 0.970 | 0.914 | 170 |
| chapman | test | AF | 0.982 | 0.908 | 339 |
| chapman | test | RBBB | 0.966 | 0.830 | 84 |
| chapman | test | LBBB | 0.999 | 0.884 | 14 |
| chapman | test | 1dAVB | 0.757 | 0.114 | 51 |
| chapman | test | SB | 0.984 | 0.967 | 777 |
| chapman | test | ST | 0.976 | 0.903 | 336 |
| sph | validation | AF | 0.983 | 0.880 | 76 |
| sph | validation | RBBB | 1.000 | 0.982 | 70 |
| sph | validation | LBBB | 1.000 | 0.967 | 10 |
| sph | validation | 1dAVB | 0.991 | 0.426 | 20 |
| sph | validation | SB | 0.984 | 0.873 | 264 |
| sph | validation | ST | 0.987 | 0.750 | 63 |
| sph | test | AF | 0.997 | 0.835 | 137 |
| sph | test | RBBB | 0.991 | 0.966 | 135 |
| sph | test | LBBB | 0.999 | 0.945 | 22 |
| sph | test | 1dAVB | 0.967 | 0.297 | 50 |
| sph | test | SB | 0.983 | 0.866 | 544 |
| sph | test | ST | 0.982 | 0.838 | 156 |

## Figures

![Overall metrics](resnet1d_internal_dataset_baseline_overall_metrics.png)

![Per-label metrics](resnet1d_internal_dataset_baseline_per_label_metrics.png)

## Files

- `results_summary.csv`
- `per_class_summary.csv`
- `resnet1d_internal_dataset_baseline_overall_metrics.png`
- `resnet1d_internal_dataset_baseline_overall_metrics.svg`
- `resnet1d_internal_dataset_baseline_per_label_metrics.png`
- `resnet1d_internal_dataset_baseline_per_label_metrics.svg`
