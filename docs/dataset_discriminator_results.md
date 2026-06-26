# Dataset Discriminator Results

`run4` finished cleanly with the new `xresnet1d` backbone.
It separates the four datasets well, but it did not beat the earlier `resnet1d_wang` run on the same uncontrolled multiclass setting.

## Test Set Comparison

| Run | Backbone | Accuracy | Balanced Accuracy | Macro F1 | AUROC |
|---|---|---:|---:|---:|---:|
| `run3` | `resnet1d_wang` | 0.9985 | 0.9935 | 0.9908 | 0.999880 |
| `run4` | `xresnet1d` | 0.9969 | 0.9771 | 0.9797 | 0.999880 |

`run4` is usable. The model trained and finished normally.
For this setting, the older `run3` baseline is still better on balanced accuracy and macro F1.

## Run4 Confusion Matrix

Rows are true labels. Columns are predicted labels.

| True \ Pred | PTBXL | CODE15 | CHAPMAN | SPH |
|---|---:|---:|---:|---:|
| PTBXL | 2146 | 3 | 30 | 19 |
| CODE15 | 2 | 69245 | 2 | 11 |
| CHAPMAN | 5 | 3 | 2006 | 115 |
| SPH | 0 | 2 | 50 | 5094 |

Most samples stay on the diagonal.
The main remaining confusion in `run4` is between `CHAPMAN` and `SPH`.

## Files

- Comparison table: `outputs/dataset-discriminator/test_metrics_comparison.csv`
- Run4 test metrics: `outputs/dataset-discriminator/run4/test_metrics.json`
- Run4 confusion matrix: `outputs/dataset-discriminator/run4/confusion_matrix.csv`
- Run4 training history: `outputs/dataset-discriminator/run4/history.json`