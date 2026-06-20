"""
Benchmark protocols for ECGShiftBench.

Three benchmark settings are provided:

1. **Covariate Shift** — evaluate how well a model trained on one dataset
   generalises to test sets of other datasets without any adaptation.

2. **Domain Generalisation** — train on multiple source datasets and
   evaluate zero-shot transfer to a held-out target dataset.

3. **Unsupervised Domain Adaptation (UDA)** — use unlabelled target-domain
   samples during training to close the covariate-shift gap.
"""

from ecgshiftbench.benchmarks.base import BenchmarkProtocol, BenchmarkResult
from ecgshiftbench.benchmarks.covariate_shift import CovariateShiftBenchmark
from ecgshiftbench.benchmarks.domain_generalization import DomainGeneralizationBenchmark
from ecgshiftbench.benchmarks.uda import UDABenchmark

__all__ = [
    "BenchmarkProtocol",
    "BenchmarkResult",
    "CovariateShiftBenchmark",
    "DomainGeneralizationBenchmark",
    "UDABenchmark",
]
