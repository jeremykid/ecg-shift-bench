# Reproducibility checklist

For each benchmark result, archive or record:

- dataset name, exact release, acquisition date, official source, license, and source checksums;
- indexing script version, raw-to-processed counts, exclusions, and missing/corrupt records;
- immutable record-to-patient/domain index and split manifests with hashes;
- label-map version, label prevalence by split/domain, and unresolved mapping assumptions;
- lead order, units, sampling rate, filters, crop/padding rule, normalization, and quality policy;
- full experiment YAML, source/target protocol, seed, dependency lock/environment, and hardware;
- model checkpoint, training logs, target-label access statement, and selection criterion;
- per-domain/per-label metrics, support, uncertainty intervals, and undefined-metric handling.

Run tests before experiments:

```bash
pip install -e ".[dev]"
pytest
```

For strict reproducibility, generate a frozen dependency environment and dataset snapshot
manifest in the experiment phase. The initialization intentionally does not lock platform-
specific Torch builds or create manifests for data that is not present.
