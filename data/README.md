# Data layout

No ECG data is distributed with this repository. Download each dataset from its official
source after accepting its license and place it under `data/raw/<dataset>/`, or override
`root` in the corresponding dataset configuration.

- `raw/`: manually downloaded, immutable source files.
- `processed/`: derived metadata and normalized record indexes.
- `splits/`: reproducible patient/domain split manifests.

Do not commit clinical data or derived artifacts that may contain identifiers. The expected
file layout for each source is documented in `docs/dataset_cards.md` and its YAML config.
