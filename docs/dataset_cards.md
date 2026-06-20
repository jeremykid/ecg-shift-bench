# Dataset cards and expected layouts

These cards document adapter assumptions, not redistribution instructions. Obtain data from
official sources and follow their current license and access requirements.

## PTB-XL

- Configured metadata: `data/raw/ptbxl/ptbxl_database.csv`
- Configured signals: `data/raw/ptbxl/records500/`
- Expected native labels: SCP statement dictionary in `scp_codes`
- TODO: validate release, WFDB units, 12-lead ordering, patient IDs, and 100/500 Hz selection.

## Chapman-Shaoxing / Ningbo

- Configured metadata: `data/raw/chapman/Diagnostics.xlsx`
- Configured signals: `data/raw/chapman/ECGData/`
- Expected native label field: `Rhythm`
- TODO: normalize release-specific column names, waveform CSV orientation, units, durations,
  patient identifiers, and the interpretation of Chapman versus Ningbo as one or two domains.

## SPH

- Configured normalized index: `data/raw/sph/metadata.csv`
- Configured signals: `data/raw/sph/records/`
- Expected native label field: `labels`
- TODO: write a release-specific index converter and validate code 82 as the prolonged-PR proxy.

## CODE-15%

- Configured annotations: `data/raw/code15/annotations.csv`
- Configured signals: `data/raw/code15/tracings/`
- Expected waveform container: HDF5; expected rate: 400 Hz
- TODO: validate HDF5 keys/chunks, exam-to-patient linkage, label capitalization, units, and
  whether public annotations expose all six labels with comparable definitions.

## Future placeholders

PhysioNet/CinC 2020/2021 and MIMIC-IV-ECG are not registered yet. New adapters must document
release identifiers, access controls, record/patient keys, lead order, signal units, native
ontology, missingness, and any site subdivision before inclusion.
