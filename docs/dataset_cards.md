# Dataset cards and expected layouts

These cards document adapter assumptions, not redistribution instructions. Obtain data from
official sources and follow their current license and access requirements.

## PTB-XL

- Configured metadata: `data/raw/ptbxl/ptbxl_database.csv`
- Configured signals: `data/raw/ptbxl/records500/`
- Native signal unit: `mV`
- Expected native labels: SCP statement dictionary in `scp_codes`
- Supported layout: official PhysioNet PTB-XL 1.0.3-style metadata with `filename_hr` paths.
- The loader reads physical WFDB samples, verifies 500 Hz, and returns I, II, III, aVR, aVL,
  aVF, V1--V6 in `(12, 5000)` order. Use `filename_lr` with a 100 Hz config if required.
- SCP dictionary values are not binary indicators. Statement presence defines the raw label;
  rhythm/form statements such as AFIB, SBRAD, and STACH commonly carry value `0.0`.

## Chapman-Shaoxing / Ningbo

- Configured metadata: `data/raw/chapman/Diagnostics.xlsx`
- Configured signals: `data/raw/chapman/ECGData/`
- Native signal unit: `uV`
- Expected native label field: `Rhythm`
- The waveform CSVs are lead-last on disk; the loader reorders them to lead-first `(12, L)`.
- Loader converts the raw waveforms from `uV` to the shared benchmark unit `mV`.
- TODO: normalize release-specific column names, waveform CSV orientation, durations,
  patient identifiers, and the interpretation of Chapman versus Ningbo as one or two domains.

## SPH

- Configured metadata: `data/raw/sph/metadata.csv`
- Configured signals: `data/raw/sph/records/`
- Native signal unit: `mV`
- Expected native label field: `AHA_Code`
- TODO: write a release-specific index converter and validate code 82 as the prolonged-PR proxy.

## CODE-15%

- Configured annotations: `data/raw/code15/exams.csv`
- Configured signals: `data/raw/code15/exams_part0.hdf5` through `exams_part17.hdf5`
- Native signal unit: `mV`
- Waveforms are float32 `(4096, 12)` arrays at 400 Hz, ordered I, II, III, aVR, aVL,
  aVF, V1--V6. The loader returns lead-first `(12, 4096)` arrays without removing the
  release's leading/trailing zero padding.
- `exams.csv` contains `patient_id`, `trace_file`, and the six boolean target columns. Patient
  IDs, not exam IDs, must be used for splitting because many patients have repeated exams.
- Each HDF5 part has an extra all-zero sentinel row with `exam_id == 0`; the loader excludes it
  from its ID index. ZIP parts must be manually extracted before their records can be read.

## Future placeholders

PhysioNet/CinC 2020/2021 and MIMIC-IV-ECG are not registered yet. New adapters must document
release identifiers, access controls, record/patient keys, lead order, signal units, native
ontology, missingness, and any site subdivision before inclusion.
