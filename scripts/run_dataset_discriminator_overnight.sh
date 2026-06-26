#!/usr/bin/env bash
set -u

ROOT="/home/shuo19/ecg-shift-bench"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dataset-discriminator/run3}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"
RETRIES="${RETRIES:-1}"

PTBXL_ROOT="${PTBXL_ROOT:-/data/padmalab_external/special_project/physionet.org/files/ptb-xl/1.0.3/}"
CODE15_ROOT="${CODE15_ROOT:-/data/padmalab_external/special_project/code_15}"
CHAPMAN_ROOT="${CHAPMAN_ROOT:-/data/padmalab_external/special_project/chapman_shaoxing}"
SPH_ROOT="${SPH_ROOT:-/data/padmalab_external/special_project/sph}"
DEVICE="${DEVICE:-cuda:0}"
MODE="${MODE:-multiclass}"
SUBSET="${SUBSET:-uncontrolled}"
CONFIG="${CONFIG:-configs/experiments/dataset_discriminator.yaml}"

mkdir -p "$(dirname "$LOG_FILE")" "$OUTPUT_DIR"

cd "$ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "===== Attempt ${attempt} started at $(date -u +"%Y-%m-%dT%H:%M:%SZ") ====="
  PYTHONPATH=src .venv/bin/python scripts/train_dataset_discriminator.py \
    --config "$CONFIG" \
    --ptbxl-root "$PTBXL_ROOT" \
    --code15-root "$CODE15_ROOT" \
    --chapman-root "$CHAPMAN_ROOT" \
    --sph-root "$SPH_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --mode "$MODE" \
    --subset "$SUBSET"
  rc=$?
  echo "===== Attempt ${attempt} finished with exit code ${rc} at $(date -u +"%Y-%m-%dT%H:%M:%SZ") ====="

  if [ "$rc" -eq 0 ]; then
    exit 0
  fi

  if [ "$attempt" -gt "$RETRIES" ]; then
    exit "$rc"
  fi

  echo "Retrying after failure..." | tee -a "$LOG_FILE"
  sleep 10
done
