#!/usr/bin/env bash
set -euo pipefail

python -m eda.cli training-dag \
  --input-path "${WINDOWS_PATH:-/app/reports/eda/features/windows.npz}" \
  --output-dir "${MODEL_OUTPUT_DIR:-/app/reports/eda/models}" \
  --series-path "${FEATURES_PATH:-/app/reports/eda/features/features.jsonl}" \
  --target-col "${TARGET_COL:-orders}" \
  --val-ratio "${VAL_RATIO:-0.2}" \
  --random-state "${RANDOM_STATE:-42}" \
  --selection-metric "${SELECTION_METRIC:-mae}"
