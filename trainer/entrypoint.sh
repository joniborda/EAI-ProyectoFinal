#!/usr/bin/env bash
set -euo pipefail

if [[ "${PREPARE_DATA:-false}" == "true" ]]; then
  DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-/app/reports/eda/data}"
  FEATURES_OUTPUT_DIR="${FEATURES_OUTPUT_DIR:-/app/reports/eda/features}"

  python -m eda.cli build-datasets \
    --output-dir "${DATA_OUTPUT_DIR}" \
    --fmt "${DATASET_FORMAT:-jsonl}"

  python -m eda.cli analyze \
    --input-dir "${DATA_OUTPUT_DIR}" \
    --plots-dir "${PLOTS_OUTPUT_DIR:-/app/reports/eda/plots}" \
    --no-with-plots

  python -m eda.cli build-features \
    --input-path "${COMBINED_PATH:-${DATA_OUTPUT_DIR}/combined.jsonl}" \
    --output-dir "${FEATURES_OUTPUT_DIR}" \
    --lags "${LAGS:-1,7,30}" \
    --target-col "${TARGET_COL:-orders}" \
    --window-size "${WINDOW_SIZE:-28}"
fi

python -m eda.cli training-dag \
  --input-path "${WINDOWS_PATH:-/app/reports/eda/features/windows.npz}" \
  --output-dir "${MODEL_OUTPUT_DIR:-/app/reports/eda/models}" \
  --series-path "${FEATURES_PATH:-/app/reports/eda/features/features.jsonl}" \
  --target-col "${TARGET_COL:-orders}" \
  --val-ratio "${VAL_RATIO:-0.2}" \
  --random-state "${RANDOM_STATE:-42}" \
  --selection-metric "${SELECTION_METRIC:-mae}"
