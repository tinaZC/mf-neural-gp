#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CODE_ROOT="${CODE_ROOT:-${REPO_ROOT}/code}"

# Main output root for ablation runs.
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/result_out/ablate_runs_tm}"
OUT_DIR="${OUT_DIR:-${RUNS_ROOT}}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/mf_sweep_datasets_nano_tm/hf100_lfx10}"
EXP_NAME="${EXP_NAME:-nano_tm_main}"

# Refactored shared training backend
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${CODE_ROOT}/mf_train_baseline/mf_train.py}"
ABLATE_SCRIPT="${ABLATE_SCRIPT:-${CODE_ROOT}/nanophotonic_tm/run_ablate_tm.py}"

EXP_DIR="${OUT_DIR}/${EXP_NAME}"

echo "[INFO] CODE_ROOT=${CODE_ROOT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] EXP_DIR=${EXP_DIR}"

SKIP_TRAIN_FLAG=""
FIRST_REPORT="$(find "${EXP_DIR}" -name report.json -type f -print -quit 2>/dev/null || true)"

if [[ -n "${FIRST_REPORT}" ]]; then
  echo "[INFO] Found existing ablation run results under:"
  echo "       ${EXP_DIR}"
  echo "[INFO] Example report:"
  echo "       ${FIRST_REPORT}"
  echo "[INFO] Skip training and summarize/plot only."
  SKIP_TRAIN_FLAG="--skip_train"
else
  echo "[INFO] No existing report.json found under:"
  echo "       ${EXP_DIR}"
  echo "[INFO] Run training + summarize + plot."
fi

"${PYTHON_BIN}" "${ABLATE_SCRIPT}" \
  --python "${PYTHON_BIN}" \
  --train_script "${TRAIN_SCRIPT}" \
  --data_dirs "${DATA_DIR}" \
  --out_dir "${OUT_DIR}" \
  --exp_name "${EXP_NAME}" \
  ${SKIP_TRAIN_FLAG}

echo "[DONE] Ablation workflow finished."
echo "       exp_dir = ${EXP_DIR}"
echo "       summary = ${EXP_DIR}/summary_runs.json"
echo "       plots   = ${EXP_DIR}/summary"