#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Switchable code root:
#   default -> ${REPO_ROOT}/code
#   test v2  -> export CODE_ROOT=${REPO_ROOT}/code_v2
CODE_ROOT="${CODE_ROOT:-${REPO_ROOT}/code}"

# Main output root for this script.
# You can override either RUNS_ROOT or OUT_ROOT; RUNS_ROOT is preferred.
RUNS_ROOT="${RUNS_ROOT:-${OUT_ROOT:-${REPO_ROOT}/result_out/mf_baseline_out_microwave_mtm_multi}}"
OUT_ROOT="${OUT_ROOT:-${RUNS_ROOT}}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/mf_dataset_mw_mtm/hf50_lfx10}"

RUN_SCRIPT="${RUN_SCRIPT:-${CODE_ROOT}/microwave_mtm/run_baseline_mtm_multi_seed.py}"
BASELINE_SCRIPT="${BASELINE_SCRIPT:-${CODE_ROOT}/microwave_mtm/mf_baseline_microwave_mtm.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-${CODE_ROOT}/microwave_mtm/plot_baseline_mtm_compare.py}"

PLOT_OUT_DIR="${PLOT_OUT_DIR:-${OUT_ROOT}/hf50_lfx10/plot_mtm_compare}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${PLOT_OUT_DIR}"

echo "[INFO] CODE_ROOT=${CODE_ROOT}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"

DATASET_RUN_ROOT="${OUT_ROOT}/hf50_lfx10"
FIRST_REPORT="$(find "${DATASET_RUN_ROOT}" -name report.json -type f -print -quit 2>/dev/null || true)"

if [[ -n "${FIRST_REPORT}" ]]; then
  echo "[1/2] Found existing MTM run results:"
  echo "      ${FIRST_REPORT}"
  echo "      Skip rerunning baseline sweep; plot only."
else
  echo "[1/2] Running MTM baseline sweep..."
  "${PYTHON_BIN}" "${RUN_SCRIPT}" \
    --python_exe "${PYTHON_BIN}" \
    --baseline_script "${BASELINE_SCRIPT}" \
    --plot_script "${PLOT_SCRIPT}" \
    --data_dir "${DATA_DIR}" \
    --out_root "${OUT_ROOT}" \
    --call_plot 0
fi

echo "[2/2] Plotting MTM baseline results..."
"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
  --runs_root "${OUT_ROOT}/hf50_lfx10" \
  --out_dir "${PLOT_OUT_DIR}" \
  --hf 50 \
  --lfx 10

echo "[DONE] MTM baseline sweep reproduction finished."
echo "       runs_root = ${OUT_ROOT}/hf50_lfx10"
echo "       plot_out  = ${PLOT_OUT_DIR}"
