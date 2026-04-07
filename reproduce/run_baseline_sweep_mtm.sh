#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/mf_dataset_mw_mtm/hf50_lfx10}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/result_out/mf_baseline_out_microwave_mtm_multi}"

RUN_SCRIPT="${RUN_SCRIPT:-${REPO_ROOT}/code/microwave_mtm/run_baseline_mtm_multi_seed.py}"
BASELINE_SCRIPT="${BASELINE_SCRIPT:-${REPO_ROOT}/code/microwave_mtm/mf_baseline_microwave_mtm.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-${REPO_ROOT}/code/microwave_mtm/plot_baseline_mtm_compare.py}"

PLOT_OUT_DIR="${PLOT_OUT_DIR:-${OUT_ROOT}/hf50_lfx10/plot_mtm_compare}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${PLOT_OUT_DIR}"

echo "[1/2] Running MTM baseline sweep..."
"${PYTHON_BIN}" "${RUN_SCRIPT}" \
  --python_exe "${PYTHON_BIN}" \
  --baseline_script "${BASELINE_SCRIPT}" \
  --plot_script "${PLOT_SCRIPT}" \
  --data_dir "${DATA_DIR}" \
  --out_root "${OUT_ROOT}" \
  --call_plot 0

echo "[2/2] Plotting MTM baseline results..."
"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
  --runs_root "${OUT_ROOT}/hf50_lfx10" \
  --out_dir "${PLOT_OUT_DIR}" \
  --hf 50 \
  --lfx 10

echo "[DONE] MTM baseline sweep reproduction finished."
echo "       runs_root = ${OUT_ROOT}/hf50_lfx10"
echo "       plot_out  = ${PLOT_OUT_DIR}"