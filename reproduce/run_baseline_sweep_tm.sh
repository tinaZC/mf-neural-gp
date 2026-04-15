#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CODE_ROOT="${CODE_ROOT:-${REPO_ROOT}/code}"

# Main output root for TM baseline sweep.
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/result_out/mf_sweep_runs_baseline_nano_tm}"
PLOT_OUT_DIR="${PLOT_OUT_DIR:-${RUNS_ROOT}/plot_result_baseline}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/mf_sweep_datasets_nano_tm}"

# Refactored shared baseline backend
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${CODE_ROOT}/mf_train_baseline/mf_baseline.py}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-${CODE_ROOT}/nanophotonic_tm/run_sweep_mf_baseline_tm.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-${CODE_ROOT}/nanophotonic_tm/plot_sweep_results_baseline_tm.py}"

echo "[INFO] CODE_ROOT=${CODE_ROOT}"
echo "[INFO] data_root=${DATA_ROOT}"
echo "[INFO] runs_root=${RUNS_ROOT}"
echo "[INFO] train_script=${TRAIN_SCRIPT}"

"${PYTHON_BIN}" "${SWEEP_SCRIPT}" \
  --data_root "${DATA_ROOT}" \
  --out_root "${RUNS_ROOT}" \
  --train_script "${TRAIN_SCRIPT}" \
  --python_bin "${PYTHON_BIN}"

"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
  --runs_root "${RUNS_ROOT}" \
  --out_dir "${PLOT_OUT_DIR}"
