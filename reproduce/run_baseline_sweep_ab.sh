#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/mf_sweep_datasets_nano_ab}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/result_out/mf_sweep_runs_baseline_nano_ab}"
PLOT_OUT_DIR="${PLOT_OUT_DIR:-${RUNS_ROOT}/plot_baseline_nano_ab}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_ROOT}/code/nanophotonic_ab/mf_baseline_ab.py}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-${REPO_ROOT}/code/nanophotonic_ab/run_sweep_mf_baseline_ab.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-${REPO_ROOT}/code/nanophotonic_ab/plot_sweep_baseline_ab.py}"

mkdir -p "${RUNS_ROOT}"
mkdir -p "${PLOT_OUT_DIR}"

echo "[1/2] Running AB baseline sweep..."
"${PYTHON_BIN}" "${SWEEP_SCRIPT}" \
  --data_root "${DATA_ROOT}" \
  --out_root "${RUNS_ROOT}" \
  --train_script "${TRAIN_SCRIPT}" \
  --python_bin "${PYTHON_BIN}"

echo "[2/2] Plotting AB baseline sweep results..."
"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
  --runs_root "${RUNS_ROOT}" \
  --out_dir "${PLOT_OUT_DIR}"

echo "[DONE] AB baseline sweep reproduction finished."
echo "       runs_root = ${RUNS_ROOT}"
echo "       plot_out  = ${PLOT_OUT_DIR}"