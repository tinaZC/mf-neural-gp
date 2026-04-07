#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/mf_sweep_datasets_nano_tm/hf100_lfx10}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/result_out/fpca_dim_sweep_tm_outputs}"

SWEEP_SCRIPT="${SWEEP_SCRIPT:-${REPO_ROOT}/code/fpca/fpca_dim_sweep_tm_with_seeds.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_ROOT}/code/fpca/mf_train_nano_tm_dim_sweep.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-${REPO_ROOT}/code/fpca/plot_fpca_dim_sweep_tm.py}"

PLOT_OUT_DIR="${PLOT_OUT_DIR:-${OUT_ROOT}/plots}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${PLOT_OUT_DIR}"

echo "[1/2] Running FPCA dim sweep..."
"${PYTHON_BIN}" "${SWEEP_SCRIPT}" \
  --python_bin "${PYTHON_BIN}" \
  --train_script "${TRAIN_SCRIPT}" \
  --data_dir "${DATA_DIR}" \
  --out_root "${OUT_ROOT}"

echo "[2/2] Plotting FPCA dim sweep results..."
"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
  --sweep_root "${OUT_ROOT}" \
  --out_dir "${PLOT_OUT_DIR}"

echo "[DONE] FPCA dim sweep reproduction finished."
echo "       sweep_root = ${OUT_ROOT}"
echo "       plot_out   = ${PLOT_OUT_DIR}"