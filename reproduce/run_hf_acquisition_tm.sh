#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CODE_ROOT="${CODE_ROOT:-${REPO_ROOT}/code}"

# Main output root for retrospective acquisition.
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/result_out/retro_acq_runs_tm}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/result_out/retro_acq_runs_tm}"

PLOT_SCRIPT="${PLOT_SCRIPT:-${CODE_ROOT}/hf_acquisition/plot_retro_acq_curve.py}"

PLOT_OUT_PATH="${PLOT_OUT_PATH:-${REPO_ROOT}/result_out/final_analysis/figures/_fig_acquisition_tm.pdf}"
SUMMARY_CSV="${OUT_DIR}/retro_acq_summary.csv"

if [[ ! -s "${SUMMARY_CSV}" ]]; then
  echo "[ERROR] Frozen acquisition summary missing or empty: ${SUMMARY_CSV}" >&2
  echo "        This wrapper plots saved results only; it does not run acquisition training." >&2
  exit 1
fi

echo "[INFO] Plotting frozen acquisition summary: ${SUMMARY_CSV}"
"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
  --retro_dir "${OUT_DIR}" \
  --out_path "${PLOT_OUT_PATH}"

echo "[DONE] Frozen acquisition figure generated."
echo "       retro_dir = ${OUT_DIR}"
echo "       summary   = ${SUMMARY_CSV}"
echo "       figure    = ${PLOT_OUT_PATH}"