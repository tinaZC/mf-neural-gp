#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CSV_PATH="${CSV_PATH:-${REPO_ROOT}/result_out/mf_sweep_runs_baseline_nano_tm/sweep_results.csv}"
EFF_ROOT="${EFF_ROOT:-${REPO_ROOT}/result_out/mf_sweep_runs_baseline_nano_tm/efficiency_out}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/result_out/mf_sweep_runs_baseline_nano_tm/efficiency_out_multi}"

SPEEDUP_SCRIPT="${SPEEDUP_SCRIPT:-${REPO_ROOT}/code/efficiency/make_efficiency_speedup_tables_multi.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-${REPO_ROOT}/code/efficiency/plot_efficiency_multi_lfx.py}"
TABLE_SCRIPT="${TABLE_SCRIPT:-${REPO_ROOT}/code/efficiency/make_efficiency_mapping_table.py}"

mkdir -p "${EFF_ROOT}"
mkdir -p "${OUT_DIR}"

echo "[1/3] Building efficiency speedup curves for lfx=5,10,15..."
"${PYTHON_BIN}" "${SPEEDUP_SCRIPT}" \
  --csv_path "${CSV_PATH}" \
  --out_dir "${EFF_ROOT}" \
  --curve_lfx_list "5,10,15"

echo "[2/3] Plotting efficiency figure..."
"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
  --npz_5 "${EFF_ROOT}/efficiency_out_lfx05/speedup_curves.npz" \
  --npz_10 "${EFF_ROOT}/efficiency_out_lfx10/speedup_curves.npz" \
  --npz_15 "${EFF_ROOT}/efficiency_out_lfx15/speedup_curves.npz" \
  --out_dir "${OUT_DIR}"

echo "[3/3] Building efficiency mapping table..."
"${PYTHON_BIN}" "${TABLE_SCRIPT}" \
  --npz_5 "${EFF_ROOT}/efficiency_out_lfx05/speedup_curves.npz" \
  --npz_10 "${EFF_ROOT}/efficiency_out_lfx10/speedup_curves.npz" \
  --npz_15 "${EFF_ROOT}/efficiency_out_lfx15/speedup_curves.npz" \
  --out_dir "${OUT_DIR}"

echo "[DONE] Efficiency reproduction finished."
echo "       speedup_npz_root = ${EFF_ROOT}"
echo "       figure/table_out = ${OUT_DIR}"