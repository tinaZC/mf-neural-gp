#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/mf_sweep_datasets_nano_tm}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/result_out/retro_acq_runs_tm}"

RETRO_SCRIPT="${RETRO_SCRIPT:-${REPO_ROOT}/code/hf_acquisition/acquisition_with_baseline_tm.py}"
PLOT_SCRIPT="${PLOT_SCRIPT:-${REPO_ROOT}/code/hf_acquisition/plot_retro_acq_curve.py}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-${REPO_ROOT}/code/nanophotonic_tm/mf_baseline_tm.py}"

PLOT_OUT_PATH="${PLOT_OUT_PATH:-${OUT_DIR}/_fig_acquisition.png}"

mkdir -p "${OUT_DIR}"

echo "[1/2] Running retrospective HF acquisition..."
"${PYTHON_BIN}" "${RETRO_SCRIPT}" \
  --root "${DATA_ROOT}" \
  --initial_subdir hf50_lfx10 \
  --max_subdir hf500_lfx10 \
  --target_split test \
  --n_targets 20 \
  --rounds 10 \
  --batch_size 5 \
  --beta 0.5 \
  --out_dir "${OUT_DIR}" \
  --compare_script "${COMPARE_SCRIPT}" \
  --python_bin "${PYTHON_BIN}" \
  --extra_args "--wl_low 380 --wl_high 750 --fpca_var_ratio 0.999 --svgp_M 64 --svgp_steps 500 --gp_ard 1 --plot_ci 0 --n_plot 0"

echo "[2/2] Plotting aggregate acquisition curve..."
"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
  --retro_dir "${OUT_DIR}" \
  --out_path "${PLOT_OUT_PATH}"

echo "[DONE] HF acquisition reproduction finished."
echo "       retro_dir = ${OUT_DIR}"
echo "       figure    = ${PLOT_OUT_PATH}"
