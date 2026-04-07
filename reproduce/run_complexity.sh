#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ABSB_ROOT="${ABSB_ROOT:-${REPO_ROOT}/data/mf_sweep_datasets_nano_ab}"
TMST_ROOT="${TMST_ROOT:-${REPO_ROOT}/data/mf_sweep_datasets_nano_tm}"
OUT_PNG="${OUT_PNG:-${REPO_ROOT}/result_out/fig_structural_complexity_2panel.png}"

PLOT_SCRIPT="${PLOT_SCRIPT:-${REPO_ROOT}/code/complexity/plot_structural_complexity.py}"

mkdir -p "$(dirname "${OUT_PNG}")"

echo "[RUN] Plotting structural complexity figure..."
"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
  --absb_root "${ABSB_ROOT}" \
  --tmst_root "${TMST_ROOT}" \
  --splits "train" \
  --r_latent 32 \
  --ridge 1e-6 \
  --out_png "${OUT_PNG}"

echo "[DONE] Structural complexity reproduction finished."
echo "       out_png = ${OUT_PNG}"