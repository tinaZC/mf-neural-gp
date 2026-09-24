#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CODE_ROOT="${CODE_ROOT:-${REPO_ROOT}/code}"

# Plot directly into the final publication figure directory.
FIGURE_DIR="${FIGURE_DIR:-${REPO_ROOT}/result_out/final_analysis/figures}"

ABSB_ROOT="${ABSB_ROOT:-${REPO_ROOT}/data/mf_sweep_datasets_nano_ab}"
TMST_ROOT="${TMST_ROOT:-${REPO_ROOT}/data/mf_sweep_datasets_nano_tm}"
OUT_PNG="${OUT_PNG:-_fig_structural_complexity.png}"

PLOT_SCRIPT="${PLOT_SCRIPT:-${CODE_ROOT}/complexity/plot_structural_complexity.py}"

mkdir -p "${FIGURE_DIR}"

echo "[INFO] CODE_ROOT=${CODE_ROOT}"
echo "[INFO] FIGURE_DIR=${FIGURE_DIR}"

echo "[RUN] Plotting structural complexity figure..."
"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
  --absb_root "${ABSB_ROOT}" \
  --tmst_root "${TMST_ROOT}" \
  --splits "train" \
  --r_latent 32 \
  --ridge 1e-6 \
  --out_dir "${FIGURE_DIR}" \
  --out_fig "${OUT_PNG}"

echo "[DONE] Structural complexity reproduction finished."

