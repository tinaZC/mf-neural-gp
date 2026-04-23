#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CODE_ROOT="${CODE_ROOT:-${REPO_ROOT}/code}"

# For this figure-only script, RUNS_ROOT is used as the parent output root.
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/result_out}"

ABSB_ROOT="${ABSB_ROOT:-${REPO_ROOT}/data/mf_sweep_datasets_nano_ab}"
TMST_ROOT="${TMST_ROOT:-${REPO_ROOT}/data/mf_sweep_datasets_nano_tm}"
OUT_PNG="${OUT_PNG:-${RUNS_ROOT}/fig_structural_complexity_2panel.png}"

PLOT_SCRIPT="${PLOT_SCRIPT:-${CODE_ROOT}/complexity/plot_structural_complexity.py}"

mkdir -p "$(dirname "${OUT_PNG}")"

echo "[INFO] CODE_ROOT=${CODE_ROOT}"
echo "[INFO] RUNS_ROOT=${RUNS_ROOT}"

echo "[RUN] Plotting structural complexity figure..."
"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
  --absb_root "${ABSB_ROOT}" \
  --tmst_root "${TMST_ROOT}" \
  --splits "train" \
  --r_latent 32 \
  --ridge 1e-6 \
  --out_dir "${RUNS_ROOT}/structural_complexity" \
  --out_lme "lme_vs_hf.png" \
  --out_rank "effective_rank_vs_hf.png"

echo "[DONE] Structural complexity reproduction finished."

