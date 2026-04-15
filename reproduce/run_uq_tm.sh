#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CODE_ROOT="${CODE_ROOT:-${REPO_ROOT}/code}"

# For UQ plotting, RUNS_ROOT points to the TM baseline sweep root that contains caches.
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/result_out/mf_sweep_runs_baseline_nano_tm}"

# Keep this script explicit / fixed-path based.
# You can still override BEST_CACHE_NPZ manually if needed.
BEST_CACHE_NPZ="${BEST_CACHE_NPZ:-${RUNS_ROOT}/hf200_lfx10/seed333/bl0r333/cache/uq_cache_v1.npz}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/result_out/figs_uq/uq_hf200x10_seed333}"
PLOT_SCRIPT="${PLOT_SCRIPT:-${CODE_ROOT}/nanophotonic_tm/plot_uq_from_cache.py}"

if [[ ! -f "${BEST_CACHE_NPZ}" ]]; then
  echo "[ERROR] BEST_CACHE_NPZ does not exist:"
  echo "        ${BEST_CACHE_NPZ}"
  echo "        Please update BEST_CACHE_NPZ or override it explicitly."
  exit 1
fi

echo "[INFO] CODE_ROOT=${CODE_ROOT}"
echo "[INFO] RUNS_ROOT=${RUNS_ROOT}"
echo "[INFO] BEST_CACHE_NPZ=${BEST_CACHE_NPZ}"
echo "[INFO] OUT_DIR=${OUT_DIR}"

"${PYTHON_BIN}" "${PLOT_SCRIPT}" \
  --best_cache_npz "${BEST_CACHE_NPZ}" \
  --out_dir "${OUT_DIR}"
