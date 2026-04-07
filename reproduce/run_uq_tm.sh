#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BEST_CACHE_NPZ="${BEST_CACHE_NPZ:-${REPO_ROOT}/result_out/mf_sweep_runs_baseline_nano_tm/hf200_lfx10/seed333/bl0r333/cache/uq_cache_v1.npz}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/result_out/figs_uq/uq_hf200x10_seed333}"

"${PYTHON_BIN}" "${REPO_ROOT}/code/nanophotonic_tm/plot_uq_from_cache.py" \
  --best_cache_npz "${BEST_CACHE_NPZ}" \
  --out_dir "${OUT_DIR}"