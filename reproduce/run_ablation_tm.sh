#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/mf_sweep_datasets_nano_tm/hf100_lfx10}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/result_out/ablate_runs_tm}"
EXP_NAME="${EXP_NAME:-nano_tm_main}"

"${PYTHON_BIN}" "${REPO_ROOT}/code/nanophotonic_tm/run_ablate_tm.py" \
  --python "${PYTHON_BIN}" \
  --train_script "${REPO_ROOT}/code/nanophotonic_tm/mf_train_tm.py" \
  --data_dirs "${DATA_DIR}" \
  --out_dir "${OUT_DIR}" \
  --exp_name "${EXP_NAME}"