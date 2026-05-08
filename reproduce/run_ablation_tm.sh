#!/usr/bin/env bash
set -euo pipefail

# Revised reproduce script for the TM A0--A3 component/pathway ablation.
#
# Output location:
#   <repo>/result_out/ablate_tm
#
# Required Python runner:
#   code/nanophotonic_tm/run_ablate_tm.py must be the revised A0--A3 runner
#   that supports:
#     --boxplot_view
#     --spectrum_view
#     --save_pdf
#     --rerun_existing
#
# If this script reports that run_ablate_tm.py is outdated, replace:
#   code/nanophotonic_tm/run_ablate_tm.py
# with the revised A0--A3 Python script before running again.

PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CODE_ROOT="${CODE_ROOT:-${REPO_ROOT}/code}"

# Make final experiment directory exactly:
#   ${REPO_ROOT}/result_out/ablate_tm
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/result_out}"
EXP_NAME="${EXP_NAME:-ablate_tm}"
EXP_DIR="${OUT_DIR}/${EXP_NAME}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/mf_sweep_datasets_nano_tm/hf100_lfx10}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${CODE_ROOT}/mf_train_baseline/mf_train.py}"
ABLATE_SCRIPT="${ABLATE_SCRIPT:-${CODE_ROOT}/nanophotonic_tm/run_ablate_tm.py}"

DEVICE="${DEVICE:-cuda}"

# Default manuscript seeds. Override as:
#   SEEDS="42 33" bash run_ablation_tm.sh
SEEDS="${SEEDS:-42 33 55 66 77 8 9 11 22 88 99 111 222 333 555}"

echo "[INFO] Revised TM A0-A3 ablation workflow"
echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] CODE_ROOT=${CODE_ROOT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] EXP_NAME=${EXP_NAME}"
echo "[INFO] EXP_DIR=${EXP_DIR}"
echo "[INFO] DATA_DIR=${DATA_DIR}"
echo "[INFO] TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[INFO] ABLATE_SCRIPT=${ABLATE_SCRIPT}"
echo "[INFO] DEVICE=${DEVICE}"
echo "[INFO] SEEDS=${SEEDS}"

if [[ ! -f "${ABLATE_SCRIPT}" ]]; then
  echo "[ERROR] ABLATE_SCRIPT does not exist:"
  echo "        ${ABLATE_SCRIPT}"
  exit 2
fi

# The old runner accepts --ablations but does not accept the revised plotting flags.
# Stop early rather than silently running the old A0-A3 ablation definition.
if ! "${PYTHON_BIN}" "${ABLATE_SCRIPT}" -h 2>&1 | grep -q -- "--boxplot_view"; then
  echo "[ERROR] The current run_ablate_tm.py is not the revised A0-A3 pathway runner."
  echo "        It does not support --boxplot_view / --spectrum_view / --save_pdf."
  echo ""
  echo "        Please replace this file first:"
  echo "        ${ABLATE_SCRIPT}"
  echo ""
  echo "        with the revised A0-A3 Python script, then rerun:"
  echo "        bash ${SCRIPT_DIR}/run_ablation_tm.sh"
  exit 2
fi

if ! grep -q "Linear LF + GP(x)" "${ABLATE_SCRIPT}"; then
  echo "[ERROR] The current run_ablate_tm.py does not appear to contain the revised A2:"
  echo "        A2 = Linear LF + GP(x)"
  echo ""
  echo "        Please replace:"
  echo "        ${ABLATE_SCRIPT}"
  echo "        with the revised A0-A3 Python script."
  exit 2
fi

RERUN_FLAG=""
if [[ "${RERUN_EXISTING:-0}" == "1" ]]; then
  RERUN_FLAG="--rerun_existing"
  echo "[INFO] RERUN_EXISTING=1 -> force retraining existing runs."
else
  echo "[INFO] Existing report.json files will be reused per variant/seed by the Python runner."
fi

read -r -a SEED_ARRAY <<< "${SEEDS}"

"${PYTHON_BIN}" "${ABLATE_SCRIPT}" \
  --python "${PYTHON_BIN}" \
  --train_script "${TRAIN_SCRIPT}" \
  --data_dirs "${DATA_DIR}" \
  --out_dir "${OUT_DIR}" \
  --exp_name "${EXP_NAME}" \
  --seeds "${SEED_ARRAY[@]}" \
  --device "${DEVICE}" \
  --boxplot_view broken \
  --spectrum_view broken \
  --save_pdf 1 \
  ${RERUN_FLAG}

echo "[DONE] Revised TM A0-A3 ablation workflow finished."
echo "       exp_dir = ${EXP_DIR}"
echo "       summary = ${EXP_DIR}/summary_runs.json"
echo "       table   = ${EXP_DIR}/summary_runs.csv"
echo "       plots   = ${EXP_DIR}/summary"
