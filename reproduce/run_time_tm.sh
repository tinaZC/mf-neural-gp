#!/usr/bin/env bash
set -euo pipefail

# Timing audit for the nanophotonic transmission benchmark.
#
# Put this file at:
#   <PROJECT_ROOT>/reproduce/run_time_tm.sh
#
# Then run:
#   cd <PROJECT_ROOT>/reproduce
#   bash run_time_tm.sh
#
# The script automatically treats <PROJECT_ROOT> as the parent directory of this reproduce/ folder.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CODE_ROOT="${PROJECT_ROOT}/code"
DATA_ROOT="${PROJECT_ROOT}/data/mf_sweep_datasets_nano_tm"
OUT_ROOT="${PROJECT_ROOT}/result_out/tm_timing"

BASELINE_SCRIPT="${CODE_ROOT}/time/mf_baseline_time.py"
TRAIN_SCRIPT="${CODE_ROOT}/time/mf_train_time.py"
TIMING_TABLE_SCRIPT="${CODE_ROOT}/time/make_timing_table.py"

# Keep imports robust after moving timing entry scripts into code/time.
# This also preserves access to shared modules that may still live under code/mf_train_baseline.
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/time:${CODE_ROOT}/mf_train_baseline${PYTHONPATH:+:${PYTHONPATH}}"

SETTINGS=(
  "hf50_lfx10"
  "hf100_lfx10"
  "hf500_lfx10"
)

SEED=42

echo "[INFO] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] BASELINE_SCRIPT=${BASELINE_SCRIPT}"
echo "[INFO] TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[INFO] TIMING_TABLE_SCRIPT=${TIMING_TABLE_SCRIPT}"

if [[ ! -f "${BASELINE_SCRIPT}" ]]; then
  echo "[ERROR] Missing baseline timing script: ${BASELINE_SCRIPT}" >&2
  echo "[HINT] Put mf_baseline_time.py under: ${CODE_ROOT}/time/" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] Missing train timing script: ${TRAIN_SCRIPT}" >&2
  echo "[HINT] Put mf_train_time.py under: ${CODE_ROOT}/time/" >&2
  exit 1
fi

if [[ ! -f "${TIMING_TABLE_SCRIPT}" ]]; then
  echo "[ERROR] Missing timing table script: ${TIMING_TABLE_SCRIPT}" >&2
  echo "[HINT] Put make_timing_table.py under: ${CODE_ROOT}/time/" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

for SUB in "${SETTINGS[@]}"; do
  DATA_DIR="${DATA_ROOT}/${SUB}"
  RUN_DIR="${OUT_ROOT}/${SUB}_seed${SEED}"

  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[ERROR] Missing data directory: ${DATA_DIR}" >&2
    exit 1
  fi

  echo
  echo "============================================================"
  echo "[RUN] TM timing setting: ${SUB}, seed=${SEED}"
  echo "[RUN] data_dir=${DATA_DIR}"
  echo "[RUN] out_dir=${RUN_DIR}"
  echo "============================================================"

  mkdir -p "${RUN_DIR}"

  python "${BASELINE_SCRIPT}" \
    --data_dir "${DATA_DIR}" \
    --out_dir "${RUN_DIR}" \
    --run_prefix "timing_tm" \
    --methods all \
    --delegate_ours_to_train 1 \
    --ours_train_script "${TRAIN_SCRIPT}" \
    --seed "${SEED}" \
    --device cuda \
    --wl_low 380 \
    --wl_high 750 \
    --fpca_var_ratio 0.999 \
    --fpca_max_dim 50 \
    --svgp_M 64 \
    --svgp_steps 2000 \
    --gp_ard 1 \
    --lf_prob 0 \
    --mc_lf_samples 1 \
    --plot_ci 0 \
    --n_plot 0 \
    --save_pred_arrays 1

  echo "[OK] Finished ${SUB}. Timing files should be under:"
  echo "     ${RUN_DIR}/timing.json"
  echo "     ${RUN_DIR}/ours/timing.json"
done

echo
echo "============================================================"
echo "[SUMMARY] Build timing tables"
echo "============================================================"

python "${TIMING_TABLE_SCRIPT}" \
  --runs_root "${OUT_ROOT}" \
  --out_dir "${OUT_ROOT}/timing_tables"

echo
echo "[DONE] Timing audit complete."
echo "[DONE] Timing tables:"
echo "  ${OUT_ROOT}/timing_tables/timing_table_long.csv"
echo "  ${OUT_ROOT}/timing_tables/timing_table_summary.csv"
echo "  ${OUT_ROOT}/timing_tables/timing_table_summary.md"
