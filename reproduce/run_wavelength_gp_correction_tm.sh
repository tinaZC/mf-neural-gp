#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_ROOT="${REPO_ROOT}/code"
FINAL_ROOT="${REPO_ROOT}/result_out/final_runs"
MODULE="comparison_methods.wavelength_gp_correction_controlled"

usage() {
  cat <<'EOF'
Usage:
  # Stage I/common preprocessing once
  bash reproduce/run_wavelength_gp_correction_tm.sh --prepare DATA_DIR OUT_DIR [ARGS...]

  # Train a wavelength chunk; omit --freq_start/--freq_end for the full axis
  bash reproduce/run_wavelength_gp_correction_tm.sh --train DATA_DIR OUT_DIR [ARGS...]

  # Aggregate after every wavelength is complete
  bash reproduce/run_wavelength_gp_correction_tm.sh --aggregate DATA_DIR OUT_DIR [ARGS...]

Examples:
  bash reproduce/run_wavelength_gp_correction_tm.sh --prepare \
    data/mf_sweep_datasets_nano_tm/hf100_lfx10 \
    result_out/final_runs/wavelength_gp/hf100_lfx10/seed200 \
    --seed 200 --device cuda

  bash reproduce/run_wavelength_gp_correction_tm.sh --train \
    data/mf_sweep_datasets_nano_tm/hf100_lfx10 \
    result_out/final_runs/wavelength_gp/hf100_lfx10/seed200 \
    --seed 200 --device cuda --freq_start 0 --freq_end 100 --resume

  bash reproduce/run_wavelength_gp_correction_tm.sh --aggregate \
    data/mf_sweep_datasets_nano_tm/hf100_lfx10 \
    result_out/final_runs/wavelength_gp/hf100_lfx10/seed200 \
    --seed 200 --device cuda
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || $# -lt 3 ]]; then
  usage
  exit 0
fi

MODE="$1"
DATA_DIR="$(realpath -e "$2")"
OUT_DIR="$(realpath -m "$3")"
shift 3

case "${DATA_DIR}/" in
  "${REPO_ROOT}/"*) ;;
  *) echo "DATA_DIR must be inside ${REPO_ROOT}: ${DATA_DIR}" >&2; exit 2 ;;
esac
case "${OUT_DIR}/" in
  "${FINAL_ROOT}/"*) ;;
  *) echo "OUT_DIR must be inside ${FINAL_ROOT}: ${OUT_DIR}" >&2; exit 2 ;;
esac

export PYTHONPATH="${CODE_ROOT}"

case "${MODE}" in
  --prepare)
    exec "${PYTHON_BIN}" -m "${MODULE}" \
      --dataset TM --data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}" \
      --prepare_stage1 "$@"
    ;;
  --train)
    exec "${PYTHON_BIN}" -m "${MODULE}" \
      --dataset TM --data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}" "$@"
    ;;
  --aggregate)
    exec "${PYTHON_BIN}" -m "${MODULE}" \
      --dataset TM --data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}" \
      --aggregate "$@"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
