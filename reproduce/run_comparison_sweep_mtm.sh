#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_ROOT="${REPO_ROOT}/code"
FINAL_ROOT="${REPO_ROOT}/result_out/final_runs"
BENCHMARK="MTM"

usage() {
  cat <<'EOF'
Usage:
  bash reproduce/run_comparison_sweep_mtm.sh --help
  bash reproduce/run_comparison_sweep_mtm.sh --run METHOD DATA_DIR OUT_DIR [METHOD_ARGS...]

Runs exactly one explicitly selected comparison case. It does not define or
launch a seed/setting sweep. METHOD is one of:
  fpca-nargp | mf-deeponet | freqwise-nargp | direct-latent

DATA_DIR must be inside this repository. OUT_DIR must be inside
result_out/final_runs/. Additional arguments are passed to the Python method.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || $# -eq 0 ]]; then
  usage
  exit 0
fi
if [[ "${1}" != "--run" || $# -lt 4 ]]; then
  usage >&2
  exit 2
fi
shift
METHOD="${1}"
DATA_DIR="$(realpath -e "${2}")"
OUT_DIR="$(realpath -m "${3}")"
shift 3

case "${DATA_DIR}/" in
  "${REPO_ROOT}/"*) ;;
  *) echo "DATA_DIR must be inside ${REPO_ROOT}: ${DATA_DIR}" >&2; exit 2 ;;
esac
case "${OUT_DIR}/" in
  "${FINAL_ROOT}/"*) ;;
  *) echo "OUT_DIR must be inside ${FINAL_ROOT}: ${OUT_DIR}" >&2; exit 2 ;;
esac

case "${METHOD}" in
  fpca-nargp) MODULE="comparison_methods.fpca_nargp" ;;
  mf-deeponet) MODULE="comparison_methods.mf_deeponet" ;;
  freqwise-nargp) MODULE="comparison_methods.freqwise_nargp" ;;
  direct-latent) MODULE="comparison_methods.direct_latent" ;;
  *) echo "Unknown METHOD: ${METHOD}" >&2; usage >&2; exit 2 ;;
esac

export PYTHONPATH="${CODE_ROOT}"
exec "${PYTHON_BIN}" -m "${MODULE}" --dataset "${BENCHMARK}" \
  --data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}" "$@"
