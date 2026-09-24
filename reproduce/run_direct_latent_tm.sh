#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash reproduce/run_direct_latent_tm.sh --help
  bash reproduce/run_direct_latent_tm.sh --run DATA_DIR OUT_DIR [METHOD_ARGS...]

Runs one explicitly selected TM direct-latent case. OUT_DIR must be under
result_out/final_runs/. No sweep is defined or launched by this script.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || $# -eq 0 ]]; then
  usage
  exit 0
fi
if [[ "${1}" != "--run" || $# -lt 3 ]]; then
  usage >&2
  exit 2
fi
shift

exec bash "${SCRIPT_DIR}/run_comparison_sweep_tm.sh" \
  --run direct-latent "$@"
