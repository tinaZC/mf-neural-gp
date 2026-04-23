#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CODE_ROOT="${CODE_ROOT:-${REPO_ROOT}/code}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/result_out/mtm_examples_runs}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/mf_dataset_mw_mtm/hf50_lfx10}"
DATASET_TAG="${DATASET_TAG:-$(basename "${DATA_DIR}")}"
RUN_DIR="${RUNS_ROOT}/${DATASET_TAG}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/fig_mtm_examples}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${CODE_ROOT}/microwave_mtm/mf_train_microwave_mtm.py}"

SEED="${SEED:-42}"
CI_KIND="${CI_KIND:-raw}"          # raw | cal
N_SELECT="${N_SELECT:-2}"
N_PLOT="${N_PLOT:-999}"
SAVE_PRED_ARRAYS="${SAVE_PRED_ARRAYS:-1}"

# Selection mode:
#   specified -> use SPECIFIED_IDXS below
#   best      -> choose top-N by max(rmse_lf - rmse_mf_student)
SELECT_MODE="${SELECT_MODE:-specified}"

# Space-separated idx list used when SELECT_MODE=specified.
# Default requested examples: idx 0 and 7
SPECIFIED_IDXS="${SPECIFIED_IDXS:-0 7}"

mkdir -p "${RUNS_ROOT}" "${OUT_DIR}"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] TRAIN_SCRIPT not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ERROR] DATA_DIR not found: ${DATA_DIR}" >&2
  exit 1
fi
if [[ "${CI_KIND}" != "raw" && "${CI_KIND}" != "cal" ]]; then
  echo "[ERROR] CI_KIND must be 'raw' or 'cal', got: ${CI_KIND}" >&2
  exit 1
fi
if [[ "${SELECT_MODE}" != "specified" && "${SELECT_MODE}" != "best" ]]; then
  echo "[ERROR] SELECT_MODE must be 'specified' or 'best', got: ${SELECT_MODE}" >&2
  exit 1
fi
if ! [[ "${N_SELECT}" =~ ^[0-9]+$ ]] || [[ "${N_SELECT}" -lt 1 ]]; then
  echo "[ERROR] N_SELECT must be a positive integer, got: ${N_SELECT}" >&2
  exit 1
fi

PLOT_SUFFIX="__ciRaw.png"
if [[ "${CI_KIND}" == "cal" ]]; then
  PLOT_SUFFIX="__ciCal.png"
fi

MANIFEST_JSONL="${OUT_DIR}/mtm_example_manifest.jsonl"
: > "${MANIFEST_JSONL}"

echo "[INFO] CODE_ROOT=${CODE_ROOT}"
echo "[INFO] DATA_DIR=${DATA_DIR}"
echo "[INFO] RUNS_ROOT=${RUNS_ROOT}"
echo "[INFO] RUN_DIR=${RUN_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[INFO] CI_KIND=${CI_KIND}"
echo "[INFO] SELECT_MODE=${SELECT_MODE}"
echo "[INFO] SPECIFIED_IDXS=${SPECIFIED_IDXS}"
echo "[INFO] N_SELECT=${N_SELECT}"
echo "[INFO] Training only MF-student branch (skip HF-only and MF-oracle)"

if [[ "${SELECT_MODE}" == "specified" ]]; then
  echo "[INFO] Selection = specified idx list"
else
  echo "[INFO] Selection = top-N by max(rmse_lf - rmse_mf_student)"
fi

# Clean rerun to avoid mixing old files with the new run.
rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}" "${OUT_DIR}"

echo "[RUN] Training MTM MF-student model and generating test-case plots..."
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --data_dir "${DATA_DIR}" \
  --out_dir "${RUN_DIR}" \
  --no_subdir 1 \
  --seed "${SEED}" \
  --freq_low 0 \
  --freq_high 30 \
  --fpca_var_ratio 0.999 \
  --svgp_M 64 \
  --svgp_steps 2000 \
  --gp_ard 1 \
  --plot_ci 1 \
  --n_plot "${N_PLOT}" \
  --save_pred_arrays "${SAVE_PRED_ARRAYS}" \
  --run_hf_only 0 \
  --run_oracle 0 \
  --run_student 1

RUN_DIR_THIS="${RUN_DIR}" \
OUT_DIR_THIS="${OUT_DIR}" \
PLOT_SUFFIX_THIS="${PLOT_SUFFIX}" \
MANIFEST_JSONL_THIS="${MANIFEST_JSONL}" \
N_SELECT_THIS="${N_SELECT}" \
DATASET_TAG_THIS="${DATASET_TAG}" \
SELECT_MODE_THIS="${SELECT_MODE}" \
SPECIFIED_IDXS_THIS="${SPECIFIED_IDXS}" \
"${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

run_dir = Path(os.environ["RUN_DIR_THIS"])
out_dir = Path(os.environ["OUT_DIR_THIS"])
plot_suffix = os.environ["PLOT_SUFFIX_THIS"]
manifest_jsonl = Path(os.environ["MANIFEST_JSONL_THIS"])
n_select = int(os.environ["N_SELECT_THIS"])
dataset_tag = os.environ["DATASET_TAG_THIS"]
select_mode = os.environ["SELECT_MODE_THIS"].strip().lower()
specified_idxs_str = os.environ.get("SPECIFIED_IDXS_THIS", "").strip()

summary_csv = run_dir / "pred_arrays" / "test" / "rmse_sample__summary.csv"
plots_dir = run_dir / "plots"

if not summary_csv.exists():
    raise SystemExit(f"[ERROR] Missing summary CSV: {summary_csv}")
if not plots_dir.exists():
    raise SystemExit(f"[ERROR] Missing plots dir: {plots_dir}")

with open(summary_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    raise SystemExit(f"[ERROR] Empty summary CSV: {summary_csv}")

fieldnames = set(rows[0].keys())
lf_key_candidates = [
    "rmse_lf",
    "rmse_lf_only",
    "rmse_student_lf",
]
lf_key = next((k for k in lf_key_candidates if k in fieldnames), None)
if lf_key is None:
    raise SystemExit(
        "[ERROR] Could not find an LF RMSE column in summary CSV. "
        f"Available columns: {sorted(fieldnames)}"
    )

idx = np.array([int(r["idx"]) for r in rows], dtype=int)
rmse_lf = np.array([float(r[lf_key]) for r in rows], dtype=float)
rmse_st = np.array([float(r["rmse_mf_student"]) for r in rows], dtype=float)

abs_gain = rmse_lf - rmse_st
valid = np.isfinite(abs_gain) & np.isfinite(rmse_lf) & np.isfinite(rmse_st)
if not np.any(valid):
    raise SystemExit(f"[ERROR] No finite LF/MF cases found in: {summary_csv}")

with np.errstate(divide="ignore", invalid="ignore"):
    rel_gain_all = abs_gain / rmse_lf
rel_gain_all = np.where(np.isfinite(rel_gain_all), rel_gain_all, np.nan)

def _fmt_num(x, ndigits=4):
    return f"{float(x):.{ndigits}f}".replace("-", "m").replace(".", "p")

def _fmt_pct(x, ndigits=1):
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{100.0 * float(x):.{ndigits}f}".replace("-", "m").replace(".", "p")

picked_positions = []
selection_rule = ""
selection_note = ""

if select_mode == "specified":
    if not specified_idxs_str:
        raise SystemExit("[ERROR] SELECT_MODE=specified but SPECIFIED_IDXS is empty")

    requested_idxs = []
    for token in specified_idxs_str.split():
        try:
            requested_idxs.append(int(token))
        except ValueError:
            raise SystemExit(
                f"[ERROR] SPECIFIED_IDXS must be a space-separated integer list, got bad token: {token}"
            )

    if not requested_idxs:
        raise SystemExit("[ERROR] No valid idx parsed from SPECIFIED_IDXS")

    for ridx in requested_idxs:
        hit = np.where(idx == int(ridx))[0]
        if hit.size == 0:
            raise SystemExit(
                f"[ERROR] Requested idx={ridx:04d} not found in {summary_csv}"
            )
        pos = int(hit[0])
        if not valid[pos]:
            raise SystemExit(
                f"[ERROR] Requested idx={ridx:04d} exists but LF/MF RMSE is not finite"
            )
        picked_positions.append(pos)

    selection_rule = "specified_idx_list"
    selection_note = f"selected by requested idx list: {requested_idxs}"

elif select_mode == "best":
    # Primary key: larger LF->MF improvement.
    # Tie-break 1: smaller MF RMSE.
    # Tie-break 2: larger LF RMSE.
    order = sorted(
        np.where(valid)[0].tolist(),
        key=lambda i: (-float(abs_gain[i]), float(rmse_st[i]), -float(rmse_lf[i]), int(idx[i]))
    )
    picked_positions = order[: min(n_select, len(order))]
    if not picked_positions:
        raise SystemExit("[ERROR] No selectable cases after ranking.")
    selection_rule = "top_n_by_max_absolute_rmse_reduction_vs_lf"
    selection_note = f"selected top-{len(picked_positions)} by LF->MF absolute RMSE reduction"

else:
    raise SystemExit(f"[ERROR] Unknown SELECT_MODE: {select_mode}")

for rank, pos in enumerate(picked_positions, start=1):
    selected_idx = int(idx[pos])
    selected_abs_gain = float(abs_gain[pos])
    selected_rel_gain = float(rel_gain_all[pos]) if np.isfinite(rel_gain_all[pos]) else None

    matches = sorted(plots_dir.glob(f"test_case*_idx{selected_idx:04d}{plot_suffix}"))
    if not matches:
        raise SystemExit(
            f"[ERROR] No plot matched idx={selected_idx:04d} suffix={plot_suffix} under {plots_dir}"
        )
    src_plot = matches[0]

    lf_tag = _fmt_num(rmse_lf[pos], 4)
    mf_tag = _fmt_num(rmse_st[pos], 4)
    red_tag = _fmt_pct(selected_rel_gain, 1)

    if select_mode == "specified":
        name_stem = (
            f"{dataset_tag}"
            f"__specified{rank:02d}"
            f"_idx{selected_idx:04d}"
            f"__lf{lf_tag}"
            f"__mf{mf_tag}"
            f"__red{red_tag}pct"
        )
    else:
        name_stem = (
            f"{dataset_tag}"
            f"__top{rank:02d}"
            f"_idx{selected_idx:04d}"
            f"__lf{lf_tag}"
            f"__mf{mf_tag}"
            f"__red{red_tag}pct"
        )

    dst_plot = out_dir / f"{name_stem}{plot_suffix}"
    shutil.copy2(src_plot, dst_plot)

    record = {
        "dataset": dataset_tag,
        "selection_mode": select_mode,
        "selection_rule": selection_rule,
        "selection_note": selection_note,
        "rank": rank,
        "lf_rmse_column": lf_key,
        "selected_position_idx": int(pos),
        "selected_sample_idx": selected_idx,
        "selected_rmse_lf": float(rmse_lf[pos]),
        "selected_rmse_mf_student": float(rmse_st[pos]),
        "absolute_rmse_reduction_vs_lf": selected_abs_gain,
        "relative_rmse_reduction_vs_lf": selected_rel_gain,
        "plot_suffix": plot_suffix,
        "source_plot": str(src_plot),
        "copied_plot": str(dst_plot),
        "name_stem": name_stem,
        "rmse_summary_row": rows[pos],
    }

    summary_json = out_dir / f"{name_stem}__summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    with open(manifest_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    rel_str = "nan" if selected_rel_gain is None else f"{selected_rel_gain:.4%}"
    print(
        f"[OK] rank={rank} idx={selected_idx:04d}, "
        f"lf={rmse_lf[pos]:.6g}, ours={rmse_st[pos]:.6g}, "
        f"abs_gain={selected_abs_gain:.6g}, rel_gain={rel_str}"
    )
    print(f"[OK] copied plot -> {dst_plot}")
    print(f"[OK] summary json -> {summary_json}")
PY

echo "[DONE] MTM example reproduction finished."
echo "       run_dir  = ${RUN_DIR}"
echo "       out_dir  = ${OUT_DIR}"
echo "       manifest = ${MANIFEST_JSONL}"