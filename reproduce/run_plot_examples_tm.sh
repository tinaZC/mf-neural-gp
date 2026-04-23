#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default to the v2 code tree used for the current TM example generation.
CODE_ROOT="${CODE_ROOT:-${REPO_ROOT}/code}"

# Intermediate per-dataset mf_train.py runs.
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/result_out/tm_examples_runs}"

# Figure outputs copied for paper use.
OUT_DIR="${OUT_DIR:-${RUNS_ROOT}/fig_tm_examples}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/mf_sweep_datasets_nano_tm}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${CODE_ROOT}/mf_train_baseline/mf_train.py}"

RUN_PREFIX="${RUN_PREFIX:-tm0}"
SEED="${SEED:-42}"
CI_KIND="${CI_KIND:-raw}"            # raw | cal
N_PLOT="${N_PLOT:-999}"
SAVE_PRED_ARRAYS="${SAVE_PRED_ARRAYS:-1}"

# Space-separated dataset subdirectories under DATA_ROOT.
DATASETS_STR="${DATASETS:-hf50_lfx10 hf100_lfx10}"

# Selection mode:
#   specified -> use EXAMPLE_IDXS mapping below
#   best      -> choose the sample with max LF->MF absolute RMSE reduction
SELECT_MODE="${SELECT_MODE:-specified}"

# Space-separated mapping: dataset_name:idx
# Default requested examples:
#   hf50_lfx10  -> idx 0001
#   hf100_lfx10 -> idx 0011
EXAMPLE_IDXS="${EXAMPLE_IDXS:-hf50_lfx10:1 hf100_lfx10:11}"

mkdir -p "${OUT_DIR}" "${RUNS_ROOT}"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] TRAIN_SCRIPT not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT not found: ${DATA_ROOT}" >&2
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

PLOT_SUFFIX="__ciRaw.png"
if [[ "${CI_KIND}" == "cal" ]]; then
  PLOT_SUFFIX="__ciCal.png"
fi

MANIFEST_JSONL="${OUT_DIR}/tm_example_manifest.jsonl"
: > "${MANIFEST_JSONL}"

echo "[INFO] CODE_ROOT=${CODE_ROOT}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] RUNS_ROOT=${RUNS_ROOT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[INFO] CI_KIND=${CI_KIND}"
echo "[INFO] SELECT_MODE=${SELECT_MODE}"
echo "[INFO] EXAMPLE_IDXS=${EXAMPLE_IDXS}"

if [[ "${SELECT_MODE}" == "specified" ]]; then
  echo "[INFO] Selection = user-specified idx per dataset"
else
  echo "[INFO] Selection = max(MF improvement over LF)"
fi

echo "[RUN] Generating TM example plots..."
for ds in ${DATASETS_STR}; do
  DATA_DIR="${DATA_ROOT}/${ds}"
  RUN_DIR="${RUNS_ROOT}/${ds}"

  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[ERROR] dataset directory not found: ${DATA_DIR}" >&2
    exit 1
  fi

  mkdir -p "${RUN_DIR}"

  echo "------------------------------------------------------------"
  echo "[DATASET] ${ds}"
  echo "[DATASET] data_dir=${DATA_DIR}"
  echo "[DATASET] run_dir=${RUN_DIR}"

  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --data_dir "${DATA_DIR}" \
    --out_dir "${RUN_DIR}" \
    --run_prefix "${RUN_PREFIX}" \
    --no_subdir 1 \
    --seed "${SEED}" \
    --wl_low 380 \
    --wl_high 750 \
    --fpca_var_ratio 0.999 \
    --svgp_M 64 \
    --svgp_steps 2000 \
    --gp_ard 1 \
    --plot_ci 1 \
    --n_plot "${N_PLOT}" \
    --save_pred_arrays "${SAVE_PRED_ARRAYS}"

  DATASET_NAME="${ds}" \
  RUN_DIR_THIS="${RUN_DIR}" \
  OUT_DIR_THIS="${OUT_DIR}" \
  PLOT_SUFFIX_THIS="${PLOT_SUFFIX}" \
  MANIFEST_JSONL_THIS="${MANIFEST_JSONL}" \
  SELECT_MODE_THIS="${SELECT_MODE}" \
  EXAMPLE_IDXS_THIS="${EXAMPLE_IDXS}" \
  "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

dataset_name = os.environ["DATASET_NAME"]
run_dir = Path(os.environ["RUN_DIR_THIS"])
out_dir = Path(os.environ["OUT_DIR_THIS"])
plot_suffix = os.environ["PLOT_SUFFIX_THIS"]
manifest_jsonl = Path(os.environ["MANIFEST_JSONL_THIS"])
select_mode = os.environ["SELECT_MODE_THIS"].strip().lower()
example_idxs_str = os.environ.get("EXAMPLE_IDXS_THIS", "").strip()

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
with np.errstate(divide="ignore", invalid="ignore"):
    rel_gain = abs_gain / rmse_lf
rel_gain = np.where(np.isfinite(rel_gain), rel_gain, np.nan)

valid = np.isfinite(abs_gain) & np.isfinite(rmse_lf) & np.isfinite(rmse_st)
if not np.any(valid):
    raise SystemExit(f"[ERROR] No finite LF/MF cases found in: {summary_csv}")

spec_map = {}
if example_idxs_str:
    for token in example_idxs_str.split():
        if ":" not in token:
            raise SystemExit(
                "[ERROR] EXAMPLE_IDXS must be space-separated dataset:idx pairs, "
                f"got bad token: {token}"
            )
        ds_name, idx_str = token.split(":", 1)
        ds_name = ds_name.strip()
        idx_str = idx_str.strip()
        if not ds_name:
            raise SystemExit(f"[ERROR] Empty dataset name in EXAMPLE_IDXS token: {token}")
        try:
            spec_map[ds_name] = int(idx_str)
        except ValueError:
            raise SystemExit(f"[ERROR] Invalid idx in EXAMPLE_IDXS token: {token}")

requested_idx = spec_map.get(dataset_name, None)

if select_mode == "specified":
    if requested_idx is None:
        raise SystemExit(
            f"[ERROR] SELECT_MODE=specified but dataset '{dataset_name}' is not present in EXAMPLE_IDXS={example_idxs_str!r}"
        )
    hit = np.where(idx == int(requested_idx))[0]
    if hit.size == 0:
        raise SystemExit(
            f"[ERROR] Requested idx={requested_idx:04d} for dataset={dataset_name} "
            f"not found in {summary_csv}"
        )
    best_pos = int(hit[0])
    selection_rule = "specified_idx"
    selection_note = f"selected by requested idx={requested_idx:04d}"
elif select_mode == "best":
    positive = valid & (abs_gain > 0)
    if np.any(positive):
        score = np.where(positive, abs_gain, -np.inf)
        selection_rule = "max_absolute_rmse_reduction_vs_lf"
        selection_note = "selected among positively improved LF->MF cases"
    else:
        score = np.where(valid, abs_gain, -np.inf)
        selection_rule = "max_absolute_rmse_reduction_vs_lf_allow_nonpositive"
        selection_note = "no positive LF->MF improvement found; selected least-bad case"
    best_pos = int(np.argmax(score))
else:
    raise SystemExit(f"[ERROR] Unknown SELECT_MODE: {select_mode}")

selected_idx = int(idx[best_pos])
selected_row = rows[best_pos]
selected_abs_gain = float(abs_gain[best_pos])
selected_rel_gain = float(rel_gain[best_pos]) if np.isfinite(rel_gain[best_pos]) else None

matches = sorted(plots_dir.glob(f"test_case*_idx{selected_idx:04d}{plot_suffix}"))
if not matches:
    raise SystemExit(
        f"[ERROR] No plot matched idx={selected_idx:04d} suffix={plot_suffix} under {plots_dir}"
    )
src_plot = matches[0]

def _fmt_num(x, ndigits=4):
    return f"{float(x):.{ndigits}f}".replace("-", "m").replace(".", "p")

def _fmt_pct(x, ndigits=1):
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{100.0 * float(x):.{ndigits}f}".replace("-", "m").replace(".", "p")

lf_tag = _fmt_num(rmse_lf[best_pos], 4)
mf_tag = _fmt_num(rmse_st[best_pos], 4)
red_tag = _fmt_pct(selected_rel_gain, 1)

name_stem = (
    f"{dataset_name}"
    f"__{selection_rule}"
    f"_idx{selected_idx:04d}"
    f"__lf{lf_tag}"
    f"__mf{mf_tag}"
    f"__red{red_tag}pct"
)

dst_plot = out_dir / f"{name_stem}{plot_suffix}"
shutil.copy2(src_plot, dst_plot)

record = {
    "dataset": dataset_name,
    "selection_mode": select_mode,
    "selection_rule": selection_rule,
    "selection_note": selection_note,
    "requested_idx": requested_idx,
    "lf_rmse_column": lf_key,
    "selected_position_idx": best_pos,
    "selected_sample_idx": selected_idx,
    "selected_rmse_lf": float(rmse_lf[best_pos]),
    "selected_rmse_mf_student": float(rmse_st[best_pos]),
    "absolute_rmse_reduction_vs_lf": selected_abs_gain,
    "relative_rmse_reduction_vs_lf": selected_rel_gain,
    "plot_suffix": plot_suffix,
    "source_plot": str(src_plot),
    "copied_plot": str(dst_plot),
    "name_stem": name_stem,
    "rmse_summary_row": selected_row,
}

summary_json = out_dir / f"{name_stem}__summary.json"
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2, ensure_ascii=False)

with open(manifest_jsonl, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")

rel_str = "nan" if selected_rel_gain is None else f"{selected_rel_gain:.4%}"
print(
    f"[OK] {dataset_name}: idx={selected_idx:04d}, "
    f"lf={rmse_lf[best_pos]:.6g}, ours={rmse_st[best_pos]:.6g}, "
    f"abs_gain={selected_abs_gain:.6g}, rel_gain={rel_str}"
)
print(f"[OK] copied plot -> {dst_plot}")
print(f"[OK] summary json -> {summary_json}")
PY

done

echo "[DONE] TM example reproduction finished."
echo "       runs_root = ${RUNS_ROOT}"
echo "       out_dir   = ${OUT_DIR}"
echo "       manifest  = ${MANIFEST_JSONL}"