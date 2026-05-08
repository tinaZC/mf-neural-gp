#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build timing tables from separate timing.json files.

Usage:
  python make_timing_table.py \
    --runs_root ../result_out_v2_timing \
    --out_dir ../result_out_v2_timing/timing_tables

This script does not read report.json or results.csv. It only scans **/timing.json,
so existing result files remain untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _f(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _dataset_from_path(s: str) -> str:
    t = s.lower()
    if "microwave" in t or "mtm" in t or "mw" in t:
        return "MTM"
    if "abs" in t or "absorption" in t or "nano_ab" in t:
        return "AB"
    if "tm" in t or "tsmt" in t or "trans" in t or "nano" in t:
        return "TM"
    return "unknown"


def _setting_from_path(s: str) -> str:
    p = Path(s)
    for part in reversed(p.parts):
        if part.startswith("hf") and "lfx" in part:
            return part
    return ""


def _load_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] failed to read {p}: {e}")
        return {}


def _row_base(timing_path: Path, timing: Dict[str, Any]) -> Dict[str, Any]:
    data_dir = str(timing.get("data_dir") or "")
    out_dir = str(timing.get("out_dir") or timing_path.parent)
    dataset = _dataset_from_path(data_dir + " " + out_dir + " " + str(timing_path))
    setting = _setting_from_path(data_dir) or _setting_from_path(out_dir) or _setting_from_path(str(timing_path))
    return {
        "dataset": dataset,
        "setting": setting,
        "run_name": timing.get("run_name") or timing_path.parent.name,
        "seed": timing.get("seed", ""),
        "device": timing.get("device", ""),
        "script": timing.get("script", ""),
        "timing_path": str(timing_path),
        "n_hf_train": timing.get("n_hf_train", ""),
        "n_lf_train_total": timing.get("n_lf_train_total", ""),
        "total_model_wall_s": _f(timing.get("total_model_wall_s")),
    }


def _add_row(
    rows: List[Dict[str, Any]],
    base: Dict[str, Any],
    method: str,
    stage1: Optional[float],
    stage2: Optional[float],
    infer: Optional[float],
    note: str = "",
) -> None:
    if stage1 is None and stage2 is None and infer is None:
        return
    parts = [v for v in (stage1, stage2, infer) if v is not None]
    overhead = float(sum(parts)) if parts else None
    row = dict(base)
    row.update({
        "method": method,
        "stage1_train_s": stage1,
        "stage2_train_s": stage2,
        "inference_val_test_s": infer,
        "model_overhead_components_s": overhead,
        "note": note,
    })
    rows.append(row)


def parse_timing_file(timing_path: Path, all_timing_paths: set[str]) -> List[Dict[str, Any]]:
    timing = _load_json(timing_path)
    if not timing:
        return []
    base = _row_base(timing_path, timing)
    stage2 = timing.get("stage2_train_s") or {}
    infer = timing.get("inference_s") or {}
    script = str(timing.get("script") or timing_path.parent.name).lower()
    rows: List[Dict[str, Any]] = []

    if "baseline" in script:
        _add_row(rows, base, "HF-only", None, _f(stage2.get("hf_only")), _f(infer.get("hf_only_val_test")))
        _add_row(rows, base, "AR1 / co-kriging", None, _f(stage2.get("ar1_total")), _f(infer.get("ar1_val_test")))

        nested_path = str(timing_path.parent / "ours" / "timing.json")
        if nested_path not in all_timing_paths:
            _add_row(
                rows,
                base,
                "Neural-GP MF (delegated total)",
                None,
                _f(stage2.get("ours_delegate_total")),
                _f(infer.get("ours_val_test")),
                note="Fallback row: detailed ours/timing.json was not found.",
            )
        return rows

    _add_row(
        rows,
        base,
        "Neural-GP MF",
        _f(timing.get("stage1_train_s")),
        _f(stage2.get("mf_student")),
        _f(infer.get("mf_student_val_test")),
    )
    _add_row(rows, base, "HF-only", None, _f(stage2.get("hf_only")), _f(infer.get("hf_only_val_test")))
    _add_row(
        rows,
        base,
        "MF-oracle",
        _f(timing.get("stage1_train_s")),
        _f(stage2.get("mf_oracle")),
        _f(infer.get("mf_oracle_val_test")),
    )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def mean_std(vals: List[Optional[float]]) -> tuple[Optional[float], Optional[float], int]:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    n = len(clean)
    if n == 0:
        return None, None, 0
    m = sum(clean) / n
    if n == 1:
        return m, 0.0, 1
    var = sum((v - m) ** 2 for v in clean) / (n - 1)
    return m, math.sqrt(var), n


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r.get("dataset", ""), r.get("setting", ""), r.get("method", ""))].append(r)

    out = []
    metrics = [
        "stage1_train_s",
        "stage2_train_s",
        "inference_val_test_s",
        "model_overhead_components_s",
        "total_model_wall_s",
    ]
    for (dataset, setting, method), rs in sorted(groups.items()):
        row = {"dataset": dataset, "setting": setting, "method": method, "n_runs": len(rs)}
        for m in metrics:
            mean, std, n = mean_std([_f(r.get(m)) for r in rs])
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_n"] = n
        row["n_hf_train"] = next((r.get("n_hf_train") for r in rs if r.get("n_hf_train") not in (None, "")), "")
        row["n_lf_train_total"] = next((r.get("n_lf_train_total") for r in rs if r.get("n_lf_train_total") not in (None, "")), "")
        out.append(row)
    return out


def _fmt(x: Any) -> str:
    v = _f(x)
    if v is None:
        return ""
    return f"{v:.3f}"


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "Dataset", "Setting", "Method", "Runs", "N_HF", "N_LF",
        "Stage I train (s)", "Stage II train (s)", "Inference val+test (s)", "Component overhead (s)",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        vals = [
            r.get("dataset", ""),
            r.get("setting", ""),
            r.get("method", ""),
            str(r.get("n_runs", "")),
            str(r.get("n_hf_train", "")),
            str(r.get("n_lf_train_total", "")),
            _fmt(r.get("stage1_train_s_mean")),
            _fmt(r.get("stage2_train_s_mean")),
            _fmt(r.get("inference_val_test_s_mean")),
            _fmt(r.get("model_overhead_components_s_mean")),
        ]
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, required=True, help="Root directory containing timing.json files.")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory. Default: <runs_root>/timing_tables")
    args = ap.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else runs_root / "timing_tables"
    timing_paths = sorted(runs_root.rglob("timing.json"))
    all_path_set = {str(p) for p in timing_paths}

    rows: List[Dict[str, Any]] = []
    for p in timing_paths:
        rows.extend(parse_timing_file(p, all_path_set))

    if not rows:
        raise SystemExit(f"No timing rows found under {runs_root}. Did you rerun after adding separate timing.json output?")

    long_fields = [
        "dataset", "setting", "method", "run_name", "seed", "device", "script",
        "n_hf_train", "n_lf_train_total",
        "stage1_train_s", "stage2_train_s", "inference_val_test_s", "model_overhead_components_s",
        "total_model_wall_s", "timing_path", "note",
    ]
    write_csv(out_dir / "timing_table_long.csv", rows, long_fields)

    summary = summarize(rows)
    summary_fields = ["dataset", "setting", "method", "n_runs", "n_hf_train", "n_lf_train_total"]
    for m in ["stage1_train_s", "stage2_train_s", "inference_val_test_s", "model_overhead_components_s", "total_model_wall_s"]:
        summary_fields += [f"{m}_mean", f"{m}_std", f"{m}_n"]
    write_csv(out_dir / "timing_table_summary.csv", summary, summary_fields)
    write_markdown(out_dir / "timing_table_summary.md", summary)

    print(f"[OK] timing files: {len(timing_paths)}")
    print(f"[OK] timing rows: {len(rows)}")
    print(f"[OK] wrote: {out_dir / 'timing_table_long.csv'}")
    print(f"[OK] wrote: {out_dir / 'timing_table_summary.csv'}")
    print(f"[OK] wrote: {out_dir / 'timing_table_summary.md'}")


if __name__ == "__main__":
    main()
