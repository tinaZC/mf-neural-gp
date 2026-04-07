#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sweep_mf_baseline_hf_delta_ar1.py  (v2)

Fix:
- Do NOT pass --no_subdir unless the training script supports it.
  (Your mf_baseline_hf_delta_ar1.py does not have this flag.)

Design:
- Thin wrapper: iterates dataset dirs, runs training script via subprocess, aggregates report.json.
- One run directory per (dataset, seed): out_root/<dataset_name>/seed<seed>/
  We pass --out_dir to that run_dir. If the training script still creates subfolders internally,
  report.json is searched under run_dir recursively as fallback.

Outputs:
- out_root/sweep_results.csv
- out_root/sweep_results.jsonl
"""

from __future__ import annotations

import os
import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _norm_path(p: str) -> str:
    p = (p or "").strip()
    if os.name != "nt":
        p = p.replace("\\", "/")
    return os.path.normpath(os.path.expanduser(p))


def is_dataset_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    if not (p / "wavelengths.npy").exists():
        return False
    for sub in ("hf", "lf_paired", "lf_unpaired"):
        if not (p / sub).is_dir():
            return False
    return True


def find_dataset_dirs(data_root: Path) -> List[Path]:
    out: List[Path] = []
    for child in sorted(data_root.iterdir()):
        if is_dataset_dir(child):
            out.append(child)
    if out:
        return out
    for p in sorted(data_root.rglob("*")):
        if is_dataset_dir(p):
            out.append(p)
    return out


def load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        kk = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, kk, sep=sep))
        else:
            out[kk] = v
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    # NOTE: Keep these keys aligned with mf_baseline_hf_delta_ar1_aligned.py's report.json.
    # Also include runner-level sweep settings (args.*) so the CSV is self-describing.
    preferred = [
        "dataset_name", "dataset_dir", "seed", "status", "runtime_sec", "run_dir",
        "args.wl_low", "args.wl_high", "args.fpca_var_ratio", "args.svgp_M", "args.svgp_steps", "args.gp_ard",
        "metrics.y_rmse.hf_only", "metrics.y_rmse.ar1", "metrics.y_rmse.ours",
        "metrics.uq.alpha.hf_only", "metrics.uq.alpha.ar1", "metrics.uq.alpha.ours",
        "metrics.uq.test.coverage_cal.hf_only", "metrics.uq.test.coverage_cal.ar1", "metrics.uq.test.coverage_cal.ours",
    ]
    pref_set = set(preferred)
    keys = preferred + [k for k in keys if k not in pref_set]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


@dataclass
class RunResult:
    row: Dict[str, Any]
    ok: bool


def _append_if_missing(extra_args: List[str], flag: str, value: str) -> None:
    if flag in extra_args:
        return
    extra_args += [flag, value]


def script_supports_flag(python_bin: str, script: Path, flag: str) -> bool:
    try:
        proc = subprocess.run(
            [python_bin, str(script), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
            check=False,
        )
        out = proc.stdout or ""
        return flag in out
    except Exception:
        return False


def find_report_json(run_dir: Path) -> Optional[Path]:
    p0 = run_dir / "report.json"
    if p0.exists():
        return p0
    cands = list(run_dir.rglob("report.json"))
    if not cands:
        return None
    # Prefer the *shallowest* report.json under run_dir, then lexicographic.
    cands.sort(key=lambda p: (len(p.relative_to(run_dir).parts), str(p)))
    return cands[0]


def run_one(
    *,
    dataset_dir: Path,
    dataset_name: str,
    out_root: Path,
    train_script: Path,
    python_bin: str,
    seed: int,
    sweep_args_flat: Dict[str, Any],
    extra_args: List[str],
    skip_if_done: bool,
    timeout_sec: Optional[int],
    use_no_subdir_if_available: bool,
) -> RunResult:
    run_dir = out_root / dataset_name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = find_report_json(run_dir)
    log_path = run_dir / "run.log"

    base_row: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "seed": int(seed),
        "run_dir": str(run_dir),
        **(sweep_args_flat or {}),
    }

    if skip_if_done and report_path is not None and report_path.exists():
        rep = load_json(report_path) or {}
        flat = flatten_dict(rep)
        # NOTE: Do NOT auto-merge config.json.
        # Some training scripts (including mf_baseline_hf_delta_ar1_aligned.py) don't emit it,
        # and the user requested avoiding config.* / args.* inference when it's absent.
        row = {**base_row, "status": "SKIP_DONE", "runtime_sec": 0.0, "report_path": str(report_path)}
        row.update(flat)
        return RunResult(row=row, ok=True)

    cmd = [
        python_bin,
        str(train_script),
        "--data_dir", str(dataset_dir),
        "--out_dir", str(run_dir),
        "--seed", str(seed),
    ]

    if use_no_subdir_if_available:
        if script_supports_flag(python_bin, train_script, "--no_subdir"):
            cmd += ["--no_subdir", "1"]

    cmd += extra_args

    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write("[CMD] " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
            logf.flush()
            proc = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                cwd=str(run_dir),
                timeout=timeout_sec,
                check=False,
            )
        dt = time.time() - t0

        if proc.returncode != 0:
            row = {**base_row, "status": f"FAIL_{proc.returncode}", "runtime_sec": float(dt)}
            return RunResult(row=row, ok=False)

        report_path = find_report_json(run_dir)
        if report_path is None:
            row = {**base_row, "status": "FAIL_NO_REPORT", "runtime_sec": float(dt)}
            return RunResult(row=row, ok=False)

        rep = load_json(report_path)
        if rep is None:
            row = {**base_row, "status": "FAIL_BAD_REPORT", "runtime_sec": float(dt), "report_path": str(report_path)}
            return RunResult(row=row, ok=False)

        flat = flatten_dict(rep)
        # NOTE: Do NOT auto-merge config.json (see comment above).
        row = {**base_row, "status": "OK", "runtime_sec": float(dt), "report_path": str(report_path)}
        row.update(flat)
        return RunResult(row=row, ok=True)

    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        row = {**base_row, "status": "TIMEOUT", "runtime_sec": float(dt)}
        return RunResult(row=row, ok=False)

    except Exception as e:
        dt = time.time() - t0
        row = {**base_row, "status": f"ERROR:{type(e).__name__}", "runtime_sec": float(dt), "error": str(e)}
        return RunResult(row=row, ok=False)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default="../../data/mf_sweep_datasets_nano_tm")
    ap.add_argument("--out_root", type=str, default="../../result_out/mf_sweep_runs_baseline_nano_tm")
    ap.add_argument("--train_script", type=str, default="./mf_baseline_tm.py")
    ap.add_argument("--python_bin", type=str, default=sys.executable)
    # ap.add_argument("--seeds", type=str, default="200,210,220,230,240,250,260,270,280,290,300,311,322,333,344,355,366,377,388,399")
    # ap.add_argument("--seeds", type=str, default="333,344,355,366,377,388,399")
    ap.add_argument("--seeds", type=str, default="333")
    ap.add_argument("--wl_low", type=float, default=380.0)
    ap.add_argument("--wl_high", type=float, default=750.0)
    ap.add_argument("--fpca_var_ratio", type=float, default=0.999)
    ap.add_argument("--svgp_M", type=int, default=64)
    ap.add_argument("--svgp_steps", type=int, default=2000)
    ap.add_argument("--gp_ard", type=int, default=1, choices=[0, 1])

    ap.add_argument("--extra_args", type=str, default="")
    ap.add_argument("--timeout_sec", type=int, default=0)
    ap.add_argument("--skip_if_done", type=int, default=1, choices=[0, 1])
    ap.add_argument("--try_no_subdir", type=int, default=0, choices=[0, 1])

    args = ap.parse_args()

    seeds: List[int] = []
    for s in (args.seeds or "").split(","):
        s = s.strip()
        if s:
            seeds.append(int(s))
    if not seeds:
        seeds = [42]

    data_root = Path(_norm_path(args.data_root)).expanduser().resolve()
    out_root = Path(_norm_path(args.out_root)).expanduser().resolve()
    train_script = Path(_norm_path(args.train_script)).expanduser().resolve()

    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] out_root={out_root}")
    print(f"[INFO] train_script={train_script}")

    if not data_root.exists():
        raise SystemExit(f"data_root not found: {data_root}")
    if not train_script.exists():
        raise SystemExit(f"train_script not found: {train_script}")

    out_root.mkdir(parents=True, exist_ok=True)

    dataset_dirs = find_dataset_dirs(data_root)
    if not dataset_dirs:
        raise SystemExit(f"No dataset dirs found under: {data_root}")

    extra_args = shlex.split(args.extra_args.strip()) if args.extra_args.strip() else []

    _append_if_missing(extra_args, "--wl_low", str(float(args.wl_low)))
    _append_if_missing(extra_args, "--wl_high", str(float(args.wl_high)))
    _append_if_missing(extra_args, "--fpca_var_ratio", str(float(args.fpca_var_ratio)))
    _append_if_missing(extra_args, "--svgp_M", str(int(args.svgp_M)))
    _append_if_missing(extra_args, "--svgp_steps", str(int(args.svgp_steps)))
    _append_if_missing(extra_args, "--gp_ard", str(int(args.gp_ard)))

    # Record sweep-level args in every row (even if the training report doesn't include them).
    sweep_args_flat: Dict[str, Any] = {
        "args.wl_low": float(args.wl_low),
        "args.wl_high": float(args.wl_high),
        "args.fpca_var_ratio": float(args.fpca_var_ratio),
        "args.svgp_M": int(args.svgp_M),
        "args.svgp_steps": int(args.svgp_steps),
        "args.gp_ard": int(args.gp_ard),
    }

    timeout_sec = int(args.timeout_sec) if int(args.timeout_sec) > 0 else None
    skip_if_done = bool(int(args.skip_if_done))
    try_no_subdir = bool(int(args.try_no_subdir))

    all_rows: List[Dict[str, Any]] = []

    print(f"[SWEEP] found {len(dataset_dirs)} datasets under {data_root}")
    print(f"[SWEEP] seeds={seeds}")
    if extra_args:
        print(f"[SWEEP] extra_args: {' '.join(extra_args)}")

    n_ok = 0
    n_fail = 0

    # Loop order: seed outer, dataset inner.
    # This guarantees that for any prefix of seeds completed, you have a full dataset sweep for each completed seed.
    for si, seed in enumerate(seeds, start=1):
        print(f"\n[SWEEP][SEED] seed={seed} ({si}/{len(seeds)})")
        for d in dataset_dirs:
            # Make dataset_name stable and collision-free under a sweep root.
            # Example: hf50_lfx30/ds_001  ->  hf50_lfx30__ds_001
            rel = d.relative_to(data_root)
            dataset_name = "__".join(rel.parts)
            print(f"\n[RUN] seed={seed} dataset={dataset_name}")
            rr = run_one(
                dataset_dir=d,
                dataset_name=dataset_name,
                out_root=out_root,
                train_script=train_script,
                python_bin=str(args.python_bin),
                seed=int(seed),
                sweep_args_flat=sweep_args_flat,
                extra_args=extra_args,
                skip_if_done=skip_if_done,
                timeout_sec=timeout_sec,
                use_no_subdir_if_available=try_no_subdir,
            )
            all_rows.append(rr.row)
            if rr.ok:
                n_ok += 1
                print(f"[OK]   -> {rr.row.get('run_dir')}")
            else:
                n_fail += 1
                print(f"[FAIL] -> {rr.row.get('run_dir')} status={rr.row.get('status')}")
                print(f"       see log: {Path(rr.row.get('run_dir')) / 'run.log'}")

        # Save partial results after each seed so interrupted runs still produce complete-per-seed stats.
        csv_path = out_root / "sweep_results.csv"
        jsonl_path = out_root / "sweep_results.jsonl"
        write_csv(csv_path, all_rows)
        write_jsonl(jsonl_path, all_rows)
        print(f"[SWEEP][SEED_DONE] seed={seed} | OK={n_ok} FAIL={n_fail} | saved partial: {csv_path.name}, {jsonl_path.name}")

    csv_path = out_root / "sweep_results.csv"
    jsonl_path = out_root / "sweep_results.jsonl"
    write_csv(csv_path, all_rows)
    write_jsonl(jsonl_path, all_rows)

    print("\n[SWEEP][DONE]")
    print(f"[SWEEP] OK={n_ok} FAIL={n_fail}")
    print(f"[SWEEP] saved: {csv_path}")
    print(f"[SWEEP] saved: {jsonl_path}")


if __name__ == "__main__":
    main()
