#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_m013_baseline_multi_seed.py

Batch runner for m013 baseline experiments.

What it does
------------
1) Runs mf_baseline_hf_delta_ar1_aligned_lfprob_microwave_m013.py over many seeds.
2) Saves outputs under a directory layout that plot_m013_baseline_compare.py can
   parse directly from paths:

   <out_root>/hf50_lfx10/
       seed1/<run_name>/report.json
       seed2/<run_name>/report.json
       ...
       plot_m013_compare/

3) Optionally calls plot_m013_baseline_compare.py after all seeds finish.

Example
-------
python run_m013_baseline_multi_seed.py \
  --data_dir /mnt/sdb/tzc/mf_dnn_sgp/mtm_013/mf_dataset_mw_m013/hf50_lfx10 \
  --out_root /mnt/sdb/tzc/mf_dnn_sgp/mtm_013/mf_baseline_out_microwave_m013_multi \
  --seeds 1-20 \
  --device cuda
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


HF_LFX_RE = re.compile(r"hf(\d+)_lfx(\d+)", re.IGNORECASE)


def parse_hf_lfx(text: str) -> Tuple[int, int, str]:
    m = HF_LFX_RE.search(text)
    if not m:
        raise ValueError(f"Cannot parse hf/lfx from path: {text}")
    hf = int(m.group(1))
    lfx = int(m.group(2))
    return hf, lfx, f"hf{hf}_lfx{lfx}"


def parse_seed_spec(seed_spec: str) -> List[int]:
    seed_spec = str(seed_spec).strip()
    if not seed_spec:
        return [42]

    out: List[int] = []
    for part in seed_spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            a = int(a.strip())
            b = int(b.strip())
            step = 1 if b >= a else -1
            out.extend(list(range(a, b + step, step)))
        else:
            out.append(int(part))

    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    if not uniq:
        raise ValueError(f"No valid seeds parsed from: {seed_spec}")
    return uniq


def run_cmd(cmd: List[str], cwd: Path | None = None) -> int:
    print("[RUN]", shlex.join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return int(proc.returncode)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline_script",
        type=str,
        default=str(Path(__file__).with_name("mf_baseline_microwave_mtm.py")),
        help="Path to mf_baseline_microwave_mtm.py",
    )
    ap.add_argument(
        "--plot_script",
        type=str,
        default=str(Path(__file__).with_name("plot_baseline_mtm_compare.py")),
        help="Path to plot_baseline_compare.py",
    )
    ap.add_argument(
        "--python_exe",
        type=str,
        default=sys.executable,
        help="Python interpreter used to run child scripts.",
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default="../../data/mf_dataset_mw_mtm/hf50_lfx10",
        help="Dataset directory. Must contain hfXX_lfxYY in its path.",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="../../result_out/mf_baseline_out_microwave_mtm_multi",
        help="Root output directory for all seeds.",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="1-20",
        help="Seed spec, e.g. '1-20' or '1,2,5,8'.",
    )
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--methods", type=str, default="hf_only,ar1,ours")
    ap.add_argument("--delegate_ours_to_train", type=int, default=1, choices=[0, 1])
    ap.add_argument("--lf_prob", type=int, default=0, choices=[0, 1])
    ap.add_argument("--n_plot", type=int, default=10)
    ap.add_argument("--plot_ci", type=int, default=1, choices=[0, 1])
    ap.add_argument("--save_pred_arrays", type=int, default=1, choices=[0, 1])
    ap.add_argument("--save_uq_cache", type=int, default=1, choices=[0, 1])
    ap.add_argument("--ci_calibrate", type=int, default=0, choices=[0, 1])
    ap.add_argument(
        "--continue_on_error",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, continue remaining seeds even if one seed fails.",
    )
    ap.add_argument(
        "--call_plot",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, call plot_m013_baseline_compare.py after all runs.",
    )
    ap.add_argument(
        "--extra_args",
        type=str,
        default="",
        help="Extra raw CLI args appended to baseline script, e.g. '--svgp_steps 3000 --student_epochs 3000'",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()

    baseline_script = Path(args.baseline_script).expanduser().resolve()
    plot_script = Path(args.plot_script).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if not baseline_script.exists():
        raise FileNotFoundError(f"Baseline script not found: {baseline_script}")
    if not plot_script.exists() and int(args.call_plot) == 1:
        raise FileNotFoundError(f"Plot script not found: {plot_script}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    hf, lfx, dataset_tag = parse_hf_lfx(str(data_dir))
    seeds = parse_seed_spec(args.seeds)

    dataset_root = out_root / dataset_tag
    dataset_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] baseline_script={baseline_script}")
    print(f"[INFO] plot_script={plot_script}")
    print(f"[INFO] data_dir={data_dir}")
    print(f"[INFO] dataset_tag={dataset_tag}")
    print(f"[INFO] seeds={seeds}")
    print(f"[INFO] dataset_root={dataset_root}")

    extra_args = shlex.split(args.extra_args) if str(args.extra_args).strip() else []

    failed: List[Tuple[int, int]] = []
    for seed in seeds:
        seed_root = dataset_root / f"seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python_exe,
            str(baseline_script),
            "--data_dir", str(data_dir),
            "--out_dir", str(seed_root),
            "--seed", str(int(seed)),
            "--device", str(args.device),
            "--methods", str(args.methods),
            "--delegate_ours_to_train", str(int(args.delegate_ours_to_train)),
            "--lf_prob", str(int(args.lf_prob)),
            "--n_plot", str(int(args.n_plot)),
            "--plot_ci", str(int(args.plot_ci)),
            "--save_pred_arrays", str(int(args.save_pred_arrays)),
            "--save_uq_cache", str(int(args.save_uq_cache)),
            "--ci_calibrate", str(int(args.ci_calibrate)),
        ] + extra_args

        ret = run_cmd(cmd)
        if ret != 0:
            failed.append((seed, ret))
            print(f"[FAIL] seed={seed} returncode={ret}")
            if int(args.continue_on_error) != 1:
                raise SystemExit(ret)
        else:
            print(f"[OK] seed={seed}")

    if failed:
        print("[WARN] Some seeds failed:")
        for s, rc in failed:
            print(f"  seed={s} rc={rc}")

    if int(args.call_plot) == 1:
        plot_out = dataset_root / "plot_mtm_compare"
        plot_out.mkdir(parents=True, exist_ok=True)
        title = f"m013 baseline comparison ({dataset_tag}, {len(seeds) - len(failed)} seeds)"
        cmd_plot = [
            args.python_exe,
            str(plot_script),
            "--runs_root", str(dataset_root),
            "--out_dir", str(plot_out),
            "--hf", str(int(hf)),
            "--lfx", str(int(lfx)),
            "--title", title,
        ]
        ret = run_cmd(cmd_plot)
        if ret != 0:
            raise SystemExit(ret)
        print(f"[DONE] plot saved under {plot_out}")

    print("[DONE] multi-seed baseline run finished.")


if __name__ == "__main__":
    main()
