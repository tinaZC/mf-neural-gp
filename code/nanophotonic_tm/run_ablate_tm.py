#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_ablate_delta_student_tm.py

Clean ablation runner for shared mf_train.py (TM).

What it does
------------
1) Runs a selected set of ablation variants (A0..A12) across one or more dataset roots and seeds.
2) Collects scalar metrics from each run's report.json.
3) Produces a summary folder with:
   (a) boxplot: RMSE over seeds for chosen ablations
   (b) spectrum RMSE curves: per-axis RMSE_k curve for each ablation (mf_student only)
   (c) residual diagnostic curves (optional): RMS_k(y_hf - base_hat) for delta-student

Notes
-----
- This script assumes the training script saves per-axis RMSE curves under:
    <run_out>/rmse_curves/axis_wavelength.npy or axis_frequency.npy or axis.npy
    <run_out>/rmse_curves/rmse_<split>__mf_student.npy

- Residual diagnostic curve is saved by training script as:
    <run_out>/rmse_curves/rmse_<split>__mf_student_residual.npy
  which equals RMS_k(y_hf - base_hat) for delta-student.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent                 # .../code/nanophotonic_tm
CODE_ROOT = SCRIPT_DIR.parent                                # .../code
PROJECT_ROOT = CODE_ROOT.parent                              # repo root


# -------------------------
# Ablation definitions
# -------------------------

@dataclass(frozen=True)
class Ablation:
    key: str
    tag: str
    desc: str
    overrides: Dict[str, Any]


def get_ablations() -> Dict[str, Ablation]:
    """Return all supported ablation variants."""

    # Base: mixfit Stage-I + xlf + delta + ARD
    BASE_BASE: Dict[str, Any] = {
        "mf_u_mode": "xlf",
        "student_mode": "delta",
        "gp_ard": 1,
        "skip_student": 0,
        "mf_student_lf_source": "student",
        "student_train_set": "mix",
        "student_val_set": "paired",
        "student_y_scaler_fit": "paired",
    }

    # Best Stage-I MLP config (A0)
    BASE_BEST: Dict[str, Any] = {
        **BASE_BASE,
        "student_hidden": [256, 256, 256],
        "student_feat_dim": 32,
        "student_epochs": 2000,
        "student_bs": 256,
        "student_patience": 100,
    }

    # Baseline Stage-I MLP config
    BASE_STAGE1_BASELINE: Dict[str, Any] = {
        **dict(BASE_BASE),
        "student_hidden": [256, 256],
        "student_feat_dim": 16,
        "student_epochs": 500,
        "student_bs": 64,
        "student_patience": 30,
    }

    return {
        # A0 (BEST): mixfit Stage-I + xlf + delta + ARD + best Stage-I MLP
        "A0": Ablation(
            key="A0",
            tag="A0_best_stage1_mlp_arch",
            desc="A0 (best): xlf + delta + ARD + Stage-I best MLP (hidden=[256,256,256], feat_dim=32, epochs=2000, bs=256, patience=100)",
            overrides=dict(BASE_BEST),
        ),

        # A1: student paired-only
        "A1": Ablation(
            key="A1",
            tag="A1_student_paired",
            desc="A1: Stage-I student paired-only (train/val/scaler=paired)",
            overrides={
                **BASE_BEST,
                "student_train_set": "paired",
                "student_val_set": "paired",
                "student_y_scaler_fit": "paired",
            },
        ),

        # A2: direct_no_delta (do not learn residual; directly learn HF)
        "A2": Ablation(
            key="A2",
            tag="A2_direct_no_delta",
            desc="A2: Stage-II learns HF directly (student_mode=direct); ARD=1",
            overrides={**BASE_BEST, "student_mode": "direct"},
        ),

        # A3: no_ARD
        "A3": Ablation(
            key="A3",
            tag="A3_no_ARD",
            desc="A3: disable ARD (gp_ard=0)",
            overrides={**BASE_BEST, "gp_ard": 0},
        ),

        # A4: no_x (Stage-II uses yl only; still delta residual)
        "A4": Ablation(
            key="A4",
            tag="A4_no_x",
            desc="A4: Stage-II u=yl only (mf_u_mode=lf); delta residual; ARD=1",
            overrides={**BASE_BEST, "mf_u_mode": "lf"},
        ),

        # A5: feature (Stage-II uses feature + yl, no x)
        "A5": Ablation(
            key="A5",
            tag="A5_feature_flf",
            desc="A5: Stage-II u=[feature,yl] (flf); delta residual; ARD=1",
            overrides={**BASE_BEST, "mf_u_mode": "flf"},
        ),

        # A6: rho_fit_source=student
        "A6": Ablation(
            key="A6",
            tag="A6_rho_fit_student",
            desc="A6: fit affine rho on LF_hat from student (rho_fit_source=student)",
            overrides={**BASE_BEST, "rho_fit_source": "student"},
        ),

        # A7: Stage-I baseline MLP
        "A7": Ablation(
            key="A7",
            tag="A7_stage1_mlp_baseline",
            desc="A7: Stage-I baseline MLP (hidden=[256,256], feat_dim=16, epochs=500, bs=64, patience=30); other settings same as A0 base (xlf+delta+ARD)",
            overrides=dict(BASE_STAGE1_BASELINE),
        ),

        # A8: Stage-II input = feature + x (no yl/lf as GP input)
        "A8": Ablation(
            key="A8",
            tag="A8_stage2_xf_no_yl",
            desc="A8: Stage-II u=[feature,x] (xf); no yl/lf in GP input; delta residual; ARD=1",
            overrides={**BASE_BEST, "mf_u_mode": "xf"},
        ),

        # A9: x|lf block kernel
        "A9": Ablation(
            key="A9",
            tag="A9_stage2_xlf_block_kernel",
            desc="A9: Stage-II kernel_struct=xlf_block (additive block kernel on u=[x|yl])",
            overrides={**BASE_BEST, "kernel_struct": "xlf_block"},
        ),

        # A10: HF-only baseline
        "A10": Ablation(
            key="A10",
            tag="A10_hf_only",
            desc="A10: HF-only baseline (primary method: hf_only)",
            overrides={},
        ),

        # A11: MF-oracle baseline
        "A11": Ablation(
            key="A11",
            tag="A11_mf_oracle",
            desc="A11: MF-oracle baseline (primary method: mf_oracle)",
            overrides={},
        ),

        # A12: oracle LF / no-MLP
        "A12": Ablation(
            key="A12",
            tag="A12_oracle_lf_no_mlp",
            desc="A12: skip Stage-I; LF_hat:=oracle (paired LF); delta residual; ARD=1 (oracle upper bound / diagnostic)",
            overrides={**BASE_BEST, "skip_student": 1, "mf_student_lf_source": "oracle"},
        ),
    }


# -------------------------
# Helpers
# -------------------------

def safe_name(s: str) -> str:
    return "".join([c if (c.isalnum() or c in "-_+.@") else "_" for c in s])


def run_cmd(cmd: List[str], dry_run: bool = False) -> int:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return 0
    p = subprocess.run(cmd, check=False)
    return int(p.returncode)


def find_latest_report_dir(seed_run_dir: Path) -> Path:
    """
    Training script may create a config subdir under seed_run_dir and write report.json there.
    Return the directory that contains the latest report.json.
    """
    direct = seed_run_dir / "report.json"
    if direct.exists():
        return seed_run_dir

    cands = list(seed_run_dir.glob("**/report.json"))
    if not cands:
        return seed_run_dir

    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0].parent


def read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rmse_curve(run_dir: Path, split: str = "test") -> Tuple[np.ndarray, np.ndarray]:
    rmse_dir = run_dir / "rmse_curves"
    axis_candidates = [
        rmse_dir / "axis_wavelength.npy",
        rmse_dir / "axis_frequency.npy",
        rmse_dir / "axis.npy",
    ]
    axis_p = next((p for p in axis_candidates if p.exists()), None)
    y_p = rmse_dir / f"rmse_{split}__mf_student.npy"

    if axis_p is None or (not y_p.exists()):
        raise FileNotFoundError(
            f"Missing RMSE curve files under {rmse_dir} for split={split}. "
            f"axis tried={[p.name for p in axis_candidates]}, rmse tried={y_p.name}"
        )
    axis = np.load(axis_p)
    y = np.load(y_p)
    return axis.astype(np.float32), y.astype(np.float32)


def load_residual_curve(run_dir: Path, split: str = "test") -> Tuple[np.ndarray, np.ndarray]:
    rmse_dir = run_dir / "rmse_curves"
    axis_candidates = [
        rmse_dir / "axis_wavelength.npy",
        rmse_dir / "axis_frequency.npy",
        rmse_dir / "axis.npy",
    ]
    axis_p = next((p for p in axis_candidates if p.exists()), None)
    y_p = rmse_dir / f"rmse_{split}__mf_student_residual.npy"

    if axis_p is None or (not y_p.exists()):
        raise FileNotFoundError(f"Missing residual curve files under {rmse_dir} for split={split}")
    axis = np.load(axis_p)
    y = np.load(y_p)
    return axis.astype(np.float32), y.astype(np.float32)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Plotting
# -------------------------

def plot_ablation_boxplot(
    out_png: Path,
    tags: List[str],
    values: List[List[float]],
    title: str,
    ylabel: str = "RMSE",
) -> None:
    plt.figure(figsize=(max(7.0, 1.2 * len(tags)), 4.6))
    ax = plt.gca()

    bp = ax.boxplot(
        values,
        labels=tags,
        showfliers=True,
        patch_artist=True,
        medianprops=dict(linewidth=2.0),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        boxprops=dict(linewidth=1.8),
        flierprops=dict(markersize=4, alpha=0.6),
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    for i in range(len(tags)):
        c = colors[i % len(colors)]
        box = bp["boxes"][i]
        box.set_facecolor(c)
        box.set_alpha(0.18)
        box.set_edgecolor(c)
        box.set_linewidth(2.0)

        med = bp["medians"][i]
        med.set_color(c)
        med.set_linewidth(2.2)

        for w in bp["whiskers"][2 * i: 2 * i + 2]:
            w.set_color(c)
            w.set_linewidth(1.6)
        for cap in bp["caps"][2 * i: 2 * i + 2]:
            cap.set_color(c)
            cap.set_linewidth(1.6)

        if "fliers" in bp and i < len(bp["fliers"]):
            bp["fliers"][i].set_markeredgecolor(c)
            bp["fliers"][i].set_markerfacecolor(c)
            bp["fliers"][i].set_alpha(0.6)

    all_vals = np.array([x for vv in values for x in vv], dtype=float)
    if all_vals.size > 0:
        y_min = float(np.min(all_vals))
        y_max = float(np.max(all_vals))
        y_rng = max(y_max - y_min, 1e-12)
        ax.set_ylim(y_min - 0.05 * y_rng, y_max + 0.18 * y_rng)
    else:
        y_rng = 1.0

    rng = np.random.default_rng(0)
    for i, v in enumerate(values, start=1):
        if len(v) == 0:
            continue
        c = colors[(i - 1) % len(colors)]
        v_arr = np.asarray(v, dtype=float)
        x = rng.normal(loc=i, scale=0.04, size=v_arr.size)
        ax.scatter(x, v_arr, s=14, alpha=0.55, linewidths=0, color=c)

    for i, v in enumerate(values, start=1):
        if len(v) == 0:
            continue
        v_arr = np.asarray(v, dtype=float)
        mu = float(np.mean(v_arr))
        sd = float(np.std(v_arr))
        q3 = float(np.quantile(v_arr, 0.75))
        y_off = 0.03 * y_rng
        ax.text(i, q3 + y_off, f"{mu:.2e}±{sd:.2e}",
                ha="center", va="bottom", fontsize=11)

    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title(title)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_spectrum_curves(
    out_png: Path,
    axis: np.ndarray,
    curves_mean: Dict[str, np.ndarray],
    curves_std: Optional[Dict[str, np.ndarray]],
    title: str,
    xlabel: str = "wavelength / frequency",
    ylabel: str = "RMSE",
    label_map: Optional[Dict[str, str]] = None,
) -> None:
    plt.figure(figsize=(7.2, 4.4))
    for tag, y_mean in curves_mean.items():
        lab = label_map.get(tag, tag) if label_map else tag
        line, = plt.plot(axis, y_mean, label=lab)
        if curves_std is not None and tag in curves_std:
            y_std = curves_std[tag]
            c = line.get_color()
            plt.fill_between(axis, y_mean - y_std, y_mean + y_std, alpha=0.18, color=c)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--python", type=str, default=os.environ.get("PYTHON_BIN", "python3"))
    ap.add_argument(
        "--train_script",
        type=str,
        default=str(CODE_ROOT / "mf_train_baseline" / "mf_train.py"),
        help="Path to shared mf_train.py",
    )
    ap.add_argument(
        "--run_prefix",
        type=str,
        default="tm0",
        help="Run prefix passed to shared mf_train.py",
    )
    ap.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        default=[str(PROJECT_ROOT / "data" / "mf_sweep_datasets_nano_tm" / "hf100_lfx10")],
        help="One or more dataset roots (each must match training script layout).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(PROJECT_ROOT / "result_out" / "ablate_runs_tm"),
        help="Root output directory for all runs.",
    )
    ap.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Experiment name subfolder under out_dir.",
    )

    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 33, 55, 66, 77, 8, 9, 11, 22, 88, 99, 111, 222, 333, 555],
    )

    ap.add_argument(
        "--ablations",
        type=str,
        nargs="*",
        default=["A0", "A1", "A2", "A3"],
        help="Ablations to run (any subset of: A0..A12).",
    )

    ap.add_argument(
        "--rmse_curve_split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split's RMSE_k curve to use in spectrum plots.",
    )

    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_train", action="store_true", help="Skip training runs and only summarize existing folders.")
    ap.add_argument("--skip_plots", action="store_true")

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--wl_low", type=float, default=380.0)
    ap.add_argument("--wl_high", type=float, default=750.0)
    ap.add_argument("--dim_reduce", type=str, default="fpca", choices=["fpca", "subsample"])
    ap.add_argument("--fpca_var_ratio", type=float, default=0.999)
    ap.add_argument("--fpca_max_dim", type=int, default=50)
    ap.add_argument("--subsample_K", type=int, default=1)

    ap.add_argument("--kernel_struct", type=str, default="block", choices=["full", "block", "xlf_block"])
    ap.add_argument("--kernel", type=str, default="matern", choices=["rbf", "matern"])
    ap.add_argument("--matern_nu", type=float, default=2.5)
    ap.add_argument("--svgp_M", type=int, default=64, help="SVGP inducing points")
    ap.add_argument("--svgp_steps", type=int, default=2000, help="SVGP training steps")

    ap.add_argument("--ci_level", type=float, default=0.95)
    ap.add_argument("--ci_calibrate", type=int, default=1, choices=[0, 1])

    ap.add_argument(
        "--extra_args",
        type=str,
        nargs="*",
        default=[],
        help="Extra args passed directly to the training script.",
    )

    return ap.parse_args()


# -------------------------
# Main
# -------------------------

def main() -> None:
    args = parse_args()

    all_defs = get_ablations()
    chosen_keys = [k.upper() for k in args.ablations]
    for k in chosen_keys:
        if k not in all_defs:
            raise ValueError(f"Unknown ablation '{k}'. Supported: {sorted(all_defs.keys())}")

    out_root = Path(args.out_dir) / safe_name(args.exp_name)
    ensure_dir(out_root)

    run_records: List[Dict[str, Any]] = []

    for seed in args.seeds:
        for data_dir in [Path(d).resolve() for d in args.data_dirs]:
            ds_tag = safe_name(data_dir.name)
            for k in chosen_keys:
                ab = all_defs[k]

                run_dir = out_root / ds_tag / ab.tag / f"seed{seed}"
                ensure_dir(run_dir)

                kernel_struct_run = str(args.kernel_struct)
                mf_u_mode_run = str(ab.overrides.get("mf_u_mode", ""))
                if mf_u_mode_run == "flf":
                    kernel_struct_run = "block"

                if not args.skip_train:
                    cmd = [
                        args.python,
                        args.train_script,
                        "--data_dir", str(data_dir),
                        "--out_dir", str(run_dir),
                        "--run_prefix", str(args.run_prefix),
                        "--seed", str(seed),
                        "--device", str(args.device),
                        "--wl_low", str(args.wl_low),
                        "--wl_high", str(args.wl_high),
                        "--dim_reduce", str(args.dim_reduce),
                        "--fpca_var_ratio", str(args.fpca_var_ratio),
                        "--fpca_max_dim", str(args.fpca_max_dim),
                        "--subsample_K", str(args.subsample_K),
                        "--kernel_struct", str(kernel_struct_run),
                        "--kernel", str(args.kernel),
                        "--matern_nu", str(args.matern_nu),
                        "--svgp_M", str(args.svgp_M),
                        "--svgp_steps", str(args.svgp_steps),
                        "--ci_level", str(args.ci_level),
                        "--ci_calibrate", str(args.ci_calibrate),
                    ]

                    run_hf_only = 0
                    run_oracle = 0
                    run_student = 1
                    if ab.key == "A10":
                        run_hf_only = 1
                        run_oracle = 0
                        run_student = 0
                    elif ab.key == "A11":
                        run_hf_only = 0
                        run_oracle = 1
                        run_student = 0
                    cmd += [
                        "--run_hf_only", str(run_hf_only),
                        "--run_oracle", str(run_oracle),
                        "--run_student", str(run_student),
                    ]

                    for kk, vv in ab.overrides.items():
                        flag = f"--{kk}"
                        if isinstance(vv, (list, tuple)):
                            cmd.append(flag)
                            cmd.extend([str(x) for x in vv])
                        else:
                            cmd.extend([flag, str(vv)])

                    cmd += list(args.extra_args)

                    rc = run_cmd(cmd, dry_run=args.dry_run)
                    if rc != 0:
                        raise RuntimeError(f"Training failed: dataset={data_dir} seed={seed} ablation={ab.key} rc={rc}")

                effective_run_dir = find_latest_report_dir(run_dir)
                report_p = effective_run_dir / "report.json"
                if not report_p.exists():
                    print(f"[WARN] Missing report.json under: {run_dir}")
                    continue

                rep = read_json(report_p)

                primary_method = "mf_student"
                if ab.key == "A10":
                    primary_method = "hf_only"
                elif ab.key == "A11":
                    primary_method = "mf_oracle"

                y_rmse_test = float(rep["metrics"]["y_rmse"][primary_method])

                run_records.append({
                    "dataset": ds_tag,
                    "data_dir": str(data_dir),
                    "seed": int(seed),
                    "ablation_key": ab.key,
                    "ablation_tag": ab.tag,
                    "primary_method": primary_method,
                    "y_rmse_test": y_rmse_test,
                    "run_dir": str(effective_run_dir),
                })

    summary_json = out_root / "summary_runs.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(run_records, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {summary_json}")

    if args.skip_plots:
        return

    by_ds: Dict[str, List[Dict[str, Any]]] = {}
    for r in run_records:
        by_ds.setdefault(r["dataset"], []).append(r)

    plot_dir = out_root / "summary"
    ensure_dir(plot_dir)

    for ds, rows in by_ds.items():
        tags: List[str] = []
        vals_list: List[List[float]] = []

        for k in chosen_keys:
            vals = [float(rr["y_rmse_test"]) for rr in rows if rr["ablation_key"] == k]
            if len(vals) == 0:
                continue
            tags.append(k)
            vals_list.append(vals)

        if len(tags) > 0:
            plot_ablation_boxplot(
                out_png=plot_dir / f"{ds}__boxplot__y_rmse_test.png",
                tags=tags,
                values=vals_list,
                # title="RMSE of ablation experiments",
                title="",
                ylabel="RMSE",
            )

        curves_mean: Dict[str, np.ndarray] = {}
        curves_std: Dict[str, np.ndarray] = {}
        axis_ref: Optional[np.ndarray] = None

        for k in chosen_keys:
            curve_list: List[np.ndarray] = []
            for rr in rows:
                if rr["ablation_key"] != k:
                    continue
                run_dir = Path(rr["run_dir"])
                try:
                    axis, y = load_rmse_curve(run_dir, split=args.rmse_curve_split)
                except FileNotFoundError as e:
                    print(f"[WARN] {e}")
                    continue
                if axis_ref is None:
                    axis_ref = axis
                else:
                    if axis.shape != axis_ref.shape or np.max(np.abs(axis - axis_ref)) > 1e-6:
                        raise ValueError(f"Axis mismatch across runs for dataset={ds}.")
                curve_list.append(y)

            if len(curve_list) == 0:
                continue
            yy = np.stack(curve_list, axis=0)
            curves_mean[k] = np.mean(yy, axis=0)
            curves_std[k] = np.std(yy, axis=0)

        if axis_ref is not None and len(curves_mean) > 0:
            label_map = {}
            for k in chosen_keys:
                vals = [float(rr["y_rmse_test"]) for rr in rows if rr["ablation_key"] == k]
                if len(vals) == 0:
                    continue
                mu = float(np.mean(vals))
                sd = float(np.std(vals))
                label_map[k] = f"{k} ({mu:.2e}±{sd:.2e})"
            plot_spectrum_curves(
                out_png=plot_dir / f"{ds}__spectrum_rmse_curves__{args.rmse_curve_split}.png",
                axis=axis_ref,
                curves_mean=curves_mean,
                curves_std=curves_std,
                # title="Spectrum RMSE curves",
                title="",
                xlabel="wavelength",
                ylabel="RMSE",
                label_map=label_map,
            )

        if "A0" in chosen_keys and "A12" in chosen_keys:
            res_mean: Dict[str, np.ndarray] = {}
            res_std: Dict[str, np.ndarray] = {}
            axis_res: Optional[np.ndarray] = None
            for k in ("A0", "A12"):
                curve_list = []
                for rr in rows:
                    if rr["ablation_key"] != k:
                        continue
                    run_dir = Path(rr["run_dir"])
                    try:
                        axis, y = load_residual_curve(run_dir, split=args.rmse_curve_split)
                    except FileNotFoundError as e:
                        print(f"[WARN] {e}")
                        continue
                    if axis_res is None:
                        axis_res = axis
                    else:
                        if axis.shape != axis_res.shape or np.max(np.abs(axis - axis_res)) > 1e-6:
                            raise ValueError(f"Axis mismatch across runs for residual curves dataset={ds}.")
                    curve_list.append(y)

                if len(curve_list) > 0:
                    yy = np.stack(curve_list, axis=0)
                    res_mean[k] = np.mean(yy, axis=0)
                    res_std[k] = np.std(yy, axis=0)

            if axis_res is not None and len(res_mean) == 2:
                plot_spectrum_curves(
                    out_png=plot_dir / f"{ds}__residual_rmse_curves__{args.rmse_curve_split}__A0_vs_A12.png",
                    axis=axis_res,
                    curves_mean=res_mean,
                    curves_std=res_std,
                    title=f"Residual RMSE ({args.rmse_curve_split}) | mean±std",
                    xlabel="wavelength",
                    ylabel="Residual RMSE",
                )

    print(f"[DONE] Plots saved under: {plot_dir}")


if __name__ == "__main__":
    main()