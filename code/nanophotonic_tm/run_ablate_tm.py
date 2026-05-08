#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_ablate_tm.py

Pathway/component analysis runner for Neural--GP MF on the nanophotonic TM benchmark.

This version implements the revised A0--A3 comparison:

  A0. Full Neural--GP MF
      z_h_hat = rho_a * z_l_hat + rho_b + GP_residual(x_s, z_l_hat)

  A1. Affine LF only
      z_h_hat = rho_a * z_l_hat + rho_b
      No residual GP is added. This diagnostic is derived from the A0 run by using
      the saved residual curve RMS(y_hf - y_base), where y_base is the pure affine
      LF-to-HF base prediction.

  A2. Linear LF + GP(x)
      z_h_hat = rho_a * z_l_hat + rho_b + GP_residual(x_s)
      This is an additive co-kriging-style control: z_l contributes only through the
      linear affine channel, while the residual GP does not see z_l.

  A3. GP LF channel only
      z_h_hat = GP(x_s, z_l_hat)
      This removes the affine base transfer and directly predicts the HF latent state
      using the GP input [x_s, z_l_hat].

Plotting conventions
--------------------
- Boxplot x-axis labels are shown as A0, A1, A2, A3.
- Spectrum-curve legends also use the A0, A1, A2, A3 order.
- Broken y-axis is used by default for both the boxplot and spectrum RMSE curve.
- Existing runs are reused automatically if report.json already exists. Use
  --rerun_existing to force retraining.

Outputs
-------
Under <out_dir>/<exp_name>/:
  summary_runs.json
  summary_runs.csv
  summary/ablation_mapping.json
  summary/<dataset>__zl_channel_boxplot__y_rmse_test.png/pdf
  summary/<dataset>__zl_channel_spectrum_rmse_curves__test.png/pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator


SCRIPT_DIR = Path(__file__).resolve().parent

# Robust default paths:
#   expected location: <repo>/code/nanophotonic_tm/run_zl_channel_ablate_tm.py
#   repo root:         <repo>
#   code root:         <repo>/code
if SCRIPT_DIR.name == "nanophotonic_tm" and SCRIPT_DIR.parent.name == "code":
    CODE_ROOT = SCRIPT_DIR.parent
    PROJECT_ROOT = CODE_ROOT.parent
else:
    # fallback for copied scripts
    PROJECT_ROOT = SCRIPT_DIR.parent
    CODE_ROOT = PROJECT_ROOT / "code"

DEFAULT_TRAIN_SCRIPT = str(CODE_ROOT / "mf_train_baseline" / "mf_train.py")
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data" / "mf_sweep_datasets_nano_tm" / "hf100_lfx10")
DEFAULT_OUT_DIR = str(PROJECT_ROOT / "result_out" / "zl_channel_ablate_tm")

# Color-blind-friendly palette used throughout the paper figures.
C_HF_BLUE = "#0072B2"
C_COKRIGING_ORANGE = "#E69F00"
C_OURS_GREEN = "#009E73"
C_RANDOM_PURPLE = "#CC79A7"

PATHWAY_COLORS = {
    "A0": C_OURS_GREEN,             # Full Neural–GP MF
    "A1": C_COKRIGING_ORANGE,       # Affine LF only
    "A2": C_HF_BLUE,                # Linear LF + GP(x)
    "A3": C_RANDOM_PURPLE,          # GP LF channel only
}


@dataclass(frozen=True)
class TrainVariant:
    key: str
    tag: str
    label: str
    desc: str
    overrides: Dict[str, Any]


def safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_+.@") else "_" for c in str(s))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_cmd(cmd: List[str], dry_run: bool = False) -> int:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return 0
    p = subprocess.run(cmd, check=False)
    return int(p.returncode)


def find_report_dir(run_dir: Path) -> Optional[Path]:
    direct = run_dir / "report.json"
    if direct.exists():
        return run_dir
    cands = list(run_dir.glob("**/report.json"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0].parent


def get_train_variants() -> Dict[str, TrainVariant]:
    """Training variants. A1 affine-only is derived from A0, so it is not trained."""

    base_best: Dict[str, Any] = {
        # Stage-II pathway settings
        "mf_u_mode": "xlf",
        "student_mode": "delta",
        "gp_ard": 1,
        "skip_student": 0,
        "mf_student_lf_source": "student",
        "rho_fit_source": "oracle",
        "rho_intercept": 1,
        # Stage-I settings matching the previous A0 setup
        "student_train_set": "mix",
        "student_val_set": "paired",
        "student_y_scaler_fit": "paired",
        "student_hidden": [256, 256, 256],
        "student_feat_dim": 32,
        "student_epochs": 2000,
        "student_bs": 256,
        "student_patience": 100,
    }

    return {
        "A0": TrainVariant(
            key="A0",
            # Keep the old P0 folder name so existing full-model runs can be reused.
            tag="A0_full_both_lf_channels",
            label="A0",
            desc="Full Neural--GP MF: affine LF base + GP residual with u=[x,z_l].",
            overrides=dict(base_best),
        ),
        "A2": TrainVariant(
            key="A2",
            tag="A2_linear_lf_plus_gp_x",
            label="A2",
            desc="Linear LF + GP(x): affine LF base plus residual GP using x only.",
            overrides={**base_best, "student_mode": "delta", "mf_u_mode": "x"},
        ),
        "A3": TrainVariant(
            key="A3",
            tag="A3_gp_lf_channel_only",
            label="A3",
            desc="GP LF channel only: remove affine LF base; directly predict z_h with GP input u=[x,z_l].",
            overrides={**base_best, "student_mode": "direct", "mf_u_mode": "xlf"},
        ),
    }


def add_overrides_to_cmd(cmd: List[str], overrides: Dict[str, Any]) -> None:
    for kk, vv in overrides.items():
        flag = f"--{kk}"
        if isinstance(vv, (list, tuple)):
            cmd.append(flag)
            cmd.extend(str(x) for x in vv)
        else:
            cmd.extend([flag, str(vv)])


def build_train_cmd(
    args: argparse.Namespace,
    *,
    data_dir: Path,
    run_dir: Path,
    seed: int,
    variant: TrainVariant,
) -> List[str]:
    cmd: List[str] = [
        str(args.python),
        str(args.train_script),
        "--data_dir", str(data_dir),
        "--out_dir", str(run_dir),
        "--run_prefix", str(args.run_prefix),
        "--seed", str(seed),
        "--device", str(args.device),
        "--no_subdir", "1",
        "--run_hf_only", "0",
        "--run_oracle", "0",
        "--run_student", "1",
        "--wl_low", str(args.wl_low),
        "--wl_high", str(args.wl_high),
        "--dim_reduce", str(args.dim_reduce),
        "--fpca_var_ratio", str(args.fpca_var_ratio),
        "--fpca_max_dim", str(args.fpca_max_dim),
        "--subsample_K", str(args.subsample_K),
        "--kernel_struct", str(args.kernel_struct),
        "--kernel", str(args.kernel),
        "--matern_nu", str(args.matern_nu),
        "--svgp_M", str(args.svgp_M),
        "--svgp_steps", str(args.svgp_steps),
        "--ci_level", str(args.ci_level),
        "--ci_calibrate", str(args.ci_calibrate),
        "--n_plot", str(args.n_plot_train),
        "--save_pred_arrays", str(args.save_pred_arrays),
    ]

    if int(args.fpca_dim) > 0:
        cmd.extend(["--fpca_dim", str(args.fpca_dim)])

    add_overrides_to_cmd(cmd, variant.overrides)

    if args.extra_args:
        cmd.extend(list(args.extra_args))

    return cmd


def metric_from_report(report_dir: Path, method: str = "mf_student") -> float:
    rep = read_json(report_dir / "report.json")
    return float(rep["metrics"]["y_rmse"][method])


def axis_file(rmse_dir: Path) -> Optional[Path]:
    for name in ("axis.npy", "axis_wavelength.npy", "axis_frequency.npy"):
        p = rmse_dir / name
        if p.exists():
            return p
    return None


def load_rmse_curve(report_dir: Path, *, split: str, method: str) -> Tuple[np.ndarray, np.ndarray]:
    rmse_dir = report_dir / "rmse_curves"
    ap = axis_file(rmse_dir)
    yp = rmse_dir / f"rmse_{split}__{method}.npy"
    if ap is None or not yp.exists():
        raise FileNotFoundError(f"Missing curve files under {rmse_dir}: axis={ap}, curve={yp.name}")
    return np.load(ap).astype(np.float32), np.load(yp).astype(np.float32)


def rmse_scalar_from_curve(curve: np.ndarray) -> float:
    curve = np.asarray(curve, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(curve))))


def collect_one_dataset_seed(
    args: argparse.Namespace,
    *,
    out_root: Path,
    data_dir: Path,
    seed: int,
) -> List[Dict[str, Any]]:
    """Run/collect A0, A2, A3 and derive A1 from A0."""
    ds_tag = safe_name(data_dir.name)
    variants = get_train_variants()
    rows: List[Dict[str, Any]] = []

    report_dirs: Dict[str, Path] = {}

    for key in ("A0", "A2", "A3"):
        variant = variants[key]
        run_dir = out_root / ds_tag / variant.tag / f"seed{seed}"
        ensure_dir(run_dir)

        report_dir_existing = find_report_dir(run_dir)

        # Default behavior: if this exact run already has report.json, reuse it and only replot.
        # Use --rerun_existing to force retraining.
        if report_dir_existing is not None and (not args.rerun_existing):
            print(f"[SKIP] Existing result found for dataset={ds_tag} seed={seed} variant={key}: {report_dir_existing}")
        elif not args.skip_train:
            cmd = build_train_cmd(args, data_dir=data_dir, run_dir=run_dir, seed=seed, variant=variant)
            rc = run_cmd(cmd, dry_run=args.dry_run)
            if rc != 0:
                raise RuntimeError(f"Training failed: dataset={data_dir} seed={seed} variant={key} rc={rc}")

        report_dir = find_report_dir(run_dir)
        if report_dir is None:
            print(f"[WARN] Missing report.json: {run_dir}")
            continue
        report_dirs[key] = report_dir

        try:
            y_rmse = metric_from_report(report_dir, method="mf_student")
        except Exception as e:
            print(f"[WARN] Cannot read y_rmse for {key}: {report_dir} | {type(e).__name__}: {e}")
            continue

        rows.append({
            "dataset": ds_tag,
            "data_dir": str(data_dir),
            "seed": int(seed),
            "pathway_key": key,
            "pathway_tag": variant.tag,
            "pathway_label": variant.label,
            "pathway_desc": variant.desc,
            "source_method": "mf_student",
            "derived_from": "",
            "y_rmse_test": float(y_rmse),
            "run_dir": str(report_dir),
        })

    # A1: affine-only LF channel, derived from A0 residual curve.
    # This intentionally has no residual GP. It is the base prediction error from A0.
    if "A0" in report_dirs:
        p0_dir = report_dirs["A0"]
        try:
            _, residual_curve = load_rmse_curve(p0_dir, split=args.rmse_curve_split, method="mf_student_residual")
            y_rmse_affine = rmse_scalar_from_curve(residual_curve)
            rows.append({
                "dataset": ds_tag,
                "data_dir": str(data_dir),
                "seed": int(seed),
                "pathway_key": "A1",
                "pathway_tag": "A1_affine_lf_only_derived_from_A0",
                "pathway_label": "A1",
                "pathway_desc": "Linear LF channel only: rho_a*z_l + rho_b; no residual GP.",
                "source_method": "mf_student_residual",
                "derived_from": "A0_full_both_lf_channels",
                "y_rmse_test": float(y_rmse_affine),
                "run_dir": str(p0_dir),
            })
        except Exception as e:
            print(f"[WARN] Cannot derive A1 affine-only from A0: {p0_dir} | {type(e).__name__}: {e}")

    return rows


def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def format_y_axis(ax, *, nbins: int = 5) -> None:
    """Use scientific notation without offset text.

    This avoids duplicated decimal labels such as 0.03, 0.03, 0.03 on
    narrow broken-axis panels, while also preventing Matplotlib offset
    notation such as "1e-12 + 9.9e-2".
    """
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))   # always use scientific notation
    formatter.set_useOffset(False)      # never use additive offset notation

    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_visible(True)



def _format_mean_std(vals: List[float], fmt: str = ".3e") -> str:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return ""
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    return f"{mu:{fmt}}±{sd:{fmt}}"


def _add_break_marks(ax_top, ax_bottom) -> None:
    """Draw diagonal marks for a broken y-axis."""
    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


def _styled_boxplot(ax, labels: List[str], values: List[List[float]], *, show_xticklabels: bool) -> Any:
    bp = ax.boxplot(
        values,
        labels=(labels if show_xticklabels else [""] * len(labels)),
        showfliers=True,
        patch_artist=True,
        medianprops={"linewidth": 1.5},
        whiskerprops={"linewidth": 1.1},
        capprops={"linewidth": 1.1},
        boxprops={"linewidth": 1.1},
        flierprops={"markersize": 3, "alpha": 0.5},
    )

    for i, lab in enumerate(labels):
        c = PATHWAY_COLORS.get(lab, f"C{i}")
        bp["boxes"][i].set_facecolor(c)
        bp["boxes"][i].set_alpha(0.18)
        bp["boxes"][i].set_edgecolor(c)
        bp["medians"][i].set_color(c)
        for w in bp["whiskers"][2 * i: 2 * i + 2]:
            w.set_color(c)
        for cap in bp["caps"][2 * i: 2 * i + 2]:
            cap.set_color(c)
        if i < len(bp.get("fliers", [])):
            bp["fliers"][i].set_markeredgecolor(c)
            bp["fliers"][i].set_markerfacecolor(c)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(values, start=1):
        if not vals:
            continue
        lab = labels[i - 1]
        c = PATHWAY_COLORS.get(lab, f"C{i - 1}")
        v = np.asarray(vals, dtype=float)
        x = rng.normal(loc=i, scale=0.035, size=v.size)
        ax.scatter(x, v, s=12, alpha=0.58, linewidths=0, color=c, zorder=3)

    return bp


def plot_boxplot(
    out_png: Path,
    labels: List[str],
    values: List[List[float]],
    ylabel: str = "RMSE",
    *,
    view: str = "broken",
    annotate: bool = True,
    value_fmt: str = ".3e",
    save_pdf: bool = True,
) -> None:
    """Plot pathway RMSE boxplot.

    view:
      - "broken": broken y-axis. Recommended when Affine LF only is much larger.
      - "plain":  single axis.
    """
    apply_plot_style()
    fig_w = max(5.0, 1.65 * len(labels))

    flat = np.asarray([x for vv in values for x in vv], dtype=float)
    med = np.asarray([np.median(np.asarray(v, dtype=float)) if len(v) else np.nan for v in values], dtype=float)
    should_break = (
        view == "broken"
        and len(values) >= 3
        and np.all(np.isfinite(med))
        and np.nanmax(med) / max(np.nanmin(med), 1e-12) > 2.5
    )

    if not should_break:
        fig, ax = plt.subplots(figsize=(fig_w, 3.45))
        _styled_boxplot(ax, labels, values, show_xticklabels=True)

        if flat.size > 0:
            ymin, ymax = float(np.min(flat)), float(np.max(flat))
            ymid = 0.5 * (ymin + ymax)
            yrng = max(ymax - ymin, 0.08 * abs(ymid), 1e-4)
            ax.set_ylim(max(0.0, ymin - 0.08 * yrng), ymax + 0.28 * yrng)

        if annotate:
            ymin, ymax = ax.get_ylim()
            yrng = ymax - ymin
            for i, vals in enumerate(values, start=1):
                if not vals:
                    continue
                y = float(np.max(vals)) + 0.035 * yrng
                ax.text(i, y, _format_mean_std(vals, value_fmt), ha="center", va="bottom", fontsize=8)

        ax.set_ylabel(ylabel)
        format_y_axis(ax)
        ax.grid(True, axis="y", alpha=0.25)
        plt.xticks(rotation=12, ha="right")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        if save_pdf:
            plt.savefig(out_png.with_suffix(".pdf"))
        plt.close()
        return

    # Broken-axis view: top panel shows the large affine-only error;
    # bottom panel zooms into GP-only and full-model errors.
    imax = int(np.nanargmax(med))
    low_vals = np.asarray(
        [x for j, vv in enumerate(values) if j != imax for x in vv],
        dtype=float,
    )
    high_vals = np.asarray(values[imax], dtype=float)

    low_min, low_max = float(np.min(low_vals)), float(np.max(low_vals))
    high_min, high_max = float(np.min(high_vals)), float(np.max(high_vals))
    low_mid = 0.5 * (low_min + low_max)
    high_mid = 0.5 * (high_min + high_max)
    # Use an absolute/value-dependent minimum range. This avoids Matplotlib offset
    # text such as "1e-12 + ..." when a group has only one value or zero spread.
    low_rng = max(low_max - low_min, 0.08 * abs(low_mid), 1e-4)
    high_rng = max(high_max - high_min, 0.08 * abs(high_mid), 1e-4)

    bottom_ylim = (max(0.0, low_min - 0.25 * low_rng), low_max + 0.65 * low_rng)
    top_ylim = (max(0.0, high_min - 0.60 * high_rng), high_max + 0.90 * high_rng)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(fig_w, 4.05),
        gridspec_kw={"height_ratios": [1.0, 1.45], "hspace": 0.06},
    )

    _styled_boxplot(ax_top, labels, values, show_xticklabels=False)
    _styled_boxplot(ax_bottom, labels, values, show_xticklabels=True)

    ax_top.set_ylim(*top_ylim)
    ax_bottom.set_ylim(*bottom_ylim)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bottom.xaxis.tick_bottom()
    _add_break_marks(ax_top, ax_bottom)

    # One shared y-label.
    fig.text(0.015, 0.52, ylabel, va="center", rotation="vertical", fontsize=9)

    for ax in (ax_top, ax_bottom):
        format_y_axis(ax)
        ax.grid(True, axis="y", alpha=0.25)

    if annotate:
        # Put each value label on the panel where the corresponding group is visible.
        for i, vals in enumerate(values, start=1):
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            lab = _format_mean_std(vals, value_fmt)
            mu = float(np.mean(arr))
            if bottom_ylim[0] <= mu <= bottom_ylim[1]:
                ax = ax_bottom
                yrng = bottom_ylim[1] - bottom_ylim[0]
                y = min(float(np.max(arr)) + 0.08 * yrng, bottom_ylim[1] - 0.05 * yrng)
            else:
                ax = ax_top
                yrng = top_ylim[1] - top_ylim[0]
                y = min(float(np.max(arr)) + 0.10 * yrng, top_ylim[1] - 0.06 * yrng)
            ax.text(i, y, lab, ha="center", va="bottom", fontsize=8)

    plt.xticks(rotation=12, ha="right")
    plt.tight_layout(rect=(0.035, 0.0, 1.0, 1.0))
    plt.savefig(out_png, dpi=300)
    if save_pdf:
        plt.savefig(out_png.with_suffix(".pdf"))
    plt.close()


def _spectrum_range_for_labels(
    mean_curves: Dict[str, np.ndarray],
    std_curves: Dict[str, np.ndarray],
    labs: List[str],
) -> Tuple[float, float]:
    ys: List[np.ndarray] = []
    for lab in labs:
        if lab not in mean_curves:
            continue
        y = np.asarray(mean_curves[lab], dtype=float)
        if lab in std_curves:
            s = np.asarray(std_curves[lab], dtype=float)
            ys.append(y - s)
            ys.append(y + s)
        else:
            ys.append(y)
    if not ys:
        return 0.0, 1.0
    arr = np.concatenate([a.ravel() for a in ys])
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    return float(np.min(arr)), float(np.max(arr))


def _plot_spectrum_lines_on_axis(
    ax,
    axis: np.ndarray,
    mean_curves: Dict[str, np.ndarray],
    std_curves: Dict[str, np.ndarray],
    labels_order: List[str],
    *,
    label_lines: bool,
) -> None:
    for lab in labels_order:
        if lab not in mean_curves:
            continue
        y = np.asarray(mean_curves[lab], dtype=float)
        c = PATHWAY_COLORS.get(lab, None)
        line, = ax.plot(axis, y, label=(lab if label_lines else "_nolegend_"), color=c)
        if lab in std_curves:
            ys = np.asarray(std_curves[lab], dtype=float)
            cc = c if c is not None else line.get_color()
            ax.fill_between(axis, y - ys, y + ys, alpha=0.16, color=cc, linewidth=0)


def plot_spectrum_curves(
    out_png: Path,
    axis: np.ndarray,
    mean_curves: Dict[str, np.ndarray],
    std_curves: Dict[str, np.ndarray],
    labels_order: List[str],
    xlabel: str = "Wavelength (nm)",
    ylabel: str = "RMSE",
    *,
    view: str = "broken",
    save_pdf: bool = True,
) -> None:
    """Plot wavelength-resolved RMSE curves.

    view:
      - "broken": broken y-axis. Recommended when affine-only error is much larger.
      - "plain":  single axis.
    """
    apply_plot_style()

    valid_labs = [lab for lab in labels_order if lab in mean_curves]
    med = np.asarray(
        [np.median(np.asarray(mean_curves[lab], dtype=float)) for lab in valid_labs],
        dtype=float,
    )
    should_break = (
        view == "broken"
        and len(valid_labs) >= 3
        and np.all(np.isfinite(med))
        and np.nanmax(med) / max(np.nanmin(med), 1e-12) > 2.5
    )

    if not should_break:
        plt.figure(figsize=(5.4, 3.35))
        ax = plt.gca()
        _plot_spectrum_lines_on_axis(
            ax, axis, mean_curves, std_curves, labels_order, label_lines=True
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        format_y_axis(ax)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="best")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        if save_pdf:
            plt.savefig(out_png.with_suffix(".pdf"))
        plt.close()
        return

    # Split by median error. The high-error curve is normally "Affine LF only";
    # the lower-error curves include A0/A2/A3.
    imax = int(np.nanargmax(med))
    high_lab = valid_labs[imax]
    low_labs = [lab for lab in valid_labs if lab != high_lab]

    low_min, low_max = _spectrum_range_for_labels(mean_curves, std_curves, low_labs)
    high_min, high_max = _spectrum_range_for_labels(mean_curves, std_curves, [high_lab])

    low_mid = 0.5 * (low_min + low_max)
    high_mid = 0.5 * (high_min + high_max)
    # Use an absolute/value-dependent minimum range. This avoids offset notation
    # when one panel contains a nearly constant curve or only one effective value.
    low_rng = max(low_max - low_min, 0.08 * abs(low_mid), 1e-4)
    high_rng = max(high_max - high_min, 0.08 * abs(high_mid), 1e-4)

    bottom_ylim = (max(0.0, low_min - 0.18 * low_rng), low_max + 0.35 * low_rng)
    top_ylim = (max(0.0, high_min - 0.12 * high_rng), high_max + 0.18 * high_rng)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(5.7, 4.15),
        gridspec_kw={"height_ratios": [1.0, 1.35], "hspace": 0.06},
    )

    _plot_spectrum_lines_on_axis(
        ax_top, axis, mean_curves, std_curves, labels_order, label_lines=True
    )
    _plot_spectrum_lines_on_axis(
        ax_bottom, axis, mean_curves, std_curves, labels_order, label_lines=False
    )

    ax_top.set_ylim(*top_ylim)
    ax_bottom.set_ylim(*bottom_ylim)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bottom.xaxis.tick_bottom()
    _add_break_marks(ax_top, ax_bottom)

    ax_bottom.set_xlabel(xlabel)
    fig.text(0.015, 0.52, ylabel, va="center", rotation="vertical", fontsize=9)

    for ax in (ax_top, ax_bottom):
        format_y_axis(ax)
        ax.grid(True, alpha=0.25)

    ax_top.legend(frameon=False, loc="best")

    plt.tight_layout(rect=(0.035, 0.0, 1.0, 1.0))
    plt.savefig(out_png, dpi=300)
    if save_pdf:
        plt.savefig(out_png.with_suffix(".pdf"))
    plt.close()


def summarize_and_plot(args: argparse.Namespace, out_root: Path, records: List[Dict[str, Any]]) -> None:
    summary_dir = out_root / "summary"
    ensure_dir(summary_dir)

    # CSV summary
    csv_path = out_root / "summary_runs.csv"
    fieldnames = [
        "dataset", "data_dir", "seed", "pathway_key", "pathway_tag", "pathway_label",
        "source_method", "derived_from", "y_rmse_test", "run_dir", "pathway_desc",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[SAVE] {csv_path}")

    if args.skip_plots:
        return

    order_keys = ["A0", "A1", "A2", "A3"]
    label_by_key = {k: k for k in order_keys}
    ablation_name_by_key = {
        "A0": "Full Neural--GP MF",
        "A1": "Affine LF only",
        "A2": "Linear LF + GP(x)",
        "A3": "GP LF channel only",
    }
    ablation_formula_by_key = {
        "A0": "z_h = rho_a * z_l + rho_b + GP(x_s, z_l)",
        "A1": "z_h = rho_a * z_l + rho_b",
        "A2": "z_h = rho_a * z_l + rho_b + GP(x_s)",
        "A3": "z_h = GP(x_s, z_l)",
    }
    labels_order = [label_by_key[k] for k in order_keys]

    mapping = {
        k: {"name": ablation_name_by_key[k], "formula": ablation_formula_by_key[k]}
        for k in order_keys
    }
    write_json(summary_dir / "ablation_mapping.json", mapping)

    by_ds: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_ds.setdefault(str(r["dataset"]), []).append(r)

    for ds, rows in by_ds.items():
        # Boxplot
        labels: List[str] = []
        vals_list: List[List[float]] = []
        for k in order_keys:
            vals = [float(r["y_rmse_test"]) for r in rows if r["pathway_key"] == k]
            if vals:
                labels.append(label_by_key[k])
                vals_list.append(vals)
        if vals_list:
            out_png = summary_dir / f"{ds}__zl_channel_boxplot__y_rmse_test.png"
            plot_boxplot(out_png, labels, vals_list, ylabel="Test RMSE", view=args.boxplot_view, annotate=bool(args.annotate_values), value_fmt=args.value_fmt, save_pdf=bool(args.save_pdf))
            print(f"[SAVE] {out_png}")

        # Spectrum curves
        curve_by_label: Dict[str, List[np.ndarray]] = {lab: [] for lab in labels_order}
        axis_ref: Optional[np.ndarray] = None
        for r in rows:
            k = str(r["pathway_key"])
            lab = label_by_key.get(k)
            if lab is None:
                continue
            method = "mf_student_residual" if k == "A1" else "mf_student"
            try:
                axis, curve = load_rmse_curve(Path(r["run_dir"]), split=args.rmse_curve_split, method=method)
            except FileNotFoundError as e:
                print(f"[WARN] {e}")
                continue
            if axis_ref is None:
                axis_ref = axis
            else:
                if axis.shape != axis_ref.shape or float(np.max(np.abs(axis - axis_ref))) > 1e-6:
                    raise ValueError(f"Axis mismatch in dataset={ds}, run_dir={r['run_dir']}")
            curve_by_label[lab].append(curve)

        if axis_ref is not None:
            mean_curves: Dict[str, np.ndarray] = {}
            std_curves: Dict[str, np.ndarray] = {}
            for lab, arrs in curve_by_label.items():
                if not arrs:
                    continue
                yy = np.stack(arrs, axis=0)
                mean_curves[lab] = np.mean(yy, axis=0)
                std_curves[lab] = np.std(yy, axis=0)
            if mean_curves:
                out_png = summary_dir / f"{ds}__zl_channel_spectrum_rmse_curves__{args.rmse_curve_split}.png"
                plot_spectrum_curves(
                    out_png,
                    axis_ref,
                    mean_curves,
                    std_curves,
                    labels_order=labels_order,
                    xlabel="Wavelength (nm)",
                    ylabel="RMSE",
                    view=args.spectrum_view,
                    save_pdf=bool(args.save_pdf),
                )
                print(f"[SAVE] {out_png}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run z_l -> z_h pathway ablation A0--A3: full, affine-only, linear-LF+GP(x), and GP-only."
    )

    ap.add_argument("--python", type=str, default=os.environ.get("PYTHON_BIN", "python3"))
    ap.add_argument("--train_script", type=str, default=DEFAULT_TRAIN_SCRIPT)
    ap.add_argument("--run_prefix", type=str, default="zl")
    ap.add_argument("--data_dirs", type=str, nargs="+", default=[DEFAULT_DATA_DIR])
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--exp_name", type=str, default="zl_channel_pathway")

    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 33, 55, 66, 77, 8, 9, 11, 22, 88, 99, 111, 222, 333, 555],
    )

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--wl_low", type=float, default=380.0)
    ap.add_argument("--wl_high", type=float, default=750.0)
    ap.add_argument("--dim_reduce", type=str, default="fpca", choices=["fpca", "subsample"])
    ap.add_argument("--fpca_dim", type=int, default=0)
    ap.add_argument("--fpca_var_ratio", type=float, default=0.999)
    ap.add_argument("--fpca_max_dim", type=int, default=50)
    ap.add_argument("--subsample_K", type=int, default=1)

    # Keep default as the old runner did. mf_train.py will force this to full when no feature block exists.
    ap.add_argument("--kernel_struct", type=str, default="block", choices=["full", "block", "xlf_block"])
    ap.add_argument("--kernel", type=str, default="matern", choices=["rbf", "matern"])
    ap.add_argument("--matern_nu", type=float, default=2.5)
    ap.add_argument("--svgp_M", type=int, default=64)
    ap.add_argument("--svgp_steps", type=int, default=2000)

    ap.add_argument("--ci_level", type=float, default=0.95)
    ap.add_argument("--ci_calibrate", type=int, default=1, choices=[0, 1])
    ap.add_argument("--rmse_curve_split", type=str, default="test", choices=["val", "test"])

    # Keep training runs lean by default; plots for the paper are produced by this runner.
    ap.add_argument("--n_plot_train", type=int, default=0)
    ap.add_argument("--save_pred_arrays", type=int, default=0, choices=[0, 1])

    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--rerun_existing", action="store_true",
                    help="Force retraining even when report.json already exists. By default, existing runs are reused and only summaries/plots are regenerated.")
    ap.add_argument("--skip_plots", action="store_true")
    ap.add_argument("--boxplot_view", type=str, default="broken", choices=["broken", "plain"],
                    help="Use a broken y-axis by default so lower-error variants are visually separated from the larger affine-only error.")
    ap.add_argument("--spectrum_view", type=str, default="broken", choices=["broken", "plain"],
                    help="Use a broken y-axis by default for wavelength-resolved RMSE curves.")
    ap.add_argument("--annotate_values", type=int, default=1, choices=[0, 1],
                    help="Annotate boxplot with mean±std values.")
    ap.add_argument("--value_fmt", type=str, default=".3e",
                    help="Python numeric format for mean±std labels, e.g. .3e or .4f.")
    ap.add_argument("--save_pdf", type=int, default=1, choices=[0, 1],
                    help="Also save PDF versions of summary figures.")
    ap.add_argument("--extra_args", type=str, nargs="*", default=[])

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    args.train_script = str(Path(args.train_script).expanduser().resolve())

    out_root = Path(args.out_dir).expanduser().resolve() / safe_name(args.exp_name)
    ensure_dir(out_root)

    write_json(out_root / "runner_config.json", vars(args))

    records: List[Dict[str, Any]] = []
    for data_dir_s in args.data_dirs:
        data_dir = Path(data_dir_s).expanduser().resolve()
        for seed in args.seeds:
            rows = collect_one_dataset_seed(args, out_root=out_root, data_dir=data_dir, seed=int(seed))
            records.extend(rows)

    summary_json = out_root / "summary_runs.json"
    write_json(summary_json, records)
    print(f"[SAVE] {summary_json}")

    summarize_and_plot(args, out_root, records)
    print(f"[DONE] Output root: {out_root}")


if __name__ == "__main__":
    main()
