#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot FPCA dim-sweep results as a publication-ready 2x2 figure.

Panels
------
(a) Reconstruction RMSE vs latent dimension
(b) Downstream y_RMSE vs latent dimension
(c) Calibrated NLL vs latent dimension
(d) Coverage + CI width vs latent dimension (dual y-axis)

Supported inputs
----------------
1) Single-seed CSV:
   fpca_dim_sweep_summary.csv
   Required columns:
     - dim
     - recon_rmse_hfval
     - y_rmse_test
     - nll_test_cal
   Optional:
     - coverage_test_cal
     - ci_width_test_cal

2) Multi-seed aggregated CSV:
   fpca_dim_sweep_summary_seedmean.csv
   Required columns:
     - dim
     - recon_rmse_hfval_mean
     - recon_rmse_hfval_std
     - y_rmse_test_mean
     - y_rmse_test_std
     - nll_test_cal_mean
     - nll_test_cal_std
   Optional:
     - coverage_test_cal_mean
     - coverage_test_cal_std
     - ci_width_test_cal_mean
     - ci_width_test_cal_std

Outputs
-------
- fpca_dim_sweep_2x2.png
- fpca_dim_sweep_2x2.pdf

Example
-------
Single-seed:
python plot_fpca_dim_sweep_tm.py \
  --csv ./fpca_dim_sweep_tm_outputs/fpca_dim_sweep_summary.csv \
  --out_dir ./fpca_dim_sweep_tm_outputs/plots

Multi-seed:
python plot_fpca_dim_sweep_tm.py \
  --csv ./fpca_dim_sweep_tm_outputs/fpca_dim_sweep_summary_seedmean.csv \
  --out_dir ./fpca_dim_sweep_tm_outputs/plots

Auto-detect from sweep root:
python plot_fpca_dim_sweep_tm.py \
  --sweep_root ./fpca_dim_sweep_tm_outputs
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# IO helpers
# -----------------------------
def pick_input_csv(sweep_root: Path) -> Path:
    agg = sweep_root / "fpca_dim_sweep_summary_seedmean.csv"
    per = sweep_root / "fpca_dim_sweep_summary.csv"
    if agg.exists():
        return agg
    if per.exists():
        return per
    raise FileNotFoundError(
        f"No summary CSV found under {sweep_root}. "
        f"Expected one of: {agg.name}, {per.name}"
    )


def infer_mode(df: pd.DataFrame) -> str:
    if "recon_rmse_hfval_mean" in df.columns:
        return "multi_seed"
    if "recon_rmse_hfval" in df.columns:
        return "single_seed"
    raise ValueError("Cannot infer CSV mode from columns.")


def _get_series(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    return df[name] if name in df.columns else None


def load_curves(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, Dict[str, Optional[pd.Series]]], str]:
    mode = infer_mode(df)
    df = df.copy()
    x_col = "fpca_dim_effective" if "fpca_dim_effective" in df.columns else "dim"
    df = df.sort_values(x_col).reset_index(drop=True)
    x = df[x_col]

    if mode == "single_seed":
        curves = {
            "recon": {
                "y": _get_series(df, "recon_rmse_hfval"),
                "std": None,
                "ylabel": "HF validation reconstruction RMSE",
            },
            "y_rmse": {
                "y": _get_series(df, "y_rmse_test"),
                "std": None,
                "ylabel": "Test RMSE",
            },
            "nll": {
                "y": _get_series(df, "nll_test_cal"),
                "std": None,
                "ylabel": "Adjusted test NLL",
            },
            "coverage": {
                "y": _get_series(df, "coverage_test_cal"),
                "std": None,
                "ylabel": "Calibrated coverage",
            },
            "width": {
                "y": _get_series(df, "ci_width_test_cal"),
                "std": None,
                "ylabel": "Mean interval width (response units)",
            },
        }
    else:
        curves = {
            "recon": {
                "y": _get_series(df, "recon_rmse_hfval_mean"),
                "std": _get_series(df, "recon_rmse_hfval_std"),
                "ylabel": "HF validation reconstruction RMSE",
            },
            "y_rmse": {
                "y": _get_series(df, "y_rmse_test_mean"),
                "std": _get_series(df, "y_rmse_test_std"),
                "ylabel": "Test RMSE",
            },
            "nll": {
                "y": _get_series(df, "nll_test_cal_mean"),
                "std": _get_series(df, "nll_test_cal_std"),
                "ylabel": "Adjusted test NLL",
            },
            "coverage": {
                "y": _get_series(df, "coverage_test_cal_mean"),
                "std": _get_series(df, "coverage_test_cal_std"),
                "ylabel": "Calibrated coverage",
            },
            "width": {
                "y": _get_series(df, "ci_width_test_cal_mean"),
                "std": _get_series(df, "ci_width_test_cal_std"),
                "ylabel": "Mean interval width (response units)",
            },
        }

    return x, curves, mode


# -----------------------------
# Plot style
# -----------------------------
def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        # overall text
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 12,

        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _finite_mask(y: pd.Series) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    return np.isfinite(arr)


def _best_idx(y: pd.Series, mode: str = "min") -> Optional[int]:
    arr = np.asarray(y, dtype=float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return None
    valid_idx = np.where(mask)[0]
    valid_arr = arr[mask]
    if mode == "min":
        j = int(np.argmin(valid_arr))
    else:
        j = int(np.argmax(valid_arr))
    return int(valid_idx[j])


def _pad_ylim(y: pd.Series, frac: float = 0.10) -> Tuple[float, float]:
    arr = np.asarray(y, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (0.0, 1.0)
    ymin = float(np.min(arr))
    ymax = float(np.max(arr))
    if np.isclose(ymin, ymax):
        eps = max(abs(ymin) * 0.1, 1e-6)
        return ymin - eps, ymax + eps
    pad = (ymax - ymin) * frac
    return ymin - pad, ymax + pad


def style_axis(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if str(title).strip():
        ax.set_title(title, pad=4)
    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_alpha(0.9)


def add_panel_label_outside(fig, ax, panel_label: str) -> None:
    pos = ax.get_position()
    fig.text(
        pos.x0, pos.y1 + 0.012, panel_label,
        ha="left", va="bottom",
        fontsize=11, fontweight="normal"
    )


# -----------------------------
# Drawing primitives
# -----------------------------
def plot_line_with_band(
    ax,
    x: pd.Series,
    y: pd.Series,
    std: Optional[pd.Series],
    color: str,
    label: str,
    lw: float = 2.2,
    marker: str = "o",
    ms: float = 5.5,
    alpha_fill: float = 0.16,
):
    mask = _finite_mask(y)
    xx = np.asarray(x[mask], dtype=float)
    yy = np.asarray(y[mask], dtype=float)

    ax.plot(
        xx, yy,
        color=color,
        linewidth=lw,
        marker=marker,
        markersize=ms,
        label=label,
    )

    if std is not None:
        ss = np.asarray(std[mask], dtype=float)
        lo = yy - ss
        hi = yy + ss
        ax.fill_between(xx, lo, hi, color=color, alpha=alpha_fill, linewidth=0)


def annotate_best(
    ax,
    x: pd.Series,
    y: pd.Series,
    std: Optional[pd.Series],
    mode: str,
    color: str,
    text_prefix: str = "best dim",
):
    idx = _best_idx(y, mode=mode)
    if idx is None:
        return

    xv = float(x.iloc[idx])
    yv = float(y.iloc[idx])

    ax.scatter(
        [xv], [yv],
        s=75,
        facecolors="white",
        edgecolors=color,
        linewidths=1.8,
        zorder=6,
    )
    ax.axvline(x=xv, linestyle="--", linewidth=1.1, alpha=0.55, color=color)

    if std is not None and pd.notna(std.iloc[idx]):
        txt = f"{text_prefix} = {int(xv)}\n{yv:.4g} ± {float(std.iloc[idx]):.2g}"
    else:
        txt = f"{text_prefix} = {int(xv)}\n{yv:.4g}"

    ax.annotate(
        txt,
        xy=(xv, yv),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=11,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.92),
        arrowprops=dict(arrowstyle="-", color=color, lw=1.0, alpha=0.8),
    )


# -----------------------------
# Main figure
# -----------------------------
def make_2x2_figure(
    x: pd.Series,
    curves: Dict[str, Dict[str, Optional[pd.Series]]],
    out_png: Path,
    out_pdf: Path,
    figure_title: str = "",
    nominal_coverage: float = 0.95,
) -> None:
    setup_style()

    # Colors chosen for clean journal-style distinction
    c_recon = "#4C72B0"
    c_rmse = "#C44E52"
    c_nll = "#55A868"
    c_cov = "#8172B2"
    # c_wid = "#CCB974"
    c_wid = "#E67E22"

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), constrained_layout=False)
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    # (a) Recon RMSE
    plot_line_with_band(
        ax_a, x,
        curves["recon"]["y"], curves["recon"]["std"],
        color=c_recon, label="Recon RMSE"
    )
    style_axis(
        ax_a,
        xlabel=r"Latent dimension $R$",
        ylabel=curves["recon"]["ylabel"],
        title="",
    )
    ax_a.set_xticks(list(map(int, x.tolist())))
    ax_a.set_ylim(*_pad_ylim(curves["recon"]["y"], frac=0.12))

    # (b) y_RMSE
    plot_line_with_band(
        ax_b, x,
        curves["y_rmse"]["y"], curves["y_rmse"]["std"],
        color=c_rmse, label="y_RMSE"
    )
    annotate_best(
        ax_b, x,
        curves["y_rmse"]["y"], curves["y_rmse"]["std"],
        mode="min", color=c_rmse, text_prefix="best dim"
    )
    style_axis(
        ax_b,
        xlabel=r"Latent dimension $R$",
        ylabel=curves["y_rmse"]["ylabel"],
        title="",
    )
    ax_b.set_xticks(list(map(int, x.tolist())))
    ax_b.set_ylim(*_pad_ylim(curves["y_rmse"]["y"], frac=0.15))

    # (c) Calibrated NLL
    plot_line_with_band(
        ax_c, x,
        curves["nll"]["y"], curves["nll"]["std"],
        color=c_nll, label="Adjusted NLL"
    )
    annotate_best(
        ax_c, x,
        curves["nll"]["y"], curves["nll"]["std"],
        mode="min", color=c_nll, text_prefix="best dim"
    )
    style_axis(
        ax_c,
        xlabel=r"Latent dimension $R$",
        ylabel=curves["nll"]["ylabel"],
        title="",
    )
    ax_c.set_xticks(list(map(int, x.tolist())))
    ax_c.set_ylim(*_pad_ylim(curves["nll"]["y"], frac=0.15))

    # (d) Coverage + CI width
    cov_y = curves["coverage"]["y"]
    wid_y = curves["width"]["y"]
    cov_std = curves["coverage"]["std"]
    wid_std = curves["width"]["std"]

    has_cov = cov_y is not None and np.any(_finite_mask(cov_y))
    has_wid = wid_y is not None and np.any(_finite_mask(wid_y))

    if not has_cov and not has_wid:
        ax_d.text(
            0.5, 0.5, "Coverage / CI width\nnot available in CSV",
            transform=ax_d.transAxes, ha="center", va="center", fontsize=11
        )
        style_axis(
            ax_d,
            xlabel=r"Latent dimension $R$",
            ylabel="Coverage / interval width",
            title="",
        )
        ax_d.set_xticks(list(map(int, x.tolist())))
    else:
        ax_d2 = ax_d.twinx()

        handles = []
        labels = []

        if has_cov:
            plot_line_with_band(
                ax_d, x, cov_y, cov_std,
                color=c_cov, label="Coverage", marker="o"
            )
            ax_d.axhline(
                nominal_coverage,
                linestyle="--",
                linewidth=1.1,
                color=c_cov,
                alpha=0.6,
            )
            handles.append(ax_d.lines[-1])
            labels.append("Coverage")
            cov_lo, cov_hi = _pad_ylim(cov_y, frac=0.10)
            cov_lo = min(cov_lo, nominal_coverage - 0.01)
            cov_hi = max(cov_hi, nominal_coverage + 0.01)
            ax_d.set_ylim(cov_lo, cov_hi)
            ax_d.set_ylabel(curves["coverage"]["ylabel"], color=c_cov)
            ax_d.tick_params(axis="y", colors=c_cov)

        if has_wid:
            mask = _finite_mask(wid_y)
            xx = np.asarray(x[mask], dtype=float)
            yy = np.asarray(wid_y[mask], dtype=float)

            ax_d2.plot(
                xx, yy,
                color=c_wid,
                linewidth=2.2,
                marker="s",
                markersize=5.0,
                label="CI width",
            )
            if wid_std is not None:
                ss = np.asarray(wid_std[mask], dtype=float)
                ax_d2.fill_between(xx, yy - ss, yy + ss, color=c_wid, alpha=0.14, linewidth=0)

            handles.append(ax_d2.lines[-1])
            labels.append("CI width")
            ax_d2.set_ylim(*_pad_ylim(wid_y, frac=0.14))
            ax_d2.set_ylabel(curves["width"]["ylabel"], color=c_wid)
            ax_d2.tick_params(axis="y", colors=c_wid)

        style_axis(
            ax_d,
            xlabel=r"Latent dimension $R$",
            ylabel=curves["coverage"]["ylabel"] if has_cov else "Coverage",
            title="",
        )
        ax_d.set_xticks(list(map(int, x.tolist())))
        ax_d.legend(handles, labels, loc="best", frameon=True)

    if figure_title.strip():
        fig.suptitle(figure_title, y=0.985)

    plt.tight_layout(rect=(0, 0, 1, 0.97 if figure_title.strip() else 1))

    add_panel_label_outside(fig, ax_a, "(a)")
    add_panel_label_outside(fig, ax_b, "(b)")
    add_panel_label_outside(fig, ax_c, "(c)")
    add_panel_label_outside(fig, ax_d, "(d)")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot FPCA dim sweep as a publication-ready 2x2 figure.")
    ap.add_argument(
        "--csv",
        type=str,
        default="",
        help="Path to summary CSV. If omitted, --sweep_root is used for auto-detection.",
    )
    ap.add_argument(
        "--sweep_root",
        type=str,
        default="../../result_out/fpca_dim_sweep_tm_outputs",
        help="Sweep output root; used when --csv is not provided.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="../../result_out/fpca_dim_sweep_tm_outputs/",
        help="Output directory for figures. Default: <csv_parent>/plots",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional figure super-title.",
    )
    ap.add_argument(
        "--nominal_coverage",
        type=float,
        default=0.95,
        help="Nominal coverage line for panel (d). Default: 0.95",
    )
    args = ap.parse_args()

    if args.csv.strip():
        csv_path = Path(args.csv).resolve()
    else:
        csv_path = pick_input_csv(Path(args.sweep_root).resolve())

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir.strip() else (csv_path.parent / "plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    x, curves, mode = load_curves(df)

    out_png = out_dir / "fpca_dim_sweep_2x2.png"
    out_pdf = out_dir / "fpca_dim_sweep_2x2.pdf"

    make_2x2_figure(
        x=x,
        curves=curves,
        out_png=out_png,
        out_pdf=out_pdf,
        figure_title=args.title,
        nominal_coverage=float(args.nominal_coverage),
    )

    print(f"[DONE] mode={mode}")
    print(f"[DONE] input_csv={csv_path}")
    print(f"[DONE] out_dir={out_dir}")
    print(f"[SAVE] {out_png}")
    print(f"[SAVE] {out_pdf}")


if __name__ == "__main__":
    main()