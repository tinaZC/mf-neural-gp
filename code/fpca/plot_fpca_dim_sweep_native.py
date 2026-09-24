#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot the revised/native FPCA latent-dimension sweep using the visual style of
the original GitHub script `code/fpca/plot_fpca_dim_sweep_tm.py`.

Scientific rule
---------------
This script DOES NOT recompute or modify any experiment result.  It only reads
the already-generated revised summary CSV:

    result_out/final_analysis/figures/fpca_dim_sweep_native_summary.csv

and redraws the 2x2 figure. Shaded bands are fixed to mean ± 1 sample SD across seeds.

Default outputs
---------------
    result_out/final_analysis/figures/fig_fpca_dim_sweep_native.png
    result_out/final_analysis/figures/fig_fpca_dim_sweep_native.pdf

The loader accepts several compatible column names because the revised/native
summary and the historical sweep used slightly different names.  Native/raw UQ
columns are preferred over calibrated columns whenever both are present.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

DEFAULT_CSV = (
    REPO_ROOT
    / "result_out"
    / "final_analysis"
    / "figures"
    / "fpca_dim_sweep_native_summary.csv"
)
DEFAULT_OUT_DIR = DEFAULT_CSV.parent


# ---------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------
def first_existing(df: pd.DataFrame, names: Iterable[str], required: bool = True) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise KeyError(
            "None of the expected columns were found.\n"
            f"Candidates: {list(names)}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    return None


def numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def resolve_dimension(df: pd.DataFrame) -> str:
    return first_existing(
        df,
        (
            "dim",
            "fpca_dim",
            "fpca_dim_requested",
            "fpca_dim_effective",
            "latent_dim",
            "R",
        ),
    )


def resolve_metric(
    df: pd.DataFrame,
    mean_candidates: Iterable[str],
    std_candidates: Iterable[str] = (),
    low_candidates: Iterable[str] = (),
    high_candidates: Iterable[str] = (),
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    mean_col = first_existing(df, mean_candidates, required=True)
    std_col = first_existing(df, std_candidates, required=False)
    low_col = first_existing(df, low_candidates, required=False)
    high_col = first_existing(df, high_candidates, required=False)

    # CI must be a complete pair.
    if (low_col is None) != (high_col is None):
        low_col = None
        high_col = None

    return mean_col, std_col, low_col, high_col


def band_bounds(
    df: pd.DataFrame,
    mean_col: str,
    std_col: Optional[str],
    low_col: Optional[str],
    high_col: Optional[str],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return mean and a mean ± 1 SD band.

    The FPCA dimension sweep is a sensitivity analysis over 10 random seeds.
    To keep the statistical meaning fixed and reproducible, the shaded band is
    always sample mean ± 1 sample standard deviation when an SD column exists.
    CI columns, even if present in a future summary table, are intentionally
    not substituted silently.
    """
    y = numeric(df, mean_col)

    if std_col is not None:
        sd = numeric(df, std_col)
        return y, y - sd, y + sd

    return y, None, None


# ---------------------------------------------------------------------
# Plot style: intentionally follows plot_fpca_dim_sweep_tm.py
# ---------------------------------------------------------------------
def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 10,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finite_xy(x: np.ndarray, y: np.ndarray):
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], mask


def padded_ylim(y: np.ndarray, frac: float = 0.10) -> Tuple[float, float]:
    arr = y[np.isfinite(y)]
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if np.isclose(lo, hi):
        eps = max(abs(lo) * 0.1, 1e-6)
        return lo - eps, hi + eps
    pad = (hi - lo) * frac
    return lo - pad, hi + pad


def style_axis(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_alpha(0.9)


def add_panel_label_outside(fig, ax, label: str) -> None:
    pos = ax.get_position()
    fig.text(
        pos.x0,
        pos.y1 + 0.008,
        label,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def plot_line_with_band(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    lo: Optional[np.ndarray],
    hi: Optional[np.ndarray],
    *,
    color: str,
    label: str,
    marker: str = "o",
    lw: float = 1.6,
    ms: float = 4.2,
    alpha_fill: float = 0.11,
):
    xx, yy, mask = finite_xy(x, y)
    line, = ax.plot(
        xx,
        yy,
        color=color,
        linewidth=lw,
        marker=marker,
        markersize=ms,
        label=label,
    )

    if lo is not None and hi is not None:
        ll = np.asarray(lo, dtype=float)[mask]
        hh = np.asarray(hi, dtype=float)[mask]
        valid = np.isfinite(ll) & np.isfinite(hh)
        if np.any(valid):
            ax.fill_between(
                xx[valid],
                ll[valid],
                hh[valid],
                color=color,
                alpha=alpha_fill,
                linewidth=0,
            )

    return line


def annotate_best(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    uncertainty: Optional[np.ndarray] = None,
) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return

    idx_valid = np.where(mask)[0]
    idx = int(idx_valid[np.argmin(y[mask])])
    xv = float(x[idx])
    yv = float(y[idx])

    ax.scatter(
        [xv],
        [yv],
        s=75,
        facecolors="white",
        edgecolors=color,
        linewidths=1.8,
        zorder=6,
    )
    ax.axvline(x=xv, linestyle="--", linewidth=1.1, alpha=0.55, color=color)

    if uncertainty is not None and np.isfinite(uncertainty[idx]):
        txt = f"best dim = {int(round(xv))}\n{yv:.4g} ± {float(uncertainty[idx]):.2g}"
    else:
        txt = f"best dim = {int(round(xv))}\n{yv:.4g}"

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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redraw revised/native FPCA dimension sweep using the historical publication style."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Summary CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--basename",
        default="fig_fpca_dim_sweep_native",
        help="Output basename without extension.",
    )
    parser.add_argument(
        "--nominal-coverage",
        type=float,
        default=0.95,
        help="Nominal coverage reference line.",
    )
    parser.add_argument(
        "--annotate-best",
        type=int,
        choices=(0, 1),
        default=0,
        help="1: show best-dimension annotation; 0: suppress it (default, cleaner publication style).",
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Summary CSV is empty: {csv_path}")

    dim_col = resolve_dimension(df)
    df = df.copy()
    df[dim_col] = pd.to_numeric(df[dim_col], errors="coerce")
    df = df.sort_values(dim_col).reset_index(drop=True)
    x = numeric(df, dim_col)

    # (a) HF validation reconstruction RMSE
    recon_cols = resolve_metric(
        df,
        mean_candidates=(
            "recon_rmse_hfval_mean",
            "fpca_recon_rmse_hfval_mean",
            "reconstruction_rmse_mean",
            "recon_mean",
            "recon_rmse_hfval",
        ),
        std_candidates=(
            "recon_rmse_hfval_std",
            "fpca_recon_rmse_hfval_std",
            "reconstruction_rmse_std",
            "recon_std",
        ),
        low_candidates=(
            "recon_rmse_hfval_ci_low",
            "recon_rmse_hfval_ci95_low",
            "reconstruction_rmse_ci_low",
        ),
        high_candidates=(
            "recon_rmse_hfval_ci_high",
            "recon_rmse_hfval_ci95_high",
            "reconstruction_rmse_ci_high",
        ),
    )

    # (b) downstream test RMSE
    rmse_cols = resolve_metric(
        df,
        mean_candidates=(
            "y_rmse_test_mean",
            "test_rmse_mean",
            "rmse_test_mean",
            "rmse_mean",
            "y_rmse_test",
        ),
        std_candidates=(
            "y_rmse_test_std",
            "test_rmse_std",
            "rmse_test_std",
            "rmse_std",
        ),
        low_candidates=(
            "y_rmse_test_ci_low",
            "y_rmse_test_ci95_low",
            "test_rmse_ci_low",
            "rmse_ci_low",
        ),
        high_candidates=(
            "y_rmse_test_ci_high",
            "y_rmse_test_ci95_high",
            "test_rmse_ci_high",
            "rmse_ci_high",
        ),
    )

    # (c) native/raw NLL preferred; calibrated aliases are last-resort compatibility.
    nll_cols = resolve_metric(
        df,
        mean_candidates=(
            "nll_test_raw_mean",
            "nll_test_native_mean",
            "native_nll_mean",
            "nll_raw_mean",
            "nll_mean",
            "nll_test_cal_mean",
            "nll_test_raw",
        ),
        std_candidates=(
            "nll_test_raw_std",
            "nll_test_native_std",
            "native_nll_std",
            "nll_raw_std",
            "nll_std",
            "nll_test_cal_std",
        ),
        low_candidates=(
            "nll_test_raw_ci_low",
            "nll_test_raw_ci95_low",
            "native_nll_ci_low",
            "nll_ci_low",
        ),
        high_candidates=(
            "nll_test_raw_ci_high",
            "nll_test_raw_ci95_high",
            "native_nll_ci_high",
            "nll_ci_high",
        ),
    )

    # (d) native/raw coverage preferred.
    coverage_cols = resolve_metric(
        df,
        mean_candidates=(
            "coverage_test_raw_mean",
            "coverage_test_native_mean",
            "native_coverage_mean",
            "coverage_raw_mean",
            "coverage_mean",
            "coverage_test_cal_mean",
            "coverage_test_raw",
        ),
        std_candidates=(
            "coverage_test_raw_std",
            "coverage_test_native_std",
            "native_coverage_std",
            "coverage_raw_std",
            "coverage_std",
            "coverage_test_cal_std",
        ),
        low_candidates=(
            "coverage_test_raw_ci_low",
            "coverage_test_raw_ci95_low",
            "native_coverage_ci_low",
            "coverage_ci_low",
        ),
        high_candidates=(
            "coverage_test_raw_ci_high",
            "coverage_test_raw_ci95_high",
            "native_coverage_ci_high",
            "coverage_ci_high",
        ),
    )

    width_cols = resolve_metric(
        df,
        mean_candidates=(
            "ci_width_test_raw_mean",
            "width_test_raw_mean",
            "width_test_native_mean",
            "native_width_mean",
            "interval_width_mean",
            "width_raw_mean",
            "width_mean",
            "ci_width_test_cal_mean",
            "ci_width_test_raw",
        ),
        std_candidates=(
            "ci_width_test_raw_std",
            "width_test_raw_std",
            "width_test_native_std",
            "native_width_std",
            "interval_width_std",
            "width_raw_std",
            "width_std",
            "ci_width_test_cal_std",
        ),
        low_candidates=(
            "ci_width_test_raw_ci_low",
            "width_test_raw_ci_low",
            "width_test_raw_ci95_low",
            "native_width_ci_low",
            "width_ci_low",
        ),
        high_candidates=(
            "ci_width_test_raw_ci_high",
            "width_test_raw_ci_high",
            "width_test_raw_ci95_high",
            "native_width_ci_high",
            "width_ci_high",
        ),
    )

    recon_y, recon_lo, recon_hi = band_bounds(df, *recon_cols)
    rmse_y, rmse_lo, rmse_hi = band_bounds(df, *rmse_cols)
    nll_y, nll_lo, nll_hi = band_bounds(df, *nll_cols)
    cov_y, cov_lo_band, cov_hi_band = band_bounds(df, *coverage_cols)
    wid_y, wid_lo, wid_hi = band_bounds(df, *width_cols)

    # For the annotation text, preserve the historical "mean ± std" convention
    # when an actual sample-SD column exists.
    rmse_std = numeric(df, rmse_cols[1]) if rmse_cols[1] else None
    nll_std = numeric(df, nll_cols[1]) if nll_cols[1] else None

    setup_style()

    # Same palette as the referenced GitHub plotting script.
    c_recon = "#4C72B0"
    c_rmse = "#C44E52"
    c_nll = "#55A868"
    c_cov = "#8172B2"
    c_wid = "#E67E22"

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), constrained_layout=False)
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    existing_dims = {int(round(v)) for v in x if np.isfinite(v)}
    xticks = [v for v in (2, 4, 10, 16, 32, 64) if v in existing_dims]
    if not xticks:
        xticks = [int(round(v)) for v in x if np.isfinite(v)]

    # (a)
    plot_line_with_band(
        ax_a, x, recon_y, recon_lo, recon_hi,
        color=c_recon, label="Recon RMSE",
    )
    style_axis(ax_a, r"Latent dimension $R$", "HF validation reconstruction RMSE")
    ax_a.set_xticks(xticks)
    ax_a.set_ylim(*padded_ylim(recon_y, frac=0.12))

    # (b)
    plot_line_with_band(
        ax_b, x, rmse_y, rmse_lo, rmse_hi,
        color=c_rmse, label="Test RMSE",
    )
    if args.annotate_best:
        annotate_best(ax_b, x, rmse_y, color=c_rmse, uncertainty=rmse_std)
    style_axis(ax_b, r"Latent dimension $R$", "Test RMSE")
    ax_b.set_xticks(xticks)
    ax_b.set_ylim(*padded_ylim(rmse_y, frac=0.15))

    # (c)
    plot_line_with_band(
        ax_c, x, nll_y, nll_lo, nll_hi,
        color=c_nll, label="Native NLL",
    )
    if args.annotate_best:
        annotate_best(ax_c, x, nll_y, color=c_nll, uncertainty=nll_std)
    style_axis(ax_c, r"Latent dimension $R$", "Native predictive NLL")
    ax_c.set_xticks(xticks)
    ax_c.set_ylim(*padded_ylim(nll_y, frac=0.15))

    # (d)
    ax_d2 = ax_d.twinx()
    line_cov = plot_line_with_band(
        ax_d, x, cov_y, cov_lo_band, cov_hi_band,
        color=c_cov, label="Coverage", marker="o",
    )
    ax_d.axhline(
        args.nominal_coverage,
        linestyle="--",
        linewidth=1.1,
        color="0.45",
        alpha=0.8,
    )

    xx, yy, mask = finite_xy(x, wid_y)
    line_wid, = ax_d2.plot(
        xx,
        yy,
        color=c_wid,
        linewidth=1.6,
        marker="s",
        markersize=4.0,
        label="Interval width",
    )
    if wid_lo is not None and wid_hi is not None:
        ll = np.asarray(wid_lo, dtype=float)[mask]
        hh = np.asarray(wid_hi, dtype=float)[mask]
        valid = np.isfinite(ll) & np.isfinite(hh)
        if np.any(valid):
            ax_d2.fill_between(
                xx[valid],
                ll[valid],
                hh[valid],
                color=c_wid,
                alpha=0.11,
                linewidth=0,
            )

    style_axis(ax_d, r"Latent dimension $R$", "Native empirical coverage")
    ax_d.set_xticks(xticks)
    cov_lo, cov_hi = padded_ylim(cov_y, frac=0.10)
    cov_lo = min(cov_lo, args.nominal_coverage - 0.01)
    cov_hi = max(cov_hi, args.nominal_coverage + 0.01)
    ax_d.set_ylim(cov_lo, cov_hi)
    ax_d.set_ylabel("Native empirical coverage", color=c_cov)
    ax_d.tick_params(axis="y", colors=c_cov)

    ax_d2.set_ylim(*padded_ylim(wid_y, frac=0.14))
    ax_d2.set_ylabel("Mean prediction interval width", color=c_wid)
    ax_d2.tick_params(axis="y", colors=c_wid)

    ax_d.legend(
        [line_cov, line_wid],
        ["Native coverage", "Native interval width"],
        loc="best",
        frameon=True,
    )

    plt.tight_layout(pad=0.7, w_pad=1.4, h_pad=1.0)

    add_panel_label_outside(fig, ax_a, "(a)")
    add_panel_label_outside(fig, ax_b, "(b)")
    add_panel_label_outside(fig, ax_c, "(c)")
    add_panel_label_outside(fig, ax_d, "(d)")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{args.basename}.png"
    out_pdf = out_dir / f"{args.basename}.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] input = {csv_path}")
    print(f"[DONE] dimension column = {dim_col}")
    print(f"[DONE] recon mean = {recon_cols[0]}")
    print(f"[DONE] RMSE mean = {rmse_cols[0]}")
    print(f"[DONE] NLL mean = {nll_cols[0]}")
    print(f"[DONE] coverage mean = {coverage_cols[0]}")
    print(f"[DONE] interval-width mean = {width_cols[0]}")
    print("[DONE] shaded bands = mean ± 1 sample SD across seeds (when SD columns are available)")
    print(f"[SAVE] {out_png}")
    print(f"[SAVE] {out_pdf}")


if __name__ == "__main__":
    main()
