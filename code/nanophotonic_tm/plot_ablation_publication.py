#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication redraw of the TM pathway/component ablation.

Data source
-----------
result_out/ablate_tm/summary_runs.csv

A0: Full Neural-GP MF
A1: Affine LF only
A2: Linear LF + GP(x)
A3: GP LF channel only

No training is performed.

Panel (a):
    15-seed test-RMSE distributions.

Panel (b):
    Wavelength-resolved test RMSE, mean +/- sample SD over the same 15 seeds.

Both panels use broken y-axes because A1 has substantially larger error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]

INPUT_CSV = REPO / "result_out" / "final_analysis" / "frozen_inputs" / "ablation" / "fig_ablation_pathway_seed_values.csv"
SPECTRUM_CSV = REPO / "result_out" / "final_analysis" / "frozen_inputs" / "ablation" / "fig_ablation_pathway_spectrum.csv"
OUT_DIR = REPO / "result_out" / "final_analysis" / "figures"

OUT_PNG = OUT_DIR / "fig_ablation_pathway.png"
OUT_PDF = OUT_DIR / "fig_ablation_pathway.pdf"
OUT_STATS = OUT_DIR / "fig_ablation_pathway_stats.csv"
OUT_SPECTRUM = OUT_DIR / "fig_ablation_pathway_spectrum.csv"

ORDER = ["A0", "A1", "A2", "A3"]

NAMES = {
    "A0": "Full Neural-GP MF",
    "A1": "Affine LF only",
    "A2": "Linear LF + GP(x)",
    "A3": "GP LF channel only",
}

LEGEND_NAMES = {
    "A0": "A0: Full Neural-GP MF",
    "A1": "A1: Affine LF only",
    "A2": "A2: Linear LF + GP(x)",
    "A3": "A3: GP LF channel only",
}

COLORS = {
    "A0": "#009E73",
    "A1": "#E69F00",
    "A2": "#0072B2",
    "A3": "#CC79A7",
}


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def add_break_marks(ax_top, ax_bottom):
    d = 0.012

    kw = dict(
        color="black",
        clip_on=False,
        linewidth=0.9,
    )

    ax_top.plot(
        (-d, +d),
        (-d, +d),
        transform=ax_top.transAxes,
        **kw,
    )
    ax_top.plot(
        (1 - d, 1 + d),
        (-d, +d),
        transform=ax_top.transAxes,
        **kw,
    )

    ax_bottom.plot(
        (-d, +d),
        (1 - d, 1 + d),
        transform=ax_bottom.transAxes,
        **kw,
    )
    ax_bottom.plot(
        (1 - d, 1 + d),
        (1 - d, 1 + d),
        transform=ax_bottom.transAxes,
        **kw,
    )


def padded_limits(lo, hi, frac=0.12, floor_zero=True):
    lo = float(lo)
    hi = float(hi)

    span = hi - lo
    if span <= 0:
        span = max(abs(lo) * 0.1, 1e-4)

    a = lo - frac * span
    b = hi + frac * span

    if floor_zero:
        a = max(0.0, a)

    return a, b


def load_curve(run_dir: Path, source_method: str):
    rmse_dir = run_dir / "rmse_curves"

    axis_file = rmse_dir / "axis.npy"
    curve_file = rmse_dir / f"rmse_test__{source_method}.npy"

    if not axis_file.is_file():
        raise FileNotFoundError(axis_file)

    if not curve_file.is_file():
        raise FileNotFoundError(curve_file)

    axis = np.load(axis_file).astype(float)
    curve = np.load(curve_file).astype(float)

    if axis.ndim != 1 or curve.ndim != 1:
        raise ValueError(
            f"Expected 1-D arrays: axis={axis.shape}, curve={curve.shape}"
        )

    if len(axis) != len(curve):
        raise ValueError(
            f"Axis/curve mismatch: {axis.shape} vs {curve.shape}"
        )

    return axis, curve


def main():
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    required = {
        "seed",
        "pathway_key",
        "y_rmse_test",
    }

    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    df = df[df["pathway_key"].isin(ORDER)].copy()

    # ------------------------------------------------------------
    # Integrity checks
    # ------------------------------------------------------------
    counts = df.groupby("pathway_key").size().reindex(ORDER)

    if counts.isna().any():
        raise RuntimeError(f"Missing pathway(s):\n{counts}")

    if not np.all(counts.values == 15):
        raise RuntimeError(
            f"Expected exactly 15 runs per pathway:\n{counts}"
        )

    seed_sets = {
        key: set(
            df.loc[df["pathway_key"] == key, "seed"]
            .astype(int)
            .tolist()
        )
        for key in ORDER
    }

    if not all(seed_sets[k] == seed_sets["A0"] for k in ORDER):
        raise RuntimeError(f"Seed mismatch: {seed_sets}")

    print("[INFO] matched seeds =", sorted(seed_sets["A0"]))

    # ------------------------------------------------------------
    # Scalar statistics for panel (a)
    # ------------------------------------------------------------
    stat_rows = []

    for key in ORDER:
        vals = (
            df.loc[df["pathway_key"] == key, "y_rmse_test"]
            .astype(float)
            .to_numpy()
        )

        n = len(vals)
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        sem = sd / np.sqrt(n)

        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        ci_lo = mean - tcrit * sem
        ci_hi = mean + tcrit * sem

        stat_rows.append({
            "pathway": key,
            "name": NAMES[key],
            "n": n,
            "mean": mean,
            "sample_std": sd,
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
        })

    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(OUT_STATS, index=False)

    print("\n===== Scalar ablation statistics =====")
    print(stats_df.to_string(index=False))

    # ------------------------------------------------------------
    # Wavelength-resolved curves for panel (b)
    # ------------------------------------------------------------
    if not SPECTRUM_CSV.is_file():
        raise FileNotFoundError(SPECTRUM_CSV)
    spectrum_df = pd.read_csv(SPECTRUM_CSV)
    required_spectrum = {"wavelength"} | {
        f"{key}_{suffix}" for key in ORDER for suffix in ("mean", "sample_std")
    }
    missing_spectrum = required_spectrum - set(spectrum_df.columns)
    if missing_spectrum:
        raise KeyError(f"Missing spectrum columns: {sorted(missing_spectrum)}")
    axis_ref = spectrum_df["wavelength"].to_numpy(dtype=float)
    curve_mean = {key: spectrum_df[f"{key}_mean"].to_numpy(dtype=float) for key in ORDER}
    curve_sd = {key: spectrum_df[f"{key}_sample_std"].to_numpy(dtype=float) for key in ORDER}
    spectrum_df.to_csv(OUT_SPECTRUM, index=False)

    # ------------------------------------------------------------
    # Dynamic broken-axis limits
    # ------------------------------------------------------------

    # Panel (a)
    low_scalar = df[
        df["pathway_key"].isin(["A0", "A2", "A3"])
    ]["y_rmse_test"].astype(float).to_numpy()

    high_scalar = df[
        df["pathway_key"] == "A1"
    ]["y_rmse_test"].astype(float).to_numpy()

    a_bottom_ylim = padded_limits(
        low_scalar.min(),
        low_scalar.max(),
        frac=0.15,
    )

    a_top_ylim = padded_limits(
        high_scalar.min(),
        high_scalar.max(),
        frac=0.25,
    )

    # Panel (b): include +/- 1 sample SD envelopes.
    low_lo = []
    low_hi = []

    for key in ["A0", "A2", "A3"]:
        low_lo.append(curve_mean[key] - curve_sd[key])
        low_hi.append(curve_mean[key] + curve_sd[key])

    low_lo = np.concatenate(low_lo)
    low_hi = np.concatenate(low_hi)

    high_lo = curve_mean["A1"] - curve_sd["A1"]
    high_hi = curve_mean["A1"] + curve_sd["A1"]

    b_bottom_ylim = padded_limits(
        np.nanmin(low_lo),
        np.nanmax(low_hi),
        frac=0.07,
    )

    b_top_ylim = padded_limits(
        np.nanmin(high_lo),
        np.nanmax(high_hi),
        frac=0.07,
    )

    print("\n===== Broken-axis ranges =====")
    print("panel a lower:", a_bottom_ylim)
    print("panel a upper:", a_top_ylim)
    print("panel b lower:", b_bottom_ylim)
    print("panel b upper:", b_top_ylim)

    if b_bottom_ylim[1] >= b_top_ylim[0]:
        print(
            "[WARN] Spectral lower and upper ranges overlap. "
            "Inspect the figure carefully before freezing."
        )

    # ------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(11.0, 4.9))

    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.92, 1.25],
        wspace=0.25,
    )

    gs_a = outer[0].subgridspec(
        2,
        1,
        height_ratios=[1.0, 1.65],
        hspace=0.055,
    )

    ax_a_top = fig.add_subplot(gs_a[0])
    ax_a_bottom = fig.add_subplot(
        gs_a[1],
        sharex=ax_a_top,
    )

    gs_b = outer[1].subgridspec(
        2,
        1,
        height_ratios=[1.0, 1.65],
        hspace=0.055,
    )

    ax_b_top = fig.add_subplot(gs_b[0])
    ax_b_bottom = fig.add_subplot(
        gs_b[1],
        sharex=ax_b_top,
    )

    # ------------------------------------------------------------
    # Panel (a): boxplot + individual runs
    # ------------------------------------------------------------
    values = [
        df.loc[
            df["pathway_key"] == key,
            "y_rmse_test",
        ].astype(float).to_numpy()
        for key in ORDER
    ]

    positions = np.arange(1, 5)

    def draw_boxplot(ax):
        bp = ax.boxplot(
            values,
            positions=positions,
            widths=0.53,
            patch_artist=True,
            showfliers=False,
            medianprops={
                "linewidth": 1.6,
            },
            whiskerprops={
                "linewidth": 1.0,
            },
            capprops={
                "linewidth": 1.0,
            },
            boxprops={
                "linewidth": 1.0,
            },
        )

        for i, key in enumerate(ORDER):
            c = COLORS[key]

            bp["boxes"][i].set_facecolor(c)
            bp["boxes"][i].set_alpha(0.18)
            bp["boxes"][i].set_edgecolor(c)

            bp["medians"][i].set_color(c)

            for j in [2 * i, 2 * i + 1]:
                bp["whiskers"][j].set_color(c)
                bp["caps"][j].set_color(c)

        rng = np.random.default_rng(2026)

        for i, (key, vals) in enumerate(
            zip(ORDER, values),
            start=1,
        ):
            jitter = rng.normal(
                loc=0.0,
                scale=0.035,
                size=len(vals),
            )

            ax.scatter(
                i + jitter,
                vals,
                s=20,
                color=COLORS[key],
                alpha=0.70,
                linewidths=0,
                zorder=3,
            )

    draw_boxplot(ax_a_top)
    draw_boxplot(ax_a_bottom)

    ax_a_top.set_ylim(*a_top_ylim)
    ax_a_bottom.set_ylim(*a_bottom_ylim)

    ax_a_top.spines["bottom"].set_visible(False)
    ax_a_bottom.spines["top"].set_visible(False)

    ax_a_top.tick_params(
        bottom=False,
        labelbottom=False,
    )

    ax_a_bottom.set_xticks(positions)
    ax_a_bottom.set_xticklabels(ORDER)

    add_break_marks(
        ax_a_top,
        ax_a_bottom,
    )

    ax_a_bottom.set_xlabel("Ablation variant")

    # Shared y label for the broken-axis panel.
    ax_a_bottom.set_ylabel("Test RMSE")
    ax_a_bottom.yaxis.set_label_coords(-0.20, 0.83)

    bbox_a = outer[0].get_position(fig)

    ax_a_bottom.text(
        0.97,
        0.06,
        "n = 15",
        transform=ax_a_bottom.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    # ------------------------------------------------------------
    # Panel (b): wavelength-resolved RMSE
    # ------------------------------------------------------------
    for key in ORDER:
        mean = curve_mean[key]
        sd = curve_sd[key]
        c = COLORS[key]

        for ax in [ax_b_top, ax_b_bottom]:
            ax.plot(
                axis_ref,
                mean,
                color=c,
                linewidth=1.9,
                label=LEGEND_NAMES[key],
            )

            ax.fill_between(
                axis_ref,
                mean - sd,
                mean + sd,
                color=c,
                alpha=0.14,
                linewidth=0,
            )

    ax_b_top.set_ylim(*b_top_ylim)
    ax_b_bottom.set_ylim(*b_bottom_ylim)

    ax_b_top.spines["bottom"].set_visible(False)
    ax_b_bottom.spines["top"].set_visible(False)

    ax_b_top.tick_params(
        bottom=False,
        labelbottom=False,
    )

    add_break_marks(
        ax_b_top,
        ax_b_bottom,
    )

    ax_b_bottom.set_xlabel("Wavelength (nm)")

    # Shared y label for the broken-axis panel.
    ax_b_bottom.set_ylabel("Wavelength-resolved RMSE")
    ax_b_bottom.yaxis.set_label_coords(-0.13, 0.83)

    bbox_b = outer[1].get_position(fig)

    # One legend only.
    handles, labels = ax_b_top.get_legend_handles_labels()

    # Remove duplicates caused by plotting on both axes.
    unique = {}
    for h, lab in zip(handles, labels):
        unique.setdefault(lab, h)

    ax_b_top.legend(
        unique.values(),
        unique.keys(),
        frameon=False,
        loc="lower right",
        ncol=2,
        columnspacing=1.0,
        handlelength=2.0,
    )

    ax_b_bottom.text(
        0.98,
        0.05,
        "mean ± sample SD, n = 15",
        transform=ax_b_bottom.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
    )

    # ------------------------------------------------------------
    # No background grids.
    # ------------------------------------------------------------
    for ax in [
        ax_a_top,
        ax_a_bottom,
        ax_b_top,
        ax_b_bottom,
    ]:
        ax.grid(False)

    # Panel labels
    fig.text(
        bbox_a.x0 - 0.015,
        bbox_a.y1 + 0.018,
        "(a)",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    fig.text(
        bbox_b.x0 - 0.015,
        bbox_b.y1 + 0.018,
        "(b)",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
    )
    fig.savefig(OUT_DIR / "_fig_ablation_pathway.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "_fig_ablation_pathway.pdf", bbox_inches="tight")

    fig.savefig(
        OUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\n[SAVE]", OUT_PNG)
    print("[SAVE]", OUT_PDF)
    print("[SAVE]", OUT_STATS)
    print("[SAVE]", OUT_SPECTRUM)


if __name__ == "__main__":
    main()
