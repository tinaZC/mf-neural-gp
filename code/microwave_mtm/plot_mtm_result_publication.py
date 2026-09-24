#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final publication-quality MTM 2x2 figure.

(a) Microwave metamaterial unit geometry used in the benchmark
(b) Representative test response, historical seed-42 run, test idx=7
(c) Formal 20-run RMSE distributions
(d) Formal paired RMSE reductions with Student-t 95% CI

Important:
- No training.
- Panels (a,b) are reconstructed directly from saved prediction arrays.
- Raw predictive bands use the original make_ci_bands_for_curve() helper.
- Panels (c,d) use exactly seed_1 ... seed_20 authoritative reports.
- Final PDF is true vector graphics.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MTM_CODE_DIR = PROJECT_ROOT / "code" / "microwave_mtm"
if str(MTM_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(MTM_CODE_DIR))

# Use the original uncertainty-band implementation.
from mf_utils import make_ci_bands_for_curve
from mf_train_microwave_mtm import (
    ri_to_mag_db,
    std_ri_to_mag_db_approx,
)


EXAMPLE_ROOT = (
    PROJECT_ROOT
    / "result_out"
    / "final_analysis"
    / "frozen_inputs"
    / "mtm"
)

PRED_TEST = EXAMPLE_ROOT / "pred_arrays" / "test"

FORMAL_CSV = EXAMPLE_ROOT / "formal_seed_values.csv"

FORMAL_ROOT = (
    PROJECT_ROOT
    / "result_out"
    / "final_runs"
    / "original"
    / "mtm"
    / "hf50_lfx10"
)

OUT_DIR = (
    PROJECT_ROOT
    / "result_out"
    / "final_analysis"
    / "figures"
)

OUT_PNG = OUT_DIR / "fig_mtm_result.png"
OUT_PDF = OUT_DIR / "fig_mtm_result.pdf"
MANUSCRIPT_PNG = OUT_DIR / "_fig_mtm_result.png"
MANUSCRIPT_PDF = OUT_DIR / "_fig_mtm_result.pdf"

OUT_CSV_C = OUT_DIR / "fig_mtm_result_panel_c_stats.csv"
OUT_CSV_D = OUT_DIR / "fig_mtm_result_panel_d_stats.csv"


# ============================================================
# Style
# ============================================================

COLOR_HF = "#0072B2"
COLOR_AR1 = "#E69F00"
COLOR_OURS = "#009E73"

# representative-spectrum colors
COLOR_TARGET = "#0072B2"
COLOR_LF = "#E69F00"
COLOR_MF = "#009E73"


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,

        "axes.linewidth": 0.8,
        "lines.linewidth": 1.7,

        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


# ============================================================
# Historical representative examples
# ============================================================

def require_file(p: Path) -> Path:
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_example_arrays():
    axis = np.load(require_file(EXAMPLE_ROOT / "pred_arrays" / "axis.npy"))

    y_true = np.load(require_file(PRED_TEST / "y_true.npy"))
    y_lf = np.load(require_file(PRED_TEST / "y_lf.npy"))
    y_pred = np.load(require_file(PRED_TEST / "y_pred__mf_student.npy"))
    std_raw = np.load(require_file(PRED_TEST / "std_raw__mf_student.npy"))

    summary_path = require_file(PRED_TEST / "rmse_sample__summary.csv")
    summary = pd.read_csv(summary_path)

    expected_shape = y_true.shape

    for name, arr in [
        ("y_lf", y_lf),
        ("y_pred", y_pred),
        ("std_raw", std_raw),
    ]:
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} shape {arr.shape} != y_true shape {expected_shape}"
            )

    if y_true.ndim != 2:
        raise ValueError(f"Expected y_true as (N,K), got {y_true.shape}")

    if axis.ndim != 1:
        raise ValueError(
            f"Expected 1-D frequency axis, got {axis.shape}"
        )

    K_axis = int(axis.size)
    y_dim = int(y_true.shape[1])

    if y_dim not in (K_axis, 2 * K_axis):
        raise ValueError(
            f"Unsupported MTM response layout: "
            f"axis has K={K_axis}, while spectra have dim={y_dim}. "
            f"Expected K or 2K."
        )

    layout = "single" if y_dim == K_axis else "real-imag"
    print(
        f"[INFO] representative response layout: {layout}; "
        f"K={K_axis}, stored_dim={y_dim}"
    )

    if "idx" not in summary.columns:
        raise ValueError(
            f"'idx' missing from {summary_path}; columns={list(summary.columns)}"
        )

    if len(summary) != y_true.shape[0]:
        raise ValueError(
            f"summary rows={len(summary)} != prediction rows={y_true.shape[0]}"
        )

    return axis, y_true, y_lf, y_pred, std_raw, summary


def frequency_axis_ghz(axis: np.ndarray) -> np.ndarray:
    """
    Convert axis to GHz only when it is clearly in Hz.
    The decision is printed explicitly.
    """
    axis = np.asarray(axis, dtype=float)

    mx = float(np.nanmax(np.abs(axis)))

    if mx > 1e6:
        print("[INFO] axis detected in Hz -> converting to GHz")
        return axis / 1e9

    if mx <= 100.0:
        print("[INFO] axis detected as GHz-scale -> using directly")
        return axis

    raise ValueError(
        f"Ambiguous frequency-axis units: max(abs(axis))={mx}"
    )


def position_for_idx(summary: pd.DataFrame, selected_idx: int) -> int:
    hits = np.where(
        summary["idx"].to_numpy(dtype=int) == int(selected_idx)
    )[0]

    if hits.size != 1:
        raise ValueError(
            f"Expected exactly one row for idx={selected_idx}; got {hits.size}"
        )

    return int(hits[0])


def plot_example_panel(
    ax,
    freq_ghz,
    y_true,
    y_lf,
    y_pred,
    std_raw,
    summary,
    selected_idx,
    ci_level=0.95,
):
    pos = position_for_idx(summary, selected_idx)

    # --------------------------------------------------------
    # Stored MTM responses may use the RI representation:
    #
    #   [Real(f_1...f_K), Imag(f_1...f_K)]
    #
    # Reproduce the exact plotting transformation used by
    # mf_train_microwave_mtm.py:
    #
    #   RI mean -> magnitude in dB
    #   RI std  -> approximate magnitude-dB std (delta method)
    #   dB mean/std -> predictive CI
    # --------------------------------------------------------

    K_axis = int(np.asarray(freq_ghz).size)

    yt_raw = np.asarray(y_true[pos], dtype=np.float32)
    yl_raw = np.asarray(y_lf[pos], dtype=np.float32)
    yp_raw = np.asarray(y_pred[pos], dtype=np.float32)
    ys_raw = np.asarray(std_raw[pos], dtype=np.float32)

    yt = ri_to_mag_db(
        yt_raw[None, :],
        K_axis,
    )[0].astype(float)

    yl = ri_to_mag_db(
        yl_raw[None, :],
        K_axis,
    )[0].astype(float)

    yp = ri_to_mag_db(
        yp_raw[None, :],
        K_axis,
    )[0].astype(float)

    std_db = std_ri_to_mag_db_approx(
        yp_raw[None, :],
        ys_raw[None, :],
        K_axis,
    )

    if std_db is None:
        raise RuntimeError(
            f"Could not convert raw RI uncertainty to dB space "
            f"for representative idx={selected_idx}"
        )

    ys_db = np.asarray(
        std_db[0],
        dtype=float,
    )

    if (
        yt.size != K_axis
        or yl.size != K_axis
        or yp.size != K_axis
        or ys_db.size != K_axis
    ):
        raise RuntimeError(
            f"Post-conversion dimension mismatch for idx={selected_idx}: "
            f"K={K_axis}, "
            f"HF={yt.size}, LF={yl.size}, MF={yp.size}, std={ys_db.size}"
        )

    bands = make_ci_bands_for_curve(
        yp,
        ys_db,
        float(ci_level),
    )

    lo = np.asarray(
        bands["lo"],
        dtype=float,
    )

    hi = np.asarray(
        bands["hi"],
        dtype=float,
    )

    ax.plot(
        freq_ghz,
        yt,
        color=COLOR_TARGET,
        linewidth=1.7,
        label="HF target",
        zorder=4,
    )

    ax.plot(
        freq_ghz,
        yl,
        color=COLOR_LF,
        linewidth=1.5,
        label="LF",
        zorder=3,
    )

    ax.fill_between(
        freq_ghz,
        lo,
        hi,
        color=COLOR_MF,
        alpha=0.15,
        linewidth=0,
        zorder=1,
    )

    ax.plot(
        freq_ghz,
        yp,
        color=COLOR_MF,
        linewidth=1.8,
        label="Neural-GP MF",
        zorder=5,
    )

    ax.set_xlabel(r"Frequency $f$ (GHz)")
    ax.set_ylabel(r"$20\log_{10}|S_{21}|$ (dB)")

    ax.grid(False)

    ax.set_xlim(
        float(np.min(freq_ghz)),
        float(np.max(freq_ghz)),
    )

    # preserve natural data extent with modest headroom
    finite_all = np.concatenate([
        yt[np.isfinite(yt)],
        yl[np.isfinite(yl)],
        yp[np.isfinite(yp)],
        lo[np.isfinite(lo)],
        hi[np.isfinite(hi)],
    ])

    ymin = float(np.min(finite_all))
    ymax = float(np.max(finite_all))
    span = max(ymax - ymin, 1e-9)

    ax.set_ylim(
        ymin - 0.05 * span,
        ymax + 0.08 * span,
    )

    ax.legend(
        frameon=False,
        loc="lower left",
    )

    # RMSE values from the historical selection summary, if present
    row = summary.iloc[pos]

    rmse_lf = (
        float(row["rmse_lf"])
        if "rmse_lf" in summary.columns
        else float("nan")
    )

    rmse_mf = (
        float(row["rmse_mf_student"])
        if "rmse_mf_student" in summary.columns
        else float("nan")
    )

    print(
        f"[INFO] representative idx={selected_idx}: "
        f"row={pos}, "
        f"LF RMSE={rmse_lf:.8g}, "
        f"MF RMSE={rmse_mf:.8g}"
    )


# ============================================================
# Formal 20-run data
# ============================================================

def load_formal_runs() -> pd.DataFrame:
    if FORMAL_CSV.is_file():
        df = pd.read_csv(FORMAL_CSV)
        required = {"seed", "hf_only_rmse", "ar1_rmse", "ours_rmse", "reduction_vs_hf_pct", "reduction_vs_ar1_pct"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{FORMAL_CSV}: missing columns {sorted(missing)}")
        if sorted(df["seed"].astype(int).tolist()) != list(range(1, 21)):
            raise ValueError(f"{FORMAL_CSV}: expected exactly seeds 1..20")
        return df.rename(columns={"hf_only_rmse": "HF-only", "ar1_rmse": "AR1 / co-kriging", "ours_rmse": "Neural-GP MF"})

    expected_names = {
        f"seed_{i}"
        for i in range(1, 21)
    }

    if not FORMAL_ROOT.is_dir():
        raise FileNotFoundError(FORMAL_ROOT)

    seed_dirs = [
        p
        for p in FORMAL_ROOT.iterdir()
        if p.is_dir()
    ]

    actual_names = {
        p.name
        for p in seed_dirs
    }

    if actual_names != expected_names:
        raise ValueError(
            "Expected exactly seed_1 ... seed_20 under "
            f"{FORMAL_ROOT}; found {sorted(actual_names)}"
        )

    rows = []

    for seed in range(1, 21):
        rp = (
            FORMAL_ROOT
            / f"seed_{seed}"
            / "report.json"
        )

        require_file(rp)

        with rp.open("r", encoding="utf-8") as f:
            report = json.load(f)

        if int(report["seed"]) != seed:
            raise ValueError(
                f"{rp}: expected seed={seed}, got {report['seed']}"
            )

        rmse = report["metrics"]["y_rmse"]

        hf = float(rmse["hf_only"])
        ar1 = float(rmse["ar1"])
        ours = float(rmse["ours"])

        if not all(
            math.isfinite(x)
            for x in (hf, ar1, ours)
        ):
            raise ValueError(
                f"{rp}: non-finite RMSE values"
            )

        if hf == 0.0 or ar1 == 0.0:
            raise ValueError(
                f"{rp}: comparator RMSE cannot be zero"
            )

        rows.append({
            "seed": seed,
            "HF-only": hf,
            "AR1 / co-kriging": ar1,
            "Neural-GP MF": ours,
            "reduction_vs_hf_pct":
                100.0 * (hf - ours) / hf,
            "reduction_vs_ar1_pct":
                100.0 * (ar1 - ours) / ar1,
        })

    df = pd.DataFrame(rows)

    if len(df) != 20:
        raise ValueError(
            f"Expected exactly 20 formal runs, got {len(df)}"
        )

    return df


# ============================================================
# Panel (c)
# ============================================================

def plot_panel_c(ax, df: pd.DataFrame):
    labels = [
        "HF-only",
        "AR1 / co-kriging",
        "Neural-GP MF",
    ]

    short_labels = [
        "HF-only",
        "AR1 /\nco-kriging",
        "Neural-GP\nMF",
    ]

    colors = [
        COLOR_HF,
        COLOR_AR1,
        COLOR_OURS,
    ]

    vals = [
        df[label].to_numpy(dtype=float)
        for label in labels
    ]

    pos = np.arange(1, 4, dtype=float)

    bp = ax.boxplot(
        vals,
        positions=pos,
        widths=0.42,
        patch_artist=True,
        showfliers=True,
        medianprops={"linewidth": 1.8},
        whiskerprops={"linewidth": 1.3},
        capprops={"linewidth": 1.3},
        boxprops={"linewidth": 1.5},
        flierprops={
            "markersize": 3.5,
            "alpha": 0.50,
        },
    )

    for i, color in enumerate(colors):
        box = bp["boxes"][i]
        box.set_facecolor(color)
        box.set_alpha(0.15)
        box.set_edgecolor(color)

        bp["medians"][i].set_color(color)

        for w in bp["whiskers"][2*i:2*i+2]:
            w.set_color(color)

        for c in bp["caps"][2*i:2*i+2]:
            c.set_color(color)

        if i < len(bp["fliers"]):
            bp["fliers"][i].set_markeredgecolor(color)
            bp["fliers"][i].set_markerfacecolor(color)

    rng = np.random.default_rng(0)

    all_vals = np.concatenate(vals)
    ymin = float(np.min(all_vals))
    ymax = float(np.max(all_vals))
    span = max(ymax - ymin, 1e-12)

    stats_rows = []

    for i, (label, arr, color) in enumerate(
        zip(labels, vals, colors),
        start=1,
    ):
        x = rng.normal(
            loc=float(i),
            scale=0.035,
            size=arr.size,
        )

        ax.scatter(
            x,
            arr,
            s=13,
            alpha=0.50,
            linewidths=0,
            color=color,
            zorder=3,
        )

        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1))

        stats_rows.append({
            "method": label,
            "n": int(arr.size),
            "mean": mean,
            "sample_std": sd,
        })

        ax.text(
            float(i),
            float(np.max(arr)) + 0.045 * span,
            f"{mean:.2e}\n± {sd:.1e}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            color=color,
        )

    ax.set_xticks(pos)
    ax.set_xticklabels(short_labels)

    ax.set_ylabel("Test RMSE")
    ax.grid(False)

    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
    )

    ax.set_xlim(
        0.45,
        3.55,
    )

    ax.set_ylim(
        ymin - 0.07 * span,
        ymax + 0.23 * span,
    )

    ax.text(
        0.97,
        0.04,
        "n=20",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    return pd.DataFrame(stats_rows)


# ============================================================
# Panel (d)
# ============================================================

def summarize_reduction(
    values: np.ndarray,
    comparator: str,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    n = int(values.size)

    if n != 20:
        raise ValueError(
            f"{comparator}: expected n=20, got {n}"
        )

    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))

    half = (
        float(t.ppf(0.975, n - 1))
        * sd
        / math.sqrt(n)
    )

    return {
        "comparator": comparator,
        "n": n,
        "mean_reduction_pct": mean,
        "sample_std_pct": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def plot_panel_d(ax, df: pd.DataFrame):
    rows = [
        summarize_reduction(
            df["reduction_vs_hf_pct"].to_numpy(dtype=float),
            "HF-only",
        ),
        summarize_reduction(
            df["reduction_vs_ar1_pct"].to_numpy(dtype=float),
            "AR1 / co-kriging",
        ),
    ]

    summary = pd.DataFrame(rows)

    means = summary[
        "mean_reduction_pct"
    ].to_numpy(dtype=float)

    ci_lo = summary[
        "ci95_low"
    ].to_numpy(dtype=float)

    ci_hi = summary[
        "ci95_high"
    ].to_numpy(dtype=float)

    yerr = np.vstack([
        means - ci_lo,
        ci_hi - means,
    ])

    x = np.array(
        [0.42, 0.74],
        dtype=float,
    )

    colors = [
        COLOR_HF,
        COLOR_AR1,
    ]

    ax.bar(
        x,
        means,
        width=0.12,
        color=colors,
        linewidth=0,
        yerr=yerr,
        error_kw={
            "ecolor": "#333333",
            "elinewidth": 1.3,
            "capsize": 4,
            "capthick": 1.3,
        },
    )

    for xi, mean, hi, color in zip(
        x,
        means,
        ci_hi,
        colors,
    ):
        ax.text(
            xi,
            hi + 0.24,
            f"{mean:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
        )

    ax.set_xlim(
        0.26,
        0.90,
    )

    ax.set_ylim(
        0.0,
        max(
            13.4,
            float(np.max(ci_hi)) + 1.1,
        ),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([
        "vs HF-only",
        "vs AR1/co-kriging",
    ])

    ax.set_ylabel(
        "Relative RMSE reduction (%)"
    )

    ax.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        alpha=0.30,
    )

    ax.grid(False)

    ax.text(
        0.97,
        0.04,
        "n=20",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    return summary


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    apply_style()

    # --------------------------------------------------------
    # Load historical representative arrays
    # --------------------------------------------------------
    (
        axis,
        y_true,
        y_lf,
        y_pred,
        std_raw,
        summary_examples,
    ) = load_example_arrays()

    freq_ghz = frequency_axis_ghz(axis)

    # --------------------------------------------------------
    # Load formal 20-run statistics
    # --------------------------------------------------------
    formal_df = load_formal_runs()

    print(
        "[INFO] formal MTM runs:",
        len(formal_df),
    )

    print(
        "[INFO] representative CI level: 0.95 "
        "(raw/native predictive uncertainty)"
    )

    # --------------------------------------------------------
    # Build true vector 2x2 figure
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.2, 7.8),
    )

    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # --------------------------------------------------------
    # Panel (a): microwave metamaterial unit geometry.
    # This is the original benchmark geometry figure used in
    # the manuscript; no model inference is involved.
    # --------------------------------------------------------
    unit_img_path = OUT_DIR / "mtm_unit_geometry.png"

    if not unit_img_path.is_file():
        raise FileNotFoundError(unit_img_path)

    unit_img = plt.imread(unit_img_path)

    ax_a.imshow(unit_img)
    ax_a.set_axis_off()
    ax_a.set_anchor("C")

    plot_example_panel(
        ax_b,
        freq_ghz,
        y_true,
        y_lf,
        y_pred,
        std_raw,
        summary_examples,
        selected_idx=7,
        ci_level=0.95,
    )

    stats_c = plot_panel_c(
        ax_c,
        formal_df,
    )

    stats_d = plot_panel_d(
        ax_d,
        formal_df,
    )

    # --------------------------------------------------------
    # Panel labels
    # --------------------------------------------------------
    for ax, label in zip(
        [ax_a, ax_b, ax_c, ax_d],
        ["(a)", "(b)", "(c)", "(d)"],
    ):
        ax.text(
            -0.12,
            1.04,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.085,
        top=0.965,
        wspace=0.26,
        hspace=0.34,
    )

    # --------------------------------------------------------
    # Save vector PDF + 300-dpi PNG
    # --------------------------------------------------------
    fig.savefig(
        OUT_PNG,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(MANUSCRIPT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(MANUSCRIPT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats_c.to_csv(
        OUT_CSV_C,
        index=False,
        encoding="utf-8",
    )

    stats_d.to_csv(
        OUT_CSV_D,
        index=False,
        encoding="utf-8",
    )

    print("\n===== Panel (c) formal statistics =====")
    print(
        stats_c.to_string(
            index=False
        )
    )

    print("\n===== Panel (d) formal statistics =====")
    print(
        stats_d.to_string(
            index=False
        )
    )

    print("\n[SAVE]", OUT_PNG)
    print("[SAVE]", OUT_PDF)
    print("[SAVE]", OUT_CSV_C)
    print("[SAVE]", OUT_CSV_D)


if __name__ == "__main__":
    main()
