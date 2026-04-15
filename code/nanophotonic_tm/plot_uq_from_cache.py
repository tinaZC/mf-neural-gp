#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_uq_from_cache.py

Generate publication-oriented UQ figures from ONE cache file (uq_cache_v1.npz):

Fig1: Reliability curve (TEST), RAW uncertainty (no post-hoc calibration curve shown)
Fig2: Coverage–Width trade-off at a target CI (TEST), RAW vs post-hoc CAL (pooled conformal on VAL)

Design choices (strict, no silent fallback):
- Reads ONE cache npz that contains y_val/y_test, predictions and std_raw for methods.
- Calibration is DONE HERE (post-hoc) using pooled conformal on VAL:
    r = |e| / std_raw
    q(ci) = quantile(r, ci)
    half_cal = q(ci) * std_raw_test

Defaults are provided so running without args is possible.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


METHODS = ("hf_only", "ar1", "ours")


# -----------------------------
# Math helpers
# -----------------------------
def _norminv(p: float) -> float:
    """
    Approximation of the inverse CDF of the standard normal distribution.
    Peter J. Acklam's algorithm (high accuracy for double precision).
    No scipy / no numpy.erfinv required.
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1), got {p}")

    a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = np.sqrt(-2.0 * np.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return float(num / den)

    if p > phigh:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return float(-(num / den))

    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return float(num / den)


def z_from_ci(ci: float) -> float:
    """Two-sided standard normal quantile: P(|Z|<=z)=ci. z = Phi^{-1}((1+ci)/2)."""
    ci = float(ci)
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0,1), got {ci}")
    p = 0.5 * (1.0 + ci)
    return _norminv(p)


def pooled_conformal_q(r_val: np.ndarray, ci: float) -> float:
    """q(ci) from pooled r on VAL, where r = |e|/std."""
    r = np.asarray(r_val, dtype=np.float64).reshape(-1)
    if r.size == 0:
        raise ValueError("Empty r_val for pooled conformal.")
    return float(np.quantile(r, float(ci)))


def coverage_and_width(e_abs: np.ndarray, half: np.ndarray) -> Tuple[float, float]:
    """Return (coverage_mean, width_mean) over all (N,K)."""
    if e_abs.shape != half.shape:
        raise ValueError(f"shape mismatch: e_abs={e_abs.shape}, half={half.shape}")
    cov = float(np.mean(e_abs <= half))
    width = float(np.mean(2.0 * half))
    return cov, width


def ece_from_curve(ci_grid: List[float], emp_cov: List[float]) -> float:
    """Expected Calibration Error (ECE) for a reliability curve: mean |empirical - nominal|."""
    if len(ci_grid) != len(emp_cov):
        raise ValueError(f"ECE length mismatch: ci_grid={len(ci_grid)} emp_cov={len(emp_cov)}")
    ci = np.asarray(ci_grid, dtype=np.float64)
    ec = np.asarray(emp_cov, dtype=np.float64)
    return float(np.mean(np.abs(ec - ci)))


# -----------------------------
# Cache IO (uq_cache_v1.npz)
# -----------------------------
def need_key(d: np.lib.npyio.NpzFile, k: str) -> np.ndarray:
    if k not in d.files:
        raise KeyError(f"Missing key '{k}'. Available keys(head50)={list(d.files)[:50]}")
    return d[k]


def load_cache(cache_npz: Path) -> Dict[str, np.ndarray]:
    if not cache_npz.exists():
        raise FileNotFoundError(f"cache not found: {cache_npz}")
    d = np.load(cache_npz, allow_pickle=False)
    out: Dict[str, np.ndarray] = {}

    out["y_val"] = need_key(d, "y_val").astype(np.float64)
    out["y_test"] = need_key(d, "y_test").astype(np.float64)

    if "axis" in d.files:
        out["axis"] = need_key(d, "axis").astype(np.float64).reshape(-1)

    for m in METHODS:
        out[f"y_pred_val__{m}"] = need_key(d, f"y_pred_val__{m}").astype(np.float64)
        out[f"y_pred_test__{m}"] = need_key(d, f"y_pred_test__{m}").astype(np.float64)
        out[f"std_raw_val__{m}"] = need_key(d, f"std_raw_val__{m}").astype(np.float64)
        out[f"std_raw_test__{m}"] = need_key(d, f"std_raw_test__{m}").astype(np.float64)

    # strict shape checks
    yv = out["y_val"]
    yt = out["y_test"]
    if yv.ndim != 2 or yt.ndim != 2:
        raise ValueError(f"y_val/y_test must be 2D, got y_val={yv.shape}, y_test={yt.shape}")
    for m in METHODS:
        for k in (f"y_pred_val__{m}", f"std_raw_val__{m}"):
            if out[k].shape != yv.shape:
                raise ValueError(f"{k} shape {out[k].shape} != y_val {yv.shape}")
        for k in (f"y_pred_test__{m}", f"std_raw_test__{m}"):
            if out[k].shape != yt.shape:
                raise ValueError(f"{k} shape {out[k].shape} != y_test {yt.shape}")

    if "axis" in out and out["axis"].size != yt.shape[1]:
        raise ValueError(f"axis length {out['axis'].size} != K={yt.shape[1]}")

    return out


# -----------------------------
# Fig1 / Fig2 from one cache
# -----------------------------
def compute_reliability_curve(cache: Dict[str, np.ndarray], ci_grid: List[float]) -> Dict[str, List[float]]:
    """
    Returns curves[method] = empirical coverage on TEST for each ci in ci_grid,
    using RAW Gaussian intervals: half = z(ci) * std_raw_test.
    """
    yt = cache["y_test"]
    curves: Dict[str, List[float]] = {}

    for m in METHODS:
        mu_t = cache[f"y_pred_test__{m}"]
        st = cache[f"std_raw_test__{m}"]
        et = np.abs(yt - mu_t)

        ys: List[float] = []
        for ci in ci_grid:
            z = z_from_ci(ci)
            half_raw = z * st
            cov_raw, _ = coverage_and_width(et, half_raw)
            ys.append(cov_raw)
        curves[m] = ys

    return curves


def compute_cov_width_points(cache: Dict[str, np.ndarray], ci_level: float) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    points[method]["raw"] = (coverage, width) on TEST at ci_level
    points[method]["cal"] = (coverage, width) on TEST at ci_level (pooled conformal q from VAL)
    """
    yv = cache["y_val"]
    yt = cache["y_test"]

    points: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for m in METHODS:
        mu_v = cache[f"y_pred_val__{m}"]
        mu_t = cache[f"y_pred_test__{m}"]
        sv = cache[f"std_raw_val__{m}"]
        st = cache[f"std_raw_test__{m}"]

        ev = np.abs(yv - mu_v)
        et = np.abs(yt - mu_t)
        rv = ev / (sv + 1e-12)

        z = z_from_ci(ci_level)
        half_raw = z * st
        cov_raw, w_raw = coverage_and_width(et, half_raw)

        q = pooled_conformal_q(rv, ci_level)
        half_cal = q * st
        cov_cal, w_cal = coverage_and_width(et, half_cal)

        points[m] = {"raw": (cov_raw, w_raw), "cal": (cov_cal, w_cal)}

    return points


def plot_fig1_reliability(out_png: Path, ci_grid: List[float], curves: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(6.8, 5.4))
    ax = plt.gca()

    ax.plot([min(ci_grid), max(ci_grid)], [min(ci_grid), max(ci_grid)], linestyle=":", linewidth=2.0, label="ideal")

    label_map = {"hf_only": "HF-only", "ar1": "AR1", "ours": "Ours"}
    ece_lines: List[str] = []

    for m in METHODS:
        raw = curves[m]
        ece = ece_from_curve(ci_grid, raw)
        ece_lines.append(f"{label_map.get(m, m)}  ECE={ece:.3f}")
        ax.plot(ci_grid, raw, linestyle="-", linewidth=2.2, label=label_map.get(m, m))

    ax.set_xlabel("Nominal coverage (CI level)")
    ax.set_ylabel("Empirical coverage (TEST)")
    ax.set_title("Reliability curve (raw uncertainty)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(min(ci_grid), max(ci_grid))

    y_all = [y for m in METHODS for y in curves[m]]
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    span = max(1e-6, y_max - y_min)
    margin = max(0.02, 0.15 * span)
    ax.set_ylim(max(0.0, y_min - margin), min(1.0, y_max + margin))

    ax.text(
        0.02,
        0.98,
        "\n".join(ece_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, linewidth=0.8),
    )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False, fontsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_fig2_cov_width_tradeoff(out_png: Path, points: Dict[str, Dict[str, Tuple[float, float]]], *, ci_level: float) -> None:
    plt.figure(figsize=(6.8, 5.4))
    ax = plt.gca()

    label_map = {"hf_only": "HF-only", "ar1": "AR1", "ours": "Ours"}

    cov_all = []
    for m in METHODS:
        cov_all.append(float(points[m]["raw"][0]))
        cov_all.append(float(points[m]["cal"][0]))
    cmin = float(np.min(cov_all))
    cmax = float(np.max(cov_all))
    span = max(1e-6, cmax - cmin)
    margin = max(0.02, 0.25 * span)
    x_lo = max(0.0, cmin - margin)
    x_hi = min(1.0, cmax + margin)

    ax.axvline(float(ci_level), linestyle=":", linewidth=2.0, label=f"target={ci_level:.2f}")

    for m in METHODS:
        (cr, wr) = points[m]["raw"]
        (cc, wc) = points[m]["cal"]
        name = label_map.get(m, m)

        ax.scatter([cr], [wr], marker="x", s=70, label=f"{name} raw")
        ax.scatter([cc], [wc], marker="o", s=55, label=f"{name} cal")
        ax.plot([cr, cc], [wr, wc], linewidth=1.4)

    ax.set_xlabel("Empirical coverage (TEST)")
    ax.set_ylabel("Mean interval width (TEST)")
    ax.set_title(f"Coverage–Width trade-off at CI={ci_level:.2f}")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(x_lo, x_hi)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False, fontsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--best_cache_npz",
        type=str,
        default="../../result_out/mf_sweep_runs_baseline_nano_tm/hf100_lfx10/seed355/bl0r355/cache/uq_cache_v1.npz",
        help="uq_cache_v1.npz for the BEST configuration (used by Fig1 & Fig2).",
    )
    ap.add_argument("--out_dir", type=str, default="../../result_out/figs_uq/uq_hf100x10_seed355", help="Output directory for PNGs.")
    ap.add_argument("--ci_level", type=float, default=0.95)
    ap.add_argument(
        "--ci_grid",
        type=str,
        default="0.50,0.60,0.70,0.80,0.90,0.95,0.97,0.98,0.99",
        help="Nominal CI grid for Fig1 reliability curve.",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ci_level = float(args.ci_level)
    ci_grid = [float(x) for x in str(args.ci_grid).split(",") if str(x).strip() != ""]
    for c in ci_grid:
        if not (0.0 < c < 1.0):
            raise ValueError(f"Invalid ci_grid value {c}")

    cache = load_cache(Path(args.best_cache_npz))

    curves = compute_reliability_curve(cache, ci_grid)
    plot_fig1_reliability(out_dir / "fig1_reliability_curve.png", ci_grid, curves)

    points = compute_cov_width_points(cache, ci_level)
    plot_fig2_cov_width_tradeoff(out_dir / "fig2_cov_width_tradeoff.png", points, ci_level=ci_level)

    print("[OK] Saved Fig1/2 to:", out_dir)


if __name__ == "__main__":
    main()
