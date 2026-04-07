#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mf_hf_delta_ar1.py

One-shot baseline comparison on the SAME MF dataset layout:
  1) HF-only (SVGP on HF)
  2) AR1 CoKrig baseline (SVGP_L on LF-all + SVGP_delta on HF residual)
  3) Ours: xLF + delta-student (Stage-I student + Stage-II SVGP residual)

Design goals (per your requirements)
-----------------------------------
- Single standalone script (NO calling other training scripts).
- Can import and use mf_utils.
- Algorithmic logic matches:
    - mf_train_fpca_cokrig_ar1.py  (AR1 baseline)
    - mf_train_fpca_sgp_delta_student.py (Ours delta-student)
- Metrics include rmse, r2, nll, nlpd and UQ coverage/width raw+cal.
- Plot 5 curves: HF_only / AR1 / Ours / HF_gt / LF.

Data layout MUST match the run-through version:
  data_dir/
    wavelengths.npy
    idx_wavelength.npy
    hf/        x_{split}.npy y_{split}.npy t_{split}.npy idx_{split}.npy
    lf_paired/ x_{split}.npy y_{split}.npy t_{split}.npy idx_{split}.npy
    lf_unpaired/ ...

Notes
-----
- We keep variable naming compatibility internally (mu_or_* etc) but reporting keys are:
    hf_only / ar1 / ours
- Plot function from mf_utils is reused as-is (oracle curve slot is used for AR1).

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import inspect
import numpy as np
import hashlib
# For conformal -> equivalent Gaussian std conversion (only for reporting NLL/NLPD).
try:
    from scipy.stats import norm  # type: ignore
except Exception:  # pragma: no cover
    norm = None
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mf_utils import (
    # basic
    set_seed, now_tag, safe_tag, save_pickle, rmse, r2_score, gaussian_nll, gaussian_nlpd, z_from_ci_level, pca_recon_rmse,
    # wavelength + subsample
    load_wavelengths, make_even_subsample_indices, pick_y_sub, upsample_y_sub_to_full, build_linear_interp_weights,
    # uncertainty
    propagate_subsample_var_to_full_y_var,
    ci_coverage_y, ci_width_y, calibrate_sigma_scale,
    # plot
    plot_case_5curves_spectrum, make_ci_bands_for_curve,
    # csv
    CsvLogger,
    # student training/infer helpers
    FeatureMLP,
    train_feature_mlp, mlp_predict_and_features,
    # svgp
    train_svgp_per_dim, predict_svgp_per_dim, save_svgp_bundle,
    # data
    load_split_block, assert_indices_match,
    # dbg
    dbg_block_stats, dbg_student_on_hf_errors,
)


# ---------------------------------------------------------------------
# Robust helpers: sklearn transformers do not accept empty (0-row) arrays.
# lfx=1 datasets can legitimately have LF-unpaired = 0, so we guard transforms.
# ---------------------------------------------------------------------
def safe_scaler_transform(scaler: StandardScaler, X: np.ndarray, *, n_features: Optional[int] = None) -> np.ndarray:
    """Like scaler.transform(X) but returns a correctly-shaped empty array if X has 0 rows."""
    if X is None:
        if n_features is None:
            n_features = int(getattr(scaler, "n_features_in_", 0))
        return np.zeros((0, n_features), dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array for transform, got shape={X.shape}")
    if n_features is None:
        n_features = int(X.shape[1])
    if X.shape[0] == 0:
        return np.zeros((0, n_features), dtype=np.float32)
    return scaler.transform(X).astype(np.float32)


def _rmse_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

def _cov_width_1d(y_gt: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    y_gt = np.asarray(y_gt, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    m = np.isfinite(y_gt) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(m):
        return float("nan"), float("nan")
    cov = float(np.mean((y_gt[m] >= lo[m]) & (y_gt[m] <= hi[m])))
    wid = float(np.mean(hi[m] - lo[m]))
    return cov, wid

# ----------------------------
# Local plot helper (do NOT modify mf_utils)
# ----------------------------
def plot_case_5curves_spectrum_named(
    wl: np.ndarray,
    y_hf_gt: np.ndarray,
    y_lf: np.ndarray,
    y_pred_hf: Optional[np.ndarray],
    y_pred_ar1: Optional[np.ndarray],
    y_pred_delta_svgp: Optional[np.ndarray],
    title: str,
    save_path: Path,
    ci_bands: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    # NEW: show discrete GT points
    show_gt_points: bool = True,
    # NEW: text box with metrics (already formatted by caller)
    metrics_text: Optional[str] = None,
) -> None:
    """Spectrum plot with unified method naming.

    - Always plots: HF_GT, LF_oracle
    - Optionally plots: Pred_HF-only, Pred_AR1(CoKrig), Pred_ours (delta_svgp)
      A prediction curve is skipped if it is None or all-NaN.

    ci_bands keys (if provided):
        - 'hf_only', 'ar1_cokrig', 'delta_svgp' each maps to {'lo':..., 'hi':...}
      A band is skipped if curve is missing.

    Enhancements:
        - CI band colors are matched to the corresponding prediction curve colors.
        - Optionally show HF_GT as scatter points (discrete measurements).
        - Optionally add a metrics annotation textbox.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5.2))
    ax = plt.gca()

    # Always plot GT and LF (no CI bands)
    ax.plot(wl, y_hf_gt, label="HF_GT", zorder=3)
    if show_gt_points:
        ax.scatter(wl, y_hf_gt, s=10, alpha=0.9, linewidths=0, label="HF_GT pts", zorder=5)

    ax.plot(wl, y_lf, label="LF_oracle", zorder=3)

    def _valid(y: Optional[np.ndarray]) -> bool:
        if y is None:
            return False
        y = np.asarray(y)
        return not np.all(np.isnan(y))

    # Plot prediction curves and keep their line handles for CI band coloring
    ln_hf = ln_ar1 = ln_ours = None
    if _valid(y_pred_hf):
        (ln_hf,) = ax.plot(wl, y_pred_hf, label="Pred_HF-only", zorder=4)
    if _valid(y_pred_ar1):
        (ln_ar1,) = ax.plot(wl, y_pred_ar1, label="Pred_AR1(CoKrig)", zorder=4)
    if _valid(y_pred_delta_svgp):
        (ln_ours,) = ax.plot(wl, y_pred_delta_svgp, label="Pred_ours", zorder=4)

    # CI bands (match color to corresponding curve)
    if ci_bands is not None:
        alpha_band = 0.18

        if ln_hf is not None and "hf_only" in ci_bands:
            ax.fill_between(
                wl,
                ci_bands["hf_only"]["lo"],
                ci_bands["hf_only"]["hi"],
                color=ln_hf.get_color(),
                alpha=alpha_band,
                linewidth=0,
                zorder=2,
            )

        if ln_ar1 is not None and "ar1_cokrig" in ci_bands:
            ax.fill_between(
                wl,
                ci_bands["ar1_cokrig"]["lo"],
                ci_bands["ar1_cokrig"]["hi"],
                color=ln_ar1.get_color(),
                alpha=alpha_band,
                linewidth=0,
                zorder=2,
            )

        if ln_ours is not None and "delta_svgp" in ci_bands:
            ax.fill_between(
                wl,
                ci_bands["delta_svgp"]["lo"],
                ci_bands["delta_svgp"]["hi"],
                color=ln_ours.get_color(),
                alpha=alpha_band,
                linewidth=0,
                zorder=2,
            )

    # NEW: metrics textbox
    if metrics_text:
        ax.text(
            0.99,
            0.99,
            metrics_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.18, linewidth=0),
            zorder=10,
        )

    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Spectrum (metric)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=9)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=160)
    plt.close()




# ----------------------------
# Baseline selection helper
# ----------------------------
def parse_methods(methods: str) -> set:
    """Parse --methods into a normalized set among: {'hf_only','ar1','ours'}."""
    if methods is None:
        return {"hf_only", "ar1", "ours"}
    s = str(methods).strip().lower()
    if s == "" or s == "all":
        return {"hf_only", "ar1", "ours"}
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    if any(p == "all" for p in parts):
        return {"hf_only", "ar1", "ours"}
    mapping = {
        "hf": "hf_only",
        "hf-only": "hf_only",
        "hf_only": "hf_only",
        "ar1": "ar1",
        "cokrig": "ar1",
        "co-krig": "ar1",
        "co_krig": "ar1",
        "ours": "ours",
        "delta": "ours",
        "delta_svgp": "ours",
    }
    out = set()
    bad = []
    for p in parts:
        if p not in mapping:
            bad.append(p)
        else:
            out.add(mapping[p])
    if bad:
        raise ValueError(f"Unknown --methods entries: {bad}. Allowed: hf_only,ar1,ours (or all).")
    if not out:
        return {"hf_only", "ar1", "ours"}
    return out

# ----------------------------
# Rewrite results.csv in-place to avoid hf/or/st confusion (do NOT modify mf_utils)
# ----------------------------
def rewrite_results_csv_inplace(results_csv: Path) -> None:
    """Rename mf_utils canonical suffixes in results.csv:
        *_hf -> *_hf_only
        *_or -> *_ar1_cokrig
        *_st -> *_delta_svgp
    """
    results_csv = Path(results_csv)
    if not results_csv.exists():
        return
    import csv
    with results_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    def ren(c: str) -> str:
        if c.endswith("_hf"):
            return c[:-3] + "_hf_only"
        if c.endswith("_or"):
            return c[:-3] + "_ar1_cokrig"
        if c.endswith("_st"):
            return c[:-3] + "_delta_svgp"
        return c
    new_fields = [ren(c) for c in fields]
    new_rows = []
    for r in rows:
        new_rows.append({ren(k): v for k, v in r.items()})
    with results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for r in new_rows:
            writer.writerow(r)



def safe_append_results_csv(results_csv: Path, row: Dict[str, Any]) -> None:
    """Append a single row to results.csv robustly.

    This avoids silent failures if mf_utils.CsvLogger has schema drift.

    Behavior:
      - If file does not exist: create with header=row.keys()
      - If file exists: union existing header with row.keys(); if header expands, rewrite file.
      - Always writes all columns in header order; missing values are left blank.

    NOTE: Values are stringified via str(); numpy scalars are converted to python scalars.
    """
    results_csv = Path(results_csv)
    import csv
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    def _to_str(v: Any) -> str:
        if v is None:
            return ""
        try:
            # numpy scalar -> python scalar
            if hasattr(v, "item") and callable(getattr(v, "item")):
                v = v.item()
        except Exception:
            pass
        if isinstance(v, float):
            # stable float repr (avoid scientific noise in csv diffs)
            return f"{v:.10g}"
        return str(v)

    if not results_csv.exists() or results_csv.stat().st_size == 0:
        fieldnames = list(row.keys())
        with results_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow({k: _to_str(row.get(k, "")) for k in fieldnames})
        return

    # read existing
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        old_fields = list(r.fieldnames or [])
        old_rows = list(r)

    new_fields = old_fields[:]
    for k in row.keys():
        if k not in new_fields:
            new_fields.append(k)

    if new_fields != old_fields:
        # rewrite with expanded header
        with results_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=new_fields)
            w.writeheader()
            for rr in old_rows:
                w.writerow({k: _to_str(rr.get(k, "")) for k in new_fields})
            w.writerow({k: _to_str(row.get(k, "")) for k in new_fields})
    else:
        # simple append
        with results_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=old_fields)
            w.writerow({k: _to_str(row.get(k, "")) for k in old_fields})



# ----------------------------
# Per-K calibration (optional): fit alpha_k per wavelength/frequency point on VAL
# ----------------------------
def calibrate_sigma_scale_per_k(
    y_gt: np.ndarray,
    y_pred: np.ndarray,
    std_raw: np.ndarray,
    ci_level: float,
    *,
    eps: float = 1e-12,
    smooth_win: int = 0,
    alpha_clip: Tuple[float, float] = (0.0, 200.0),
) -> np.ndarray:
    """Return alpha_k (K,) so that mean_k[ 1{|e| <= z * alpha_k * std} ] ~= ci_level on VAL.

    We use a simple per-dimension quantile fit:
        ratio_{i,k} = |e_{i,k}| / (z * std_{i,k} + eps)
        alpha_k = quantile_i(ratio_{i,k}, ci_level)

    Optionally smooth alpha_k with a moving average of window `smooth_win` (odd >=3 recommended).
    """
    y_gt = np.asarray(y_gt, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    std_raw = np.asarray(std_raw, dtype=np.float32)
    if y_gt.shape != y_pred.shape or y_gt.shape != std_raw.shape:
        raise ValueError(f"calibrate_sigma_scale_per_k shape mismatch: y_gt={y_gt.shape} y_pred={y_pred.shape} std={std_raw.shape}")
    if y_gt.ndim != 2:
        raise ValueError(f"calibrate_sigma_scale_per_k expects (N,K), got {y_gt.shape}")

    z = float(z_from_ci_level(float(ci_level)))
    denom = z * np.maximum(std_raw, 0.0) + float(eps)
    ratio = np.abs(y_gt - y_pred) / denom  # (N,K)

    # robust quantile across samples (axis=0)
    alpha_k = np.quantile(ratio, float(ci_level), axis=0).astype(np.float32)

    lo, hi = float(alpha_clip[0]), float(alpha_clip[1])
    if lo is not None:
        alpha_k = np.maximum(alpha_k, lo)
    if hi is not None:
        alpha_k = np.minimum(alpha_k, hi)

    # optional smoothing (moving average)
    sw = int(smooth_win or 0)
    if sw >= 3:
        if sw % 2 == 0:
            sw += 1
        kernel = np.ones(sw, dtype=np.float32) / float(sw)
        pad = sw // 2
        a_pad = np.pad(alpha_k, (pad, pad), mode="edge")
        alpha_k = np.convolve(a_pad, kernel, mode="valid").astype(np.float32)

    return alpha_k



def conformal_q_norm_pooled(
    y_gt: np.ndarray,
    y_pred: np.ndarray,
    std_raw: np.ndarray,
    ci_level: float,
    *,
    eps: float = 1e-12,
) -> float:
    """Normalized split conformal pooled across (N,K).

    score = |e| / (std + eps). We compute a finite-sample corrected quantile using the
    conformal recipe q = ceil((n+1)*ci_level)/n with 'higher' interpolation to avoid
    under-coverage due to quantile interpolation.
    """
    y_gt = np.asarray(y_gt, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    std_raw = np.asarray(std_raw, dtype=np.float32)
    if y_gt.shape != y_pred.shape or y_gt.shape != std_raw.shape:
        raise ValueError(f"shape mismatch: y_gt={y_gt.shape} y_pred={y_pred.shape} std={std_raw.shape}")
    if y_gt.ndim != 2:
        raise ValueError(f"expects (N,K), got {y_gt.shape}")

    denom = np.maximum(std_raw, 0.0) + float(eps)
    scores = (np.abs(y_gt - y_pred) / denom).reshape(-1)  # (N*K,)
    n = int(scores.size)
    if n <= 0:
        return 1.0

    import math
    q_level = math.ceil((n + 1) * float(ci_level)) / float(n)
    q_level = float(min(1.0, max(0.0, q_level)))

    try:
        q = float(np.quantile(scores, q_level, method="higher"))
    except TypeError:
        q = float(np.quantile(scores, q_level, interpolation="higher"))
    return max(0.0, q)



def _build_strata_from_x(
    x: np.ndarray,
    dim: int,
    n_bins: int,
) -> Dict[str, Any]:
    """Build strata definition from x[:,dim].

    If x[:,dim] appears categorical with small unique count, use categorical bins.
    Otherwise use quantile bins.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"x must be (N,D), got {x.shape}")
    D = int(x.shape[1])
    dim = int(dim)
    if dim < 0 or dim >= D:
        raise ValueError(f"stratify dim out of range: dim={dim} D={D}")

    v = x[:, dim].astype(np.float32)
    u = np.unique(v)
    # Treat as categorical if few unique values (common for discrete parameter grids).
    if u.size <= max(12, int(n_bins)):
        u_sorted = np.sort(u)
        cat2bin = {float(val): int(i) for i, val in enumerate(u_sorted)}
        return {"mode": "categorical", "dim": dim, "cats": u_sorted.astype(np.float32), "cat2bin": cat2bin}

    # Quantile bins (continuous)
    n_bins = int(max(2, n_bins))
    qs = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    edges = np.quantile(v, qs)
    edges = np.unique(edges.astype(np.float32))
    if edges.size < 3:
        # fallback to categorical-like behavior
        u_sorted = np.sort(u)
        cat2bin = {float(val): int(i) for i, val in enumerate(u_sorted)}
        return {"mode": "categorical", "dim": dim, "cats": u_sorted.astype(np.float32), "cat2bin": cat2bin}
    return {"mode": "quantile", "dim": dim, "edges": edges}


def _build_coarse_strata_from_x(
    x: np.ndarray,
    dim: int,
    *,
    n_coarse: int = 3,
    n_bins_fallback: int = 3,
) -> Dict[str, Any]:
    """Build a *coarse* strata definition from x[:,dim].

    Goal: make stratified conformal usable with tiny validation sets.

    - If x[:,dim] looks categorical (few uniques), we *merge* categories into
      n_coarse groups by sorting unique values and splitting into nearly-equal
      sized groups.
    - Otherwise we use n_coarse quantile bins.

    Returns a dict compatible with _assign_strata_ids/conformal_q_apply_stratified.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"x must be (N,D), got {x.shape}")
    D = int(x.shape[1])
    dim = int(dim)
    if dim < 0 or dim >= D:
        raise ValueError(f"stratify dim out of range: dim={dim} D={D}")

    n_coarse = int(max(1, n_coarse))

    v = x[:, dim].astype(np.float32)
    u = np.unique(v)

    # Categorical-ish: merge uniques into coarse groups
    if u.size <= max(12, int(n_bins_fallback)):
        u_sorted = np.sort(u).astype(np.float32)
        # if fewer uniques than requested bins, keep as-is
        nb = int(min(n_coarse, int(u_sorted.size)))
        # split indices into nb groups
        groups = np.array_split(np.arange(int(u_sorted.size)), nb)
        cat2bin: Dict[float, int] = {}
        cats_groups: List[List[float]] = []
        for bi, g in enumerate(groups):
            vals = [float(u_sorted[j]) for j in g.tolist()]
            cats_groups.append(vals)
            for val in vals:
                cat2bin[float(val)] = int(bi)
        return {
            "mode": "categorical",
            "dim": dim,
            "cats": u_sorted,
            "cat2bin": cat2bin,
            "coarse_bins": int(nb),
            "cats_groups": cats_groups,
        }

    # Continuous: quantile bins with n_coarse
    nb = int(max(2, n_coarse))
    qs = np.linspace(0.0, 1.0, nb + 1, dtype=np.float32)
    edges = np.quantile(v, qs)
    edges = np.unique(edges.astype(np.float32))
    if edges.size < 3:
        # fallback to ordinary strata builder (categorical-like)
        return _build_strata_from_x(x, dim, n_bins=int(max(2, n_bins_fallback)))
    return {"mode": "quantile", "dim": dim, "edges": edges, "coarse_bins": int(edges.size - 1)}


def _assign_strata_ids(x: np.ndarray, strata: Dict[str, Any]) -> np.ndarray:
    x = np.asarray(x)
    dim = int(strata["dim"])
    v = x[:, dim].astype(np.float32)

    if strata["mode"] == "categorical":
        cat2bin = strata["cat2bin"]
        # unknown categories -> -1
        ids = np.full((v.shape[0],), -1, dtype=np.int32)
        for i, val in enumerate(v):
            ids[i] = int(cat2bin.get(float(val), -1))
        return ids

    edges = np.asarray(strata["edges"], dtype=np.float32)
    # bins: [edges[i], edges[i+1]) except last includes right edge
    ids = np.searchsorted(edges, v, side="right") - 1
    ids = np.clip(ids, 0, int(edges.size) - 2).astype(np.int32)
    return ids


def conformal_q_norm_stratified(
    y_gt: np.ndarray,
    y_pred: np.ndarray,
    std_raw: np.ndarray,
    x_val: np.ndarray,
    ci_level: float,
    *,
    strat_dim: int = 0,
    n_bins: int = 8,
    min_n: int = 30,
    eps: float = 1e-12,
) -> Tuple[float, Dict[str, Any]]:
    """Normalized split conformal with stratification by x[:,strat_dim].

    Returns:
      q_global: pooled q on all val points
      strat: dict containing strata definition and per-bin q values

    Notes:
      - We fit a separate q_bin when that bin has at least `min_n` pooled points (N_bin*K).
      - Otherwise we fall back to q_global for that bin.
    """
    y_gt = np.asarray(y_gt, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    std_raw = np.asarray(std_raw, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)

    if y_gt.shape != y_pred.shape or y_gt.shape != std_raw.shape:
        raise ValueError(f"shape mismatch: y_gt={y_gt.shape} y_pred={y_pred.shape} std={std_raw.shape}")
    if y_gt.ndim != 2:
        raise ValueError(f"expects (N,K), got {y_gt.shape}")
    if x_val.ndim != 2 or x_val.shape[0] != y_gt.shape[0]:
        raise ValueError(f"x_val must be (N,D) matching y, got x_val={x_val.shape} y={y_gt.shape}")

    q_global = float(conformal_q_norm_pooled(y_gt, y_pred, std_raw, ci_level, eps=eps))

    strata = _build_strata_from_x(x_val, strat_dim, n_bins)
    ids = _assign_strata_ids(x_val, strata)

    if strata["mode"] == "categorical":
        n_strata = int(np.asarray(strata["cats"]).size)
    else:
        n_strata = int(np.asarray(strata["edges"]).size) - 1

    q_bins = np.full((n_strata,), q_global, dtype=np.float32)

    # precompute per-sample pooled count in each bin: N_bin*K
    K = int(y_gt.shape[1])
    for b in range(n_strata):
        m = (ids == b)
        n_pool = int(np.sum(m)) * K
        if n_pool >= int(min_n):
            q_bins[b] = float(conformal_q_norm_pooled(y_gt[m], y_pred[m], std_raw[m], ci_level, eps=eps))
        else:
            q_bins[b] = float(q_global)

    strat = dict(strata)
    strat["q_global"] = float(q_global)
    strat["q_bins"] = q_bins
    strat["min_n"] = int(min_n)
    strat["n_bins_req"] = int(n_bins)
    return float(q_global), strat


def conformal_q_norm_stratified_with_strata(
    y_gt: np.ndarray,
    y_pred: np.ndarray,
    std_raw: np.ndarray,
    x_cal: np.ndarray,
    ci_level: float,
    *,
    strata: Dict[str, Any],
    n_bins_req: int = 3,
    min_n: int = 30,
    eps: float = 1e-12,
) -> Tuple[float, Dict[str, Any]]:
    """Same as conformal_q_norm_stratified, but uses a provided strata dict."""
    y_gt = np.asarray(y_gt, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    std_raw = np.asarray(std_raw, dtype=np.float32)
    x_cal = np.asarray(x_cal, dtype=np.float32)
    if y_gt.shape != y_pred.shape or y_gt.shape != std_raw.shape:
        raise ValueError(f"shape mismatch: y_gt={y_gt.shape} y_pred={y_pred.shape} std={std_raw.shape}")
    if y_gt.ndim != 2:
        raise ValueError(f"expects (N,K), got {y_gt.shape}")
    if x_cal.ndim != 2 or x_cal.shape[0] != y_gt.shape[0]:
        raise ValueError(f"x_cal must be (N,D) matching y, got x_cal={x_cal.shape} y={y_gt.shape}")

    q_global = float(conformal_q_norm_pooled(y_gt, y_pred, std_raw, ci_level, eps=eps))
    ids = _assign_strata_ids(x_cal, strata)

    if strata.get("mode") == "categorical":
        # if coarse, prefer provided coarse_bins, else cats size
        n_strata = int(strata.get("coarse_bins", np.asarray(strata.get("cats", [])).size))
        if n_strata <= 0:
            n_strata = int(np.asarray(strata.get("cats", [])).size)
    else:
        n_strata = int(np.asarray(strata.get("edges")).size) - 1
    n_strata = int(max(1, n_strata))

    q_bins = np.full((n_strata,), q_global, dtype=np.float32)
    K = int(y_gt.shape[1])
    for b in range(n_strata):
        m = (ids == b)
        n_pool = int(np.sum(m)) * K
        if n_pool >= int(min_n):
            q_bins[b] = float(conformal_q_norm_pooled(y_gt[m], y_pred[m], std_raw[m], ci_level, eps=eps))
        else:
            q_bins[b] = float(q_global)

    strat = dict(strata)
    strat["q_global"] = float(q_global)
    strat["q_bins"] = q_bins
    strat["min_n"] = int(min_n)
    strat["n_bins_req"] = int(n_bins_req)
    return float(q_global), strat


def _stratified_sample_indices(
    x_pool: np.ndarray,
    *,
    n: int,
    dim: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Stratified sampling of row indices by discrete-ish x[:,dim].

    Works best when x[:,dim] is categorical (few unique values). For continuous
    values, it still performs a simple quantile-stratified approximation via bins.
    """
    x_pool = np.asarray(x_pool, dtype=np.float32)
    N = int(x_pool.shape[0])
    n = int(max(1, min(int(n), N)))
    v = x_pool[:, int(dim)].astype(np.float32)
    u = np.unique(v)

    # If few uniques -> treat as categorical
    if u.size <= 20:
        cats = np.sort(u)
        by_cat = {float(c): np.where(v == c)[0] for c in cats.tolist()}
        # ensure at least 1 per cat if possible
        chosen: List[int] = []
        for c in cats.tolist():
            idxs = by_cat[float(c)]
            if idxs.size > 0 and len(chosen) < n:
                chosen.append(int(rng.choice(idxs, size=1, replace=False)[0]))
        # fill remaining proportionally
        remaining = n - len(chosen)
        if remaining > 0:
            all_idx = np.arange(N)
            mask = np.ones((N,), dtype=bool)
            mask[np.asarray(chosen, dtype=np.int64)] = False
            pool = all_idx[mask]
            if pool.size > 0:
                extra = rng.choice(pool, size=min(int(remaining), int(pool.size)), replace=False)
                chosen.extend([int(i) for i in extra.tolist()])
        return np.asarray(sorted(set(chosen)), dtype=np.int64)

    # Continuous: quantile bins
    nb = int(min(5, max(2, n // 5)))
    qs = np.linspace(0.0, 1.0, nb + 1, dtype=np.float32)
    edges = np.unique(np.quantile(v, qs).astype(np.float32))
    if edges.size < 3:
        return rng.choice(np.arange(N), size=n, replace=False).astype(np.int64)
    # assign bins
    ids = np.searchsorted(edges, v, side="right") - 1
    ids = np.clip(ids, 0, int(edges.size) - 2).astype(np.int32)
    chosen: List[int] = []
    for b in range(int(edges.size) - 1):
        idxs = np.where(ids == b)[0]
        if idxs.size > 0 and len(chosen) < n:
            chosen.append(int(rng.choice(idxs, size=1, replace=False)[0]))
    remaining = n - len(chosen)
    if remaining > 0:
        all_idx = np.arange(N)
        mask = np.ones((N,), dtype=bool)
        mask[np.asarray(chosen, dtype=np.int64)] = False
        pool = all_idx[mask]
        if pool.size > 0:
            extra = rng.choice(pool, size=min(int(remaining), int(pool.size)), replace=False)
            chosen.extend([int(i) for i in extra.tolist()])
    return np.asarray(sorted(set(chosen)), dtype=np.int64)


def _radius_to_std_equiv(radius: np.ndarray, ci_level: float, eps: float = 1e-12) -> np.ndarray:
    """Convert a CI half-width (radius) to an equivalent Gaussian std under z(ci_level)."""
    if norm is None:
        # scipy is unavailable; fall back to a common approximation for 95%.
        # This keeps the script runnable; note it only affects NLL/NLPD reporting.
        if abs(float(ci_level) - 0.95) < 1e-6:
            z = 1.959963984540054
        else:
            raise RuntimeError("scipy is required for radius->std conversion when ci_level != 0.95")
    else:
        z = float(norm.ppf(0.5 + 0.5 * float(ci_level)))
    return (np.asarray(radius, dtype=np.float32) / max(z, eps)).astype(np.float32)


def conformal_q_apply_stratified(
    x: np.ndarray,
    strat: Dict[str, Any],
) -> np.ndarray:
    """Return per-sample q(x) (N,) from fitted strat dict."""
    x = np.asarray(x, dtype=np.float32)
    ids = _assign_strata_ids(x, strat)
    q_bins = np.asarray(strat["q_bins"], dtype=np.float32)
    q = np.full((x.shape[0],), float(strat.get("q_global", 1.0)), dtype=np.float32)
    ok = ids >= 0
    q[ok] = q_bins[ids[ok]]
    return q


def ci_coverage_from_radius(y_gt: np.ndarray, y_pred: np.ndarray, radius: np.ndarray) -> float:
    y_gt = np.asarray(y_gt, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    radius = np.asarray(radius, dtype=np.float32)
    if y_gt.shape != y_pred.shape or y_gt.shape != radius.shape:
        raise ValueError(f"shape mismatch: y_gt={y_gt.shape} y_pred={y_pred.shape} radius={radius.shape}")
    return float(np.mean(np.abs(y_gt - y_pred) <= radius))


def ci_width_from_radius(radius: np.ndarray) -> float:
    radius = np.asarray(radius, dtype=np.float32)
    return float(np.mean(2.0 * radius))


def make_csv_logger(results_path: Path):
    """Create and OPEN mf_utils.CsvLogger with proper trace/results paths.

    mf_utils.CsvLogger in this repo expects:
        CsvLogger(trace_path: Path, results_path: Path)
    and requires calling .open() before any write_*.
    Some older variants may accept a single path; we keep a small compatibility shim.
    """
    results_path = Path(results_path)
    trace_path = results_path.parent / "trace.csv"
    try:
        sig = inspect.signature(CsvLogger)
        params = list(sig.parameters.values())
        if len(params) == 1:
            lg = CsvLogger(results_path)
            return lg
        # default: (trace_path, results_path)
        lg = CsvLogger(trace_path, results_path)
        return lg
    except Exception:
        # fallback attempts
        try:
            return CsvLogger(trace_path, results_path)
        except TypeError:
            return CsvLogger(results_path)


# ============================================================
# Signature-adaptive wrappers for mf_utils API drift
# ============================================================
def _call_train_feature_mlp(fn, *, x_train, y_train, x_val, y_val, **hp):
    """Call mf_utils.train_feature_mlp with robust argument mapping.

    mf_utils.train_feature_mlp has evolved across files:
      - some versions: train_feature_mlp(X_train, y_train, X_val, y_val, ...)
      - current mf_utils.py: train_feature_mlp(model, X_train, y_train, X_val, y_val, ...)

    This wrapper:
      1) Detects whether a `model` positional arg is required.
      2) If required and not provided, builds a FeatureMLP from (x_train,y_train) shapes
         using hp: hidden/layers/feat_dim/feat_act[/dropout].
      3) Maps common hyperparameter names (steps -> max_epochs, etc.) and filters kwargs
         to only those accepted by the target function (unless it has **kwargs).
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    names = list(params.keys())
    name_set = set(names)

    # ---- Build kwargs filtered/mapped to target signature ----
    # canonical -> aliases (first match wins)
    mapping = {
        # architecture-ish (used only for building FeatureMLP; not forwarded to trainer)
        "hidden": ["hidden", "hid", "width", "n_hidden"],
        "layers": ["layers", "n_layers", "depth"],
        "feat_dim": ["feat_dim", "feature_dim", "d_feat", "feat"],
        "feat_act": ["feat_act", "feature_act", "act_feat", "act"],
        "dropout": ["dropout", "drop"],

        # training-ish
        "lr": ["lr", "learning_rate"],
        "batch_size": ["batch_size", "bs", "batch"],
        "weight_decay": ["weight_decay", "wd"],
        "patience": ["patience", "early_stop_patience", "es_patience"],
        "min_delta": ["min_delta", "early_stop_min_delta"],
        "steps": ["steps", "iters", "iterations", "n_steps", "epochs", "max_epochs"],
        "device": ["device", "dev"],
        "tag": ["tag", "name", "run_name"],
        "print_every": ["print_every", "log_every"],
        "eval_sets": ["eval_sets"],
        "eval_batch_size": ["eval_batch_size"],
        "early_stop_metric": ["early_stop_metric"],
        "early_stop_set": ["early_stop_set"],
    }

    # function accepts arbitrary **kwargs?
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    # We will:
    # - extract architecture params from hp for possible model construction
    arch_hp = {
        "hidden": hp.get("hidden", None),
        "layers": hp.get("layers", None),
        "feat_dim": hp.get("feat_dim", None),
        "feat_act": hp.get("feat_act", None),
        "dropout": hp.get("dropout", 0.0),
    }

    # map steps -> max_epochs if needed
    # (we keep both in hp; filtering below will pick the accepted one)
    if "steps" in hp and "max_epochs" not in hp:
        hp["max_epochs"] = hp["steps"]

    kwargs = {}
    for k, v in hp.items():
        # architecture keys should not be forwarded to train_feature_mlp
        if k in ("hidden", "layers", "feat_dim", "feat_act", "dropout"):
            continue

        if k in mapping:
            for alias in mapping[k]:
                if alias in name_set:
                    kwargs[alias] = v
                    break
            else:
                if has_varkw:
                    kwargs[k] = v
        elif k in name_set:
            kwargs[k] = v
        elif has_varkw:
            kwargs[k] = v

    # ---- Decide call convention (model-first or not) ----
    needs_model = (len(names) >= 1 and names[0] in ("model", "net"))  # stable in our repo
    if needs_model:
        model = hp.get("model", None)
        if model is None:
            in_dim = int(x_train.shape[1])
            out_dim = int(y_train.shape[1]) if getattr(y_train, "ndim", 1) >= 2 else 1

            hidden = int(arch_hp["hidden"] if arch_hp["hidden"] is not None else 256)
            layers = int(arch_hp["layers"] if arch_hp["layers"] is not None else 2)
            feat_dim = int(arch_hp["feat_dim"] if arch_hp["feat_dim"] is not None else 16)
            act = str(arch_hp["feat_act"] if arch_hp["feat_act"] is not None else "relu")
            dropout = float(arch_hp.get("dropout", 0.0) or 0.0)

            model = FeatureMLP(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden=tuple([hidden] * layers),
                feat_dim=feat_dim,
                act=act,
                dropout=dropout,
            )

        out = fn(model, x_train, y_train, x_val, y_val, **kwargs)
    else:
        out = fn(x_train, y_train, x_val, y_val, **kwargs)

    if isinstance(out, tuple):
        if len(out) >= 2:
            return out[0], out[1]
        return out[0], {}
    return out, {}


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = str(SCRIPT_DIR / "../../data/mf_sweep_datasets_nano_tm/hf100_lfx10")
DEFAULT_OUT_DIR = str(SCRIPT_DIR / "../../result_out/mf_baseline_nano_tm_out")


# ============================================================
# FPCA (discrete functional PCA) - same as delta-student script
# ============================================================
class FPCA:
    """FPCA for discretized curves y(wavelength) sampled on a fixed grid."""

    def __init__(
        self,
        n_components: int = 0,
        var_ratio: float = 0.999,
        max_components: int = 64,
        ridge: float = 1e-8,
        random_state: Optional[int] = None,
    ):
        self.n_components = int(n_components)
        self.var_ratio = float(var_ratio)
        self.max_components = int(max_components)
        self.ridge = float(ridge)
        self.random_state = random_state

        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "FPCA":
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"FPCA.fit expects 2D array, got shape={X.shape}")
        n, K = X.shape
        if n < 2:
            raise ValueError("FPCA requires at least 2 samples.")

        self.mean_ = X.mean(axis=0).astype(np.float32)
        Xc = (X - self.mean_).astype(np.float32)

        C = (Xc.T @ Xc) / float(n - 1)
        if self.ridge > 0:
            C = C + (self.ridge * np.eye(K, dtype=np.float32))

        eigvals, eigvecs = np.linalg.eigh(C.astype(np.float64))
        eigvals = eigvals.astype(np.float32)
        eigvecs = eigvecs.astype(np.float32)

        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        eigvals_clip = np.maximum(eigvals, 0.0)
        total = float(np.sum(eigvals_clip))
        if total <= 0:
            raise ValueError("FPCA: total variance <= 0")
        evr = eigvals_clip / total

        rank_cap = min(K, n - 1)
        if self.max_components > 0:
            rank_cap = min(rank_cap, int(self.max_components))

        if self.n_components > 0:
            R = min(self.n_components, rank_cap)
        else:
            cum = np.cumsum(evr)
            R = int(np.searchsorted(cum, self.var_ratio) + 1)
            R = max(1, min(R, rank_cap))

        self.n_components_ = int(R)
        self.components_ = eigvecs[:, :R].T.copy()
        self.explained_variance_ = eigvals_clip[:R].copy()
        self.explained_variance_ratio_ = evr[:R].copy()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("FPCA.transform before fit")
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mean_) @ self.components_.T).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("FPCA.inverse_transform before fit")
        Z = np.asarray(Z, dtype=np.float32)
        return (Z @ self.components_ + self.mean_).astype(np.float32)


def fpca_propagate_var_to_y(var_z_unscaled: np.ndarray, fpca: FPCA, scaler_y: StandardScaler) -> np.ndarray:
    """Diagonal var in unscaled FPCA score space -> diagonal var in original y(K)."""
    if fpca.components_ is None:
        raise RuntimeError("fpca_propagate_var_to_y: fpca not fit")
    comp = fpca.components_.astype(np.float32)  # (R,K)
    comp2 = np.square(comp).astype(np.float32)  # (R,K)
    var_z_unscaled = np.asarray(var_z_unscaled, dtype=np.float32)  # (N,R)
    var_y_scaled = var_z_unscaled @ comp2  # (N,K)
    scale2 = np.square(np.asarray(scaler_y.scale_, dtype=np.float32))[None, :]
    return (var_y_scaled * scale2).astype(np.float32)


def fit_affine_rho(lf: np.ndarray, y: np.ndarray, ridge: float = 1e-6, use_intercept: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-dim affine mapping y ≈ rho*lf + b on paired HF-train (both in SAME target space).

    lf, y: (N,R)
    Returns:
      rho: (R,)
      b  : (R,)
    """
    lf = np.asarray(lf, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if lf.ndim != 2 or y.ndim != 2 or lf.shape != y.shape:
        raise ValueError(f"fit_affine_rho expects lf,y with same shape (N,R). got lf={lf.shape} y={y.shape}")
    if use_intercept:
        mlf = lf.mean(axis=0)
        my = y.mean(axis=0)
        lf_c = lf - mlf[None, :]
        y_c = y - my[None, :]
        cov = np.mean(lf_c * y_c, axis=0)
        var = np.mean(lf_c * lf_c, axis=0) + float(ridge)
        rho = cov / var
        b = my - rho * mlf
    else:
        num = np.sum(lf * y, axis=0)
        den = np.sum(lf * lf, axis=0) + float(ridge)
        rho = num / den
        b = np.zeros_like(rho)
    return rho.astype(np.float32), b.astype(np.float32)


def ridge_rho(z_lf: np.ndarray, z_hf: np.ndarray, ridge: float) -> np.ndarray:
    """AR1 baseline rho fit (NO intercept): rho=(lf^T hf)/(lf^T lf + ridge), per-dim."""
    z_lf = np.asarray(z_lf, dtype=np.float32)
    z_hf = np.asarray(z_hf, dtype=np.float32)
    num = np.sum(z_lf * z_hf, axis=0)
    den = np.sum(z_lf * z_lf, axis=0) + float(ridge)
    return (num / den).astype(np.float32)


def _dbg_head_tail(arr: np.ndarray, n: int = 5) -> Tuple[List[float], List[float]]:
    arr = np.asarray(arr).astype(float).ravel()
    if arr.size <= 2 * n:
        return arr.tolist(), []
    return arr[:n].tolist(), arr[-n:].tolist()


def dbg_print_wavelengths(wl_full: np.ndarray, idx_keep: Optional[np.ndarray], wl_used: np.ndarray) -> None:
    h_full, t_full = _dbg_head_tail(wl_full, 5)
    h_used, t_used = _dbg_head_tail(wl_used, 5)
    print(f"[DEBUG][WL] wl_full head5={h_full} tail5={t_full}")
    print(f"[DEBUG][WL] wl_used head5={h_used} tail5={t_used}")
    if idx_keep is not None:
        wl_check = wl_full[idx_keep]
        assert wl_check.shape == wl_used.shape
        assert np.allclose(wl_check.astype(np.float32), wl_used.astype(np.float32)), "wl_used != wl_full[idx_keep]"

def build_run_name(args: argparse.Namespace) -> str:
    """
    Short run-name rule:
      - default baseline: bl0
      - only append short codes for NON-default settings
      - if name grows beyond 10 chars, fall back to bl0 + 7-char hash
    """
    base = "bl0"
    tokens = []

    def add(tok: str) -> None:
        if tok:
            tokens.append(str(tok))

    # non-default dim reduction
    if str(args.dim_reduce) != "fpca":
        add(f"s{int(args.subsample_K)}")
    else:
        if int(args.fpca_dim) > 0:
            add(f"f{int(args.fpca_dim)}")
        elif float(args.fpca_var_ratio) != 0.999:
            vr = str(args.fpca_var_ratio).replace("0.", "").replace(".", "")
            add(f"v{vr[:2] or '0'}")
        if int(args.fpca_max_dim) != 64:
            add(f"m{int(args.fpca_max_dim)}")

    # non-default model knobs
    if str(args.mf_u_mode) != "xlf":
        u_map = {"lf": "u0", "x": "ux"}
        add(u_map.get(str(args.mf_u_mode), "u1"))
    if str(args.student_mode) != "delta":
        add("d0")
    if str(getattr(args, "student_y_scaler_fit", "paired")) != "paired":
        add("ym")
    if str(args.rho_fit_source) != "oracle":
        add("rs")
    if int(args.rho_intercept) != 1:
        add("i0")

    # kernel / gp
    if str(args.kernel_struct) != "full":
        ks_map = {"block": "kb", "xlf_block": "kx"}
        add(ks_map.get(str(args.kernel_struct), "k1"))
    if str(args.kernel) != "matern":
        add("rb")
    else:
        if float(args.matern_nu) != 2.5:
            nu_map = {0.5: "n0", 1.5: "n1", 2.5: "n2"}
            add(nu_map.get(float(args.matern_nu), "n2"))
    if int(args.gp_ard) != 1:
        add("a0")
    if int(args.svgp_M) != 64:
        add(f"M{int(args.svgp_M)}")
    if int(args.svgp_steps) != 2000:
        steps = int(args.svgp_steps)
        if steps % 1000 == 0:
            add(f"t{steps // 1000}k")
        else:
            add(f"t{steps}")

    # ci / seed only when non-default
    if float(args.ci_level) != 0.95:
        ci = str(args.ci_level).replace("0.", "").replace(".", "")
        add(f"c{ci[:2] or '0'}")
    if int(args.ci_calibrate) != 0:
        add("cc")
    if int(args.seed) != 42:
        add(f"r{int(args.seed)}")

    name = base + "".join(tokens)
    if len(name) <= 10:
        return name

    payload = json.dumps(
        {
            "dim_reduce": args.dim_reduce,
            "fpca_dim": int(args.fpca_dim),
            "fpca_var_ratio": float(args.fpca_var_ratio),
            "fpca_max_dim": int(args.fpca_max_dim),
            "subsample_K": int(args.subsample_K),
            "mf_u_mode": str(args.mf_u_mode),
            "student_mode": str(args.student_mode),
            "student_y_scaler_fit": str(getattr(args, "student_y_scaler_fit", "paired")),
            "rho_fit_source": str(args.rho_fit_source),
            "rho_intercept": int(args.rho_intercept),
            "kernel_struct": str(args.kernel_struct),
            "kernel": str(args.kernel),
            "matern_nu": float(args.matern_nu),
            "gp_ard": int(args.gp_ard),
            "svgp_M": int(args.svgp_M),
            "svgp_steps": int(args.svgp_steps),
            "ci_level": float(args.ci_level),
            "ci_calibrate": int(args.ci_calibrate),
            "seed": int(args.seed),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:7]
    return f"{base}{digest}"
	
# def build_run_name(args: argparse.Namespace) -> str:
#     if args.dim_reduce == "fpca":
#         if int(args.fpca_dim) > 0:
#             red = f"fpca{int(args.fpca_dim)}"
#         else:
#             red = f"fpcaAutoV{str(args.fpca_var_ratio).replace('.','p')}_m{int(args.fpca_max_dim)}"
#         dim_tag = f"dimfpca_{red}"
#     else:
#         dim_tag = f"dimsub_subK{int(args.subsample_K)}"
#
#     parts = [
#         dim_tag,
#         f"u{safe_tag(str(args.mf_u_mode))}",
#         f"student{safe_tag(str(args.student_mode))}",
#         f"ysfit{safe_tag(str(getattr(args, 'student_y_scaler_fit', 'mix')))}",
#         f"rhoSrc{safe_tag(str(args.rho_fit_source))}",
#         f"rhoInt{int(args.rho_intercept)}",
#         f"k{safe_tag(str(args.kernel_struct))}",
#         f"{safe_tag(str(args.kernel))}" + (f"_nu{str(args.matern_nu).replace('.','p')}" if args.kernel == "matern" else ""),
#         f"ard{int(args.gp_ard)}",
#         f"M{int(args.svgp_M)}",
#         f"steps{int(args.svgp_steps)}",
#         f"seed{int(args.seed)}",
#         f"ci{str(args.ci_level).replace('.','p')}",
#         f"cal{int(args.ci_calibrate)}",
#         "z2scaler1",
#         "AR1",
#     ]
#     return "__".join(parts)


# ----------------------------
# Delegate OURS to A0 training script (mf_train_fpca_sgp_delta_student.py)
# ----------------------------
def run_ours_via_a0(args: argparse.Namespace, out_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Run OURS by invoking mf_train_fpca_sgp_delta_student.py and return (report, pred_root).

    Notes:
    - Output folder is out_dir / "ours" (no "delegate" naming).
    - Only passes arguments that exist in the training script CLI.
    """
    if int(getattr(args, "delegate_ours_to_train", 1)) != 1:
        return None, None
    script = Path(getattr(args, "ours_train_script")).expanduser().resolve()
    if not script.exists():
        raise FileNotFoundError(f"ours_train_script not found: {script}")

    ours_out = Path(out_dir) / "ours"
    ours_out.mkdir(parents=True, exist_ok=True)

    argv = [str(script),
            "--data_dir", str(Path(args.data_dir).expanduser().resolve()),
            "--out_dir", str(ours_out),
            "--no_subdir", "1",
            "--seed", str(int(args.seed)),
            "--device", str(args.device),
            "--dim_reduce", str(args.dim_reduce),
            "--mf_u_mode", str(args.mf_u_mode),
            "--student_mode", str(args.student_mode),
            "--rho_fit_source", str(args.rho_fit_source),
            "--rho_intercept", str(int(args.rho_intercept)),
            "--kernel_struct", str(args.kernel_struct),
            "--kernel", str(args.kernel),
            "--matern_nu", str(args.matern_nu),
            "--gp_ard", str(int(args.gp_ard)),
            "--svgp_M", str(int(args.svgp_M)),
            "--svgp_steps", str(int(args.svgp_steps)),
            "--svgp_lr", str(float(args.svgp_lr)),
            "--print_every", str(int(args.print_every)),
            "--ci_level", str(float(args.ci_level)),
            "--ci_calibrate", str(int(args.ci_calibrate)),
            "--plot_ci", str(int(args.plot_ci)),
            "--n_plot", str(int(args.n_plot)),
            "--save_pred_arrays", str(int(args.save_pred_arrays)),
            # run only student path by default
            "--run_hf_only", "0",
            "--run_oracle", "0",
            "--run_student", "1",
            "--lf_prob", str(int(args.lf_prob)),
            "--mc_lf_samples", str(int(args.mc_lf_samples)),
            ]

    # optional wl crop
    if args.wl_low is not None:
        argv += ["--wl_low", str(float(args.wl_low))]
    if args.wl_high is not None:
        argv += ["--wl_high", str(float(args.wl_high))]
    # fpca / subsample params
    if str(args.dim_reduce) == "fpca":
        argv += ["--fpca_var_ratio", str(float(args.fpca_var_ratio)),
                 "--fpca_max_dim", str(int(args.fpca_max_dim)),
                 "--fpca_ridge", str(float(args.fpca_ridge))]
        if getattr(args, "fpca_dim", None) is not None:
            argv += ["--fpca_dim", str(int(args.fpca_dim))]
    else:
        argv += ["--subsample_K", str(int(args.subsample_K))]

    # student set / yscale knobs (exist in training script)
    argv += ["--student_train_set", str(args.student_train_set),
             "--student_val_set", str(args.student_val_set),
             "--student_yscale", str(int(args.student_yscale)),
             "--student_y_scaler_fit", str(args.student_y_scaler_fit),
             "--student_feat_dim", str(int(args.student_feat_dim)),
             "--student_act", str(args.student_act),
             "--student_feat_act", str(args.student_feat_act),
             "--student_feat_leaky_slope", str(float(args.student_feat_leaky_slope)),
             "--student_dropout", str(float(args.student_dropout)),
             "--student_lr", str(float(args.student_lr)),
             "--student_wd", str(float(args.student_wd)),
             "--student_bs", str(int(args.student_bs)),
             "--student_epochs", str(int(args.student_epochs)),
             "--student_patience", str(int(args.student_patience)),
             "--student_min_delta", str(float(args.student_min_delta)),
             "--student_print_every", str(int(args.student_print_every)),
             "--skip_student", "0",
             "--mf_student_lf_source", str(args.mf_student_lf_source),
             "--rho_ridge", str(float(args.rho_ridge)),
             ]

    # student_hidden is nargs+
    if getattr(args, "student_hidden", None) is not None and len(args.student_hidden) > 0:
        argv += ["--student_hidden"] + [str(int(x)) for x in args.student_hidden]

    # run in-process via import to preserve environment; reset sys.argv temporarily
    import importlib.util, sys
    module_name = script.stem  # <<< key: use real filename stem, not hard-coded

    # make sure the script directory is importable (needed for pickle to re-import)
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    spec = importlib.util.spec_from_file_location(module_name, str(script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import training script: {script}")

    mod = importlib.util.module_from_spec(spec)
    # register before exec so pickle can resolve module path later
    sys.modules[module_name] = mod

    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        spec.loader.exec_module(mod)  # type: ignore
        if not hasattr(mod, "main"):
            raise RuntimeError("Training script has no main()")
        mod.main()
    finally:
        sys.argv = old_argv

    report_path = ours_out / "report.json"
    report = None
    if report_path.exists():
        import json
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    pred_root = ours_out / "pred_arrays"
    return report, (pred_root if pred_root.exists() else None)
def save_uq_cache(out_dir: Path, stem: str, pack: Dict[str, np.ndarray]) -> None:
    """Save arrays needed for UQ-only recalibration (no training, no calibration here)."""
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{stem}.npz"
    man_path = cache_dir / f"{stem}.json"
    np.savez_compressed(npz_path, **pack)
    manifest = {
        "npz": str(npz_path),
        "keys": sorted(pack.keys()),
        "shapes": {k: list(v.shape) for k, v in pack.items()},
        "note": "UQ cache for uq_residual_conformal_from_cache_kfoldsafe.py / plot_uq_from_cache.py. Baseline does NOT run UQ calibration.",
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[UQ][CACHE] Saved: {npz_path}")
    print(f"[UQ][CACHE] Manifest: {man_path}")



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # paths
    ap.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="MF dataset root")
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Output root directory")

    # which baselines to run
    ap.add_argument("--methods", type=str, default="hf_only,ar1,ours",
                    help="Comma-separated subset to run: hf_only, ar1, ours. Synonyms: hf,hf-only,cokrig,delta,delta_svgp,all")

    # Delegate OURS to A0 training script (mf_train_fpca_sgp_delta_student.py)
    ap.add_argument("--delegate_ours_to_train", type=int, default=1, choices=[0, 1],
                    help="If 1 (default) and ours enabled, run OURS via mf_train_fpca_sgp_delta_student.py (A0 path) and load its outputs. Set to 1 only for legacy in-file OURS implementation.")
    ap.add_argument("--ours_train_script", type=str, default=str(Path(__file__).parent / "mf_train_tm.py"),
                    help="Path to mf_train_fpca_sgp_delta_student.py used when delegating OURS.")

    # seeds
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default="", help="Optional: comma-separated seeds, overrides --seed")

    # device
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Training device")

    # wavelength band crop
    ap.add_argument("--wl_low", type=float, default=None, help="Crop wavelength low (nm)")
    ap.add_argument("--wl_high", type=float, default=None, help="Crop wavelength high (nm)")

    # dim reduce
    ap.add_argument("--dim_reduce", type=str, default="fpca", choices=["fpca", "subsample"])
    ap.add_argument("--fpca_dim", type=int, default=0)
    ap.add_argument("--fpca_var_ratio", type=float, default=0.999)
    ap.add_argument("--fpca_max_dim", type=int, default=64)
    ap.add_argument("--fpca_ridge", type=float, default=1e-8)
    ap.add_argument("--subsample_K", type=int, default=50)

    # Stage-I student (delegated script-compatible)
    ap.add_argument(
        "--mf_student_lf_source",
        type=str,
        default="student",
        choices=["student", "oracle"],
        help="LF_hat source used by MF-student/Stage-II: 'student' uses Stage-I predicted LF; 'oracle' uses paired true LF (ablation upper bound)."
    )

    # Stage-II target mode (ours): direct HF vs residual/delta
    ap.add_argument("--student_mode", type=str, default="delta", choices=["direct", "delta"],
                    help="MF-student Stage-II target: direct predicts HF; delta predicts residual and adds back affine(rho)*LF_hat.")
    # Stage-I student training set selection
    ap.add_argument("--student_train_set", type=str, default="mix", choices=["paired", "mix"],
                    help="Stage-I student training set: mix uses (lf_paired+lf_unpaired)/train; paired uses lf_paired/train only.")
    ap.add_argument("--student_val_set", type=str, default="paired", choices=["paired", "mix"],
                    help="Stage-I student early-stop validation set: paired uses lf_paired/val; mix uses (lf_paired+lf_unpaired)/val.")
    ap.add_argument("--student_yscale", type=int, default=1, choices=[0, 1],
                    help="If 1, apply StandardScaler on student targets y for Stage-I; if 0, train on raw y.")
    ap.add_argument("--student_y_scaler_fit", type=str, default="paired", choices=["mix", "paired"],
                    help="When student_yscale=1, fit y scaler on mix or paired train targets.")
    # MLP architecture
    ap.add_argument("--student_hidden", type=int, nargs="*", default=[256, 256, 256],
                    help="Hidden layer widths for Stage-I MLP, e.g., --student_hidden 256 256 256")
    ap.add_argument("--student_feat_dim", type=int, default=32)
    ap.add_argument("--student_act", type=str, default="relu", choices=["relu", "tanh", "gelu"])
    ap.add_argument("--student_feat_act", type=str, default="leakyrelu", choices=["leakyrelu", "identity", "same"])
    ap.add_argument("--student_feat_leaky_slope", type=float, default=0.01)
    ap.add_argument("--student_dropout", type=float, default=0.0)
    ap.add_argument("--student_lr", type=float, default=3e-4)
    ap.add_argument("--student_wd", type=float, default=1e-4)
    ap.add_argument("--student_bs", type=int, default=256)
    ap.add_argument("--student_epochs", type=int, default=2000)
    ap.add_argument("--student_patience", type=int, default=100)
    ap.add_argument("--student_min_delta", type=float, default=1e-4)
    ap.add_argument("--student_print_every", type=int, default=20)

    # Ours stage-II input mode
    ap.add_argument("--mf_u_mode", type=str, default="xlf", choices=["lf", "xlf", "x"],
                    help="Ours Stage-II input: lf only, [x,lf], or x only")

    # rho fit for Ours delta base
    ap.add_argument("--rho_fit_source", type=str, default="oracle", choices=["oracle", "student"],
                    help="Ours: fit affine(rho) using true LF or student LF_hat")
    ap.add_argument("--rho_ridge", type=float, default=1e-6)
    ap.add_argument("--rho_intercept", type=int, default=1, choices=[0, 1])

    # AR1 rho ridge
    ap.add_argument("--ar1_rho_ridge", type=float, default=1e-8)

    # GP/SVGP
    ap.add_argument("--kernel_struct", type=str, default="full", choices=["full", "block", "xlf_block"],
                    help="block supported only if you pass feature blocks; here we default to full")
    ap.add_argument("--kernel", type=str, default="matern", choices=["rbf", "matern"])
    ap.add_argument("--matern_nu", type=float, default=2.5)
    ap.add_argument("--gp_ard", type=int, default=1, choices=[0, 1])
    ap.add_argument("--svgp_M", type=int, default=64)
    ap.add_argument("--svgp_steps", type=int, default=2000)
    ap.add_argument("--svgp_lr", type=float, default=5e-3)
    ap.add_argument("--print_every", type=int, default=200)

    # UQ
    # Stage-I LF uncertainty toggle (passed to mf_train_fpca_sgp_delta_student_lfprob.py)
    ap.add_argument("--lf_prob", type=int, default=0, choices=[0, 1],
                    help="If 1, Stage-I predicts (mu, sigma^2) for LF and propagates UQ. If 0, deterministic LF (sigma=0).")

    # (optional but recommended) MC marginalization samples when lf_prob=1
    ap.add_argument("--mc_lf_samples", type=int, default=1,
                    help="MC samples for marginalizing LF uncertainty (S=1 means no marginalization).")
    ap.add_argument("--ci_level", type=float, default=0.95)
    ap.add_argument("--ci_calibrate", type=int, default=1, choices=[0, 1])
    ap.add_argument(
        "--ci_cal_mode",
        type=str,
        default="conformal_norm_pooled",
        choices=[
            "conformal_norm_pooled",
            "conformal_norm_dim0_bins",
            "conformal_norm_dim0_coarsebins_traincal",
            "global",
            "per_k",
        ],
        help=(
            "CI calibration mode: global uses a single alpha; per_k fits alpha_k per wavelength point on VAL (scale-only; no mean change). "
            "conformal_norm_pooled uses split conformal on |e|/std pooled over (N,K). "
            "conformal_norm_dim0_bins stratifies by x[:,ci_stratify_dim]. "
            "conformal_norm_dim0_coarsebins_traincal uses a coarse stratification on dim0 and calibrates q on a stratified subset sampled from TRAIN."
        ),
    )
    ap.add_argument("--ci_per_k_smooth", type=int, default=0,
                    help="If ci_cal_mode=per_k, optional moving-average smoothing window for alpha_k (0 disables; use odd >=3).")


    ap.add_argument("--ci_stratify_dim", type=int, default=0,
                help="If ci_cal_mode=conformal_norm_dim0_bins, stratify calibration by this x dimension (default 0).")
    ap.add_argument("--ci_stratify_bins", type=int, default=8,
                help="If ci_cal_mode=conformal_norm_dim0_bins and strat_dim is continuous, number of quantile bins (ignored for categorical stratification).")
    ap.add_argument("--ci_stratify_min_n", type=int, default=30,
                help="Minimum number of pooled points per stratum to fit a separate q; otherwise fall back to global q.")

    # For conformal_norm_dim0_coarsebins_traincal
    ap.add_argument("--ci_cal_source", type=str, default="train", choices=["train", "val", "train+val"],
                    help="Calibration source split for conformal_norm_dim0_coarsebins_traincal. Default: train.")
    ap.add_argument("--ci_cal_n", type=int, default=40,
                    help="Number of HF samples to use for calibration (sampled from ci_cal_source), stratified by dim0. Default: 40.")
    ap.add_argument("--ci_coarse_bins", type=int, default=3,
                    help="Number of coarse bins (groups) for dim0 stratification in conformal_norm_dim0_coarsebins_traincal. Default: 3.")

    # plots (delegated script-compatible)
    ap.add_argument("--plot_ci", type=int, default=1, choices=[0, 1])
    ap.add_argument("--n_plot", type=int, default=10)
    ap.add_argument("--save_pred_arrays", type=int, default=1, choices=[0, 1],
                    help="If 1, save prediction/uncertainty arrays (val/test) for downstream plotting.")
    ap.add_argument("--save_uq_cache", type=int, default=1, choices=[0, 1],
                    help="If 1, save UQ cache arrays (x/y_gt/y_pred/std_raw for val/test) under out_dir/cache for fast post-hoc UQ tuning.")
    ap.add_argument("--uq_cache_name", type=str, default="uq_cache_v1",
                    help="Cache file stem under out_dir/cache (writes .npz and .json).")



    return ap.parse_args()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_seed_list(args: argparse.Namespace) -> List[int]:
    s = str(args.seeds).strip()
    if not s:
        return [int(args.seed)]
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out if out else [int(args.seed)]


def run_once(args: argparse.Namespace, seed: int) -> None:
    args = argparse.Namespace(**vars(args))
    args.seed = int(seed)

    set_seed(int(args.seed))
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    data_dir = Path(args.data_dir).resolve()
    out_root = Path(args.out_dir).resolve()

    run_id = now_tag()
    run_name = build_run_name(args)
    out_dir = out_root / run_name
    _ensure_dir(out_dir)

    print("[BOOT] mf_hf_delta_ar1.py is running...")
    run_methods = parse_methods(getattr(args, 'methods', 'all'))
    print('[METHODS] enabled =', ','.join(sorted(run_methods)))
    # For delegated OURS
    ours_report = None
    ours_pred_root = None


    # CSV logger
    csv_logger = make_csv_logger(out_dir / "trace_results.csv")
    csv_logger.open()
    try:

            # ------------- load data blocks
            # HF
            x_hf_tr, y_hf_tr, t_hf_tr, idx_hf_tr = load_split_block(data_dir, "hf", "train")
            x_hf_va, y_hf_va, t_hf_va, idx_hf_va = load_split_block(data_dir, "hf", "val")
            x_hf_te, y_hf_te, t_hf_te, idx_hf_te = load_split_block(data_dir, "hf", "test")

            # LF paired
            x_lfp_tr, y_lfp_tr, t_lfp_tr, idx_lfp_tr = load_split_block(data_dir, "lf_paired", "train")
            x_lfp_va, y_lfp_va, t_lfp_va, idx_lfp_va = load_split_block(data_dir, "lf_paired", "val")
            x_lfp_te, y_lfp_te, t_lfp_te, idx_lfp_te = load_split_block(data_dir, "lf_paired", "test")

            # LF unpaired
            x_lfu_tr, y_lfu_tr, t_lfu_tr, idx_lfu_tr = load_split_block(data_dir, "lf_unpaired", "train")
            x_lfu_va, y_lfu_va, t_lfu_va, idx_lfu_va = load_split_block(data_dir, "lf_unpaired", "val")
            x_lfu_te, y_lfu_te, t_lfu_te, idx_lfu_te = load_split_block(data_dir, "lf_unpaired", "test")

            # sanity (HF vs LF paired indices should match per split)
            assert_indices_match(idx_hf_tr, idx_lfp_tr, name_a="hf/train", name_b="lf_paired/train")
            assert_indices_match(idx_hf_va, idx_lfp_va, name_a="hf/val", name_b="lf_paired/val")
            assert_indices_match(idx_hf_te, idx_lfp_te, name_a="hf/test", name_b="lf_paired/test")

            # wavelengths
            wl_full, idx_wl_full = load_wavelengths(data_dir)
            K_full = int(wl_full.size)

            # optional crop band
            if (args.wl_low is not None) and (args.wl_high is not None):
                lo = float(args.wl_low)
                hi = float(args.wl_high)
                keep = np.where((wl_full >= lo) & (wl_full <= hi))[0]
                if keep.size <= 1:
                    raise ValueError(f"Empty/too small wl crop [{lo},{hi}]")
                wl = wl_full[keep].astype(np.float32)
                idx_wl = idx_wl_full[keep].astype(np.int64)
                y_hf_tr = y_hf_tr[:, keep]
                y_hf_va = y_hf_va[:, keep]
                y_hf_te = y_hf_te[:, keep]
                y_lfp_tr = y_lfp_tr[:, keep]
                y_lfp_va = y_lfp_va[:, keep]
                y_lfp_te = y_lfp_te[:, keep]
                y_lfu_tr = y_lfu_tr[:, keep]
                y_lfu_va = y_lfu_va[:, keep]
                y_lfu_te = y_lfu_te[:, keep]
                print(f"[CROP] wl_range=[{lo},{hi}] nm | K_full={K_full} -> K={int(wl.size)}")
                dbg_print_wavelengths(wl_full, keep, wl)
            else:
                wl = wl_full.astype(np.float32)
                idx_wl = idx_wl_full.astype(np.int64)

            K = int(wl.size)
            print(f"[INFO] device={device}")
            print(f"[INFO] data_dir={data_dir}")
            print(f"[INFO] out_dir={out_dir}")
            print(f"[INFO] wavelength K={K} | wl=[{float(wl.min()):.6g},{float(wl.max()):.6g}]")
            print(f"[INFO] HF train/val/test = {x_hf_tr.shape[0]}/{x_hf_va.shape[0]}/{x_hf_te.shape[0]}")
            print(f"[INFO] LF paired train/val/test = {x_lfp_tr.shape[0]}/{x_lfp_va.shape[0]}/{x_lfp_te.shape[0]}")
            print(f"[INFO] LF unpaired train/val/test = {x_lfu_tr.shape[0]}/{x_lfu_va.shape[0]}/{x_lfu_te.shape[0]}")
            print(f"[INFO] CI level={float(args.ci_level)} z≈{z_from_ci_level(float(args.ci_level)):.4f} cal={bool(args.ci_calibrate)}")

            # save basics
            with open(out_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump({"run_id": run_id, "run_name": run_name, "args": vars(args),
                           "data_dir": str(data_dir), "out_dir": str(out_dir)},
                          f, ensure_ascii=False, indent=2)
            np.save(out_dir / "wavelengths.npy", wl.astype(np.float32))
            np.save(out_dir / "idx_wavelength.npy", idx_wl.astype(np.int64))

            # ============================================================
            if "ours" in run_methods:
                # Stage-I (Ours): student x -> LF (K)
                #   Train on mixed LF (paired+unpaired) train split.
                # ============================================================
                print("[STAGE-I] Train student: x -> LF (K)")
                
                # mix LF train/val/test blocks
                x_lf_tr = np.concatenate([x_lfp_tr, x_lfu_tr], axis=0).astype(np.float32)
                y_lf_tr = np.concatenate([y_lfp_tr, y_lfu_tr], axis=0).astype(np.float32)
                x_lf_va = np.concatenate([x_lfp_va, x_lfu_va], axis=0).astype(np.float32)
                y_lf_va = np.concatenate([y_lfp_va, y_lfu_va], axis=0).astype(np.float32)
                x_lf_te = np.concatenate([x_lfp_te, x_lfu_te], axis=0).astype(np.float32)
                y_lf_te = np.concatenate([y_lfp_te, y_lfu_te], axis=0).astype(np.float32)
                
                # student x scaler: fit on mix_train
                student_scaler_x = StandardScaler(with_mean=True, with_std=True).fit(x_lf_tr)
                save_pickle(out_dir / "student_scaler_x.pkl", student_scaler_x)
                
                x_lf_tr_s = safe_scaler_transform(student_scaler_x, x_lf_tr)
                x_lf_va_s = safe_scaler_transform(student_scaler_x, x_lf_va)
                x_lf_te_s = safe_scaler_transform(student_scaler_x, x_lf_te)                # student y scaler: fit on selected set (mix or paired-only)
                _yfit_mode = str(getattr(args, "student_y_scaler_fit", "mix")).lower().strip()
                if _yfit_mode == "paired":
                    y_fit = y_lfp_tr
                else:
                    y_fit = y_lf_tr  # mix_train = paired + unpaired
                student_scaler_y = StandardScaler(with_mean=True, with_std=True).fit(y_fit)
                save_pickle(out_dir / "student_scaler_y.pkl", student_scaler_y)

                y_lf_tr_s = safe_scaler_transform(student_scaler_y, y_lf_tr)
                y_lf_va_s = safe_scaler_transform(student_scaler_y, y_lf_va)
                y_lf_te_s = safe_scaler_transform(student_scaler_y, y_lf_te)
                
                # choose validation set for early stopping
                if str(args.student_val_set).lower().strip() == "paired":
                    x_val_es = safe_scaler_transform(student_scaler_x, x_lfp_va.astype(np.float32), n_features=int(getattr(student_scaler_x, 'n_features_in_', x_lfp_va.shape[1])))
                    y_val_es = safe_scaler_transform(student_scaler_y, y_lfp_va.astype(np.float32), n_features=int(getattr(student_scaler_y, 'n_features_in_', y_lfp_va.shape[1])))
                else:
                    x_val_es = x_lf_va_s
                    y_val_es = y_lf_va_s
                
                # train student
                student_model, student_train_log = _call_train_feature_mlp(
                    train_feature_mlp,
                    x_train=x_lf_tr_s,
                    y_train=y_lf_tr_s,
                    x_val=x_val_es,
                    y_val=y_val_es,
                    hidden=int(args.student_hidden[0]),
                    layers=int(len(args.student_hidden)),
                    feat_dim=int(args.student_feat_dim),
                    feat_act=str(args.student_feat_act),
                    lr=float(args.student_lr),
                    steps=int(args.student_epochs),
                    batch_size=int(args.student_bs),
                    patience=int(args.student_patience),
                    device=device,
                    tag="student",
                )
                save_pickle(out_dir / "student_train_log.pkl", student_train_log)
                torch.save(student_model.state_dict(), out_dir / "student_model.pt")
                
                # predict LF_hat on HF splits (paired x are identical to HF x per assert above)
                x_hf_tr_st_s = safe_scaler_transform(student_scaler_x, x_hf_tr.astype(np.float32)).astype(np.float32)
                x_hf_va_st_s = safe_scaler_transform(student_scaler_x, x_hf_va.astype(np.float32)).astype(np.float32)
                x_hf_te_st_s = safe_scaler_transform(student_scaler_x, x_hf_te.astype(np.float32)).astype(np.float32)
                
                yhat_lf_tr_s, feat_tr = mlp_predict_and_features(student_model, x_hf_tr_st_s, device=device)
                yhat_lf_va_s, feat_va = mlp_predict_and_features(student_model, x_hf_va_st_s, device=device)
                yhat_lf_te_s, feat_te = mlp_predict_and_features(student_model, x_hf_te_st_s, device=device)
                
                # inverse student y scaler to get LF_hat in original y(K)
                yhat_lf_tr = student_scaler_y.inverse_transform(yhat_lf_tr_s.astype(np.float32)).astype(np.float32)
                yhat_lf_va = student_scaler_y.inverse_transform(yhat_lf_va_s.astype(np.float32)).astype(np.float32)
                yhat_lf_te = student_scaler_y.inverse_transform(yhat_lf_te_s.astype(np.float32)).astype(np.float32)
                
                # debug: student on HF (paired) errors
                dbg_student_metrics = dbg_student_on_hf_errors(
                    yhat_lf_va, y_lfp_va,
                    yhat_lf_te, y_lfp_te
                )
                
                # ============================================================
                # Stage-II target pipeline (HF-train): y(K)->scaler_y->FPCA->scaler_z
                # ============================================================
                dim_reduce = str(args.dim_reduce).lower().strip()
                
                scaler_y: Optional[StandardScaler] = None
                fpca: Optional[FPCA] = None
                scaler_z: Optional[StandardScaler] = None
                scaler_hf_target: Optional[StandardScaler] = None
                W_full_from_sub = None
                
                fpca_dim_effective: Optional[int] = None
                fpca_evr_sum: Optional[float] = None
                fpca_recon_rmse_hftr: Optional[float] = None
                fpca_recon_rmse_hfval: Optional[float] = None
                
                # outputs of target pipeline
                Y_tr = Y_va = Y_te = None
                target_te_ref = None
                
                # Ours: LF representations (unscaled z) for paired HF splits
                z_lf_tr_repr = z_lf_va_repr = z_lf_te_repr = None
                z_hat_tr_repr = z_hat_va_repr = z_hat_te_repr = None
                
                # AR1: LF targets in target space (scaled) for train
                z_lfp_tr_s = None
                z_lfu_tr_s = None
                
                if dim_reduce == "fpca":
                    scaler_y = StandardScaler(with_mean=True, with_std=True)
                    y_hf_tr_n = scaler_y.fit_transform(y_hf_tr.astype(np.float32)).astype(np.float32)
                
                    fpca = FPCA(
                        n_components=int(args.fpca_dim),
                        var_ratio=float(args.fpca_var_ratio),
                        max_components=int(args.fpca_max_dim),
                        ridge=float(args.fpca_ridge),
                        random_state=int(args.seed),
                    )
                    z_hf_tr = fpca.fit_transform(y_hf_tr_n).astype(np.float32)
                    R = int(fpca.n_components_)
                    fpca_dim_effective = R
                    fpca_evr_sum = float(np.sum(fpca.explained_variance_ratio_)) if fpca.explained_variance_ratio_ is not None else None
                    fpca_recon_rmse_hftr = float(pca_recon_rmse(y_hf_tr, scaler_y, fpca))
                    fpca_recon_rmse_hfval = float(pca_recon_rmse(y_hf_va, scaler_y, fpca))
                    print(f"[FPCA] R={R} EVR_sum={fpca_evr_sum:.6f} recon_rmse tr={fpca_recon_rmse_hftr:.6g} val={fpca_recon_rmse_hfval:.6g}")
                    save_pickle(out_dir / "scaler_y.pkl", scaler_y)
                    save_pickle(out_dir / "fpca.pkl", fpca)
                
                    def to_fpca_z_unscaled(y: np.ndarray) -> np.ndarray:
                        assert scaler_y is not None and fpca is not None
                        y = np.asarray(y, dtype=np.float32)
                        if y.ndim != 2:
                            raise ValueError(f"to_fpca_z_unscaled expects 2D y, got shape={y.shape}")
                        if y.shape[0] == 0:
                            R = int(getattr(fpca, 'n_components_', 0))
                            return np.zeros((0, R), dtype=np.float32)
                        y_n = safe_scaler_transform(scaler_y, y, n_features=int(getattr(scaler_y, 'n_features_in_', y.shape[1])))
                        return fpca.transform(y_n).astype(np.float32)
                
                    z_hf_va = to_fpca_z_unscaled(y_hf_va)
                    z_hf_te = to_fpca_z_unscaled(y_hf_te)
                
                    # Ours: LF repr on HF x (paired)
                    z_lf_tr_repr = to_fpca_z_unscaled(y_lfp_tr)
                    z_lf_va_repr = to_fpca_z_unscaled(y_lfp_va)
                    z_lf_te_repr = to_fpca_z_unscaled(y_lfp_te)
                
                    z_hat_tr_repr = to_fpca_z_unscaled(yhat_lf_tr)
                    z_hat_va_repr = to_fpca_z_unscaled(yhat_lf_va)
                    z_hat_te_repr = to_fpca_z_unscaled(yhat_lf_te)
                
                    # AR1: LF paired/unpaired train targets
                    z_lfp_tr = to_fpca_z_unscaled(y_lfp_tr)
                    z_lfu_tr = to_fpca_z_unscaled(y_lfu_tr)
                
                    scaler_z = StandardScaler(with_mean=True, with_std=True)
                    Y_tr = scaler_z.fit_transform(z_hf_tr).astype(np.float32)
                    Y_va = scaler_z.transform(z_hf_va).astype(np.float32)
                    Y_te = scaler_z.transform(z_hf_te).astype(np.float32)
                    save_pickle(out_dir / "scaler_z.pkl", scaler_z)
                
                    z_lfp_tr_s = scaler_z.transform(z_lfp_tr).astype(np.float32)
                    z_lfu_tr_s = safe_scaler_transform(scaler_z, z_lfu_tr, n_features=Y_tr.shape[1]).astype(np.float32)
                
                    def inv_target_to_y_full(mu_target_scaled: np.ndarray) -> np.ndarray:
                        assert scaler_y is not None and fpca is not None and scaler_z is not None
                        mu_target_scaled = mu_target_scaled.astype(np.float32)
                        z_unscaled = scaler_z.inverse_transform(mu_target_scaled).astype(np.float32)
                        y_n = fpca.inverse_transform(z_unscaled)
                        y = scaler_y.inverse_transform(y_n.astype(np.float32)).astype(np.float32)
                        return y
                
                    inv_target_to_y_full_fn = inv_target_to_y_full
                    target_te_ref = Y_te
                
                elif dim_reduce == "subsample":
                    Ks_int = int(args.subsample_K)
                    idx_k = make_even_subsample_indices(Kb=K, Ks=Ks_int)
                    wl_sub = wl[idx_k]
                    np.save(out_dir / "idx_subsample_k.npy", idx_k.astype(np.int64))
                    np.save(out_dir / "wavelengths_subsample.npy", wl_sub.astype(np.float32))
                
                    W_full_from_sub = build_linear_interp_weights(x_full=wl, x_sub=wl_sub)
                    save_pickle(out_dir / "interp_weights_full_from_sub.pkl", W_full_from_sub)
                
                    y_hf_tr_sub = pick_y_sub(y_hf_tr, idx_k)
                    y_hf_va_sub = pick_y_sub(y_hf_va, idx_k)
                    y_hf_te_sub = pick_y_sub(y_hf_te, idx_k)
                
                    scaler_hf_target = StandardScaler(with_mean=True, with_std=True)
                    Y_tr = scaler_hf_target.fit_transform(y_hf_tr_sub).astype(np.float32)
                    Y_va = scaler_hf_target.transform(y_hf_va_sub).astype(np.float32)
                    Y_te = scaler_hf_target.transform(y_hf_te_sub).astype(np.float32)
                    save_pickle(out_dir / "scaler_hf_y_sub.pkl", scaler_hf_target)
                
                    # Ours: LF repr on HF x (paired)
                    z_lf_tr_repr = pick_y_sub(y_lfp_tr, idx_k)
                    z_lf_va_repr = pick_y_sub(y_lfp_va, idx_k)
                    z_lf_te_repr = pick_y_sub(y_lfp_te, idx_k)
                    z_hat_tr_repr = pick_y_sub(yhat_lf_tr, idx_k)
                    z_hat_va_repr = pick_y_sub(yhat_lf_va, idx_k)
                    z_hat_te_repr = pick_y_sub(yhat_lf_te, idx_k)
                
                    # AR1: match mf_train_fpca_cokrig_ar1.py subsample behavior (LF targets NOT scaled)
                    z_lfp_tr_s = pick_y_sub(y_lfp_tr, idx_k)
                    z_lfu_tr_s = pick_y_sub(y_lfu_tr, idx_k)
                
                    def inv_target_to_y_full(mu_target_scaled: np.ndarray) -> np.ndarray:
                        assert scaler_hf_target is not None
                        y_sub = scaler_hf_target.inverse_transform(mu_target_scaled.astype(np.float32)).astype(np.float32)
                        return upsample_y_sub_to_full(y_sub, idx_k=idx_k, wl_full=wl)
                
                    inv_target_to_y_full_fn = inv_target_to_y_full
                    target_te_ref = Y_te
                
                else:
                    raise ValueError(f"Unknown dim_reduce={dim_reduce}")
                
                assert Y_tr is not None and Y_va is not None and Y_te is not None
                assert z_lf_tr_repr is not None and z_hat_tr_repr is not None
                assert target_te_ref is not None
                
                # ============================================================
                # Stage-II inputs
                #   HF-only: X
                #   Ours: U=[x, LF_hat_repr] or LF_hat only
                # ============================================================
                sx_hf = StandardScaler(with_mean=True, with_std=True).fit(x_hf_tr.astype(np.float32))
                X_hf_tr_s = sx_hf.transform(x_hf_tr.astype(np.float32)).astype(np.float32)
                X_hf_va_s = sx_hf.transform(x_hf_va.astype(np.float32)).astype(np.float32)
                X_hf_te_s = sx_hf.transform(x_hf_te.astype(np.float32)).astype(np.float32)
                save_pickle(out_dir / "scaler_x_hf.pkl", sx_hf)
                
                mode_u = str(args.mf_u_mode).lower().strip()
                use_x = (mode_u in ["xlf", "x"])
                use_lf = (mode_u in ["xlf", "lf"])
                
                def _build_u(x_s: np.ndarray, lf_repr: np.ndarray) -> np.ndarray:
                    if use_x and use_lf:
                        return np.concatenate([x_s, lf_repr], axis=1).astype(np.float32)
                    if use_x and (not use_lf):
                        return x_s.astype(np.float32)
                    return lf_repr.astype(np.float32)
                
                U_st_tr = _build_u(X_hf_tr_s, z_hat_tr_repr)
                U_st_va = _build_u(X_hf_va_s, z_hat_va_repr)
                U_st_te = _build_u(X_hf_te_s, z_hat_te_repr)
                
                su_st = StandardScaler(with_mean=True, with_std=True).fit(U_st_tr)
                U_st_tr_s = su_st.transform(U_st_tr).astype(np.float32)
                U_st_va_s = su_st.transform(U_st_va).astype(np.float32)
                U_st_te_s = su_st.transform(U_st_te).astype(np.float32)
                save_pickle(out_dir / "scaler_u_student.pkl", su_st)
                
                # ============================================================
                # Ours: delta base_hat in target space
                # ============================================================
                student_mode = str(args.student_mode).lower().strip()
                rho_fit_source = str(args.rho_fit_source).lower().strip()
                rho_intercept = bool(int(args.rho_intercept))
                
                rho_a = rho_b = None
                base_hat_tr = base_hat_va = base_hat_te = None
                
                if student_mode == "delta":
                    if dim_reduce == "fpca":
                        assert scaler_z is not None
                        lf_or_tr_t = scaler_z.transform(z_lf_tr_repr.astype(np.float32)).astype(np.float32)
                        lf_or_va_t = scaler_z.transform(z_lf_va_repr.astype(np.float32)).astype(np.float32)
                        lf_or_te_t = scaler_z.transform(z_lf_te_repr.astype(np.float32)).astype(np.float32)
                
                        lf_hat_tr_t = scaler_z.transform(z_hat_tr_repr.astype(np.float32)).astype(np.float32)
                        lf_hat_va_t = scaler_z.transform(z_hat_va_repr.astype(np.float32)).astype(np.float32)
                        lf_hat_te_t = scaler_z.transform(z_hat_te_repr.astype(np.float32)).astype(np.float32)
                    else:
                        # subsample: match existing behavior by using raw repr as "target" for base term
                        lf_or_tr_t = z_lf_tr_repr.astype(np.float32)
                        lf_or_va_t = z_lf_va_repr.astype(np.float32)
                        lf_or_te_t = z_lf_te_repr.astype(np.float32)
                
                        lf_hat_tr_t = z_hat_tr_repr.astype(np.float32)
                        lf_hat_va_t = z_hat_va_repr.astype(np.float32)
                        lf_hat_te_t = z_hat_te_repr.astype(np.float32)
                
                    lf_fit = lf_or_tr_t if rho_fit_source == "oracle" else lf_hat_tr_t
                    rho_a, rho_b = fit_affine_rho(
                        lf_fit, Y_tr,
                        ridge=float(args.rho_ridge),
                        use_intercept=rho_intercept,
                    )
                    np.save(out_dir / "rho_a.npy", rho_a.astype(np.float32))
                    np.save(out_dir / "rho_b.npy", rho_b.astype(np.float32))
                
                    print(
                        f"[OURS][rho] fit_source={rho_fit_source} intercept={rho_intercept} ridge={float(args.rho_ridge):.3g} | "
                        f"rho_a mean={float(np.mean(rho_a)):.4g} std={float(np.std(rho_a)):.4g} min={float(np.min(rho_a)):.4g} max={float(np.max(rho_a)):.4g} | "
                        f"rho_b mean={float(np.mean(rho_b)):.4g} std={float(np.std(rho_b)):.4g} min={float(np.min(rho_b)):.4g} max={float(np.max(rho_b)):.4g}"
                    )
                
                    base_hat_tr = (lf_hat_tr_t * rho_a[None, :] + rho_b[None, :]).astype(np.float32)
                    base_hat_va = (lf_hat_va_t * rho_a[None, :] + rho_b[None, :]).astype(np.float32)
                    base_hat_te = (lf_hat_te_t * rho_a[None, :] + rho_b[None, :]).astype(np.float32)
                
                # debug blocks
                dbg_block_stats("Y_tr", Y_tr)
                dbg_block_stats("z_lf_tr_repr", z_lf_tr_repr)
                dbg_block_stats("z_hat_tr_repr", z_hat_tr_repr)
                dbg_block_stats("U_st_tr_s", U_st_tr_s)
                
                # ============================================================
            else:
                print("[STAGE-I] skip student (ours not selected)")
                student_info = {}
                base_hat_tr = base_hat_va = base_hat_te = None
                U_st_tr_s = U_st_va_s = U_st_te_s = None
                rho_a = rho_b = None
            # Stage-II: Train models
            #   HF-only SVGP
            #   AR1: SVGP_L + SVGP_delta
            #   Ours: SVGP on residual (or direct)
            # ============================================================
            ard = bool(int(args.gp_ard))

            if "hf_only" in run_methods:
                # ---- HF-only
                print("[STAGE-II] Train HF-only SVGPs ...")
                svgp_hf = train_svgp_per_dim(
                    Xtr=X_hf_tr_s, Ytr=Y_tr,
                    device=device, inducing_M=int(args.svgp_M),
                    steps=int(args.svgp_steps), lr=float(args.svgp_lr),
                    ard=ard, kernel_struct="full",
                    kernel_name=str(args.kernel), matern_nu=float(args.matern_nu),
                    feat_dim=None, print_every=int(args.print_every),
                    tag="hf_only",
                    csv_logger=csv_logger,
                    csv_run_meta={"run_id": run_id, "run_name": run_name, "baseline": "hf_delta_ar1", "seed": int(args.seed)},
                )
                save_svgp_bundle(out_dir, "svgp_hf_only", svgp_hf)
            else:
                print("[STAGE-II] skip hf_only")
                svgp_hf = None

            if "ar1" in run_methods:
                # ---- AR1
                print("[STAGE-II] Train AR1(CoKrig): SVGP_L on LF-all + SVGP_delta on HF residual ...")
                
                # LF x scaler and LF train targets
                X_lf_tr_all = np.concatenate([x_lfp_tr, x_lfu_tr], axis=0).astype(np.float32)
                Z_lf_tr_all = np.concatenate([z_lfp_tr_s, z_lfu_tr_s], axis=0).astype(np.float32)
                
                sx_lf = StandardScaler(with_mean=True, with_std=True).fit(X_lf_tr_all)
                X_lf_tr_all_s = sx_lf.transform(X_lf_tr_all).astype(np.float32)
                save_pickle(out_dir / "scaler_x_lf.pkl", sx_lf)
                
                svgp_lf = train_svgp_per_dim(
                    Xtr=X_lf_tr_all_s, Ytr=Z_lf_tr_all,
                    device=device, inducing_M=int(args.svgp_M),
                    steps=int(args.svgp_steps), lr=float(args.svgp_lr),
                    ard=ard, kernel_struct="full",
                    kernel_name=str(args.kernel), matern_nu=float(args.matern_nu),
                    feat_dim=None, print_every=int(args.print_every),
                    tag="AR1(CoKrig)-LF",
                    csv_logger=csv_logger,
                    csv_run_meta={"run_id": run_id, "run_name": run_name, "baseline": "hf_delta_ar1", "seed": int(args.seed)},
                )
                save_svgp_bundle(out_dir, "svgp_ar1_lf", svgp_lf)
                
                rho_ar1 = ridge_rho(z_lfp_tr_s, Y_tr, ridge=float(args.ar1_rho_ridge))
                np.save(out_dir / "rho_ar1.npy", rho_ar1.astype(np.float32))
                print(f"[AR1][rho] ridge={float(args.ar1_rho_ridge):.3g} | mean={float(np.mean(rho_ar1)):.4g} std={float(np.std(rho_ar1)):.4g} min={float(np.min(rho_ar1)):.4g} max={float(np.max(rho_ar1)):.4g}")
                
                Y_delta_tr = (Y_tr - z_lfp_tr_s * rho_ar1[None, :]).astype(np.float32)
                svgp_delta = train_svgp_per_dim(
                    Xtr=X_hf_tr_s, Ytr=Y_delta_tr,
                    device=device, inducing_M=int(args.svgp_M),
                    steps=int(args.svgp_steps), lr=float(args.svgp_lr),
                    ard=ard, kernel_struct="full",
                    kernel_name=str(args.kernel), matern_nu=float(args.matern_nu),
                    feat_dim=None, print_every=int(args.print_every),
                    tag="AR1(CoKrig)-Delta",
                    csv_logger=csv_logger,
                    csv_run_meta={"run_id": run_id, "run_name": run_name, "baseline": "hf_delta_ar1", "seed": int(args.seed)},
                )
                save_svgp_bundle(out_dir, "svgp_ar1_delta", svgp_delta)
            else:
                print("[STAGE-II] skip ar1")
                svgp_lf = None
                svgp_delta = None
                sx_lf = None
                rho_ar1 = None

            if "ours" in run_methods:
                if int(getattr(args, "delegate_ours_to_train", 1)) == 1:
                    print("[OURS] Delegating OURS to A0 training script ...")
                    ours_report, ours_pred_root = run_ours_via_a0(args, out_dir)
                    svgp_ours = None
                    Y_tr_ours = None
                else:
                    # ---- delta_svGP (in-file)
                    print("[STAGE-II] Train delta_svGP SVGPs ...")
                    if student_mode == "delta":
                        if base_hat_tr is None:
                            raise RuntimeError("student_mode=delta but base_hat_tr is None")
                        Y_tr_ours = (Y_tr - base_hat_tr).astype(np.float32)
                    else:
                        Y_tr_ours = Y_tr

                    svgp_ours = train_svgp_per_dim(
                        Xtr=U_st_tr_s, Ytr=Y_tr_ours,
                        device=device, inducing_M=int(args.svgp_M),
                        steps=int(args.svgp_steps), lr=float(args.svgp_lr),
                        ard=ard, kernel_struct="full",
                        kernel_name=str(args.kernel), matern_nu=float(args.matern_nu),
                        feat_dim=None, print_every=int(args.print_every),
                        tag="delta_svgp",
                        csv_logger=csv_logger,
                        csv_run_meta={"run_id": run_id, "run_name": run_name, "baseline": "hf_delta_ar1", "seed": int(args.seed)},
                    )
                    save_svgp_bundle(out_dir, "svgp_ours", svgp_ours)
            else:
                print("[STAGE-II] skip ours")
                svgp_ours = None
                Y_tr_ours = None

            # ============================================================
            # If a method was skipped, fill its predictions with NaNs (so downstream metrics become NaN).
            R = int(Y_tr.shape[1])
            nan_va = np.full((int(x_hf_va.shape[0]), R), np.nan, dtype=np.float32)
            nan_te = np.full((int(x_hf_te.shape[0]), R), np.nan, dtype=np.float32)
            nan_var_va = np.full((int(x_hf_va.shape[0]), R), np.nan, dtype=np.float32)
            nan_var_te = np.full((int(x_hf_te.shape[0]), R), np.nan, dtype=np.float32)

            # Predict in target space (VAL + TEST)
            # ============================================================
            # HF-only
            if svgp_hf is None:
                mu_hf_va, var_hf_va = nan_va, nan_var_va
                mu_hf_te, var_hf_te = nan_te, nan_var_te
            else:
                mu_hf_va, var_hf_va = predict_svgp_per_dim(svgp_hf, X_hf_va_s, device=device)
                mu_hf_te, var_hf_te = predict_svgp_per_dim(svgp_hf, X_hf_te_s, device=device)

            # AR1
            if (svgp_lf is None) or (svgp_delta is None) or (sx_lf is None) or (rho_ar1 is None):
                mu_ar1_va, var_ar1_va = nan_va, nan_var_va
                mu_ar1_te, var_ar1_te = nan_te, nan_var_te
            else:
                X_hf_va_in_lf = sx_lf.transform(x_hf_va.astype(np.float32)).astype(np.float32)
                X_hf_te_in_lf = sx_lf.transform(x_hf_te.astype(np.float32)).astype(np.float32)

                mu_l_va, var_l_va = predict_svgp_per_dim(svgp_lf, X_hf_va_in_lf, device=device)
                mu_l_te, var_l_te = predict_svgp_per_dim(svgp_lf, X_hf_te_in_lf, device=device)

                mu_d_va, var_d_va = predict_svgp_per_dim(svgp_delta, X_hf_va_s, device=device)
                mu_d_te, var_d_te = predict_svgp_per_dim(svgp_delta, X_hf_te_s, device=device)

                mu_ar1_va = (mu_l_va * rho_ar1[None, :] + mu_d_va).astype(np.float32)
                mu_ar1_te = (mu_l_te * rho_ar1[None, :] + mu_d_te).astype(np.float32)

                var_ar1_va = (var_l_va * (rho_ar1[None, :] ** 2) + var_d_va).astype(np.float32)
                var_ar1_te = (var_l_te * (rho_ar1[None, :] ** 2) + var_d_te).astype(np.float32)

            # Ours
            if ours_pred_root is not None:
                # Load delegated predictions (A0 script). We only use the mf_student branch.
                d_va = Path(ours_pred_root) / "val"
                d_te = Path(ours_pred_root) / "test"
                ours_from_delegate = True
                # NOTE: delegated training script saves predictions in FULL y-space (N,K)
                mu_st_va = np.load(d_va / "y_pred__mf_student.npy").astype(np.float32)
                mu_st_te = np.load(d_te / "y_pred__mf_student.npy").astype(np.float32)

                # Use raw std here; this script performs calibration consistently for all methods.
                std_st_va = np.load(d_va / "std_raw__mf_student.npy").astype(np.float32)
                std_st_te = np.load(d_te / "std_raw__mf_student.npy").astype(np.float32)
                # STRICT: delegate outputs must match full y shapes
                assert mu_st_va.shape == y_hf_va.shape, (mu_st_va.shape, y_hf_va.shape)
                assert mu_st_te.shape == y_hf_te.shape, (mu_st_te.shape, y_hf_te.shape)
                assert std_st_va.shape == y_hf_va.shape, (std_st_va.shape, y_hf_va.shape)
                assert std_st_te.shape == y_hf_te.shape, (std_st_te.shape, y_hf_te.shape)

                var_st_va = (std_st_va ** 2).astype(np.float32)
                var_st_te = (std_st_te ** 2).astype(np.float32)
            elif (svgp_ours is None) or (U_st_va_s is None) or (U_st_te_s is None):
                mu_st_va, var_st_va = nan_va, nan_var_va
                mu_st_te, var_st_te = nan_te, nan_var_te
            else:
                mu_st_va, var_st_va = predict_svgp_per_dim(svgp_ours, U_st_va_s, device=device)
                mu_st_te, var_st_te = predict_svgp_per_dim(svgp_ours, U_st_te_s, device=device)

                if student_mode == "delta":
                    if base_hat_va is None or base_hat_te is None:
                        raise RuntimeError("student_mode=delta but base_hat_va/base_hat_te is None")
                    mu_st_va = (mu_st_va + base_hat_va).astype(np.float32)
                    mu_st_te = (mu_st_te + base_hat_te).astype(np.float32)

            # target_rmse_hf = float(rmse(mu_hf_te, target_te_ref))
            # target_rmse_ar1 = float(rmse(mu_ar1_te, target_te_ref))
            # target_rmse_ours = float(rmse(mu_st_te, target_te_ref))

            # ============================================================
            # Inverse mean to full y
            # ============================================================
            ypred_hf_va = inv_target_to_y_full_fn(mu_hf_va)
            ypred_ar1_va = inv_target_to_y_full_fn(mu_ar1_va)
            # ypred_st_va = inv_target_to_y_full_fn(mu_st_va)
            if 'ours_from_delegate' in locals() and ours_from_delegate:
                ypred_st_va = mu_st_va
            else:
                ypred_st_va = inv_target_to_y_full_fn(mu_st_va)

            ypred_hf_te = inv_target_to_y_full_fn(mu_hf_te)
            ypred_ar1_te = inv_target_to_y_full_fn(mu_ar1_te)
            # ypred_st_te = inv_target_to_y_full_fn(mu_st_te)
            if 'ours_from_delegate' in locals() and ours_from_delegate:
                ypred_st_te = mu_st_te
            else:
                ypred_st_te = inv_target_to_y_full_fn(mu_st_te)

            target_rmse_hf = float(rmse(ypred_hf_te, y_hf_te))
            target_rmse_ar1 = float(rmse(ypred_ar1_te, y_hf_te))
            target_rmse_ours = float(rmse(ypred_st_te, y_hf_te))
            # ============================================================
            # Variance propagation to y-space
            # ============================================================
            if dim_reduce == "fpca":
                assert fpca is not None and scaler_y is not None and scaler_z is not None
                scale2_z = (np.asarray(scaler_z.scale_, dtype=np.float32) ** 2)[None, :]
                var_hf_va_z_unscaled = (var_hf_va * scale2_z).astype(np.float32)
                var_or_va_z_unscaled = (var_ar1_va * scale2_z).astype(np.float32)
                # var_st_va_z_unscaled = (var_st_va * scale2_z).astype(np.float32)
                if 'ours_from_delegate' in locals() and ours_from_delegate:
                    # delegate already gives y-space variance
                    var_st_va_y = var_st_va
                else:
                    var_st_va_z_unscaled = (var_st_va * scale2_z).astype(np.float32)
                var_hf_te_z_unscaled = (var_hf_te * scale2_z).astype(np.float32)
                var_or_te_z_unscaled = (var_ar1_te * scale2_z).astype(np.float32)
                # var_st_te_z_unscaled = (var_st_te * scale2_z).astype(np.float32)
                if 'ours_from_delegate' in locals() and ours_from_delegate:
                    var_st_te_y = var_st_te
                else:
                    var_st_te_z_unscaled = (var_st_te * scale2_z).astype(np.float32)

                var_hf_va_y = fpca_propagate_var_to_y(var_hf_va_z_unscaled, fpca, scaler_y)
                var_or_va_y = fpca_propagate_var_to_y(var_or_va_z_unscaled, fpca, scaler_y)
                # var_st_va_y = fpca_propagate_var_to_y(var_st_va_z_unscaled, fpca, scaler_y)
                if not ('ours_from_delegate' in locals() and ours_from_delegate):
                    var_st_va_y = fpca_propagate_var_to_y(var_st_va_z_unscaled, fpca, scaler_y)
                var_hf_te_y = fpca_propagate_var_to_y(var_hf_te_z_unscaled, fpca, scaler_y)
                var_or_te_y = fpca_propagate_var_to_y(var_or_te_z_unscaled, fpca, scaler_y)
                # var_st_te_y = fpca_propagate_var_to_y(var_st_te_z_unscaled, fpca, scaler_y)
                if not ('ours_from_delegate' in locals() and ours_from_delegate):
                    var_st_te_y = fpca_propagate_var_to_y(var_st_te_z_unscaled, fpca, scaler_y)
            else:
                assert scaler_hf_target is not None and W_full_from_sub is not None
                var_hf_va_y = propagate_subsample_var_to_full_y_var(var_hf_va, scaler_hf_target, W_full_from_sub)
                var_or_va_y = propagate_subsample_var_to_full_y_var(var_ar1_va, scaler_hf_target, W_full_from_sub)
                # var_st_va_y = propagate_subsample_var_to_full_y_var(var_st_va, scaler_hf_target, W_full_from_sub)
                if 'ours_from_delegate' in locals() and ours_from_delegate:
                    var_st_va_y = var_st_va
                else:
                    var_st_va_y = propagate_subsample_var_to_full_y_var(var_st_va, scaler_hf_target,W_full_from_sub)
                var_hf_te_y = propagate_subsample_var_to_full_y_var(var_hf_te, scaler_hf_target, W_full_from_sub)
                var_or_te_y = propagate_subsample_var_to_full_y_var(var_ar1_te, scaler_hf_target, W_full_from_sub)
                # var_st_te_y = propagate_subsample_var_to_full_y_var(var_st_te, scaler_hf_target, W_full_from_sub)
                if 'ours_from_delegate' in locals() and ours_from_delegate:
                    var_st_te_y = var_st_te
                else:
                    var_st_te_y = propagate_subsample_var_to_full_y_var(var_st_te, scaler_hf_target,
                                                                                     W_full_from_sub)
            std_hf_va_raw = np.sqrt(np.maximum(var_hf_va_y, 0.0)).astype(np.float32)
            std_ar1_va_raw = np.sqrt(np.maximum(var_or_va_y, 0.0)).astype(np.float32)
            std_st_va_raw = np.sqrt(np.maximum(var_st_va_y, 0.0)).astype(np.float32)

            std_hf_te_raw = np.sqrt(np.maximum(var_hf_te_y, 0.0)).astype(np.float32)
            std_ar1_te_raw = np.sqrt(np.maximum(var_or_te_y, 0.0)).astype(np.float32)
            std_st_te_raw = np.sqrt(np.maximum(var_st_te_y, 0.0)).astype(np.float32)

            
            # ============================================================
            # Calibration on VAL (UQ)
            #   Default: Normalized split conformal (pooled across (N,K))
            # ============================================================
            ci_lvl = float(args.ci_level)
            do_cal = bool(int(args.ci_calibrate))
            
            ci_cal_mode = str(getattr(args, "ci_cal_mode", "conformal_norm_pooled")).lower().strip()
            ci_per_k_smooth = int(getattr(args, "ci_per_k_smooth", 0))  # legacy per_k smoothing
            
            # We output scalar 'alpha' for CSV/report compatibility:
            #   - conformal_norm_pooled: alpha := q (conformal quantile of |e|/std)
            #   - global/per_k: alpha := scale factor on std (legacy Gaussian z*std)
            alpha_hf_k = alpha_ar1_k = alpha_ours_k = None
            uq_stratified = None
            
            if do_cal:
                if ci_cal_mode == "conformal_norm_pooled":
                    # pooled normalized conformal: radius = q * std_raw
                    alpha_hf = float(conformal_q_norm_pooled(y_hf_va, ypred_hf_va, std_hf_va_raw, ci_lvl))
                    alpha_ar1 = float(conformal_q_norm_pooled(y_hf_va, ypred_ar1_va, std_ar1_va_raw, ci_lvl))
                    alpha_ours = float(conformal_q_norm_pooled(y_hf_va, ypred_st_va, std_st_va_raw, ci_lvl))
                    
                    std_hf_va_cal = (alpha_hf * std_hf_va_raw).astype(np.float32)
                    std_ar1_va_cal = (alpha_ar1 * std_ar1_va_raw).astype(np.float32)
                    std_st_va_cal = (alpha_ours * std_st_va_raw).astype(np.float32)
                    
                    std_hf_te_cal = (alpha_hf * std_hf_te_raw).astype(np.float32)
                    std_ar1_te_cal = (alpha_ar1 * std_ar1_te_raw).astype(np.float32)
                    std_st_te_cal = (alpha_ours * std_st_te_raw).astype(np.float32)
                elif ci_cal_mode == "conformal_norm_dim0_bins":
                    # Stratified normalized conformal: fit q per bin of x[:,ci_stratify_dim] on VAL, apply q(x)*std.
                    strat_dim = int(getattr(args, "ci_stratify_dim", 0))
                    strat_bins = int(getattr(args, "ci_stratify_bins", 8))
                    strat_min_n = int(getattr(args, "ci_stratify_min_n", 30))

                    alpha_hf, strat_hf = conformal_q_norm_stratified(
                        y_hf_va, ypred_hf_va, std_hf_va_raw, x_hf_va, ci_lvl,
                        strat_dim=strat_dim, n_bins=strat_bins, min_n=strat_min_n,
                    )
                    alpha_ar1, strat_ar1 = conformal_q_norm_stratified(
                        y_hf_va, ypred_ar1_va, std_ar1_va_raw, x_hf_va, ci_lvl,
                        strat_dim=strat_dim, n_bins=strat_bins, min_n=strat_min_n,
                    )
                    alpha_ours, strat_ours = conformal_q_norm_stratified(
                        y_hf_va, ypred_st_va, std_st_va_raw, x_hf_va, ci_lvl,
                        strat_dim=strat_dim, n_bins=strat_bins, min_n=strat_min_n,
                    )

                    q_hf_va = conformal_q_apply_stratified(x_hf_va, strat_hf)   # (N,)
                    q_ar1_va = conformal_q_apply_stratified(x_hf_va, strat_ar1)
                    q_st_va = conformal_q_apply_stratified(x_hf_va, strat_ours)

                    q_hf_te = conformal_q_apply_stratified(x_hf_te, strat_hf)
                    q_ar1_te = conformal_q_apply_stratified(x_hf_te, strat_ar1)
                    q_st_te = conformal_q_apply_stratified(x_hf_te, strat_ours)

                    std_hf_va_cal = (std_hf_va_raw * q_hf_va[:, None]).astype(np.float32)
                    std_ar1_va_cal = (std_ar1_va_raw * q_ar1_va[:, None]).astype(np.float32)
                    std_st_va_cal = (std_st_va_raw * q_st_va[:, None]).astype(np.float32)

                    std_hf_te_cal = (std_hf_te_raw * q_hf_te[:, None]).astype(np.float32)
                    std_ar1_te_cal = (std_ar1_te_raw * q_ar1_te[:, None]).astype(np.float32)
                    std_st_te_cal = (std_st_te_raw * q_st_te[:, None]).astype(np.float32)

                    # store per-bin qs for report (optional; does not break runners)
                    uq_stratified = {
                        "dim": int(strat_dim),
                        "mode": str(strat_hf.get("mode", "unknown")),
                        "min_n": int(strat_min_n),
                        "n_bins_req": int(strat_bins),
                        "hf_only": {"q_bins": np.asarray(strat_hf.get("q_bins")).astype(float).tolist()},
                        "ar1": {"q_bins": np.asarray(strat_ar1.get("q_bins")).astype(float).tolist()},
                        "ours": {"q_bins": np.asarray(strat_ours.get("q_bins")).astype(float).tolist()},
                    }
                    if "cats" in strat_hf:
                        uq_stratified["cats"] = np.asarray(strat_hf["cats"]).astype(float).tolist()
                    if "edges" in strat_hf:
                        uq_stratified["edges"] = np.asarray(strat_hf["edges"]).astype(float).tolist()

                elif ci_cal_mode == "conformal_norm_dim0_coarsebins_traincal":
                    # Coarse stratification + calibration set sampled from TRAIN (or VAL) to stabilize q with tiny val.
                    strat_dim = int(getattr(args, "ci_stratify_dim", 0))
                    strat_min_n = int(getattr(args, "ci_stratify_min_n", 30))
                    n_coarse = int(getattr(args, "ci_coarse_bins", 3))
                    cal_source = str(getattr(args, "ci_cal_source", "train")).lower().strip()
                    cal_n = int(getattr(args, "ci_cal_n", 40))

                    rng = np.random.RandomState(int(args.seed) + 999)

                    # Choose calibration pool
                    if cal_source == "val":
                        x_pool = x_hf_va
                        y_pool = y_hf_va
                        pool_tag = "val"
                        pool_idx = _stratified_sample_indices(x_pool, n=cal_n, dim=strat_dim, rng=rng)
                        x_cal = x_pool[pool_idx]
                        y_cal = y_pool[pool_idx]
                        # means/stds on this pool are simply slices
                        ypred_hf_cal = ypred_hf_va[pool_idx]
                        ypred_ar1_cal = ypred_ar1_va[pool_idx]
                        ypred_st_cal = ypred_st_va[pool_idx]
                        std_hf_cal_raw = std_hf_va_raw[pool_idx]
                        std_ar1_cal_raw = std_ar1_va_raw[pool_idx]
                        std_st_cal_raw = std_st_va_raw[pool_idx]
                    else:
                        # default: train or train+val
                        if cal_source == "train+val":
                            x_pool = np.concatenate([x_hf_tr, x_hf_va], axis=0)
                            y_pool = np.concatenate([y_hf_tr, y_hf_va], axis=0)
                            pool_tag = "train+val"
                        else:
                            x_pool = x_hf_tr
                            y_pool = y_hf_tr
                            pool_tag = "train"

                        pool_idx = _stratified_sample_indices(x_pool, n=cal_n, dim=strat_dim, rng=rng)
                        x_cal = x_pool[pool_idx]
                        y_cal = y_pool[pool_idx]

                        # --- Predict mean/var on calibration set (no training changes) ---
                        # HF-only
                        if svgp_hf is None:
                            mu_hf_cal, var_hf_cal = np.full((x_cal.shape[0], R), np.nan, np.float32), np.full((x_cal.shape[0], R), np.nan, np.float32)
                        else:
                            X_cal_s = sx_hf.transform(x_cal.astype(np.float32)).astype(np.float32)
                            mu_hf_cal, var_hf_cal = predict_svgp_per_dim(svgp_hf, X_cal_s, device=device)

                        # AR1
                        if (svgp_lf is None) or (svgp_delta is None) or (sx_lf is None) or (rho_ar1 is None):
                            mu_ar1_cal, var_ar1_cal = np.full((x_cal.shape[0], R), np.nan, np.float32), np.full((x_cal.shape[0], R), np.nan, np.float32)
                        else:
                            X_cal_in_lf = sx_lf.transform(x_cal.astype(np.float32)).astype(np.float32)
                            mu_l_cal, var_l_cal = predict_svgp_per_dim(svgp_lf, X_cal_in_lf, device=device)
                            X_cal_s = sx_hf.transform(x_cal.astype(np.float32)).astype(np.float32)
                            mu_d_cal, var_d_cal = predict_svgp_per_dim(svgp_delta, X_cal_s, device=device)
                            mu_ar1_cal = (mu_l_cal * rho_ar1[None, :] + mu_d_cal).astype(np.float32)
                            var_ar1_cal = (var_l_cal * (rho_ar1[None, :] ** 2) + var_d_cal).astype(np.float32)

                        # OURS (if delegated predictions exist, we cannot reliably get TRAIN std/mean here; fall back to VAL pool)
                        if ours_pred_root is not None:
                            # fallback: use full VAL as calibration pool (stable and available)
                            pool_tag = pool_tag + "+ours_val_fallback"
                            x_cal = x_hf_va
                            y_cal = y_hf_va
                            ypred_hf_cal = ypred_hf_va
                            ypred_ar1_cal = ypred_ar1_va
                            ypred_st_cal = ypred_st_va
                            std_hf_cal_raw = std_hf_va_raw
                            std_ar1_cal_raw = std_ar1_va_raw
                            std_st_cal_raw = std_st_va_raw
                        else:
                            if (svgp_ours is None) or (U_st_tr_s is None and U_st_va_s is None and U_st_te_s is None):
                                mu_st_cal, var_st_cal = np.full((x_cal.shape[0], R), np.nan, np.float32), np.full((x_cal.shape[0], R), np.nan, np.float32)
                            else:
                                # If calibration pool is train-only, we can reuse U_st_tr_s for matching rows.
                                if cal_source == "train" and U_st_tr_s is not None and x_pool is x_hf_tr:
                                    # map pool_idx into train indices (direct)
                                    U_cal_s = U_st_tr_s[pool_idx]
                                else:
                                    # generic: compute student features for x_cal
                                    if U_st_tr_s is not None and x_pool.shape[0] == x_hf_tr.shape[0] and x_cal.shape[0] == pool_idx.shape[0] and cal_source == "train":
                                        U_cal_s = U_st_tr_s[pool_idx]
                                    else:
                                        # compute via student pipeline when available
                                        if student_scaler_x is None or student_model is None:
                                            U_cal_s = None
                                        else:
                                            x_cal_st_s = safe_scaler_transform(student_scaler_x, x_cal.astype(np.float32)).astype(np.float32)
                                            _, feat_cal = mlp_predict_and_features(student_model, x_cal_st_s, device=device)
                                            # U is either oracle feature or student feature depending on flags; here we follow U_st_* convention
                                            U_cal_s = feat_cal.astype(np.float32)

                                if U_cal_s is None:
                                    mu_st_cal, var_st_cal = np.full((x_cal.shape[0], R), np.nan, np.float32), np.full((x_cal.shape[0], R), np.nan, np.float32)
                                else:
                                    mu_st_cal, var_st_cal = predict_svgp_per_dim(svgp_ours, U_cal_s, device=device)
                                    if student_mode == "delta":
                                        # base_hat for calibration points: if we used train pool, base_hat_tr aligns.
                                        if cal_source == "train" and base_hat_tr is not None and x_pool is x_hf_tr:
                                            mu_st_cal = (mu_st_cal + base_hat_tr[pool_idx]).astype(np.float32)
                                        elif cal_source == "train+val":
                                            # approximate: if base_hat_tr exists, use zeros for val part; this is a fallback.
                                            mu_st_cal = mu_st_cal.astype(np.float32)
                                        else:
                                            mu_st_cal = mu_st_cal.astype(np.float32)

                            # Convert to y-space
                            ypred_hf_cal = inv_target_to_y_full_fn(mu_hf_cal)
                            ypred_ar1_cal = inv_target_to_y_full_fn(mu_ar1_cal)
                            ypred_st_cal = inv_target_to_y_full_fn(mu_st_cal)

                            # Propagate var to y-space
                            if dim_reduce == "fpca":
                                assert fpca is not None and scaler_y is not None and scaler_z is not None
                                scale2_z = (np.asarray(scaler_z.scale_, dtype=np.float32) ** 2)[None, :]
                                var_hf_cal_y = fpca_propagate_var_to_y((var_hf_cal * scale2_z).astype(np.float32), fpca, scaler_y)
                                var_ar1_cal_y = fpca_propagate_var_to_y((var_ar1_cal * scale2_z).astype(np.float32), fpca, scaler_y)
                                var_st_cal_y = fpca_propagate_var_to_y((var_st_cal * scale2_z).astype(np.float32), fpca, scaler_y)
                            else:
                                assert scaler_hf_target is not None and W_full_from_sub is not None
                                var_hf_cal_y = propagate_subsample_var_to_full_y_var(var_hf_cal, scaler_hf_target, W_full_from_sub)
                                var_ar1_cal_y = propagate_subsample_var_to_full_y_var(var_ar1_cal, scaler_hf_target, W_full_from_sub)
                                var_st_cal_y = propagate_subsample_var_to_full_y_var(var_st_cal, scaler_hf_target, W_full_from_sub)

                            std_hf_cal_raw = np.sqrt(np.maximum(var_hf_cal_y, 0.0)).astype(np.float32)
                            std_ar1_cal_raw = np.sqrt(np.maximum(var_ar1_cal_y, 0.0)).astype(np.float32)
                            std_st_cal_raw = np.sqrt(np.maximum(var_st_cal_y, 0.0)).astype(np.float32)

                    # Build coarse strata on calibration x
                    strata_coarse = _build_coarse_strata_from_x(x_cal, strat_dim, n_coarse=n_coarse)

                    alpha_hf, strat_hf = conformal_q_norm_stratified_with_strata(
                        y_cal, ypred_hf_cal, std_hf_cal_raw, x_cal, ci_lvl,
                        strata=strata_coarse, n_bins_req=n_coarse, min_n=strat_min_n,
                    )
                    alpha_ar1, strat_ar1 = conformal_q_norm_stratified_with_strata(
                        y_cal, ypred_ar1_cal, std_ar1_cal_raw, x_cal, ci_lvl,
                        strata=strata_coarse, n_bins_req=n_coarse, min_n=strat_min_n,
                    )
                    alpha_ours, strat_ours = conformal_q_norm_stratified_with_strata(
                        y_cal, ypred_st_cal, std_st_cal_raw, x_cal, ci_lvl,
                        strata=strata_coarse, n_bins_req=n_coarse, min_n=strat_min_n,
                    )

                    q_hf_va = conformal_q_apply_stratified(x_hf_va, strat_hf)
                    q_ar1_va = conformal_q_apply_stratified(x_hf_va, strat_ar1)
                    q_st_va = conformal_q_apply_stratified(x_hf_va, strat_ours)

                    q_hf_te = conformal_q_apply_stratified(x_hf_te, strat_hf)
                    q_ar1_te = conformal_q_apply_stratified(x_hf_te, strat_ar1)
                    q_st_te = conformal_q_apply_stratified(x_hf_te, strat_ours)

                    # NOTE: for conformal modes, these arrays are *radii* (half-widths), not Gaussian std.
                    std_hf_va_cal = (std_hf_va_raw * q_hf_va[:, None]).astype(np.float32)
                    std_ar1_va_cal = (std_ar1_va_raw * q_ar1_va[:, None]).astype(np.float32)
                    std_st_va_cal = (std_st_va_raw * q_st_va[:, None]).astype(np.float32)

                    std_hf_te_cal = (std_hf_te_raw * q_hf_te[:, None]).astype(np.float32)
                    std_ar1_te_cal = (std_ar1_te_raw * q_ar1_te[:, None]).astype(np.float32)
                    std_st_te_cal = (std_st_te_raw * q_st_te[:, None]).astype(np.float32)

                    uq_stratified = {
                        "dim": int(strat_dim),
                        "mode": str(strata_coarse.get("mode", "unknown")),
                        "min_n": int(strat_min_n),
                        "n_bins_req": int(n_coarse),
                        "cal_source": str(pool_tag),
                        "hf_only": {"q_bins": np.asarray(strat_hf.get("q_bins")).astype(float).tolist()},
                        "ar1": {"q_bins": np.asarray(strat_ar1.get("q_bins")).astype(float).tolist()},
                        "ours": {"q_bins": np.asarray(strat_ours.get("q_bins")).astype(float).tolist()},
                    }
                    if "cats_groups" in strata_coarse:
                        uq_stratified["cats_groups"] = strata_coarse["cats_groups"]
                    if "cats" in strata_coarse:
                        uq_stratified["cats"] = np.asarray(strata_coarse["cats"]).astype(float).tolist()
                    if "edges" in strata_coarse:
                        uq_stratified["edges"] = np.asarray(strata_coarse["edges"]).astype(float).tolist()

                elif ci_cal_mode == "per_k":
                    # legacy: per-k Gaussian scale-only calibration (unstable for small N)
                    alpha_hf_k = calibrate_sigma_scale_per_k(y_hf_va, ypred_hf_va, std_hf_va_raw, ci_lvl, smooth_win=ci_per_k_smooth)
                    alpha_ar1_k = calibrate_sigma_scale_per_k(y_hf_va, ypred_ar1_va, std_ar1_va_raw, ci_lvl, smooth_win=ci_per_k_smooth)
                    alpha_ours_k = calibrate_sigma_scale_per_k(y_hf_va, ypred_st_va, std_st_va_raw, ci_lvl, smooth_win=ci_per_k_smooth)
                    alpha_hf = float(np.mean(alpha_hf_k))
                    alpha_ar1 = float(np.mean(alpha_ar1_k))
                    alpha_ours = float(np.mean(alpha_ours_k))
                    
                    std_hf_va_cal = (alpha_hf_k[None, :] * std_hf_va_raw).astype(np.float32)
                    std_ar1_va_cal = (alpha_ar1_k[None, :] * std_ar1_va_raw).astype(np.float32)
                    std_st_va_cal = (alpha_ours_k[None, :] * std_st_va_raw).astype(np.float32)
                    
                    std_hf_te_cal = (alpha_hf_k[None, :] * std_hf_te_raw).astype(np.float32)
                    std_ar1_te_cal = (alpha_ar1_k[None, :] * std_ar1_te_raw).astype(np.float32)
                    std_st_te_cal = (alpha_ours_k[None, :] * std_st_te_raw).astype(np.float32)
                else:
                    # legacy: global Gaussian scale-only calibration
                    alpha_hf = float(calibrate_sigma_scale(y_hf_va, ypred_hf_va, std_hf_va_raw, ci_lvl))
                    alpha_ar1 = float(calibrate_sigma_scale(y_hf_va, ypred_ar1_va, std_ar1_va_raw, ci_lvl))
                    alpha_ours = float(calibrate_sigma_scale(y_hf_va, ypred_st_va, std_st_va_raw, ci_lvl))
                    
                    std_hf_va_cal = (alpha_hf * std_hf_va_raw).astype(np.float32)
                    std_ar1_va_cal = (alpha_ar1 * std_ar1_va_raw).astype(np.float32)
                    std_st_va_cal = (alpha_ours * std_st_va_raw).astype(np.float32)
                    
                    std_hf_te_cal = (alpha_hf * std_hf_te_raw).astype(np.float32)
                    std_ar1_te_cal = (alpha_ar1 * std_ar1_te_raw).astype(np.float32)
                    std_st_te_cal = (alpha_ours * std_st_te_raw).astype(np.float32)
            else:
                ci_cal_mode = "none"
                alpha_hf, alpha_ar1, alpha_ours = 1.0, 1.0, 1.0
                std_hf_va_cal = std_hf_va_raw
                std_ar1_va_cal = std_ar1_va_raw
                std_st_va_cal = std_st_va_raw
                std_hf_te_cal = std_hf_te_raw
                std_ar1_te_cal = std_ar1_te_raw
                std_st_te_cal = std_st_te_raw
            
            # ============================================================
            # UQ metrics
            # ============================================================

            cov_val_raw_hf = float(ci_coverage_y(y_hf_va, ypred_hf_va, std_hf_va_raw, ci_lvl))
            cov_val_raw_ar1 = float(ci_coverage_y(y_hf_va, ypred_ar1_va, std_ar1_va_raw, ci_lvl))
            cov_val_raw_ours = float(ci_coverage_y(y_hf_va, ypred_st_va, std_st_va_raw, ci_lvl))

            is_conformal = str(ci_cal_mode).startswith("conformal_norm_")

            if is_conformal:
                # In conformal modes, *_cal arrays are *radii* (half-widths) already.
                cov_val_cal_hf = ci_coverage_from_radius(y_hf_va, ypred_hf_va, std_hf_va_cal)
                cov_val_cal_ar1 = ci_coverage_from_radius(y_hf_va, ypred_ar1_va, std_ar1_va_cal)
                cov_val_cal_ours = ci_coverage_from_radius(y_hf_va, ypred_st_va, std_st_va_cal)
            else:
                cov_val_cal_hf = float(ci_coverage_y(y_hf_va, ypred_hf_va, std_hf_va_cal, ci_lvl))
                cov_val_cal_ar1 = float(ci_coverage_y(y_hf_va, ypred_ar1_va, std_ar1_va_cal, ci_lvl))
                cov_val_cal_ours = float(ci_coverage_y(y_hf_va, ypred_st_va, std_st_va_cal, ci_lvl))

            wid_val_raw_hf = float(ci_width_y(std_hf_va_raw, ci_lvl))
            wid_val_raw_ar1 = float(ci_width_y(std_ar1_va_raw, ci_lvl))
            wid_val_raw_ours = float(ci_width_y(std_st_va_raw, ci_lvl))

            if is_conformal:
                wid_val_cal_hf = ci_width_from_radius(std_hf_va_cal)
                wid_val_cal_ar1 = ci_width_from_radius(std_ar1_va_cal)
                wid_val_cal_ours = ci_width_from_radius(std_st_va_cal)
            else:
                wid_val_cal_hf = float(ci_width_y(std_hf_va_cal, ci_lvl))
                wid_val_cal_ar1 = float(ci_width_y(std_ar1_va_cal, ci_lvl))
                wid_val_cal_ours = float(ci_width_y(std_st_va_cal, ci_lvl))

            cov_test_raw_hf = float(ci_coverage_y(y_hf_te, ypred_hf_te, std_hf_te_raw, ci_lvl))
            cov_test_raw_ar1 = float(ci_coverage_y(y_hf_te, ypred_ar1_te, std_ar1_te_raw, ci_lvl))
            cov_test_raw_ours = float(ci_coverage_y(y_hf_te, ypred_st_te, std_st_te_raw, ci_lvl))

            if is_conformal:
                cov_test_cal_hf = ci_coverage_from_radius(y_hf_te, ypred_hf_te, std_hf_te_cal)
                cov_test_cal_ar1 = ci_coverage_from_radius(y_hf_te, ypred_ar1_te, std_ar1_te_cal)
                cov_test_cal_ours = ci_coverage_from_radius(y_hf_te, ypred_st_te, std_st_te_cal)
            else:
                cov_test_cal_hf = float(ci_coverage_y(y_hf_te, ypred_hf_te, std_hf_te_cal, ci_lvl))
                cov_test_cal_ar1 = float(ci_coverage_y(y_hf_te, ypred_ar1_te, std_ar1_te_cal, ci_lvl))
                cov_test_cal_ours = float(ci_coverage_y(y_hf_te, ypred_st_te, std_st_te_cal, ci_lvl))

            wid_test_raw_hf = float(ci_width_y(std_hf_te_raw, ci_lvl))
            wid_test_raw_ar1 = float(ci_width_y(std_ar1_te_raw, ci_lvl))
            wid_test_raw_ours = float(ci_width_y(std_st_te_raw, ci_lvl))

            if is_conformal:
                wid_test_cal_hf = ci_width_from_radius(std_hf_te_cal)
                wid_test_cal_ar1 = ci_width_from_radius(std_ar1_te_cal)
                wid_test_cal_ours = ci_width_from_radius(std_st_te_cal)
            else:
                wid_test_cal_hf = float(ci_width_y(std_hf_te_cal, ci_lvl))
                wid_test_cal_ar1 = float(ci_width_y(std_ar1_te_cal, ci_lvl))
                wid_test_cal_ours = float(ci_width_y(std_st_te_cal, ci_lvl))

            print("[UQ] alpha HF-only/AR1(CoKrig)/delta_svgp =", alpha_hf, alpha_ar1, alpha_ours)
            print("[UQ] VAL coverage raw  HF-only/AR1(CoKrig)/delta_svgp =", cov_val_raw_hf, cov_val_raw_ar1, cov_val_raw_ours)
            print("[UQ] VAL coverage cal  HF-only/AR1(CoKrig)/delta_svgp =", cov_val_cal_hf, cov_val_cal_ar1, cov_val_cal_ours)
            print("[UQ] TEST coverage raw HF-only/AR1(CoKrig)/delta_svgp =", cov_test_raw_hf, cov_test_raw_ar1, cov_test_raw_ours)
            print("[UQ] TEST coverage cal HF-only/AR1(CoKrig)/delta_svgp =", cov_test_cal_hf, cov_test_cal_ar1, cov_test_cal_ours)

            # ============================================================
            # Point metrics
            # ============================================================
            y_rmse_hf = float(rmse(ypred_hf_te, y_hf_te))
            y_rmse_ar1 = float(rmse(ypred_ar1_te, y_hf_te))
            y_rmse_ours = float(rmse(ypred_st_te, y_hf_te))

            y_r2_val_hf = float(r2_score(y_hf_va, ypred_hf_va))
            y_r2_val_ar1 = float(r2_score(y_hf_va, ypred_ar1_va))
            y_r2_val_ours = float(r2_score(y_hf_va, ypred_st_va))

            y_r2_test_hf = float(r2_score(y_hf_te, ypred_hf_te))
            y_r2_test_ar1 = float(r2_score(y_hf_te, ypred_ar1_te))
            y_r2_test_ours = float(r2_score(y_hf_te, ypred_st_te))

            nll_val_raw_hf = float(gaussian_nll(y_hf_va, ypred_hf_va, std=std_hf_va_raw))
            nll_val_raw_ar1 = float(gaussian_nll(y_hf_va, ypred_ar1_va, std=std_ar1_va_raw))
            nll_val_raw_ours = float(gaussian_nll(y_hf_va, ypred_st_va, std=std_st_va_raw))

            if is_conformal:
                nll_val_cal_hf = float(gaussian_nll(y_hf_va, ypred_hf_va, std=_radius_to_std_equiv(std_hf_va_cal, ci_lvl)))
                nll_val_cal_ar1 = float(gaussian_nll(y_hf_va, ypred_ar1_va, std=_radius_to_std_equiv(std_ar1_va_cal, ci_lvl)))
                nll_val_cal_ours = float(gaussian_nll(y_hf_va, ypred_st_va, std=_radius_to_std_equiv(std_st_va_cal, ci_lvl)))
            else:
                nll_val_cal_hf = float(gaussian_nll(y_hf_va, ypred_hf_va, std=std_hf_va_cal))
                nll_val_cal_ar1 = float(gaussian_nll(y_hf_va, ypred_ar1_va, std=std_ar1_va_cal))
                nll_val_cal_ours = float(gaussian_nll(y_hf_va, ypred_st_va, std=std_st_va_cal))

            nll_test_raw_hf = float(gaussian_nll(y_hf_te, ypred_hf_te, std=std_hf_te_raw))
            nll_test_raw_ar1 = float(gaussian_nll(y_hf_te, ypred_ar1_te, std=std_ar1_te_raw))
            nll_test_raw_ours = float(gaussian_nll(y_hf_te, ypred_st_te, std=std_st_te_raw))

            if is_conformal:
                nll_test_cal_hf = float(gaussian_nll(y_hf_te, ypred_hf_te, std=_radius_to_std_equiv(std_hf_te_cal, ci_lvl)))
                nll_test_cal_ar1 = float(gaussian_nll(y_hf_te, ypred_ar1_te, std=_radius_to_std_equiv(std_ar1_te_cal, ci_lvl)))
                nll_test_cal_ours = float(gaussian_nll(y_hf_te, ypred_st_te, std=_radius_to_std_equiv(std_st_te_cal, ci_lvl)))
            else:
                nll_test_cal_hf = float(gaussian_nll(y_hf_te, ypred_hf_te, std=std_hf_te_cal))
                nll_test_cal_ar1 = float(gaussian_nll(y_hf_te, ypred_ar1_te, std=std_ar1_te_cal))
                nll_test_cal_ours = float(gaussian_nll(y_hf_te, ypred_st_te, std=std_st_te_cal))

            nlpd_val_raw_hf = float(gaussian_nlpd(y_hf_va, ypred_hf_va, std=std_hf_va_raw))
            nlpd_val_raw_ar1 = float(gaussian_nlpd(y_hf_va, ypred_ar1_va, std=std_ar1_va_raw))
            nlpd_val_raw_ours = float(gaussian_nlpd(y_hf_va, ypred_st_va, std=std_st_va_raw))

            if is_conformal:
                nlpd_val_cal_hf = float(gaussian_nlpd(y_hf_va, ypred_hf_va, std=_radius_to_std_equiv(std_hf_va_cal, ci_lvl)))
                nlpd_val_cal_ar1 = float(gaussian_nlpd(y_hf_va, ypred_ar1_va, std=_radius_to_std_equiv(std_ar1_va_cal, ci_lvl)))
                nlpd_val_cal_ours = float(gaussian_nlpd(y_hf_va, ypred_st_va, std=_radius_to_std_equiv(std_st_va_cal, ci_lvl)))
            else:
                nlpd_val_cal_hf = float(gaussian_nlpd(y_hf_va, ypred_hf_va, std=std_hf_va_cal))
                nlpd_val_cal_ar1 = float(gaussian_nlpd(y_hf_va, ypred_ar1_va, std=std_ar1_va_cal))
                nlpd_val_cal_ours = float(gaussian_nlpd(y_hf_va, ypred_st_va, std=std_st_va_cal))

            nlpd_test_raw_hf = float(gaussian_nlpd(y_hf_te, ypred_hf_te, std=std_hf_te_raw))
            nlpd_test_raw_ar1 = float(gaussian_nlpd(y_hf_te, ypred_ar1_te, std=std_ar1_te_raw))
            nlpd_test_raw_ours = float(gaussian_nlpd(y_hf_te, ypred_st_te, std=std_st_te_raw))

            if is_conformal:
                nlpd_test_cal_hf = float(gaussian_nlpd(y_hf_te, ypred_hf_te, std=_radius_to_std_equiv(std_hf_te_cal, ci_lvl)))
                nlpd_test_cal_ar1 = float(gaussian_nlpd(y_hf_te, ypred_ar1_te, std=_radius_to_std_equiv(std_ar1_te_cal, ci_lvl)))
                nlpd_test_cal_ours = float(gaussian_nlpd(y_hf_te, ypred_st_te, std=_radius_to_std_equiv(std_st_te_cal, ci_lvl)))
            else:
                nlpd_test_cal_hf = float(gaussian_nlpd(y_hf_te, ypred_hf_te, std=std_hf_te_cal))
                nlpd_test_cal_ar1 = float(gaussian_nlpd(y_hf_te, ypred_ar1_te, std=std_ar1_te_cal))
                nlpd_test_cal_ours = float(gaussian_nlpd(y_hf_te, ypred_st_te, std=std_st_te_cal))

            # ============================================================
            # Plotting: 5 curves (HF_only / AR1 / Ours / HF_gt / LF)
            # ============================================================
            print("[PLOT] saving 5-curve comparison plots ...")

            n_te = int(y_hf_te.shape[0])
            n_side = int(getattr(args, "n_plot", 0))  # 与 mf_train_* 对齐

            if n_side <= 0:
                print("[PLOT] n_plot<=0, skip case plots.")
            else:
                n_side = max(1, min(n_side, n_te // 2 if n_te >= 2 else 1))
                # 下面 best/worst 选择与循环保持不变

            def _per_sample_rmse(yhat: np.ndarray, ygt: np.ndarray) -> np.ndarray:
                yhat = np.asarray(yhat, dtype=np.float32)
                ygt = np.asarray(ygt, dtype=np.float32)
                if yhat.ndim != 2 or ygt.ndim != 2 or yhat.shape != ygt.shape:
                    raise ValueError(f"_per_sample_rmse expects (N,K) arrays with same shape, got {yhat.shape} vs {ygt.shape}")
                mse = np.mean((yhat - ygt) ** 2, axis=1)
                out = np.sqrt(np.maximum(mse, 0.0))
                # if NaN predictions (skipped methods), push them to the end
                out = np.where(np.isfinite(out), out, np.inf)
                return out

            # priority for ranking: ours -> hf_only (fallback)
            if "ours" in run_methods and svgp_ours is not None:
                rmse_rank = _per_sample_rmse(ypred_st_te, y_hf_te)
                rank_name = "ours"
            elif "hf_only" in run_methods and svgp_hf is not None:
                rmse_rank = _per_sample_rmse(ypred_hf_te, y_hf_te)
                rank_name = "hf_only"
            else:
                rmse_rank = _per_sample_rmse(ypred_ar1_te, y_hf_te)
                rank_name = "ar1"

            order = np.argsort(rmse_rank)  # ascending
            best_idx = order[:n_side].tolist()
            worst_idx = order[::-1][:n_side].tolist()

            # keep deterministic + unique ordering: best first, then worst
            idx_cases = []
            for ii in best_idx + worst_idx:
                if int(ii) not in idx_cases:
                    idx_cases.append(int(ii))

            print(f"[PLOT][TEST] case selection by per-sample RMSE ({rank_name}) | best={best_idx} worst={worst_idx}")

            for j, i in enumerate(idx_cases):
                y_gt = y_hf_te[i]
                y_lf = y_lfp_te[i]  # paired LF at same idx

                y_hf_pred = ypred_hf_te[i]
                y_ar1_pred = ypred_ar1_te[i]
                y_ours_pred = ypred_st_te[i]

                # raw bands
                b = make_ci_bands_for_curve(y_hf_pred, std_hf_te_raw[i], ci_lvl)
                lo_hf_raw, hi_hf_raw = b["lo"], b["hi"]
                b = make_ci_bands_for_curve(y_ar1_pred, std_ar1_te_raw[i], ci_lvl)
                lo_ar1_raw, hi_ar1_raw = b["lo"], b["hi"]
                b = make_ci_bands_for_curve(y_ours_pred, std_st_te_raw[i], ci_lvl)
                lo_ours_raw, hi_ours_raw = b["lo"], b["hi"]

                # cal bands
                if is_conformal:
                    # In conformal modes, *_cal are CI radii already: band = y ± radius (no extra z).
                    r = np.asarray(std_hf_te_cal[i], dtype=np.float32)
                    lo_hf_cal, hi_hf_cal = y_hf_pred - r, y_hf_pred + r
                    r = np.asarray(std_ar1_te_cal[i], dtype=np.float32)
                    lo_ar1_cal, hi_ar1_cal = y_ar1_pred - r, y_ar1_pred + r
                    r = np.asarray(std_st_te_cal[i], dtype=np.float32)
                    lo_ours_cal, hi_ours_cal = y_ours_pred - r, y_ours_pred + r
                else:
                    b = make_ci_bands_for_curve(y_hf_pred, std_hf_te_cal[i], ci_lvl)
                    lo_hf_cal, hi_hf_cal = b["lo"], b["hi"]
                    b = make_ci_bands_for_curve(y_ar1_pred, std_ar1_te_cal[i], ci_lvl)
                    lo_ar1_cal, hi_ar1_cal = b["lo"], b["hi"]
                    b = make_ci_bands_for_curve(y_ours_pred, std_st_te_cal[i], ci_lvl)
                    lo_ours_cal, hi_ours_cal = b["lo"], b["hi"]

                # ---------- NEW: per-case RMSE (shared by all three plots) ----------
                rmse_hf = _rmse_1d(y_hf_pred, y_gt)
                rmse_ar1 = _rmse_1d(y_ar1_pred, y_gt)
                rmse_ours = _rmse_1d(y_ours_pred, y_gt)

                # =========================
                # base (noCI)
                # =========================
                metrics_text_base = (
                    f"RMSE: hf={rmse_hf:.4f}  ar1={rmse_ar1:.4f}  ours={rmse_ours:.4f}"
                )

                plot_case_5curves_spectrum_named(
                    save_path=Path(str(out_dir / f"case{j:02d}_noCI.png")),
                    wl=wl,
                    y_hf_gt=y_gt,
                    y_lf=y_lf,
                    y_pred_hf=y_hf_pred,
                    y_pred_ar1=y_ar1_pred,  # oracle slot reused as AR1
                    y_pred_delta_svgp=y_ours_pred,
                    title=f"TEST case{j:02d} (idx={i}) | noCI",
                    metrics_text=metrics_text_base,  # NEW
                    show_gt_points=True,  # NEW（如果你启用了点+线）
                )

                # =========================
                # raw
                # =========================
                cov_hf_raw, wid_hf_raw = _cov_width_1d(y_gt, lo_hf_raw, hi_hf_raw)
                cov_ar1_raw, wid_ar1_raw = _cov_width_1d(y_gt, lo_ar1_raw, hi_ar1_raw)
                cov_ours_raw, wid_ours_raw = _cov_width_1d(y_gt, lo_ours_raw, hi_ours_raw)

                metrics_text_raw = (
                    f"RMSE: hf={rmse_hf:.4f}  ar1={rmse_ar1:.4f}  ours={rmse_ours:.4f}\n"
                    f"Cov@{ci_lvl:.2f} (raw): hf={cov_hf_raw:.3f}  ar1={cov_ar1_raw:.3f}  ours={cov_ours_raw:.3f}\n"
                    f"Width (raw): hf={wid_hf_raw:.4f}  ar1={wid_ar1_raw:.4f}  ours={wid_ours_raw:.4f}"
                )

                plot_case_5curves_spectrum_named(
                    save_path=Path(str(out_dir / f"case{j:02d}_ciRaw.png")),
                    wl=wl,
                    y_hf_gt=y_gt,
                    y_lf=y_lf,
                    y_pred_hf=y_hf_pred,
                    y_pred_ar1=y_ar1_pred,
                    y_pred_delta_svgp=y_ours_pred,
                    ci_bands={
                        "hf_only": {"lo": lo_hf_raw, "hi": hi_hf_raw},
                        "ar1_cokrig": {"lo": lo_ar1_raw, "hi": hi_ar1_raw},
                        "delta_svgp": {"lo": lo_ours_raw, "hi": hi_ours_raw},
                    },
                    title=f"TEST case{j:02d} (idx={i}) | CI raw (lvl={ci_lvl})",
                    metrics_text=metrics_text_raw,  # NEW
                    show_gt_points=True,  # NEW
                )

                # =========================
                # cal
                # =========================
                cov_hf_cal, wid_hf_cal = _cov_width_1d(y_gt, lo_hf_cal, hi_hf_cal)
                cov_ar1_cal, wid_ar1_cal = _cov_width_1d(y_gt, lo_ar1_cal, hi_ar1_cal)
                cov_ours_cal, wid_ours_cal = _cov_width_1d(y_gt, lo_ours_cal, hi_ours_cal)

                metrics_text_cal = (
                    f"RMSE: hf={rmse_hf:.4f}  ar1={rmse_ar1:.4f}  ours={rmse_ours:.4f}\n"
                    f"Cov@{ci_lvl:.2f} (cal): hf={cov_hf_cal:.3f}  ar1={cov_ar1_cal:.3f}  ours={cov_ours_cal:.3f}\n"
                    f"Width (cal): hf={wid_hf_cal:.4f}  ar1={wid_ar1_cal:.4f}  ours={wid_ours_cal:.4f}"
                )

                plot_case_5curves_spectrum_named(
                    save_path=Path(str(out_dir / f"case{j:02d}_ciCal.png")),
                    wl=wl,
                    y_hf_gt=y_gt,
                    y_lf=y_lf,
                    y_pred_hf=y_hf_pred,
                    y_pred_ar1=y_ar1_pred,
                    y_pred_delta_svgp=y_ours_pred,
                    ci_bands={
                        "hf_only": {"lo": lo_hf_cal, "hi": hi_hf_cal},
                        "ar1_cokrig": {"lo": lo_ar1_cal, "hi": hi_ar1_cal},
                        "delta_svgp": {"lo": lo_ours_cal, "hi": hi_ours_cal},
                    },
                    title=f"TEST case{j:02d} (idx={i}) | CI cal (lvl={ci_lvl})",
                    metrics_text=metrics_text_cal,  # NEW
                    show_gt_points=True,  # NEW
                )
            # ============================================================
            # Report
            # ============================================================
            report: Dict[str, Any] = {
                "run_id": run_id,
                "run_name": run_name,
                "seed": int(args.seed),
                "data_dir": str(data_dir),
                "out_dir": str(out_dir),
                "baseline": "hf_only + ar1 + ours(delta-student)",
                "dim_reduce": dim_reduce,
                "fpca": None,
                "student": {
                    "mode": str(args.student_mode),
                    "u_mode": str(args.mf_u_mode),
                    "feat_dim": int(args.student_feat_dim),
                    "feat_act": str(args.student_feat_act),
                    "train_log": None,
                    **dbg_student_metrics,
                },
                "metrics": {
                    "target_rmse": {"hf_only": target_rmse_hf, "ar1": target_rmse_ar1, "ours": target_rmse_ours},
                    "y_rmse": {"hf_only": y_rmse_hf, "ar1": y_rmse_ar1, "ours": y_rmse_ours},
                    "r2": {
                        "val": {"hf_only": y_r2_val_hf, "ar1": y_r2_val_ar1, "ours": y_r2_val_ours},
                        "test": {"hf_only": y_r2_test_hf, "ar1": y_r2_test_ar1, "ours": y_r2_test_ours},
                    },
                    "nll": {
                        "val": {
                            "raw": {"hf_only": nll_val_raw_hf, "ar1": nll_val_raw_ar1, "ours": nll_val_raw_ours},
                            "cal": {"hf_only": nll_val_cal_hf, "ar1": nll_val_cal_ar1, "ours": nll_val_cal_ours},
                        },
                        "test": {
                            "raw": {"hf_only": nll_test_raw_hf, "ar1": nll_test_raw_ar1, "ours": nll_test_raw_ours},
                            "cal": {"hf_only": nll_test_cal_hf, "ar1": nll_test_cal_ar1, "ours": nll_test_cal_ours},
                        },
                    },
                    "nlpd": {
                        "val": {
                            "raw": {"hf_only": nlpd_val_raw_hf, "ar1": nlpd_val_raw_ar1, "ours": nlpd_val_raw_ours},
                            "cal": {"hf_only": nlpd_val_cal_hf, "ar1": nlpd_val_cal_ar1, "ours": nlpd_val_cal_ours},
                        },
                        "test": {
                            "raw": {"hf_only": nlpd_test_raw_hf, "ar1": nlpd_test_raw_ar1, "ours": nlpd_test_raw_ours},
                            "cal": {"hf_only": nlpd_test_cal_hf, "ar1": nlpd_test_cal_ar1, "ours": nlpd_test_cal_ours},
                        },
                    },
                    "uq": {
                        "ci_level": float(ci_lvl),
                "ci_calibrate": int(args.ci_calibrate),
                "ci_cal_mode": str(getattr(args, "ci_cal_mode", "conformal_norm_pooled")),
                        "stratified": uq_stratified,
                        "alpha": {"hf_only": float(alpha_hf), "ar1": float(alpha_ar1), "ours": float(alpha_ours)},
                        "val": {
                            "coverage_raw": {"hf_only": cov_val_raw_hf, "ar1": cov_val_raw_ar1, "ours": cov_val_raw_ours},
                            "coverage_cal": {"hf_only": cov_val_cal_hf, "ar1": cov_val_cal_ar1, "ours": cov_val_cal_ours},
                            "width_raw": {"hf_only": wid_val_raw_hf, "ar1": wid_val_raw_ar1, "ours": wid_val_raw_ours},
                            "width_cal": {"hf_only": wid_val_cal_hf, "ar1": wid_val_cal_ar1, "ours": wid_val_cal_ours},
                        },
                        "test": {
                            "coverage_raw": {"hf_only": cov_test_raw_hf, "ar1": cov_test_raw_ar1, "ours": cov_test_raw_ours},
                            "coverage_cal": {"hf_only": cov_test_cal_hf, "ar1": cov_test_cal_ar1, "ours": cov_test_cal_ours},
                            "width_raw": {"hf_only": wid_test_raw_hf, "ar1": wid_test_raw_ar1, "ours": wid_test_raw_ours},
                            "width_cal": {"hf_only": wid_test_cal_hf, "ar1": wid_test_cal_ar1, "ours": wid_test_cal_ours},
                        },
                    },
                },
            }

            if dim_reduce == "fpca":
                report["fpca"] = {
                    "fpca_dim": int(args.fpca_dim),
                    "fpca_var_ratio": float(args.fpca_var_ratio),
                    "fpca_max_dim": int(args.fpca_max_dim),
                    "fpca_ridge": float(args.fpca_ridge),
                    "fpca_dim_effective": int(fpca_dim_effective) if fpca_dim_effective is not None else None,
                    "fpca_evr_sum": float(fpca_evr_sum) if fpca_evr_sum is not None else None,
                    "fpca_recon_rmse_hftr": float(fpca_recon_rmse_hftr) if fpca_recon_rmse_hftr is not None else None,
                    "fpca_recon_rmse_hfval": float(fpca_recon_rmse_hfval) if fpca_recon_rmse_hfval is not None else None,
                }            # -------------------------------------------------------------
            # [UQ][CACHEONLY] Save UQ cache arrays for fast post-hoc calibration/debug.
            if int(getattr(args, "save_uq_cache", 0)) == 1:
                try:
                    pack = {
                        "x_val": np.asarray(x_hf_va, dtype=np.float32),
                        "y_val": np.asarray(y_hf_va, dtype=np.float32),
                        "x_test": np.asarray(x_hf_te, dtype=np.float32),
                        "y_test": np.asarray(y_hf_te, dtype=np.float32),

                        "y_pred_val__hf_only": np.asarray(ypred_hf_va, dtype=np.float32),
                        "std_raw_val__hf_only": np.asarray(std_hf_va_raw, dtype=np.float32),
                        "y_pred_test__hf_only": np.asarray(ypred_hf_te, dtype=np.float32),
                        "std_raw_test__hf_only": np.asarray(std_hf_te_raw, dtype=np.float32),

                        "y_pred_val__ar1": np.asarray(ypred_ar1_va, dtype=np.float32),
                        "std_raw_val__ar1": np.asarray(std_ar1_va_raw, dtype=np.float32),
                        "y_pred_test__ar1": np.asarray(ypred_ar1_te, dtype=np.float32),
                        "std_raw_test__ar1": np.asarray(std_ar1_te_raw, dtype=np.float32),

                        "y_pred_val__ours": np.asarray(ypred_st_va, dtype=np.float32),
                        "std_raw_val__ours": np.asarray(std_st_va_raw, dtype=np.float32),
                        "y_pred_test__ours": np.asarray(ypred_st_te, dtype=np.float32),
                        "std_raw_test__ours": np.asarray(std_st_te_raw, dtype=np.float32),
                    }
                    save_uq_cache(out_dir, str(getattr(args, "uq_cache_name", "uq_cache_v1")), pack)
                except Exception as _e:
                    print(f"[UQ][CACHEONLY][WARN] Failed to save UQ cache: {_e}")
            # -------------------------------------------------------------



            with open(out_dir / "report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            # results.csv row
            results_row = {
                "run_id": run_id,
                "run_name": run_name,
                "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                "data_dir": str(args.data_dir),

                # config
                "seed": int(args.seed),
                "dim_reduce": dim_reduce,
                "gp_ard": int(args.gp_ard),
                "svgp_M": int(args.svgp_M),
                "svgp_steps": int(args.svgp_steps),
                "ci_level": float(ci_lvl),
                "ci_calibrate": int(args.ci_calibrate),
                        "ci_cal_mode": str(getattr(args, "ci_cal_mode", "conformal_norm_pooled")),

                # point metrics (test)
                "y_rmse_hf_only": y_rmse_hf,
                "y_rmse_ar1_cokrig": y_rmse_ar1,
                "y_rmse_delta_svgp": y_rmse_ours,

                "r2_test_hf_only": y_r2_test_hf,
                "r2_test_ar1_cokrig": y_r2_test_ar1,
                "r2_test_delta_svgp": y_r2_test_ours,

                # probabilistic metrics (test, calibrated)
                "nll_test_cal_hf_only": nll_test_cal_hf,
                "nll_test_cal_ar1_cokrig": nll_test_cal_ar1,
                "nll_test_cal_delta_svgp": nll_test_cal_ours,

                "nlpd_test_cal_hf_only": nlpd_test_cal_hf,
                "nlpd_test_cal_ar1_cokrig": nlpd_test_cal_ar1,
                "nlpd_test_cal_delta_svgp": nlpd_test_cal_ours,

                # calibration
                "ci_alpha_hf_only": float(alpha_hf),
                "ci_alpha_ar1_cokrig": float(alpha_ar1),
                "ci_alpha_delta_svgp": float(alpha_ours),

                # UQ summaries (test, calibrated)
                "ci_cov_test_cal_hf_only": cov_test_cal_hf,
                "ci_cov_test_cal_ar1_cokrig": cov_test_cal_ar1,
                "ci_cov_test_cal_delta_svgp": cov_test_cal_ours,

                "ci_wid_test_cal_hf_only": wid_test_cal_hf,
                "ci_wid_test_cal_ar1_cokrig": wid_test_cal_ar1,
                "ci_wid_test_cal_delta_svgp": wid_test_cal_ours,
            }
            # Write a single results row (robust schema union). Do NOT double-write via CsvLogger.

            try:
                safe_append_results_csv(out_dir / "results.csv", results_row)
            except Exception as _e:
                print(f"[WARN] safe_append_results_csv failed: {_e}")


            print("\n[DONE] metrics:")
            print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))

    finally:
        csv_logger.close()
    # Rewrite results.csv in-place with unambiguous method suffixes
    try:
        rewrite_results_csv_inplace(out_dir / "results.csv")
    except Exception as _e:
        print(f"[WARN] results.csv rename skipped: {_e}")

def main() -> None:
    args = parse_args()
    # -------------------------------------------------------------
    # [UQ][CACHEONLY] Calibration is DISABLED in this script.
    # Tune calibration parameters with uq_residual_conformal_from_cache_kfoldsafe.py / plot_uq_from_cache.py (no retraining).
    if int(getattr(args, "ci_calibrate", 0)) != 0:
        print("[UQ][CACHEONLY] Forcing --ci_calibrate=0 (baseline will NOT calibrate).")
        args.ci_calibrate = 0
    # -------------------------------------------------------------

    for s in _parse_seed_list(args):
        run_once(args, int(s))


if __name__ == "__main__":
    main()