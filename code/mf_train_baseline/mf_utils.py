#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mf_utils.py
Shared utilities for mf_train_pca_band_feat.py (nanophotonic MF dataset)

Data layout (UNCHANGED, same as your "run-through" version):
  data_dir/
    wavelengths.npy
    idx_wavelength.npy
    hf/        x_{split}.npy y_{split}.npy t_{split}.npy idx_{split}.npy
    lf_paired/ x_{split}.npy y_{split}.npy t_{split}.npy idx_{split}.npy
    lf_unpaired/ ...
"""

from __future__ import annotations

import csv
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import gpytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Basic utils
# ============================================================
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean((a - b) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    Coefficient of determination R^2 for possibly multi-output targets.

    For arrays shaped (N, K) (or any shape), this computes the *global* R^2 by
    flattening all elements:
        R^2 = 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)

    If the total variance is ~0, returns 0.0 (degenerate case).
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"r2_score: shape mismatch y_true={yt.shape} vs y_pred={yp.shape}")
    y = yt.reshape(-1)
    yhat = yp.reshape(-1)
    ss_res = float(np.sum((y - yhat) ** 2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    if ss_tot <= float(eps):
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def gaussian_nll(
    y_true: np.ndarray,
    mu: np.ndarray,
    *,
    std: Optional[np.ndarray] = None,
    var: Optional[np.ndarray] = None,
    eps: float = 1e-12,
    reduce: str = "mean",
) -> float:
    """
    Gaussian negative log-likelihood (a.k.a. NLPD for Gaussian predictive density).

    Assumes element-wise independent Normal:
        y ~ N(mu, sigma^2)

    Parameters
    ----------
    y_true, mu: same shape, e.g. (N, K)
    std or var: predictive standard deviation or variance, same shape as y_true
    reduce: 'mean' (default) or 'sum' over all elements

    Returns
    -------
    float
        Aggregated NLL/NLPD value.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    mm = np.asarray(mu, dtype=np.float64)
    if yt.shape != mm.shape:
        raise ValueError(f"gaussian_nll: shape mismatch y_true={yt.shape} vs mu={mm.shape}")
    if (std is None) and (var is None):
        raise ValueError("gaussian_nll: provide std or var")
    if var is None:
        ss = np.asarray(std, dtype=np.float64)
        if ss.shape != yt.shape:
            raise ValueError(f"gaussian_nll: std shape mismatch std={ss.shape} vs y_true={yt.shape}")
        vv = np.maximum(ss ** 2, float(eps))
    else:
        vv = np.asarray(var, dtype=np.float64)
        if vv.shape != yt.shape:
            raise ValueError(f"gaussian_nll: var shape mismatch var={vv.shape} vs y_true={yt.shape}")
        vv = np.maximum(vv, float(eps))

    resid2 = (yt - mm) ** 2
    nll_elem = 0.5 * (np.log(2.0 * np.pi * vv) + (resid2 / vv))
    if reduce == "sum":
        return float(np.sum(nll_elem))
    if reduce == "mean":
        return float(np.mean(nll_elem))
    raise ValueError("gaussian_nll: reduce must be 'mean' or 'sum'")


def gaussian_nlpd(*args, **kwargs) -> float:
    """Alias for gaussian_nll (NLPD == NLL for Gaussian predictive density)."""
    return gaussian_nll(*args, **kwargs)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_pickle(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def safe_tag(s: str) -> str:
    return "".join([c if (c.isalnum() or c in "._-") else "_" for c in s])


def z_from_ci_level(ci: float) -> float:
    # standard normal quantiles for common CI levels
    if abs(ci - 0.80) < 1e-12: return 1.2815515655446004
    if abs(ci - 0.90) < 1e-12: return 1.6448536269514722
    if abs(ci - 0.95) < 1e-12: return 1.959963984540054
    if abs(ci - 0.975) < 1e-12: return 2.241402727604947
    if abs(ci - 0.99) < 1e-12: return 2.5758293035489004
    return 1.959963984540054


def must_exist(p: Path, what: str) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"Missing {what}: {p}")
    return p


# ============================================================
# Wavelengths + subsample / interpolation
# ============================================================
def load_wavelengths(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    wl = np.load(must_exist(data_dir / "wavelengths.npy", "wavelengths.npy")).astype(np.float32).reshape(-1)
    idx = np.load(must_exist(data_dir / "idx_wavelength.npy", "idx_wavelength.npy")).astype(np.int64).reshape(-1)
    if wl.size >= 2 and np.any(np.diff(wl) < 0):
        order = np.argsort(wl)
        wl = wl[order]
        idx = idx[order]
    return wl, idx


def make_even_subsample_indices(Kb: int, Ks: int) -> np.ndarray:
    if Ks < 2:
        raise ValueError("subsample_K must be >= 2")
    if Ks > Kb:
        raise ValueError(f"subsample_K cannot exceed K ({Ks} > {Kb})")
    idx = np.round(np.linspace(0, Kb - 1, Ks)).astype(np.int64)
    idx[0] = 0
    idx[-1] = Kb - 1
    idx = np.clip(idx, 0, Kb - 1)
    return idx


def pick_y_sub(y_full: np.ndarray, idx_k: np.ndarray) -> np.ndarray:
    return y_full[:, idx_k].astype(np.float32)


def upsample_y_sub_to_full(
    y_sub: np.ndarray,
    idx_k: np.ndarray,
    wl_full: np.ndarray,
) -> np.ndarray:
    y_sub = np.asarray(y_sub, dtype=np.float32)
    idx_k = idx_k.astype(np.int64)
    Kb = int(wl_full.size)
    Ks = int(idx_k.size)
    if y_sub.shape[1] != Ks:
        raise ValueError(f"y_sub dim mismatch: {y_sub.shape} vs Ks={Ks}")

    x_full = wl_full.astype(np.float32)
    x_sub = x_full[idx_k]

    y_full = np.empty((y_sub.shape[0], Kb), dtype=np.float32)
    for i in range(y_sub.shape[0]):
        y_full[i] = np.interp(x_full, x_sub, y_sub[i]).astype(np.float32)
    return y_full


def build_linear_interp_weights(x_full: np.ndarray, x_sub: np.ndarray) -> np.ndarray:
    """
    Build weights W (Kb,Ks) such that:
      y_full[i] ~= sum_j W[i,j] * y_sub[j]
    with piecewise-linear interpolation.
    """
    xf = np.asarray(x_full, dtype=np.float64).reshape(-1)
    xs = np.asarray(x_sub, dtype=np.float64).reshape(-1)
    Kb = xf.size
    Ks = xs.size
    if Ks < 2:
        raise ValueError("x_sub must have >=2 points")
    if not (np.all(np.diff(xs) >= 0) and np.all(np.diff(xf) >= 0)):
        raise ValueError("x arrays must be non-decreasing")
    if xs[0] > xf[0] + 1e-12 or xs[-1] < xf[-1] - 1e-12:
        raise ValueError("x_sub must cover full range (include endpoints)")

    W = np.zeros((Kb, Ks), dtype=np.float64)
    j = 0
    for i in range(Kb):
        x = xf[i]
        while (j < Ks - 2) and (x > xs[j + 1]):
            j += 1
        x0 = xs[j]
        x1 = xs[j + 1]
        if abs(x1 - x0) < 1e-15:
            W[i, j] = 1.0
            continue
        t = (x - x0) / (x1 - x0)
        t = min(max(t, 0.0), 1.0)
        W[i, j] = 1.0 - t
        W[i, j + 1] = t
    return W.astype(np.float32)


# ============================================================
# Uncertainty propagation + CI + calibration
# ============================================================
def propagate_pca_target_var_to_y_var(
    var_target_scaled: np.ndarray,        # (N,R) variance in scaled PCA target space
    scaler_hf_target: StandardScaler,     # fitted in PCA space
    pca: PCA,                             # fitted on HF train after scaler_y
    scaler_y: StandardScaler,             # fitted on HF train y (K)
) -> np.ndarray:
    """
    y_n = (y - mean_y)/scale_y
    z_pca = PCA(y_n)
    z_s   = (z_pca - mean_z)/scale_z

    Var(y_d) = sum_r (A_{d,r}^2 * Var(z_s_r))
    A_{d,r} = scale_y[d] * components_[r,d] * scale_z[r]
    """
    vt = np.asarray(var_target_scaled, dtype=np.float64)
    if vt.ndim != 2:
        raise ValueError(f"var_target_scaled must be (N,R), got {vt.shape}")

    s_z = np.asarray(scaler_hf_target.scale_, dtype=np.float64).reshape(-1)  # (R,)
    s_y = np.asarray(scaler_y.scale_, dtype=np.float64).reshape(-1)          # (K,)
    C = np.asarray(pca.components_, dtype=np.float64)                        # (R,K)

    R = int(C.shape[0])
    K = int(C.shape[1])
    if vt.shape[1] != R:
        raise ValueError(f"var_target_scaled dim mismatch: {vt.shape[1]} vs R={R}")

    A2 = (C.T ** 2) * (s_z.reshape(1, R) ** 2) * (s_y.reshape(K, 1) ** 2)  # (K,R)
    var_y = vt @ A2.T  # (N,K)
    return var_y.astype(np.float32)


def propagate_subsample_var_to_full_y_var(
    var_target_scaled: np.ndarray,      # (N,Ks)
    scaler_hf_target: StandardScaler,   # fitted on y_sub targets
    W_full_from_sub: np.ndarray,        # (Kb,Ks)
) -> np.ndarray:
    """
    Var(y_sub_unscaled_j) = Var(y_sub_scaled_j) * scale_j^2
    y_full_i = sum_j W[i,j] * y_sub_j
    Var(y_full_i) = sum_j W[i,j]^2 * Var(y_sub_j)
    """
    vt = np.asarray(var_target_scaled, dtype=np.float64)
    if vt.ndim != 2:
        raise ValueError(f"var_target_scaled must be (N,Ks), got {vt.shape}")

    s = np.asarray(scaler_hf_target.scale_, dtype=np.float64).reshape(-1)  # (Ks,)
    if s.size != vt.shape[1]:
        raise ValueError(f"scaler scale dim mismatch: {s.size} vs {vt.shape[1]}")

    var_unscaled = vt * (s.reshape(1, -1) ** 2)  # (N,Ks)
    W2T = (np.asarray(W_full_from_sub, dtype=np.float64) ** 2).T  # (Ks,Kb)
    var_full = var_unscaled @ W2T  # (N,Kb)
    return var_full.astype(np.float32)


def ci_coverage_y(
    y_true: np.ndarray,      # (N,K)
    y_mean: np.ndarray,      # (N,K)
    y_std: np.ndarray,       # (N,K)
    ci_level: float,
) -> float:
    z = z_from_ci_level(float(ci_level))
    yt = np.asarray(y_true, dtype=np.float32)
    ym = np.asarray(y_mean, dtype=np.float32)
    ys = np.maximum(np.asarray(y_std, dtype=np.float32), 1e-12)
    inside = (np.abs(yt - ym) <= (z * ys)).astype(np.float32)
    return float(inside.mean())


def ci_width_y(
    y_std: np.ndarray,       # (N,K)
    ci_level: float,
) -> float:
    z = z_from_ci_level(float(ci_level))
    ys = np.maximum(np.asarray(y_std, dtype=np.float32), 1e-12)
    width = 2.0 * z * ys
    return float(width.mean())


def calibrate_sigma_scale(
    y_true: np.ndarray,    # (N,K)
    y_mean: np.ndarray,    # (N,K)
    y_std_raw: np.ndarray, # (N,K)
    ci_level: float,
    lo: float = 0.05,
    hi: float = 20.0,
    iters: int = 48,
) -> float:
    """
    Find global alpha s.t. component-wise coverage on VAL matches ci_level.
    """
    z = z_from_ci_level(float(ci_level))
    yt = np.asarray(y_true, dtype=np.float32)
    ym = np.asarray(y_mean, dtype=np.float32)
    ys = np.maximum(np.asarray(y_std_raw, dtype=np.float32), 1e-12)
    err = np.abs(yt - ym)

    def cov(alpha: float) -> float:
        return float((err <= (z * float(alpha) * ys)).mean())

    a_lo, a_hi = float(lo), float(hi)
    c_lo = cov(a_lo)
    c_hi = cov(a_hi)
    if c_lo > ci_level:
        a_hi = a_lo
        a_lo = 1e-6
    if c_hi < ci_level:
        a_lo = a_hi
        a_hi = 200.0

    for _ in range(int(iters)):
        mid = 0.5 * (a_lo + a_hi)
        if cov(mid) < ci_level:
            a_lo = mid
        else:
            a_hi = mid
    return 0.5 * (a_lo + a_hi)


# ============================================================
# Plotting
# ============================================================
def make_ci_bands_for_curve(
    mean_y: np.ndarray,      # (K,)
    std_y: np.ndarray,       # (K,)
    ci_level: float,
) -> Dict[str, np.ndarray]:
    z = z_from_ci_level(float(ci_level))
    m = np.asarray(mean_y, dtype=np.float32).reshape(-1)
    s = np.maximum(np.asarray(std_y, dtype=np.float32).reshape(-1), 1e-12)
    lo = (m - z * s).astype(np.float32)
    hi = (m + z * s).astype(np.float32)
    return {"lo": lo, "hi": hi}

import matplotlib.pyplot as plt

# def plot_case_3curves_spectrum(
#     save_path: Path,
#     wl: np.ndarray,
#     y_hf_gt: np.ndarray,
#     y_lf: np.ndarray,
#     y_mf: np.ndarray,
#     title: str = "",
#     mf_band: Optional[dict] = None,   # {"lo":..., "hi":...}
# ):
#     wl = np.asarray(wl)
#     y_hf_gt = np.asarray(y_hf_gt)
#     y_lf = np.asarray(y_lf)
#     y_mf = np.asarray(y_mf)
#
#     fig = plt.figure(figsize=(6.0, 3.5))
#     ax = plt.gca()
#
#     ax.plot(wl, y_hf_gt, label="HF", linewidth=2.0)
#     ax.plot(wl, y_lf,    label="LF", linewidth=2.0)
#     ax.plot(wl, y_mf,    label="MF", linewidth=2.0)
#
#     if mf_band is not None:
#         lo = np.asarray(mf_band["lo"])
#         hi = np.asarray(mf_band["hi"])
#         # 避免全 NaN 的 band 触发奇怪的渲染
#         if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
#             ax.fill_between(wl, lo, hi, alpha=0.20, linewidth=0)
#
#     ax.set_xlabel("Wavelength (nm)")
#     ax.set_ylabel("Response")
#     if title:
#         ax.set_title(title)
#
#     ax.legend(frameon=False)
#     fig.tight_layout()
#     fig.savefig(save_path, dpi=200)
#     plt.close(fig)

# ===== unified npj-style figure settings =====
COLOR_HF = "#0072B2"        # blue
COLOR_COK = "#E69F00"       # orange
COLOR_OURS = "#009E73"      # green
COLOR_RANDOM = "#CC79A7"    # purple


def apply_npj_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
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
        "lines.linewidth": 1.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })

def plot_case_3curves_spectrum(
    *,
    save_path,
    wl=None,
    freq=None,
    y_hf_gt=None,
    y_lf=None,
    y_mf=None,
    title="",
    mf_band=None,
    x_label=None,
    y_label=None,
    curve_labels=None,
    band_label=None,
    font_family="DejaVu Sans",
    font_size=9,
    tick_size=8,
    legend_size=8,
):
    """
    Unified journal-style 3-curve spectrum plotter.

    Backward compatible with existing calls:
      - TM : plot_case_3curves_spectrum(save_path=..., wl=...,   y_hf_gt=..., y_lf=..., y_mf=..., title=..., mf_band=...)
      - MTM: plot_case_3curves_spectrum(save_path=..., freq=..., y_hf_gt=..., y_lf=..., y_mf=..., title=..., mf_band=...)

    New optional kwargs:
      - x_label
      - y_label
      - curve_labels: {"hf": "...", "lf": "...", "mf": "..."}
      - band_label
      - font_family
      - font_size
      - tick_size
      - legend_size
    """
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if wl is None and freq is None:
        raise ValueError("plot_case_3curves_spectrum requires either wl=... or freq=...")
    if wl is not None and freq is not None:
        raise ValueError("plot_case_3curves_spectrum accepts only one of wl=... or freq=...")

    x = np.asarray(wl if wl is not None else freq, dtype=float).reshape(-1)
    y_hf_gt = np.asarray(y_hf_gt, dtype=float).reshape(-1)
    y_lf = np.asarray(y_lf, dtype=float).reshape(-1)
    y_mf = np.asarray(y_mf, dtype=float).reshape(-1)

    n = x.shape[0]
    if y_hf_gt.shape[0] != n or y_lf.shape[0] != n or y_mf.shape[0] != n:
        raise ValueError(
            f"Length mismatch: len(x)={n}, len(y_hf_gt)={y_hf_gt.shape[0]}, "
            f"len(y_lf)={y_lf.shape[0]}, len(y_mf)={y_mf.shape[0]}"
        )

    # ===== unified labels =====
    if curve_labels is None:
        curve_labels = {
            "hf": "HF target",
            "lf": "LF",
            "mf": "Neural-GP MF",
        }

    if x_label is None:
        x_label = r"Wavelength $\lambda$ (nm)" if wl is not None else r"Frequency $f$ (GHz)"

    if y_label is None:
        y_label = "Response"

    # ===== unified style =====
    apply_npj_style()
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
    })

    # ===== unified colors: keep consistent with paper caption =====
    c_hf = COLOR_HF       # blue
    c_lf = COLOR_COK      # orange
    c_mf = COLOR_OURS     # green

    fig, ax = plt.subplots(figsize=(4.4, 3.2))

    # HF target
    ax.plot(
        x, y_hf_gt,
        color=c_hf,
        linewidth=1.8,
        linestyle="-",
        label=curve_labels.get("hf", "HF target"),
        zorder=3,
    )

    # LF
    ax.plot(
        x, y_lf,
        color=c_lf,
        linewidth=1.6,
        linestyle="-",
        label=curve_labels.get("lf", "LF"),
        zorder=2,
    )

    # Neural-GP MF
    ax.plot(
        x, y_mf,
        color=c_mf,
        linewidth=1.8,
        linestyle="-",
        label=curve_labels.get("mf", "Neural-GP MF"),
        zorder=4,
    )

    # Predictive interval (MF only)
    lo = hi = None
    if mf_band is not None:
        if not isinstance(mf_band, dict) or ("lo" not in mf_band) or ("hi" not in mf_band):
            raise ValueError("mf_band must be a dict with keys {'lo', 'hi'}")
        lo = np.asarray(mf_band["lo"], dtype=float).reshape(-1)
        hi = np.asarray(mf_band["hi"], dtype=float).reshape(-1)
        if lo.shape[0] != n or hi.shape[0] != n:
            raise ValueError(
                f"Band length mismatch: len(x)={n}, len(lo)={lo.shape[0]}, len(hi)={hi.shape[0]}"
            )

        ax.fill_between(
            x, lo, hi,
            color=c_mf,
            alpha=0.16,
            linewidth=0.0,
            label=band_label,
            zorder=1,
        )

    # No in-figure title for paper figures unless explicitly requested
    if str(title).strip():
        ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    # y-limits with a small margin
    y_all = [y_hf_gt, y_lf, y_mf]
    if lo is not None and hi is not None:
        y_all.extend([lo, hi])
    y_all = np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in y_all])
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size > 0:
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        y_span = max(y_max - y_min, 1e-12)
        ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.08 * y_span)

    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_case_5curves_spectrum(
    wl: np.ndarray,
    y_hf_gt: np.ndarray,     # (K,)
    y_lf: np.ndarray,        # (K,)
    y_pred_hf: np.ndarray,   # (K,)
    y_pred_or: np.ndarray,   # (K,)
    y_pred_st: np.ndarray,   # (K,)
    title: str,
    save_path: Path,
    ci_bands: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5.2))
    ax = plt.gca()
    ax.plot(wl, y_hf_gt, label="HF_GT")
    ax.plot(wl, y_lf, label="LF_oracle")
    ax.plot(wl, y_pred_hf, label="Pred_HFonly")
    ax.plot(wl, y_pred_or, label="Pred_MF_oracle")
    ax.plot(wl, y_pred_st, label="Pred_MF_student")

    if ci_bands is not None:
        if "hf" in ci_bands:
            ax.fill_between(wl, ci_bands["hf"]["lo"], ci_bands["hf"]["hi"], alpha=0.18)
        if "or" in ci_bands:
            ax.fill_between(wl, ci_bands["or"]["lo"], ci_bands["or"]["hi"], alpha=0.18)
        if "st" in ci_bands:
            ax.fill_between(wl, ci_bands["st"]["lo"], ci_bands["st"]["hi"], alpha=0.18)

    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Spectrum (metric)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=11)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=160)
    plt.close()


# ============================================================
# CSV logging
# ============================================================
TRACE_FIELDS = [
    "run_id", "run_name", "timestamp",
    "dim_reduce", "subsample_K",
    "kernel", "matern_nu", "kernel_struct", "gp_ard",
    "use_feature_student", "use_feature_oracle", "pca_dim", "pca_whiten",
    "svgp_M", "svgp_steps", "svgp_lr",
    "model_tag", "dim", "step",
    "neg_elbo_per_n", "noise",
    "var", "ell_min", "ell_mean", "ell_max",
    "kf_var", "kf_ell_min", "kf_ell_mean", "kf_ell_max",
    "ky_var", "ky_ell_min", "ky_ell_mean", "ky_ell_max",
]

RESULT_FIELDS = [
    "run_id", "run_name", "timestamp",
    "data_dir",
    "dim_reduce", "subsample_K",
    "kernel", "matern_nu", "kernel_struct", "gp_ard",
    "use_feature_student", "use_feature_oracle",
    "seed",
    "K", "Ks", "xdim",
    "n_hf_train", "n_hf_val", "n_hf_test",
    "n_lf_train_total", "n_lf_val_total",
    "pca_dim_effective", "pca_whiten", "pca_evr_sum",
    "student_best_val_mse", "student_epochs_ran",
    "student_mse_on_hfval", "student_rmse_on_hfval",
    "student_mse_on_hftest", "student_rmse_on_hftest",
    "target_rmse_scaled_hf", "target_rmse_scaled_or", "target_rmse_scaled_st",
    "y_rmse_hf", "y_rmse_or", "y_rmse_st",
    "y_r2_val_hf", "y_r2_val_or", "y_r2_val_st",
    "y_r2_test_hf", "y_r2_test_or", "y_r2_test_st",
    "nll_val_raw_hf", "nll_val_raw_or", "nll_val_raw_st",
    "nll_val_cal_hf", "nll_val_cal_or", "nll_val_cal_st",
    "nll_test_raw_hf", "nll_test_raw_or", "nll_test_raw_st",
    "nll_test_cal_hf", "nll_test_cal_or", "nll_test_cal_st",
    "nlpd_val_raw_hf", "nlpd_val_raw_or", "nlpd_val_raw_st",
    "nlpd_val_cal_hf", "nlpd_val_cal_or", "nlpd_val_cal_st",
    "nlpd_test_raw_hf", "nlpd_test_raw_or", "nlpd_test_raw_st",
    "nlpd_test_cal_hf", "nlpd_test_cal_or", "nlpd_test_cal_st",
    "ci_level", "ci_calibrate",
    "ci_alpha_hf", "ci_alpha_or", "ci_alpha_st",
    "ci_cov_val_raw_hf", "ci_cov_val_raw_or", "ci_cov_val_raw_st",
    "ci_cov_val_cal_hf", "ci_cov_val_cal_or", "ci_cov_val_cal_st",
    "ci_wid_val_raw_hf", "ci_wid_val_raw_or", "ci_wid_val_raw_st",
    "ci_wid_val_cal_hf", "ci_wid_val_cal_or", "ci_wid_val_cal_st",
    "ci_cov_test_raw_hf", "ci_cov_test_raw_or", "ci_cov_test_raw_st",
    "ci_cov_test_cal_hf", "ci_cov_test_cal_or", "ci_cov_test_cal_st",
    "ci_wid_test_raw_hf", "ci_wid_test_raw_or", "ci_wid_test_raw_st",
    "ci_wid_test_cal_hf", "ci_wid_test_cal_or", "ci_wid_test_cal_st",
]


class CsvLogger:
    def __init__(self, trace_path: Path, results_path: Path):
        self.trace_path = trace_path
        self.results_path = results_path
        self._trace_f = None
        self._trace_w = None
        self._res_f = None
        self._res_w = None

    def open(self):
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace_f = open(self.trace_path, "w", newline="", encoding="utf-8")
        self._trace_w = csv.DictWriter(self._trace_f, fieldnames=TRACE_FIELDS)
        self._trace_w.writeheader()

        self._res_f = open(self.results_path, "w", newline="", encoding="utf-8")
        self._res_w = csv.DictWriter(self._res_f, fieldnames=RESULT_FIELDS)
        self._res_w.writeheader()

    def close(self):
        if self._trace_f is not None:
            self._trace_f.close()
        if self._res_f is not None:
            self._res_f.close()

    def write_trace(self, row: Dict[str, Any]):
        assert self._trace_w is not None
        out = {k: row.get(k, "") for k in TRACE_FIELDS}
        self._trace_w.writerow(out)
        self._trace_f.flush()

    def write_result(self, row: Dict[str, Any]):
        assert self._res_w is not None
        out = {k: row.get(k, "") for k in RESULT_FIELDS}
        self._res_w.writerow(out)
        self._res_f.flush()


# ============================================================
# Stage-I: Feature MLP (x -> y_lf)
# ============================================================
class FeatureMLP(nn.Module):
    """
    Training: forward(x) -> yhat
    Inference: extract_features(x) -> f
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Tuple[int, ...] = (256, 256),
        feat_dim: int = 16,
        act: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        Act = acts.get(act, nn.ReLU)

        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        self.trunk = nn.Sequential(*layers)
        self.feat_layer = nn.Linear(prev, feat_dim)
        self.feat_act = Act()
        self.head = nn.Linear(feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        f = self.feat_act(self.feat_layer(h))
        y = self.head(f)
        return y

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        h = self.trunk(x)
        f = self.feat_act(self.feat_layer(h))
        return f


import math
from typing import Optional, Dict, Tuple, Any, List

def train_feature_mlp(
    model: FeatureMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 3e-4,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    max_epochs: int = 500,
    patience: int = 30,
    min_delta: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    print_every: int = 20,
    # --- NEW ---
    eval_sets: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    eval_batch_size: int = 4096,
    early_stop_metric: str = "mse",  # "mse" or "rmse" (rmse is monotonic to mse)
    early_stop_set: str = "mix_val", # name in eval_sets; if missing -> use X_val/y_val
) -> Tuple[FeatureMLP, Dict]:
    """
    X_train/y_train, X_val/y_val 通常传 mix_train / mix_val。
    eval_sets 可额外传入：
      {
        "mix_train": (X_mix_tr, y_mix_tr),
        "mix_val":   (X_mix_va, y_mix_va),
        "mix_test":  (X_mix_te, y_mix_te),
        "paired_train": (X_p_tr, y_p_tr),
        "paired_val":   (X_p_va, y_p_va),
        "paired_test":  (X_p_te, y_p_te),
        "hf_paired_train": (X_hf_tr, y_lfp_tr),
        ...
      }
    """

    def _eval_mse_rmse(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        model.eval()
        se_sum = 0.0
        n_elem = 0
        with torch.no_grad():
            Xt = torch.from_numpy(X.astype(np.float32))
            yt = torch.from_numpy(y.astype(np.float32))
            n = Xt.shape[0]
            for s in range(0, n, int(eval_batch_size)):
                xb = Xt[s:s+int(eval_batch_size)].to(device)
                yb = yt[s:s+int(eval_batch_size)].to(device)
                pred = model(xb)
                diff = pred - yb
                se_sum += float((diff * diff).sum().item())
                n_elem += int(diff.numel())
        mse = se_sum / max(n_elem, 1)
        rmse = math.sqrt(mse)
        return mse, rmse

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    Xtr = torch.from_numpy(X_train.astype(np.float32))
    ytr = torch.from_numpy(y_train.astype(np.float32))
    Xva = torch.from_numpy(X_val.astype(np.float32)).to(device)
    yva = torch.from_numpy(y_val.astype(np.float32)).to(device)

    n = Xtr.shape[0]
    best_val = float("inf")
    best_state = None
    bad = 0

    hist: Dict[str, Any] = {
        "mix_train_mse": [], "mix_train_rmse": [],
        "mix_val_mse": [],   "mix_val_rmse": [],
        "extra": {}  # name -> {"mse":[], "rmse":[]}
    }

    # default eval_sets：至少包含 mix_train/mix_val，便于统一早停/打印
    if eval_sets is None:
        eval_sets = {}
    if "mix_train" not in eval_sets:
        eval_sets["mix_train"] = (X_train, y_train)
    if "mix_val" not in eval_sets:
        eval_sets["mix_val"] = (X_val, y_val)

    early_stop_metric = str(early_stop_metric).lower().strip()
    if early_stop_metric not in ("mse", "rmse"):
        raise ValueError("early_stop_metric must be 'mse' or 'rmse'")

    for ep in range(1, max_epochs + 1):
        # -------- train one epoch (MSE loss) --------
        model.train()
        perm = torch.randperm(n)
        sum_loss = 0.0
        cnt = 0

        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb = Xtr[idx].to(device)
            yb = ytr[idx].to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            sum_loss += float(loss.item()) * int(xb.shape[0])
            cnt += int(xb.shape[0])

        # 训练期的 epoch-avg MSE（按 batch loss 聚合）
        train_mse_epoch = sum_loss / max(cnt, 1)

        # -------- validation (mix_val) --------
        model.eval()
        with torch.no_grad():
            val_pred = model(Xva)
            val_mse_epoch = float(loss_fn(val_pred, yva).item())
        val_rmse_epoch = math.sqrt(max(val_mse_epoch, 0.0))

        hist["mix_train_mse"].append(float(train_mse_epoch))
        hist["mix_train_rmse"].append(float(math.sqrt(max(train_mse_epoch, 0.0))))
        hist["mix_val_mse"].append(float(val_mse_epoch))
        hist["mix_val_rmse"].append(float(val_rmse_epoch))

        # -------- optional: eval extra sets (RMSE/MSE) --------
        do_print = (ep == 1) or ((print_every > 0) and (ep % max(1, int(print_every)) == 0))
        extra_metrics_line: List[str] = []
        if do_print and eval_sets is not None:
            for name, (Xe, ye) in eval_sets.items():
                # mix_train/mix_val 已经有了；这里只补充其余集合，或你也可以全部算一遍以更严谨
                if name in ("mix_train", "mix_val"):
                    continue
                mse_e, rmse_e = _eval_mse_rmse(Xe, ye)
                if name not in hist["extra"]:
                    hist["extra"][name] = {"mse": [], "rmse": []}
                hist["extra"][name]["mse"].append(float(mse_e))
                hist["extra"][name]["rmse"].append(float(rmse_e))
                extra_metrics_line.append(f"{name}_rmse={rmse_e:.6g}")

        if do_print:
            print(
                f"[STAGE-I][MLP] ep={ep:4d} "
                f"mix_train_mse={train_mse_epoch:.6g} mix_train_rmse={math.sqrt(max(train_mse_epoch,0.0)):.6g} | "
                f"mix_val_mse={val_mse_epoch:.6g} mix_val_rmse={val_rmse_epoch:.6g} | "
                f"best={best_val:.6g} bad={bad}"
                + ("" if not extra_metrics_line else " | " + " ".join(extra_metrics_line))
            )

        # -------- early stopping on chosen set+metric --------
        # 默认 early_stop_set="mix_val"：等价于你当前逻辑
        if early_stop_set in eval_sets:
            mse_es, rmse_es = _eval_mse_rmse(*eval_sets[early_stop_set]) if do_print else (val_mse_epoch, val_rmse_epoch)
            cur_metric = rmse_es if early_stop_metric == "rmse" else mse_es
        else:
            cur_metric = val_rmse_epoch if early_stop_metric == "rmse" else val_mse_epoch

        if (best_val - cur_metric) > float(min_delta):
            best_val = float(cur_metric)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                print(f"[STAGE-I][MLP] Early stop: ep={ep}  best_{early_stop_set}_{early_stop_metric}={best_val:.6g}")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    meta = {
        "best_val_metric": float(best_val),
        "early_stop_metric": str(early_stop_metric),
        "early_stop_set": str(early_stop_set),
        "epochs_ran": len(hist["mix_val_mse"]),
        # 兼容旧字段：仍给 best_val_mse（若 early_stop_metric=rmse，这里就不等价了，所以只在 mse 时严格一致）
        "best_val_mse": float(min(hist["mix_val_mse"])) if hist["mix_val_mse"] else float("inf"),
    }
    return model, {"history": hist, "meta": meta}



@torch.no_grad()
def mlp_predict_and_features(model: FeatureMLP, X: np.ndarray, device: torch.device, batch_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust inference without relying on forward_with_features:
      yhat = model(x)
      feat = model.extract_features(x)
    """
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    N = X.shape[0]
    ys, fs = [], []
    printed = False

    def _uniq_rows(a):
        return np.unique(np.round(a, 6), axis=0).shape[0]

    for s in range(0, N, batch_size):
        xb = torch.from_numpy(X[s:s + batch_size]).to(device)
        yb = model(xb)
        fb = model.extract_features(xb)

        yb_np = yb.detach().cpu().numpy()
        fb_np = fb.detach().cpu().numpy()

        if not printed:
            printed = True
            print("[DBG][mlp_predict_and_features] first batch shapes:",
                  "xb", tuple(xb.shape), "yb", yb_np.shape, "fb", fb_np.shape)
            print("[DBG][mlp_predict_and_features] first batch unique_rows:",
                  "yb", _uniq_rows(yb_np), "/", yb_np.shape[0],
                  "fb", _uniq_rows(fb_np), "/", fb_np.shape[0])
            print("[DBG][mlp_predict_and_features] first row sample:",
                  "yb[0,:5]", yb_np[0, :5], "fb[0,:5]", fb_np[0, :5])

        ys.append(yb_np)
        fs.append(fb_np)

    y = np.concatenate(ys, axis=0)
    f = np.concatenate(fs, axis=0)
    return y, f


# ============================================================
# Stage-II: SVGP + kernels
# ============================================================
def pca_recon_rmse(y: np.ndarray, scaler_y: StandardScaler, pca: PCA) -> float:
    """
    y: (N,K) in original y-space
    scaler_y: fitted on HF-tr y
    pca: fitted on HF-tr y (after scaler_y)
    """
    y_n = scaler_y.transform(y.astype(np.float32))
    z = pca.transform(y_n)
    y_n_hat = pca.inverse_transform(z)
    y_hat = scaler_y.inverse_transform(y_n_hat)
    return float(rmse(y_hat.astype(np.float32), y.astype(np.float32)))


def init_inducing_points(X: np.ndarray, M: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if X.shape[0] <= M:
        return X.copy()
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=M, random_state=seed, n_init="auto")
        km.fit(X)
        return km.cluster_centers_.astype(np.float32)
    except Exception:
        idx = rng.choice(X.shape[0], size=M, replace=False)
        return X[idx].astype(np.float32)


class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, covar_module: gpytorch.kernels.Kernel):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



@dataclass
class SVGPBundle:
    models: List[SVGPModel]
    likes: List[gpytorch.likelihoods.GaussianLikelihood]


def _make_base_kernel(
    kernel_name: str,
    input_dim: int,
    ard: bool,
    matern_nu: float,
    active_dims: Optional[Tuple[int, ...]] = None,
) -> gpytorch.kernels.Kernel:
    kernel_name = kernel_name.lower().strip()
    ard_num_dims = input_dim if ard else None

    if kernel_name == "rbf":
        return gpytorch.kernels.RBFKernel(
            ard_num_dims=ard_num_dims,
            active_dims=active_dims,
        )
    if kernel_name == "matern":
        return gpytorch.kernels.MaternKernel(
            nu=float(matern_nu),
            ard_num_dims=ard_num_dims,
            active_dims=active_dims,
        )
    raise ValueError(f"Unknown kernel_name={kernel_name}. Use rbf or matern.")


def make_single_kernel(
    x_dim: int,
    ard: bool,
    kernel_name: str,
    matern_nu: float,
) -> gpytorch.kernels.Kernel:
    base = _make_base_kernel(
        kernel_name=kernel_name,
        input_dim=x_dim,
        ard=ard,
        matern_nu=matern_nu,
        active_dims=None,
    )
    return gpytorch.kernels.ScaleKernel(base)


def make_additive_block_kernel(
    feat_dim: int,
    y_dim: int,
    ard_in_blocks: bool,
    kernel_name: str,
    matern_nu: float,
) -> gpytorch.kernels.Kernel:
    kf_base = _make_base_kernel(
        kernel_name=kernel_name,
        input_dim=feat_dim,
        ard=ard_in_blocks,
        matern_nu=matern_nu,
        active_dims=tuple(range(0, feat_dim)),
    )
    kf = gpytorch.kernels.ScaleKernel(kf_base)

    ky_base = _make_base_kernel(
        kernel_name=kernel_name,
        input_dim=y_dim,
        ard=ard_in_blocks,
        matern_nu=matern_nu,
        active_dims=tuple(range(feat_dim, feat_dim + y_dim)),
    )
    ky = gpytorch.kernels.ScaleKernel(ky_base)

    return gpytorch.kernels.AdditiveKernel(kf, ky)


def _scale_kernel_params(scale_kernel: gpytorch.kernels.ScaleKernel) -> Tuple[float, np.ndarray]:
    var = float(scale_kernel.outputscale.detach().cpu().item())
    base = scale_kernel.base_kernel
    ell = base.lengthscale.detach().cpu().view(-1).numpy().astype(np.float64)
    return var, ell


def _summarize_scale_kernel(scale_kernel: gpytorch.kernels.ScaleKernel) -> Tuple[float, float, float, float]:
    var, ell = _scale_kernel_params(scale_kernel)
    return var, float(ell.min()), float(ell.mean()), float(ell.max())


def train_svgp_per_dim(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    device: torch.device,
    inducing_M: int,
    steps: int,
    lr: float,
    ard: bool,
    kernel_struct: str,
    kernel_name: str,
    matern_nu: float,
    feat_dim: Optional[int],
    print_every: int,
    tag: str,
    csv_logger: CsvLogger,
    csv_run_meta: Dict[str, Any],
) -> SVGPBundle:
    Xtr = Xtr.astype(np.float32)
    Ytr = Ytr.astype(np.float32)
    Xt = torch.from_numpy(Xtr).to(device)

    R = Ytr.shape[1]
    models: List[SVGPModel] = []
    likes: List[gpytorch.likelihoods.GaussianLikelihood] = []

    M_eff = int(min(inducing_M, Xtr.shape[0]))
    Z0 = init_inducing_points(
        Xtr, M=M_eff,
        seed=int(csv_run_meta["seed"]) + (11 if tag == "HF" else 22 if tag == "OR" else 33)
    )
    Zt0 = torch.from_numpy(Z0).to(device)

    for r in range(R):
        yt = torch.from_numpy(Ytr[:, r]).to(device)

        if kernel_struct == "full":
            covar = make_single_kernel(
                x_dim=Xtr.shape[1],
                ard=ard,
                kernel_name=kernel_name,
                matern_nu=matern_nu,
            )
            is_block = False
        elif kernel_struct == "block":
            if feat_dim is None:
                raise ValueError("feat_dim must be provided for block kernel")
            y_dim = Xtr.shape[1] - feat_dim
            if y_dim <= 0:
                raise ValueError(f"Bad split: X_dim={Xtr.shape[1]} feat_dim={feat_dim}")
            covar = make_additive_block_kernel(
                feat_dim=feat_dim,
                y_dim=y_dim,
                ard_in_blocks=ard,
                kernel_name=kernel_name,
                matern_nu=matern_nu,
            )
            is_block = True
        else:
            raise ValueError(f"Unknown kernel_struct: {kernel_struct}")

        model = SVGPModel(inducing_points=Zt0.clone(), covar_module=covar).to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(
            [{"params": model.parameters()}, {"params": likelihood.parameters()}],
            lr=lr,
        )
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Xtr.shape[0])

        print(f"[SVGP][{tag}] dim={r:03d} start  M={M_eff}  steps={steps}  lr={lr:g}  kernel_struct={kernel_struct}")

        for s in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            out = model(Xt)
            loss = -mll(out, yt)
            loss.backward()
            optimizer.step()

            do_log = (print_every > 0) and (s == 1 or s == steps or (s % print_every == 0))
            if do_log:
                neg_elbo_per_n = float(loss.detach().cpu().item()) / float(Xtr.shape[0])
                noise = float(likelihood.noise.detach().cpu().item())

                row = dict(csv_run_meta)
                row.update({
                    "model_tag": tag,
                    "dim": int(r),
                    "step": int(s),
                    "neg_elbo_per_n": neg_elbo_per_n,
                    "noise": noise,
                })

                if not is_block:
                    ker = model.covar_module
                    var, ell_min, ell_mean, ell_max = _summarize_scale_kernel(ker)  # type: ignore
                    row.update({"var": var, "ell_min": ell_min, "ell_mean": ell_mean, "ell_max": ell_max})
                    print(
                        f"[SVGP][{tag}] dim={r:03d} step={s:5d}  -ELBO/N={neg_elbo_per_n:.6g}  noise={noise:.4g}  "
                        f"var={var:.4g}  ell(min/mean/max)={ell_min:.4g}/{ell_mean:.4g}/{ell_max:.4g}"
                    )
                else:
                    ker = model.covar_module
                    assert isinstance(ker, gpytorch.kernels.AdditiveKernel)
                    kf = ker.kernels[0]
                    ky = ker.kernels[1]
                    assert isinstance(kf, gpytorch.kernels.ScaleKernel)
                    assert isinstance(ky, gpytorch.kernels.ScaleKernel)
                    kf_var, kf_emin, kf_emean, kf_emax = _summarize_scale_kernel(kf)
                    ky_var, ky_emin, ky_emean, ky_emax = _summarize_scale_kernel(ky)
                    row.update({
                        "kf_var": kf_var, "kf_ell_min": kf_emin, "kf_ell_mean": kf_emean, "kf_ell_max": kf_emax,
                        "ky_var": ky_var, "ky_ell_min": ky_emin, "ky_ell_mean": ky_emean, "ky_ell_max": ky_emax,
                    })
                    print(f"[SVGP][{tag}] dim={r:03d} step={s:5d}  -ELBO/N={neg_elbo_per_n:.6g}  noise={noise:.4g}")
                    print(f"                 kf: var={kf_var:.4g}  ell(min/mean/max)={kf_emin:.4g}/{kf_emean:.4g}/{kf_emax:.4g}")
                    print(f"                 ky: var={ky_var:.4g}  ell(min/mean/max)={ky_emin:.4g}/{ky_emean:.4g}/{ky_emax:.4g}")

                csv_logger.write_trace(row)

        models.append(model)
        likes.append(likelihood)

    return SVGPBundle(models=models, likes=likes)


@torch.no_grad()
def predict_svgp_per_dim(bundle: SVGPBundle, Xte: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    Xte = Xte.astype(np.float32)
    Xt = torch.from_numpy(Xte).to(device)

    R = len(bundle.models)
    mu = np.empty((Xte.shape[0], R), dtype=np.float32)
    var = np.empty((Xte.shape[0], R), dtype=np.float32)

    for r in range(R):
        model = bundle.models[r]
        lik = bundle.likes[r]
        model.eval()
        lik.eval()
        with gpytorch.settings.fast_pred_var():
            pred = lik(model(Xt))
            mu[:, r] = pred.mean.detach().cpu().float().numpy()
            var[:, r] = pred.variance.detach().cpu().float().numpy()

    return mu, var


def save_svgp_bundle(out_dir: Path, name: str, bundle: SVGPBundle) -> None:
    (out_dir / name).mkdir(parents=True, exist_ok=True)
    for i, (m, l) in enumerate(zip(bundle.models, bundle.likes)):
        torch.save(m.state_dict(), out_dir / name / f"svgp_{i:03d}.pth")
        torch.save(l.state_dict(), out_dir / name / f"lik_{i:03d}.pth")


# ============================================================
# Data loading (RUN-THROUGH version: subdirs + fixed names)
# ============================================================
def load_split_block(root: Path, sub: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = root / sub
    x = np.load(must_exist(d / f"x_{split}.npy", f"{sub}/x_{split}.npy")).astype(np.float32)
    y = np.load(must_exist(d / f"y_{split}.npy", f"{sub}/y_{split}.npy")).astype(np.float32)
    t = np.load(must_exist(d / f"t_{split}.npy", f"{sub}/t_{split}.npy")).astype(np.float32).reshape(-1)
    idx = np.load(must_exist(d / f"idx_{split}.npy", f"{sub}/idx_{split}.npy")).astype(np.int64).reshape(-1)
    return x, y, t, idx


def assert_indices_match(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    if a.shape != b.shape or not np.all(a == b):
        raise ValueError(f"Index mismatch between {name_a} and {name_b}: shapes {a.shape} vs {b.shape}")


# ============================================================
# DBG helpers
# ============================================================
def dbg_block_stats(name: str, A: np.ndarray, round_decimals: int = 6) -> None:
    A = np.asarray(A)
    if A.ndim != 2:
        print(f"[DBG] {name}: ndim={A.ndim} shape={A.shape} (skip stats; expecting 2D)")
        return
    std = A.std(axis=0)
    std_min = float(std.min()) if std.size else float("nan")
    std_mean = float(std.mean()) if std.size else float("nan")
    std_max = float(std.max()) if std.size else float("nan")

    Ar = np.round(A, round_decimals)
    try:
        uniq = np.unique(Ar, axis=0).shape[0]
    except Exception:
        uniq = -1

    print(f"[DBG] {name}: shape={A.shape}  std(min/mean/max)={std_min:.3g}/{std_mean:.3g}/{std_max:.3g}  unique_rows@1e-{round_decimals}={uniq}/{A.shape[0]}")


def dbg_student_on_hf_errors(
    yhat_lf_va: np.ndarray, y_lf_va: np.ndarray,
    yhat_lf_te: np.ndarray, y_lf_te: np.ndarray
) -> Dict[str, float]:
    mse_va = mse(yhat_lf_va, y_lf_va)
    rmse_va = rmse(yhat_lf_va, y_lf_va)
    mse_te = mse(yhat_lf_te, y_lf_te)
    rmse_te = rmse(yhat_lf_te, y_lf_te)

    print(f"[DBG][STUDENT@HF] VAL:  mse={mse_va:.6g}  rmse={rmse_va:.6g}")
    print(f"[DBG][STUDENT@HF] TEST: mse={mse_te:.6g}  rmse={rmse_te:.6g}")

    return {
        "student_mse_on_hfval": float(mse_va),
        "student_rmse_on_hfval": float(rmse_va),
        "student_mse_on_hftest": float(mse_te),
        "student_rmse_on_hftest": float(rmse_te),
    }
