#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structural complexity figures (split into two separate images):

1) lme_vs_hf.png
   - LME vs HF budget

2) effective_rank_vs_hf.png
   - Effective rank of A vs HF budget
   - Includes rank-1 reference line
   - Annotates the value at HF budget = 200 for the Transmission curve
     (or falls back to the Transmission maximum if HF=200 is unavailable)

This version is intended for cases where the two panels will be combined manually later,
so it does NOT add "(a)" / "(b)" labels inside the figures.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ===== unified npj-style figure settings =====
COLOR_ABS = "#0072B2"    # blue
COLOR_TRANS = "#E69F00"  # orange
COLOR_REF = "#56B4E9"    # light blue


def apply_npj_style():
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


# -----------------------------
# Publication label conventions
# -----------------------------
DATASET_LEGEND = {
    "absb": "Absorption",
    "tmst": "Transmission",
}


def ds_legend(ds_key: str) -> str:
    return DATASET_LEGEND.get(ds_key, ds_key)


# -----------------------------
# Utilities
# -----------------------------
def _read_npy(p: Path) -> np.ndarray:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return np.load(p)


def _safe_mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x, ddof=1) if x.size > 1 else 0.0)


def _pearson_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def corr_stats(Y_h: np.ndarray, Y_l: np.ndarray) -> Tuple[float, float]:
    """
    Per-dimension Pearson corr across samples.
    Return (corr_mean, corr_min) ignoring NaNs.
    """
    if Y_h.shape != Y_l.shape:
        raise ValueError(f"corr_stats shape mismatch: HF {Y_h.shape} vs LF {Y_l.shape}")
    _, K = Y_h.shape
    corrs = np.empty((K,), dtype=float)
    for k in range(K):
        corrs[k] = _pearson_corr_1d(Y_h[:, k], Y_l[:, k])
    corrs = corrs[np.isfinite(corrs)]
    if corrs.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(corrs)), float(np.min(corrs))


def pca_fit_transform(Y: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA via SVD on centered data:
      Y ~ mean + Z @ V
      Z = (Y-mean) @ V.T
    Return mean (K,), V (r,K), Z (N,r)
    """
    Y = np.asarray(Y, dtype=float)
    mean = np.mean(Y, axis=0, keepdims=True)
    X = Y - mean
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    r_eff = min(r, Vt.shape[0])
    V = Vt[:r_eff, :]
    Z = X @ V.T
    return mean.ravel(), V, Z


def ridge_linear_map(Z_l: np.ndarray, Z_h: np.ndarray, ridge: float) -> np.ndarray:
    """
    Solve A in: Z_h ≈ Z_l @ A^T  (A: r_h x r_l)
    Ridge: min ||Z_h - Z_l A^T||^2 + ridge ||A||^2
    Closed-form: A^T = (Z_l^T Z_l + ridge I)^(-1) Z_l^T Z_h
    """
    Z_l = np.asarray(Z_l, dtype=float)
    Z_h = np.asarray(Z_h, dtype=float)
    if Z_l.shape[0] != Z_h.shape[0]:
        raise ValueError(f"ridge_linear_map N mismatch: {Z_l.shape} vs {Z_h.shape}")
    r_l = Z_l.shape[1]
    XtX = Z_l.T @ Z_l
    XtY = Z_l.T @ Z_h
    reg = ridge * np.eye(r_l, dtype=float)
    A_T = np.linalg.solve(XtX + reg, XtY)
    A = A_T.T
    return A


def effective_rank_from_singular_values(s: np.ndarray) -> float:
    """
    Effective rank (participation ratio): (sum s)^2 / sum s^2
    """
    s = np.asarray(s, dtype=float)
    s = s[s > 0]
    if s.size == 0:
        return float("nan")
    return float((np.sum(s) ** 2) / np.sum(s ** 2))


def lme_from_latents(Z_l: np.ndarray, Z_h: np.ndarray, A: np.ndarray) -> float:
    """
    Linear mapping error:
      LME = ||Z_h - Z_l A^T||_F^2 / ||Z_h||_F^2
    """
    pred = Z_l @ A.T
    num = np.sum((Z_h - pred) ** 2)
    den = np.sum(Z_h ** 2)
    if den <= 0:
        return float("nan")
    return float(num / den)


# -----------------------------
# Dataset discovery / loading
# -----------------------------
@dataclass
class MFDir:
    root: Path
    hf: int
    lfx: int
    path: Path


def parse_mf_dirs(root: Path) -> List[MFDir]:
    """
    Find subdirs like hf300_lfx10 under root.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    out: List[MFDir] = []
    pat = re.compile(r"^hf(\d+)_lfx(\d+)$")
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        hf = int(m.group(1))
        lfx = int(m.group(2))
        out.append(MFDir(root=root, hf=hf, lfx=lfx, path=p))
    if not out:
        raise RuntimeError(f"No hf*_lfx* dirs found under {root}")
    return out


def load_concat_splits(mfdir: MFDir, splits: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and concatenate y_{split}.npy from:
      mfdir/hf/y_{split}.npy
      mfdir/lf_paired/y_{split}.npy
    Return (Y_h, Y_l) with same shape.
    """
    Ys_h: List[np.ndarray] = []
    Ys_l: List[np.ndarray] = []
    for sp in splits:
        ph = mfdir.path / "hf" / f"y_{sp}.npy"
        pl = mfdir.path / "lf_paired" / f"y_{sp}.npy"
        Yh = _read_npy(ph)
        Yl = _read_npy(pl)
        if Yh.shape != Yl.shape:
            raise ValueError(f"Shape mismatch in {mfdir.path.name} split={sp}: {Yh.shape} vs {Yl.shape}")
        Ys_h.append(Yh)
        Ys_l.append(Yl)
    Y_h = np.concatenate(Ys_h, axis=0) if len(Ys_h) > 1 else Ys_h[0]
    Y_l = np.concatenate(Ys_l, axis=0) if len(Ys_l) > 1 else Ys_l[0]
    return Y_h, Y_l


# -----------------------------
# Metric computation
# -----------------------------
@dataclass
class Metrics:
    hf: int
    lfx: int
    N: int
    K: int
    lme: float
    eff_rank: float
    corr_mean: float
    corr_min: float


def compute_metrics_for_dir(
    mfdir: MFDir,
    *,
    splits: Sequence[str],
    r_latent: int,
    ridge: float,
) -> Metrics:
    Y_h, Y_l = load_concat_splits(mfdir, splits)
    N, K = Y_h.shape

    _, _, Z_l = pca_fit_transform(Y_l, r_latent)
    _, _, Z_h = pca_fit_transform(Y_h, r_latent)

    A = ridge_linear_map(Z_l, Z_h, ridge=ridge)
    s = np.linalg.svd(A, compute_uv=False)
    eff_rank = effective_rank_from_singular_values(s)
    lme = lme_from_latents(Z_l, Z_h, A)

    cm, cmin = corr_stats(Y_h, Y_l)

    return Metrics(
        hf=mfdir.hf,
        lfx=mfdir.lfx,
        N=N,
        K=K,
        lme=lme,
        eff_rank=eff_rank,
        corr_mean=cm,
        corr_min=cmin,
    )


def aggregate_over_lfx(metrics: List[Metrics]) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """
    Aggregate by HF over different lfx.
    """
    by_hf: Dict[int, List[Metrics]] = {}
    for m in metrics:
        by_hf.setdefault(m.hf, []).append(m)

    out: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for hf, ms in sorted(by_hf.items()):
        lme = np.array([x.lme for x in ms], float)
        rk = np.array([x.eff_rank for x in ms], float)
        cmi = np.array([x.corr_min for x in ms], float)
        cme = np.array([x.corr_mean for x in ms], float)
        out[hf] = {
            "lme": _safe_mean_std(lme),
            "rank": _safe_mean_std(rk),
            "corr_min": _safe_mean_std(cmi),
            "corr_mean": _safe_mean_std(cme),
        }
    return out


# -----------------------------
# Plot helpers
# -----------------------------
def _annotate_rank_point(ax, x: float, y: float, color: str) -> None:
    ax.scatter([x], [y], s=28, color=color, zorder=5)
    ax.annotate(
        f"{y:.2f}",
        xy=(x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        color=color,
    )


def plot_lme_figure(
    hfs: List[int],
    absb_lme_m: np.ndarray,
    absb_lme_s: np.ndarray,
    tmst_lme_m: np.ndarray,
    tmst_lme_s: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    apply_npj_style()
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.2))

    ax.errorbar(
        hfs, absb_lme_m, yerr=absb_lme_s,
        marker="o", capsize=3, color=COLOR_ABS,
        label=ds_legend("absb")
    )
    ax.errorbar(
        hfs, tmst_lme_m, yerr=tmst_lme_s,
        marker="o", capsize=3, color=COLOR_TRANS,
        label=ds_legend("tmst")
    )

    ax.set_xlabel(r"HF budget $N_h$")
    ax.set_ylabel("LME")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_effective_rank_figure(
    hfs: List[int],
    absb_rk_m: np.ndarray,
    absb_rk_s: np.ndarray,
    tmst_rk_m: np.ndarray,
    tmst_rk_s: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    apply_npj_style()
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.2))

    ax.errorbar(
        hfs, absb_rk_m, yerr=absb_rk_s,
        marker="o", capsize=3, color=COLOR_ABS,
        label=ds_legend("absb")
    )
    ax.errorbar(
        hfs, tmst_rk_m, yerr=tmst_rk_s,
        marker="o", capsize=3, color=COLOR_TRANS,
        label=ds_legend("tmst")
    )
    ax.axhline(
        1.0, linestyle="--", linewidth=1.2,
        alpha=0.9, color=COLOR_REF,
        label="Rank-1 reference"
    )

    # annotate the Transmission point at HF=200 if present;
    # otherwise fall back to the Transmission maximum
    hfs_arr = np.asarray(hfs, dtype=int)
    if np.any(hfs_arr == 200):
        idx = int(np.where(hfs_arr == 200)[0][0])
    else:
        idx = int(np.nanargmax(tmst_rk_m))
    _annotate_rank_point(ax, float(hfs_arr[idx]), float(tmst_rk_m[idx]), COLOR_TRANS)

    ax.set_xlabel(r"HF budget $N_h$")
    ax.set_ylabel(r"Effective rank of $A$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--absb_root", type=str,
        default="../../data/mf_sweep_datasets_nano_ab",
        help="Root directory containing absorption hf*_lfx* folders."
    )
    ap.add_argument(
        "--tmst_root", type=str,
        default="../../data/mf_sweep_datasets_nano_tm",
        help="Root directory containing transmission hf*_lfx* folders."
    )

    ap.add_argument(
        "--splits", type=str, default="train",
        help="Comma-separated splits to concatenate, e.g. 'train' or 'train,val,dev'."
    )

    ap.add_argument(
        "--r_latent", type=int, default=32,
        help="Latent dimension for PCA (used for A, LME, eff_rank)."
    )
    ap.add_argument(
        "--ridge", type=float, default=1e-6,
        help="Ridge regularization for estimating A."
    )

    ap.add_argument(
        "--out_dir", type=str,
        default="../../result_out/structural_complexity_split",
        help="Output directory for the two separate images."
    )
    ap.add_argument(
        "--out_lme", type=str, default="lme_vs_hf.png",
        help="Filename for the LME figure."
    )
    ap.add_argument(
        "--out_rank", type=str, default="effective_rank_vs_hf.png",
        help="Filename for the effective-rank figure."
    )

    args = ap.parse_args()

    absb_root = Path(args.absb_root)
    tmst_root = Path(args.tmst_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    absb_dirs = parse_mf_dirs(absb_root)
    tmst_dirs = parse_mf_dirs(tmst_root)

    absb_metrics: List[Metrics] = []
    for d in absb_dirs:
        absb_metrics.append(
            compute_metrics_for_dir(
                d,
                splits=splits,
                r_latent=args.r_latent,
                ridge=args.ridge,
            )
        )

    tmst_metrics: List[Metrics] = []
    for d in tmst_dirs:
        tmst_metrics.append(
            compute_metrics_for_dir(
                d,
                splits=splits,
                r_latent=args.r_latent,
                ridge=args.ridge,
            )
        )

    absb_ag = aggregate_over_lfx(absb_metrics)
    tmst_ag = aggregate_over_lfx(tmst_metrics)

    hfs = sorted(set(absb_ag.keys()) & set(tmst_ag.keys()))
    if not hfs:
        raise RuntimeError("No overlapping HF budgets between absb and tmst roots.")

    def series(ag, key):
        mean = np.array([ag[h][key][0] for h in hfs], float)
        std = np.array([ag[h][key][1] for h in hfs], float)
        return mean, std

    absb_lme_m, absb_lme_s = series(absb_ag, "lme")
    tmst_lme_m, tmst_lme_s = series(tmst_ag, "lme")
    absb_rk_m, absb_rk_s = series(absb_ag, "rank")
    tmst_rk_m, tmst_rk_s = series(tmst_ag, "rank")

    out_lme = out_dir / args.out_lme
    out_rank = out_dir / args.out_rank

    plot_lme_figure(
        hfs=hfs,
        absb_lme_m=absb_lme_m,
        absb_lme_s=absb_lme_s,
        tmst_lme_m=tmst_lme_m,
        tmst_lme_s=tmst_lme_s,
        out_path=out_lme,
    )

    plot_effective_rank_figure(
        hfs=hfs,
        absb_rk_m=absb_rk_m,
        absb_rk_s=absb_rk_s,
        tmst_rk_m=tmst_rk_m,
        tmst_rk_s=tmst_rk_s,
        out_path=out_rank,
    )

    print(f"[SAVE] {out_lme.resolve()}")
    print(f"[SAVE] {out_rank.resolve()}")


if __name__ == "__main__":
    main()
