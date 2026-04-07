#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.2 Structural complexity (publication-ready, 1x3 panels):
(a) LME vs HF (mean±std over LF multipliers)
(b) effective rank(A) vs HF (mean±std over LF multipliers) + y=1 rank-1 reference
(c) corr_min distribution (Abs vs Trans), same y-axis scale + range annotations

Directory layout (per hfXX_lfxYY folder):
  <root>/hfXXX_lfx10/
    hf/
      y_train.npy, y_val.npy, y_dev.npy, ...
    lf_paired/
      y_train.npy, y_val.npy, y_dev.npy, ...
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Publication label conventions
# -----------------------------

DATASET_LABELS = {
    "absb": "Absorption",
    "tmst": "Transmission",
}
DATASET_LEGEND = {
    "absb": "Abs.",
    "tmst": "Trans.",
}

def ds_label(ds_key: str) -> str:
    return DATASET_LABELS.get(ds_key, ds_key)

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
    V = Vt[:r_eff, :]  # (r_eff, K)
    Z = X @ V.T        # (N, r_eff)
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
    A_T = np.linalg.solve(XtX + reg, XtY)  # (r_l, r_h)
    A = A_T.T                               # (r_h, r_l)
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

    # PCA on LF and HF separately (cap to r_latent)
    _, _, Z_l = pca_fit_transform(Y_l, r_latent)
    _, _, Z_h = pca_fit_transform(Y_h, r_latent)

    # Map Z_l -> Z_h
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
    Aggregate by HF over different lfx: return per HF:
      lme_mean,std ; rank_mean,std ; corrmin_mean,std ; corrmean_mean,std
    """
    by_hf: Dict[int, List[Metrics]] = {}
    for m in metrics:
        by_hf.setdefault(m.hf, []).append(m)

    out: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for hf, ms in sorted(by_hf.items()):
        lme = np.array([x.lme for x in ms], float)
        rk  = np.array([x.eff_rank for x in ms], float)
        cmi = np.array([x.corr_min for x in ms], float)
        cme = np.array([x.corr_mean for x in ms], float)
        out[hf] = {
            "lme": _safe_mean_std(lme),
            "rank": _safe_mean_std(rk),
            "corr_min": _safe_mean_std(cmi),
            "corr_mean": _safe_mean_std(cme),
        }
    return out

def annotate_errorbar_points(ax, xs, ys, yerrs=None, *, fmt="mean±std", dy=6, fontsize=8):
    """
    在 errorbar 的每个点上标注数值。
    fmt:
      - "mean"      -> 0.12
      - "mean±std"  -> 0.12±0.03
    dy: 像素级向上偏移，避免盖住 marker
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if yerrs is None:
        yerrs = np.zeros_like(ys)
    else:
        yerrs = np.asarray(yerrs)

    for x, y, e in zip(xs, ys, yerrs):
        if not np.isfinite(y):
            continue
        if fmt == "mean":
            s = f"{y:.3f}"
        else:
            s = f"{y:.3f}±{e:.3f}"
        ax.annotate(
            s,
            xy=(x, y),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )

# -----------------------------
# Plotting (Fig.2, 1x3)
# -----------------------------
def plot_fig2_3panel_pub(
    absb_metrics: List[Metrics],
    tmst_metrics: List[Metrics],
    *,
    out_png: Path,
    title: str = "Structural complexity of cross-fidelity mappings",
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    absb_ag = aggregate_over_lfx(absb_metrics)
    tmst_ag = aggregate_over_lfx(tmst_metrics)

    hfs = sorted(set(absb_ag.keys()) & set(tmst_ag.keys()))
    if not hfs:
        raise RuntimeError("No overlapping HF budgets between absb and tmst roots.")

    def series(ag, key):
        mean = np.array([ag[h][key][0] for h in hfs], float)
        std  = np.array([ag[h][key][1] for h in hfs], float)
        return mean, std

    absb_lme_m, absb_lme_s = series(absb_ag, "lme")
    tmst_lme_m, tmst_lme_s = series(tmst_ag, "lme")

    absb_rk_m, absb_rk_s = series(absb_ag, "rank")
    tmst_rk_m, tmst_rk_s = series(tmst_ag, "rank")

    # --------- layout (1x2)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.8, 4.4))
    fig.suptitle(title, fontsize=14, y=1.02)

    # 只标注少数关键 HF，避免太挤
    key_hf = {50, 100, 300, 500}
    hfs_np = np.asarray(hfs)
    idx = [i for i, h in enumerate(hfs_np) if int(h) in key_hf]
    hfs_k = hfs_np[idx]

    # (a) LME vs HF
    ax_a.errorbar(
        hfs, absb_lme_m, yerr=absb_lme_s,
        marker="o", capsize=3, label=ds_legend("absb")
    )
    ax_a.errorbar(
        hfs, tmst_lme_m, yerr=tmst_lme_s,
        marker="o", capsize=3, label=ds_legend("tmst")
    )
    annotate_errorbar_points(
        ax_a, hfs_k, np.asarray(absb_lme_m)[idx], np.asarray(absb_lme_s)[idx],
        fmt="mean", dy=6, fontsize=9
    )
    annotate_errorbar_points(
        ax_a, hfs_k, np.asarray(tmst_lme_m)[idx], np.asarray(tmst_lme_s)[idx],
        fmt="mean", dy=18, fontsize=9
    )
    ax_a.set_title("(a) Linear mapping error (LME)")
    ax_a.set_xlabel("HF budget")
    ax_a.set_ylabel("LME (↓ better linear explainability)")
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(frameon=True, fontsize=9, title="Dataset", title_fontsize=9, loc="best")

    # (b) Effective rank vs HF
    ax_b.errorbar(
        hfs, absb_rk_m, yerr=absb_rk_s,
        marker="o", capsize=3, label=ds_legend("absb")
    )
    ax_b.errorbar(
        hfs, tmst_rk_m, yerr=tmst_rk_s,
        marker="o", capsize=3, label=ds_legend("tmst")
    )
    annotate_errorbar_points(
        ax_b, hfs_k, np.asarray(absb_rk_m)[idx], np.asarray(absb_rk_s)[idx],
        fmt="mean", dy=6, fontsize=9
    )
    annotate_errorbar_points(
        ax_b, hfs_k, np.asarray(tmst_rk_m)[idx], np.asarray(tmst_rk_s)[idx],
        fmt="mean", dy=18, fontsize=9
    )
    ax_b.axhline(1.0, linestyle="--", linewidth=1.2, alpha=0.7, label="rank-1 ref.")
    ax_b.set_title("(b) Effective rank of transfer operator A")
    ax_b.set_xlabel("HF budget")
    ax_b.set_ylabel("eff_rank(A) (↑ stronger mode mixing)")
    ax_b.grid(True, alpha=0.25)
    ax_b.legend(frameon=True, fontsize=9, title="Dataset", title_fontsize=9, loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

# def plot_fig2_3panel_pub(
#     absb_metrics: List[Metrics],
#     tmst_metrics: List[Metrics],
#     *,
#     out_png: Path,
#     title: str = "Structural complexity of cross-fidelity mappings",
# ) -> None:
#     out_png.parent.mkdir(parents=True, exist_ok=True)
#
#     absb_ag = aggregate_over_lfx(absb_metrics)
#     tmst_ag = aggregate_over_lfx(tmst_metrics)
#
#     hfs = sorted(set(absb_ag.keys()) & set(tmst_ag.keys()))
#     if not hfs:
#         raise RuntimeError("No overlapping HF budgets between absb and tmst roots.")
#
#     def series(ag, key):
#         mean = np.array([ag[h][key][0] for h in hfs], float)
#         std  = np.array([ag[h][key][1] for h in hfs], float)
#         return mean, std
#
#     absb_lme_m, absb_lme_s = series(absb_ag, "lme")
#     tmst_lme_m, tmst_lme_s = series(tmst_ag, "lme")
#
#     absb_rk_m, absb_rk_s = series(absb_ag, "rank")
#     tmst_rk_m, tmst_rk_s = series(tmst_ag, "rank")
#
#     # corr_min distributions across all HF×lfx
#     absb_corrmin_all = np.array([m.corr_min for m in absb_metrics], float)
#     tmst_corrmin_all = np.array([m.corr_min for m in tmst_metrics], float)
#     absb_corrmin_all = absb_corrmin_all[np.isfinite(absb_corrmin_all)]
#     tmst_corrmin_all = tmst_corrmin_all[np.isfinite(tmst_corrmin_all)]
#
#     # Unified y-limits for corr_min violin
#     y_min = float(np.nanmin(np.concatenate([absb_corrmin_all, tmst_corrmin_all])))
#     y_max = float(np.nanmax(np.concatenate([absb_corrmin_all, tmst_corrmin_all])))
#     pad = 0.05 * (y_max - y_min + 1e-12)
#     ylim_c = (max(-1.0, y_min - pad), min(1.0, y_max + pad))
#
#     # --------- layout (1x3)
#     fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15.8, 4.6))
#     fig.suptitle(title, fontsize=14, y=1.02)
#
#     # --- only annotate key HF budgets to avoid clutter ---
#     key_hf = {50, 100, 300, 500}
#     hfs = np.asarray(hfs)
#
#     idx = [i for i, h in enumerate(hfs) if int(h) in key_hf]  # int() 防止 50.0 vs 50 的坑
#     hfs_k = hfs[idx]
#
#     # (a) LME vs HF
#     ax_a.errorbar(hfs, absb_lme_m, yerr=absb_lme_s, marker="o", capsize=3, label=ds_legend("absb"))
#     ax_a.errorbar(hfs, tmst_lme_m, yerr=tmst_lme_s, marker="o", capsize=3, label=ds_legend("tmst"))
#     # annotate_errorbar_points(ax_a, hfs, absb_lme_m, absb_lme_s, fmt="mean±std", dy=6, fontsize=8)
#     # annotate_errorbar_points(ax_a, hfs, tmst_lme_m, tmst_lme_s, fmt="mean±std", dy=18, fontsize=8)
#     annotate_errorbar_points(ax_a, hfs_k, np.asarray(absb_lme_m)[idx], np.asarray(absb_lme_s)[idx], fmt="mean",
#                              dy=6, fontsize=9)
#     annotate_errorbar_points(ax_a, hfs_k, np.asarray(tmst_lme_m)[idx], np.asarray(tmst_lme_s)[idx], fmt="mean",
#                              dy=18, fontsize=9)
#     ax_a.set_title("(a) Linear mapping error (LME)")
#     ax_a.set_xlabel("HF budget")
#     ax_a.set_ylabel("LME (↓ better linear explainability)")
#     ax_a.grid(True, alpha=0.25)
#     ax_a.legend(frameon=True, fontsize=9, title="Dataset", title_fontsize=9, loc="best")
#
#     # (b) Effective rank vs HF + rank-1 reference
#     ax_b.errorbar(hfs, absb_rk_m, yerr=absb_rk_s, marker="o", capsize=3, label=ds_legend("absb"))
#     ax_b.errorbar(hfs, tmst_rk_m, yerr=tmst_rk_s, marker="o", capsize=3, label=ds_legend("tmst"))
#     # annotate_errorbar_points(ax_b, hfs, absb_rk_m, absb_rk_s, fmt="mean±std", dy=6, fontsize=8)
#     # annotate_errorbar_points(ax_b, hfs, tmst_rk_m, tmst_rk_s, fmt="mean±std", dy=18, fontsize=8)
#     annotate_errorbar_points(ax_b, hfs_k, np.asarray(absb_rk_m)[idx], np.asarray(absb_rk_s)[idx], fmt="mean", dy=6,
#                              fontsize=9)
#     annotate_errorbar_points(ax_b, hfs_k, np.asarray(tmst_rk_m)[idx], np.asarray(tmst_rk_s)[idx], fmt="mean", dy=18,
#                              fontsize=9)
#     ax_b.axhline(1.0, linestyle="--", linewidth=1.2, alpha=0.7, label="rank-1 ref.")
#     ax_b.set_title("(b) Effective rank of transfer operator A")
#     ax_b.set_xlabel("HF budget")
#     ax_b.set_ylabel("eff_rank(A) (↑ stronger mode mixing)")
#     ax_b.grid(True, alpha=0.25)
#     ax_b.legend(frameon=True, fontsize=9, title="Dataset", title_fontsize=9, loc="best")
#
#     # (c) corr_min violin + same y scale + range
#     data = [absb_corrmin_all, tmst_corrmin_all]
#     ax_c.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
#     ax_c.set_xticks([1, 2])
#     ax_c.set_xticklabels([ds_legend("absb"), ds_legend("tmst")])
#     ax_c.set_title("(c) Local misalignment: corr_min distribution")
#     ax_c.set_ylabel("corr_min over wavelength (↑ better local alignment)")
#     ax_c.grid(True, axis="y", alpha=0.25)
#     ax_c.set_ylim(*ylim_c)
#
#     # annotate median + range
#     for i, arr in enumerate(data, start=1):
#         if arr.size == 0:
#             continue
#         med = float(np.median(arr))
#         amin = float(np.min(arr))
#         amax = float(np.max(arr))
#         ax_c.text(i, med, f"median={med:.2f}", ha="center", va="bottom", fontsize=9)
#         ax_c.text(
#             i,
#             ylim_c[0] + 0.02*(ylim_c[1]-ylim_c[0]),
#             f"min={amin:.2f}\nmax={amax:.2f}",
#             ha="center",
#             va="bottom",
#             fontsize=8,
#         )
#
#     fig.tight_layout()
#     fig.savefig(out_png, dpi=200, bbox_inches="tight")
#     plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--absb_root", type=str,
                    default="../../data/mf_sweep_datasets_nano_ab",
                    help="Root directory containing absorption hf*_lfx* folders.")
    ap.add_argument("--tmst_root", type=str,
                    default="../../data/mf_sweep_datasets_nano_tm",
                    help="Root directory containing transmission hf*_lfx* folders.")

    ap.add_argument("--splits", type=str, default="train",
                    help="Comma-separated splits to concatenate, e.g. 'train' or 'train,val,dev'.")

    ap.add_argument("--r_latent", type=int, default=32,
                    help="Latent dimension for PCA (used for A, LME, eff_rank).")
    ap.add_argument("--ridge", type=float, default=1e-6,
                    help="Ridge regularization for estimating A.")

    ap.add_argument("--out_png", type=str, default="../../result_out/fig_structural_complexity_2panel.png",
                    help="Output PNG path.")

    args = ap.parse_args()

    absb_root = Path(args.absb_root)
    tmst_root = Path(args.tmst_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out_png = Path(args.out_png)

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

    plot_fig2_3panel_pub(
        absb_metrics,
        tmst_metrics,
        out_png=out_png,
    )

    print(f"[SAVE] {out_png.resolve()}")


if __name__ == "__main__":
    main()