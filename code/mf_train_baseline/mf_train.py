#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mf_train_fpca_sgp_delta_student.py  (Nanophotonic, FPCA default, PCA removed)

Nanophotonic MF dataset: SVGP + FPCA/SubSampling + kernel ablations
+ Predictive uncertainty (raw + calibrated CI)
+ Full val raw/cal recording and report.json + results.csv

Key changes vs old PCA version
------------------------------
1) Remove all PCA logic entirely.
2) FPCA is used for dim reduction when --dim_reduce=fpca (default).
3) FPCA pipeline:
     y (K) --(StandardScaler on HF-train only)--> y_n --(FPCA)--> z (R)
     z (R) --(StandardScaler on HF-train only)--> z_s (R)  [SECOND LAYER, NEW]
   IMPORTANT:
     - scale BEFORE FPCA (StandardScaler on y)
     - SECOND scaler AFTER FPCA (StandardScaler on z)  <-- per your request
4) Variance propagation for FPCA includes BOTH scalers:
     var(z_unscaled) = var(z_scaled) * (scaler_z.scale_^2)
     var(y) = fpca_var_map(var(z_unscaled)) * (scaler_y.scale_^2)

Data layout MUST match the run-through version:
  data_dir/
    wavelengths.npy
    idx_wavelength.npy
    hf/        x_{split}.npy y_{split}.npy t_{split}.npy idx_{split}.npy
    lf_paired/ x_{split}.npy y_{split}.npy t_{split}.npy idx_{split}.npy
    lf_unpaired/ ...
"""

from __future__ import annotations

import argparse
import json
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

try:
    from .mf_utils import (
        set_seed, now_tag, safe_tag, save_pickle, rmse, r2_score, gaussian_nll, gaussian_nlpd, z_from_ci_level, pca_recon_rmse,
        load_wavelengths, make_even_subsample_indices, pick_y_sub, upsample_y_sub_to_full, build_linear_interp_weights,
        propagate_subsample_var_to_full_y_var,
        ci_coverage_y, ci_width_y, calibrate_sigma_scale,
        plot_case_3curves_spectrum, make_ci_bands_for_curve,
        CsvLogger,
        train_feature_mlp, mlp_predict_and_features,
        train_svgp_per_dim, predict_svgp_per_dim, save_svgp_bundle,
        load_split_block, assert_indices_match,
        dbg_block_stats, dbg_student_on_hf_errors,
    )
except ImportError:
    from mf_utils import (
        set_seed, now_tag, safe_tag, save_pickle, rmse, r2_score, gaussian_nll, gaussian_nlpd, z_from_ci_level, pca_recon_rmse,
        load_wavelengths, make_even_subsample_indices, pick_y_sub, upsample_y_sub_to_full, build_linear_interp_weights,
        propagate_subsample_var_to_full_y_var,
        ci_coverage_y, ci_width_y, calibrate_sigma_scale,
        plot_case_3curves_spectrum, make_ci_bands_for_curve,
        CsvLogger,
        train_feature_mlp, mlp_predict_and_features,
        train_svgp_per_dim, predict_svgp_per_dim, save_svgp_bundle,
        load_split_block, assert_indices_match,
        dbg_block_stats, dbg_student_on_hf_errors,
    )

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ""
DEFAULT_OUT_DIR = ""
DEFAULT_RUN_PREFIX = "mf"

@dataclass
class TrainDefaults:
    data_dir: str = DEFAULT_DATA_DIR
    out_dir: str = DEFAULT_OUT_DIR
    run_prefix: str = DEFAULT_RUN_PREFIX

# =========================
# Debug helpers (lightweight)
# =========================
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

def dbg_print_stage2_inputs(X_hf_tr_s: np.ndarray, X_hf_va_s: np.ndarray, X_hf_te_s: np.ndarray,
                           U_or_tr_s: np.ndarray, U_or_va_s: np.ndarray, U_or_te_s: np.ndarray,
                           U_st_tr_s: np.ndarray, U_st_va_s: np.ndarray, U_st_te_s: np.ndarray,
                           dim_reduce: str, fpca_dim_effective: Optional[int]) -> None:
    print(f"[DEBUG][StageII][X] HF-only  Xtr/Xva/Xte = {X_hf_tr_s.shape} / {X_hf_va_s.shape} / {X_hf_te_s.shape}")
    print(f"[DEBUG][StageII][X] MF-oracle Utr/Uva/Ute = {U_or_tr_s.shape} / {U_or_va_s.shape} / {U_or_te_s.shape}")
    print(f"[DEBUG][StageII][X] MF-stud   Utr/Uva/Ute = {U_st_tr_s.shape} / {U_st_va_s.shape} / {U_st_te_s.shape}")
    if dim_reduce == "fpca":
        print(f"[DEBUG][FPCA] fpca_dim_effective={fpca_dim_effective}")

def dbg_print_kernel_effective(svgp_bundle: Any, tag: str) -> None:
    """
    Best-effort: print the actual kernel object representation (gpytorch models show covar_module).
    Does not assume a fixed bundle structure.
    """
    try:
        obj = svgp_bundle
        if isinstance(obj, dict):
            # common patterns: {"models": [...]} or {"bundle": [...]}
            for k in ("models", "bundle", "svgp", "gps", "gp_list"):
                if k in obj:
                    obj = obj[k]
                    break
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            obj0 = obj[0]
        else:
            obj0 = obj

        # unwrap dict item
        model = obj0
        if isinstance(obj0, dict):
            for k in ("model", "m", "gp", "svgp", "net"):
                if k in obj0:
                    model = obj0[k]
                    break

        print(f"[DEBUG][KERNEL_EFFECTIVE][{tag}] bundle_type={type(svgp_bundle)} item0_type={type(obj0)} model_type={type(model)}")
        cov = getattr(model, "covar_module", None) or getattr(model, "kernel", None)
        if cov is not None:
            print(f"[DEBUG][KERNEL_EFFECTIVE][{tag}] covar_module={cov}")
        else:
            print(f"[DEBUG][KERNEL_EFFECTIVE][{tag}] covar_module=<not found>")
    except Exception as e:
        print(f"[DEBUG][KERNEL_EFFECTIVE][{tag}] failed: {type(e).__name__}: {e}")




# ============================================================
# FPCA (discrete functional PCA) - lightweight sklearn-like
# ============================================================
class FPCA:
    """
    FPCA for discretized curves y(wavelength) sampled on a fixed grid.

    - Fit eigendecomposition of covariance across grid points (K).
    - Provides sklearn-like API: fit/transform/inverse_transform and attributes:
        components_ : (R, K)
        explained_variance_ratio_ : (R,)
        mean_ : (K,)   (mean in the space FPCA is fit on)
        n_components_ : int
    """
    def __init__(
        self,
        n_components: int = 0,         # 0 => auto by var_ratio
        var_ratio: float = 0.999,      # cumulative EVR threshold for auto selection
        max_components: int = 64,      # cap for auto selection
        ridge: float = 1e-8,           # numerical stability on covariance
        random_state: Optional[int] = None,
    ):
        self.n_components = int(n_components)
        self.var_ratio = float(var_ratio)
        self.max_components = int(max_components)
        self.ridge = float(ridge)
        self.random_state = random_state

        # learned
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
        # center
        self.mean_ = X.mean(axis=0).astype(np.float32)
        Xc = (X - self.mean_).astype(np.float32)

        # covariance across K points
        C = (Xc.T @ Xc) / float(n - 1)
        if self.ridge > 0:
            C = C + (self.ridge * np.eye(K, dtype=np.float32))

        # symmetric EVD
        eigvals, eigvecs = np.linalg.eigh(C.astype(np.float64))
        eigvals = eigvals.astype(np.float32)
        eigvecs = eigvecs.astype(np.float32)

        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        eigvals_clip = np.maximum(eigvals, 0.0)
        total = float(np.sum(eigvals_clip))
        if total <= 0:
            raise ValueError("FPCA: non-positive total variance; check input.")
        evr = eigvals_clip / total

        # rank constraints
        rank_cap = min(K, n - 1)
        rank_cap = min(rank_cap, int(self.max_components)) if self.max_components > 0 else rank_cap

        if self.n_components > 0:
            R = min(self.n_components, rank_cap)
        else:
            cum = np.cumsum(evr)
            R = int(np.searchsorted(cum, self.var_ratio) + 1)
            R = max(1, min(R, rank_cap))

        self.n_components_ = int(R)
        self.components_ = eigvecs[:, :R].T.copy()              # (R, K)
        self.explained_variance_ = eigvals[:R].copy()
        self.explained_variance_ratio_ = evr[:R].copy()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("FPCA.transform called before fit.")
        X = np.asarray(X, dtype=np.float32)
        Xc = (X - self.mean_).astype(np.float32)
        Z = Xc @ self.components_.T
        return Z.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("FPCA.inverse_transform called before fit.")
        Z = np.asarray(Z, dtype=np.float32)
        Xc_hat = Z @ self.components_
        X_hat = Xc_hat + self.mean_
        return X_hat.astype(np.float32)


def fpca_propagate_var_to_y(
    var_z_unscaled: np.ndarray,
    fpca: FPCA,
    scaler_y: StandardScaler,
) -> np.ndarray:
    """
    Map per-dim independent variance in FPCA score space (UNSCALED z) -> variance in original y space.

    Pipeline:
      y -> scaler_y -> y_n
      y_n -> FPCA -> z  (unscaled, in FPCA score coordinates)

    Given var_z_unscaled (N,R), assume diagonal covariance in z:
      var_y_n[k] = sum_r (components_[r,k]^2 * var_z_unscaled[r])
      var_y[k]   = var_y_n[k] * (scaler_y.scale_[k]^2)
    """
    if fpca.components_ is None:
        raise RuntimeError("fpca_propagate_var_to_y: fpca not fit.")
    comp = fpca.components_.astype(np.float32)           # (R, K)
    comp2 = np.square(comp).astype(np.float32)          # (R, K)

    var_z_unscaled = np.asarray(var_z_unscaled, dtype=np.float32)
    if var_z_unscaled.ndim != 2:
        raise ValueError(f"var_z_unscaled must be 2D (N,R), got {var_z_unscaled.shape}")
    if var_z_unscaled.shape[1] != comp2.shape[0]:
        raise ValueError(
            f"var_z_unscaled R mismatch: var_z_unscaled.shape[1]={var_z_unscaled.shape[1]} "
            f"vs fpca.R={comp2.shape[0]}"
        )

    var_y_n = var_z_unscaled @ comp2                      # (N, K)
    scale2 = np.square(np.asarray(scaler_y.scale_, dtype=np.float32))[None, :]  # (1, K)
    var_y = var_y_n * scale2
    return var_y.astype(np.float32)


# ============================================================
# Safer student MLP (avoid feature ReLU dead zone)
# ============================================================
class FeatureMLP(nn.Module):
    """
    Training: forward(x) -> yhat (K)
    Inference: extract_features(x) -> f (feat_dim)

    Key change vs old version:
      - feature activation can be LeakyReLU or Identity (default: LeakyReLU)
      - hidden activation remains configurable (relu/tanh/gelu)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Tuple[int, ...] = (256, 256),
        feat_dim: int = 16,
        act: str = "relu",
        dropout: float = 0.0,
        feat_act: str = "leakyrelu",   # leakyrelu|identity|same
        leaky_slope: float = 0.01,
    ):
        super().__init__()
        acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        Act = acts.get(act, nn.ReLU)

        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)

        self.trunk = nn.Sequential(*layers)
        self.feat_layer = nn.Linear(prev, int(feat_dim))

        feat_act = str(feat_act).lower().strip()
        if feat_act == "leakyrelu":
            self.feat_act = nn.LeakyReLU(negative_slope=float(leaky_slope))
        elif feat_act == "identity":
            self.feat_act = nn.Identity()
        elif feat_act == "same":
            self.feat_act = Act()
        else:
            raise ValueError(f"Unknown feat_act={feat_act}. Use leakyrelu/identity/same")

        self.head = nn.Linear(int(feat_dim), int(out_dim))

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



# -------------------------
# Probabilistic LF head helpers (optional)
# -------------------------
def _split_mu_logvar(y_out: torch.Tensor, K: int, logvar_clip: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split concatenated head output into (mu, logvar) with optional clipping."""
    mu = y_out[:, :K]
    logvar = y_out[:, K:2*K]
    if logvar_clip is not None and float(logvar_clip) > 0:
        logvar = torch.clamp(logvar, -float(logvar_clip), float(logvar_clip))
    return mu, logvar


def _gaussian_nll_torch(y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Elementwise Gaussian NLL (up to an additive constant). Returns mean over batch."""
    var = torch.exp(logvar).clamp_min(1e-12)
    return torch.mean(0.5 * (logvar + (y - mu) ** 2 / var))


def train_feature_mlp_prob(
    model: FeatureMLP,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float,
    batch_size: int,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    device: str,
    print_every: int,
    K: int,
    logvar_clip: float,
) -> Tuple[FeatureMLP, Dict[str, Any]]:
    """Minimal NLL training loop matching train_feature_mlp()'s early-stopping semantics."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    Xtr = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
    ytr = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    Xva = torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(device)
    yva = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).to(device)

    n = int(Xtr.shape[0])
    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, int(max_epochs) + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, int(batch_size)):
            idx = perm[i:i+int(batch_size)]
            xb = Xtr[idx].to(device)
            yb = ytr[idx].to(device)

            out = model(xb)
            mu, logvar = _split_mu_logvar(out, K=K, logvar_clip=logvar_clip)
            loss = _gaussian_nll_torch(yb, mu, logvar)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            out_va = model(Xva)
            mu_va, logvar_va = _split_mu_logvar(out_va, K=K, logvar_clip=logvar_clip)
            val_nll = float(_gaussian_nll_torch(yva, mu_va, logvar_va).item())

        if (ep == 1) or (ep % int(print_every) == 0):
            print(f"[STAGE-I][MLP][NLL] ep={ep:4d}  val_nll={val_nll:.6g}")

        if val_nll < best_val - float(min_delta):
            best_val = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "meta": {
            "best_val_nll": float(best_val),
            "epochs_ran": int(ep),
        }
    }
    return model, info


@torch.no_grad()
def mlp_predict_mu_logvar_and_features(
    model: FeatureMLP,
    X: np.ndarray,
    *,
    device: str,
    batch_size: int,
    K: int,
    logvar_clip: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mu, var, feat) in the *model output space* (i.e., scaled y if student_yscale=1)."""
    model = model.to(device).eval()
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    n = int(Xt.shape[0])
    mus = []
    vars_ = []
    feats = []
    for i in range(0, n, int(batch_size)):
        xb = Xt[i:i+int(batch_size)].to(device)
        out = model(xb)
        mu, logvar = _split_mu_logvar(out, K=K, logvar_clip=logvar_clip)
        var = torch.exp(logvar).clamp_min(1e-12)
        f = model.extract_features(xb)
        mus.append(mu.detach().cpu().numpy())
        vars_.append(var.detach().cpu().numpy())
        feats.append(f.detach().cpu().numpy())
    return (
        np.concatenate(mus, axis=0).astype(np.float32),
        np.concatenate(vars_, axis=0).astype(np.float32),
        np.concatenate(feats, axis=0).astype(np.float32),
    )


def _uniq_rows(a: np.ndarray, dec: int = 6) -> int:
    ar = np.round(np.asarray(a, dtype=np.float32), dec)
    try:
        return int(np.unique(ar, axis=0).shape[0])
    except Exception:
        return -1


def _collapse_guard(name: str, yhat: np.ndarray, feat: np.ndarray, n_expect: int) -> None:
    uy = _uniq_rows(yhat)
    uf = _uniq_rows(feat)
    sfeat = float(np.std(feat))
    print(f"[CHK][student-collapse][{name}] yhat uniq={uy}/{n_expect} | feat uniq={uf}/{n_expect} | feat std={sfeat:.3g}")
    if (uy <= 1) or (uf <= 1) or (sfeat <= 1e-12):
        raise RuntimeError(
            "Student inference collapsed (constant yhat and/or zero features). "
            "Fix: enable student x-scaling, and/or set student_feat_act=leakyrelu/identity."
        )



def fit_affine_rho(lf: np.ndarray, y: np.ndarray, ridge: float = 1e-6, use_intercept: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-dim affine mapping y ≈ rho*lf + b on paired HF-train (both in the SAME target space).

    lf, y: (N, R)
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

def build_run_name(args: argparse.Namespace) -> str:
    """
    Short run-name rule (<=10 chars preferred).

    Prefix comes from --run_prefix, e.g.:
      tm0 / ab0 / mf

    Only append codes for NON-default settings.
    Common codes:
      f6    -> fpca fixed dim = 6
      s50   -> subsample K = 50
      kb    -> kernel_struct = block
      kx    -> kernel_struct = xlf_block
      rb    -> kernel = rbf
      a0    -> gp_ard = 0
      m32   -> svgp_M = 32
      t1k   -> svgp_steps = 1000
      c0    -> ci_calibrate = 0
      p1    -> lf_prob = 1
      o     -> mf_student_lf_source = oracle
      r43   -> seed = 43

    If too many non-default options make the name longer than 10 chars,
    fall back to a stable hashed short name: <prefix> + hex suffix.
    """

    def _num_tag(v: int) -> str:
        v = int(v)
        if v >= 1000 and (v % 1000 == 0):
            return f"{v // 1000}k"
        return str(v)

    def _ratio_tag(v: float) -> str:
        s = f"{float(v):.4f}".rstrip("0").rstrip(".")
        return s.replace("0.", "").replace(".", "p")

    prefix = safe_tag(str(getattr(args, "run_prefix", "mf")).strip())
    if not prefix:
        prefix = "mf"

    codes = []

    # dim reduction
    if str(args.dim_reduce).lower().strip() == "subsample":
        codes.append(f"s{_num_tag(int(args.subsample_K))}")
    else:
        # default FPCA baseline: fpca_dim=0, var_ratio=0.999, max_dim=50
        if int(args.fpca_dim) > 0:
            codes.append(f"f{_num_tag(int(args.fpca_dim))}")
        else:
            if abs(float(args.fpca_var_ratio) - 0.999) > 1e-12:
                codes.append(f"v{_ratio_tag(float(args.fpca_var_ratio))}")
            if int(args.fpca_max_dim) != 50:
                codes.append(f"u{_num_tag(int(args.fpca_max_dim))}")

    # kernel structure / kernel
    ks = str(args.kernel_struct).lower().strip()
    if ks == "block":
        codes.append("kb")
    elif ks == "xlf_block":
        codes.append("kx")

    ker = str(args.kernel).lower().strip()
    if ker == "rbf":
        codes.append("rb")
    elif ker != "matern":
        codes.append(safe_tag(ker)[:2])

    # GP / SVGP
    if int(args.gp_ard) != 1:
        codes.append(f"a{int(args.gp_ard)}")
    if int(args.svgp_M) != 64:
        codes.append(f"m{_num_tag(int(args.svgp_M))}")
    if int(args.svgp_steps) != 2000:
        codes.append(f"t{_num_tag(int(args.svgp_steps))}")

    # UQ / student routing
    if int(args.ci_calibrate) != 1:
        codes.append(f"c{int(args.ci_calibrate)}")
    if int(getattr(args, "lf_prob", 0)) != 0:
        codes.append(f"p{int(getattr(args, 'lf_prob', 0))}")

    lf_src = str(getattr(args, "mf_student_lf_source", "student")).lower().strip()
    if lf_src == "oracle":
        codes.append("o")
    elif lf_src != "student":
        codes.append(safe_tag(lf_src)[:1])

    if int(args.seed) != 42:
        codes.append(f"r{_num_tag(int(args.seed))}")

    name = safe_tag(prefix + "".join(codes))
    if len(name) <= 10:
        return name

    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:7]
    keep = max(1, 10 - len(digest))
    return safe_tag(f"{prefix[:keep]}{digest}")[:10]


def main(argv: Optional[Sequence[str]] = None, defaults: Optional[TrainDefaults] = None):
    print("[BOOT] mf_train_fpca_band_feat.py is running...")

    d = defaults or TrainDefaults()
    ap = argparse.ArgumentParser()

    # --- paths
    ap.add_argument("--data_dir", type=str, default=d.data_dir,
                    help="Dataset root (contains wavelengths.npy and hf/ lf_paired/ lf_unpaired/).")
    ap.add_argument("--out_dir", type=str, default=d.out_dir,
                    help="Output root. A subdir will be created unless --no_subdir=1.")
    ap.add_argument("--run_prefix", type=str, default=d.run_prefix,
                    help="Short run prefix used by auto run-name generation, e.g. tm / ab / mf.")
    ap.add_argument("--exp_name", type=str, default="",
                    help="Optional explicit subdir name under out_dir. If empty, auto-generated from config.")
    ap.add_argument("--no_subdir", type=int, default=0, choices=[0, 1],
                    help="If 1, write directly into --out_dir (no subdir).")

    # --- wavelength crop (optional)
    ap.add_argument("--wl_low", type=float, default=380.0,
                    help="If set together with --wl_high, crop wavelength points to the inclusive range [wl_low, wl_high] (nm).")
    ap.add_argument("--wl_high", type=float, default=750.0,
                    help="Upper bound for wavelength crop (nm). Must be used together with --wl_low.")
    ap.add_argument("--wl_crop_strict", type=int, default=1, choices=[0, 1],
                    help="If 1, error out when the crop range yields 0 points; if 0, silently disable cropping in that case.")

    # --- reproducibility/device
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # --- dim reduction (DEFAULT: fpca)
    ap.add_argument("--dim_reduce", type=str, default="fpca", choices=["fpca", "subsample"])

    # --- FPCA config
    ap.add_argument("--fpca_dim", type=int, default=0,
                    help="FPCA fixed dim. If 0, use adaptive dim by fpca_var_ratio (default).")
    ap.add_argument("--fpca_var_ratio", type=float, default=0.999,
                    help="FPCA adaptive dim: choose smallest R s.t. cumulative EVR >= fpca_var_ratio.")
    ap.add_argument("--fpca_max_dim", type=int, default=50,
                    help="FPCA adaptive dim upper bound.")
    ap.add_argument("--fpca_ridge", type=float, default=1e-8,
                    help="FPCA covariance ridge for numerical stability.")

    # --- subsample config
    ap.add_argument("--subsample_K", type=int, default=50)

    # --- MF input composition

    # --- MF calibration (in z_scaled / target space)
    ap.add_argument("--mf_u_mode", type=str, default="xlf", choices=["lf","xlf","flf","xflf","x","xhf"],
                    help="Stage-II MF input u: 'lf' uses LF_repr only; 'xlf' concatenates x_scaled + LF_repr; 'x'/'xhf' uses x_scaled only.")

    # delta-student (AR1-style residual learning for MF-student)
    ap.add_argument("--student_mode", type=str, default="delta", choices=["direct", "delta"],
                    help="MF-student Stage-II target: 'direct' predicts HF target directly; 'delta' predicts residual (HF - affine(rho)*LF_hat) and then adds back affine(rho)*LF_hat.")
    ap.add_argument("--rho_fit_source", type=str, default="oracle", choices=["oracle", "student"],
                    help="When --student_mode delta: fit affine(rho) on paired HF-train using LF_repr from 'oracle' (true paired LF) or 'student' (Stage-I predicted LF).")
    ap.add_argument("--rho_ridge", type=float, default=1e-6,
                    help="Ridge added to var(LF) when fitting rho in delta-student.")
    ap.add_argument("--rho_intercept", type=int, default=1, choices=[0, 1],
                    help="Use affine fit y ≈ rho*lf + b (1) or force b=0 (0) for delta-student.")


    # --- GP kernels  #block是针对有feature，才会生效，否则退化为full
    ap.add_argument("--kernel_struct", type=str, default="full", choices=["full", "block", "xlf_block"],
                    help="Kernel input structure. 'xlf_block' means u=[x,lf] uses an additive block kernel split as x|lf.")
    ap.add_argument("--kernel", type=str, default="matern", choices=["rbf", "matern"])
    ap.add_argument("--matern_nu", type=float, default=2.5, choices=[0.5, 1.5, 2.5])
    ap.add_argument("--gp_ard", type=int, default=1, choices=[0, 1])

    # --- student settings
    ap.add_argument("--student_train_set", type=str, default="mix", choices=["paired", "mix"],
                    help="Stage-I student MLP training set: 'mix' uses (lf_paired+lf_unpaired)/train; "
                         "'paired' uses lf_paired/train only.")
    # Default validation selection aligned with the best-performing A0 setup.
    ap.add_argument("--student_val_set", type=str, default="paired", choices=["paired", "mix"],
                    help="Student early-stopping validation set: "
                         "'paired' uses lf_paired/val only; 'mix' uses (lf_paired+lf_unpaired)/val.")
    ap.add_argument("--student_yscale", type=int, default=1, choices=[0, 1],
                    help="If 1: apply StandardScaler on student targets y for Stage-I; if 0: train on raw y.")
    ap.add_argument("--student_y_scaler_fit", type=str, default="paired", choices=["mix", "paired"],
                    help="When --student_yscale=1, fit y StandardScaler on 'mix' train targets or 'paired' train targets.")
    ap.add_argument("--student_hidden", type=int, nargs="*", default=[256, 256, 256])
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


    # --- LF probabilistic head (uncertainty for LF propagation)
    ap.add_argument("--lf_prob", type=int, default=0, choices=[0, 1],
                    help="Route-2 (Gaussian head): If 1, Stage-I MLP head outputs (mu, logvar) per point and is trained with Gaussian NLL. This provides an explicit p(y_l|x)=N(mu, sigma^2) for uncertainty propagation. Set 0 to use deterministic MSE head (no LF UQ). "
                         "enables LF uncertainty propagation via --mc_lf_samples.")
    ap.add_argument("--mc_lf_samples", type=int, default=32,
                    help="If >1 and --lf_prob=1, do Monte Carlo marginalization over LF uncertainty during Stage-II prediction (VAL/TEST).")
    ap.add_argument("--lf_logvar_clip", type=float, default=10.0,
                    help="Clip log-variance predicted by Stage-I to [-lf_logvar_clip, +lf_logvar_clip] for numerical stability.")
    # --- ablations / runner hooks
    ap.add_argument("--skip_student", type=int, default=0, choices=[0, 1],
                    help="If 1: skip Stage-I student MLP training and inference; use LF paired y as LF_hat for HF points (A4).")

    ap.add_argument("--run_hf_only", type=int, default=0, choices=[0, 1],
                    help="If 1: train/eval HF-only SVGP branch (Stage-II).")
    ap.add_argument("--run_oracle", type=int, default=0, choices=[0, 1],
                    help="If 1: train/eval MF-oracle SVGP branch (Stage-II).")
    ap.add_argument("--run_student", type=int, default=1, choices=[0, 1],
                    help="If 1: train/eval MF-student SVGP branch (Stage-II).")

    ap.add_argument("--mf_student_lf_source", type=str, default="student", choices=["student", "oracle"],
                    help="Source of LF representation used inside MF-student Stage-II: 'student' uses Stage-I LF_hat; 'oracle' uses true paired LF (A4).")

    # --- SVGP
    ap.add_argument("--svgp_M", type=int, default=64)
    ap.add_argument("--svgp_steps", type=int, default=2000)
    ap.add_argument("--svgp_lr", type=float, default=5e-3)
    # ap.add_argument("--svgp_lr", type=float, default=1e-2)
    ap.add_argument("--print_every", type=int, default=200)

    # --- UQ
    ap.add_argument("--ci_level", type=float, default=0.95)
    ap.add_argument("--ci_calibrate", type=int, default=1, choices=[0, 1])

    # --- plotting
    ap.add_argument("--plot_ci", type=int, default=1, choices=[0, 1])
    ap.add_argument("--n_plot", type=int, default=10)
    ap.add_argument("--save_pred_arrays", type=int, default=0, choices=[0, 1],
                    help="If 1, save prediction/uncertainty arrays (val/test) for downstream plotting (e.g., baseline best/worst cases).")

    args = ap.parse_args(argv)
    if not str(args.data_dir).strip():
        raise SystemExit("--data_dir must be provided explicitly (or via wrapper default).")
    if not str(args.out_dir).strip():
        raise SystemExit("--out_dir must be provided explicitly (or via wrapper default).")
    if not str(getattr(args, "run_prefix", "")).strip():
        raise SystemExit("--run_prefix must be provided explicitly (or via wrapper default).")
    set_seed(int(args.seed))

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    run_id = now_tag()
    run_name = args.exp_name.strip() if args.exp_name.strip() else build_run_name(args)
    out_dir = out_root if int(args.no_subdir) == 1 else (out_root / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_logger = CsvLogger(
        trace_path=out_dir / "trace.csv",
        results_path=out_dir / "results.csv",
    )
    csv_logger.open()

    # -------------------------
    # Load wavelengths + data splits
    # -------------------------
    wl_full, idx_wl_full = load_wavelengths(data_dir)
    K_full = int(wl_full.size)

    # Decide wavelength crop indices early (apply after y is loaded)
    idx_keep_wl = None
    if (args.wl_low is not None) or (args.wl_high is not None):
        if (args.wl_low is None) or (args.wl_high is None):
            raise SystemExit("--wl_low and --wl_high must be provided together.")
        lo = float(min(args.wl_low, args.wl_high))
        hi = float(max(args.wl_low, args.wl_high))
        mask = (wl_full >= lo) & (wl_full <= hi)
        idx_keep_wl = np.where(mask)[0].astype(np.int64)
        if idx_keep_wl.size == 0:
            if bool(int(args.wl_crop_strict)):
                raise SystemExit(f"Wavelength crop [{lo},{hi}] nm keeps 0 points (K_full={K_full}).")
            idx_keep_wl = None


    x_hf_tr, y_hf_tr, t_hf_tr, idx_hf_tr = load_split_block(data_dir, "hf", "train")
    x_hf_va, y_hf_va, t_hf_va, idx_hf_va = load_split_block(data_dir, "hf", "val")
    x_hf_te, y_hf_te, t_hf_te, idx_hf_te = load_split_block(data_dir, "hf", "test")

    x_lfp_tr, y_lfp_tr, t_lfp_tr, idx_lfp_tr = load_split_block(data_dir, "lf_paired", "train")
    x_lfp_va, y_lfp_va, t_lfp_va, idx_lfp_va = load_split_block(data_dir, "lf_paired", "val")
    x_lfp_te, y_lfp_te, t_lfp_te, idx_lfp_te = load_split_block(data_dir, "lf_paired", "test")

    assert_indices_match(idx_hf_tr, idx_lfp_tr, "hf/train", "lf_paired/train")
    assert_indices_match(idx_hf_va, idx_lfp_va, "hf/val", "lf_paired/val")
    assert_indices_match(idx_hf_te, idx_lfp_te, "hf/test", "lf_paired/test")

    x_lfu_tr, y_lfu_tr, t_lfu_tr, idx_lfu_tr = load_split_block(data_dir, "lf_unpaired", "train")
    x_lfu_va, y_lfu_va, t_lfu_va, idx_lfu_va = load_split_block(data_dir, "lf_unpaired", "val")
    x_lfu_te, y_lfu_te, t_lfu_te, idx_lfu_te = load_split_block(data_dir, "lf_unpaired", "test")

    # --- apply optional wavelength crop
    if idx_keep_wl is not None:
        def _crop_y(y: np.ndarray, name: str) -> np.ndarray:
            if y is None:
                return y
            y = np.asarray(y)
            if y.ndim != 2:
                raise SystemExit(f"Expect y to be 2D for {name}, got shape={y.shape}")
            if y.shape[1] != K_full:
                raise SystemExit(f"y second-dim mismatch for {name}: y.shape[1]={y.shape[1]} vs K_full={K_full}")
            return y[:, idx_keep_wl].astype(np.float32, copy=False)

        # crop y in-place (HF + LF paired/unpaired, all splits)
        y_hf_tr = _crop_y(y_hf_tr, "hf/train")
        y_hf_va = _crop_y(y_hf_va, "hf/val")
        y_hf_te = _crop_y(y_hf_te, "hf/test")

        y_lfp_tr = _crop_y(y_lfp_tr, "lf_paired/train")
        y_lfp_va = _crop_y(y_lfp_va, "lf_paired/val")
        y_lfp_te = _crop_y(y_lfp_te, "lf_paired/test")

        y_lfu_tr = _crop_y(y_lfu_tr, "lf_unpaired/train")
        y_lfu_va = _crop_y(y_lfu_va, "lf_unpaired/val")
        y_lfu_te = _crop_y(y_lfu_te, "lf_unpaired/test")

        wl = wl_full[idx_keep_wl].astype(np.float32)
        idx_wl = idx_wl_full[idx_keep_wl].astype(np.int64)
        K = int(wl.size)

        lo = float(min(args.wl_low, args.wl_high))
        hi = float(max(args.wl_low, args.wl_high))
        print(f"[CROP] wl_range=[{lo},{hi}] nm | K_full={K_full} -> K={K}")
        dbg_print_wavelengths(wl_full=wl_full, idx_keep=idx_keep_wl, wl_used=wl)
    else:
        wl = wl_full.astype(np.float32)
        idx_wl = idx_wl_full.astype(np.int64)
        K = int(wl.size)

        dbg_print_wavelengths(wl_full=wl_full, idx_keep=idx_keep_wl, wl_used=wl)

    xdim = int(x_hf_tr.shape[1])

    print(f"[INFO] device={device}")
    print(f"[INFO] data_dir={data_dir}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] run_name={run_name} run_id={run_id}")
    print(f"[INFO] wavelength K={K} | wl=[{float(wl.min()):.6g},{float(wl.max()):.6g}]")
    # hard asserts: HF/LF y must match wavelength length after crop/subsample
    for _name, _y in [
        ("hf/train", y_hf_tr), ("hf/val", y_hf_va), ("hf/test", y_hf_te),
        ("lf_paired/train", y_lfp_tr), ("lf_paired/val", y_lfp_va), ("lf_paired/test", y_lfp_te),
        ("lf_unpaired/train", y_lfu_tr), ("lf_unpaired/val", y_lfu_va), ("lf_unpaired/test", y_lfu_te),
    ]:
        assert _y.shape[1] == K, f"{_name} y.shape[1]={_y.shape[1]} != K={K}"
    # HF/LF use identical wavelengths (single source of truth)
    assert wl.shape[0] == K

    print(f"[INFO] HF split train/val/test = {x_hf_tr.shape[0]}/{x_hf_va.shape[0]}/{x_hf_te.shape[0]}")
    print(f"[INFO] LF paired split train/val/test = {x_lfp_tr.shape[0]}/{x_lfp_va.shape[0]}/{x_lfp_te.shape[0]}")
    print(f"[INFO] LF unpaired split train/val/test = {x_lfu_tr.shape[0]}/{x_lfu_va.shape[0]}/{x_lfu_te.shape[0]}")
    print(f"[INFO] CI: level={float(args.ci_level)}  z≈{z_from_ci_level(float(args.ci_level)):.4f}  calibrate={bool(args.ci_calibrate)}")

    # config.json
    config = {
        "run_id": run_id, "run_name": run_name,
        "args": vars(args),
        "data_dir": str(data_dir), "out_dir": str(out_dir),
        "K": K, "xdim": xdim,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    np.save(out_dir / "wavelengths.npy", wl.astype(np.float32))
    np.save(out_dir / "idx_wavelength.npy", idx_wl.astype(np.int64))

    # -------------------------
    # Stage-I: student (x -> LF oracle y)
    # -------------------------
    print("[STAGE-I] Train student (feature+head): x -> LF oracle y (K)")

    # concat mix train/val/test
    X_lf_tr_all = np.concatenate([x_lfp_tr, x_lfu_tr], axis=0).astype(np.float32)
    y_lf_tr_all = np.concatenate([y_lfp_tr, y_lfu_tr], axis=0).astype(np.float32)
    X_lf_va_all = np.concatenate([x_lfp_va, x_lfu_va], axis=0).astype(np.float32)
    y_lf_va_all = np.concatenate([y_lfp_va, y_lfu_va], axis=0).astype(np.float32)
    X_lf_te_all = np.concatenate([x_lfp_te, x_lfu_te], axis=0).astype(np.float32)
    y_lf_te_all = np.concatenate([y_lfp_te, y_lfu_te], axis=0).astype(np.float32)
    # defaults for reporting (in case student_yscale=0)
    y_fit_on = ""
    y_fit_n = 0

    # fit student x-scaler on mix_train only
    sx_student: Optional[StandardScaler] = None
    if True:  # always x-scale for student
        sx_student = StandardScaler(with_mean=True, with_std=True)
        sx_student.fit(X_lf_tr_all)
        X_lf_tr_all_s = sx_student.transform(X_lf_tr_all).astype(np.float32)
        X_lf_va_all_s = sx_student.transform(X_lf_va_all).astype(np.float32)
        X_lf_te_all_s = sx_student.transform(X_lf_te_all).astype(np.float32)
        save_pickle(out_dir / "student_scaler_x.pkl", sx_student)
        print("[STAGE-I] student_xscale=1 | saved student_scaler_x.pkl")
    else:
        X_lf_tr_all_s = X_lf_tr_all
        X_lf_va_all_s = X_lf_va_all
        X_lf_te_all_s = X_lf_te_all
        print("[STAGE-I] student_xscale=0 | WARNING: training student on raw x (may collapse)")

    def _student_x(x: np.ndarray) -> np.ndarray:
        xx = x.astype(np.float32)
        if sx_student is None:
            return xx
        return sx_student.transform(xx).astype(np.float32)


    # --- optional: student y scaling (stability + conditioning)
    sy_student: Optional[StandardScaler] = None
    if bool(int(args.student_yscale)):
        sy_student = StandardScaler(with_mean=True, with_std=True)

        # Default: fit on Stage-I training targets (mix: paired + optional LF_unpaired).
        # Optional: fit only on HF-paired train targets.
        fit_mode = str(args.student_y_scaler_fit).lower().strip()
        if fit_mode not in ("mix", "paired"):
            fit_mode = "mix"

        if fit_mode == "paired":
            y_fit = y_lfp_tr
            y_fit_on = "y_lf_tr (paired-only)"
        else:
            y_fit = y_lf_tr_all
            y_fit_on = "mix_train (paired+unpaired)" if (y_fit.shape[0] != y_lfp_tr.shape[0]) else "mix_train (paired-only)"

        sy_student.fit(y_fit.astype(np.float32))
        y_fit_n = int(y_fit.shape[0])

        y_lf_tr_all_y = sy_student.transform(y_lf_tr_all.astype(np.float32)).astype(np.float32)
        y_lf_va_all_y = sy_student.transform(y_lf_va_all.astype(np.float32)).astype(np.float32)
        y_lf_te_all_y = sy_student.transform(y_lf_te_all.astype(np.float32)).astype(np.float32)
        save_pickle(out_dir / "student_scaler_y.pkl", sy_student)
        print(f"[STAGE-I] student_yscale=1 | saved student_scaler_y.pkl | fit_mode={fit_mode} | fit_on={y_fit_on} | N={y_fit_n}")
    else:
        y_lf_tr_all_y = y_lf_tr_all.astype(np.float32)
        y_lf_va_all_y = y_lf_va_all.astype(np.float32)
        y_lf_te_all_y = y_lf_te_all.astype(np.float32)
        print("[STAGE-I] student_yscale=0 | training student on raw y (may be ill-conditioned)")

    def _student_y_inv(yhat: np.ndarray) -> np.ndarray:
        yy = yhat.astype(np.float32)
        if sy_student is None:
            return yy
        return sy_student.inverse_transform(yy).astype(np.float32)

    def _student_y_fwd(y: np.ndarray) -> np.ndarray:
        yy = y.astype(np.float32)
        if sy_student is None:
            return yy
        return sy_student.transform(yy).astype(np.float32)

    skip_student = bool(int(args.skip_student))
    if skip_student:
        print("[STAGE-I] skip_student=1 | NO student MLP; set LF_hat on HF points to true paired LF (A4)")

        # For mix metrics: identity (predict exactly what you train on)
        yhat_mix_tr = y_lf_tr_all.astype(np.float32)
        yhat_mix_va = y_lf_va_all.astype(np.float32)
        yhat_mix_te = y_lf_te_all.astype(np.float32)
        stage1_rmse_mix_tr = 0.0
        stage1_rmse_mix_va = 0.0
        stage1_rmse_mix_te = 0.0

        # For HF points: treat LF_hat == LF_paired (oracle)
        yhat_lf_tr = y_lfp_tr.astype(np.float32)
        yhat_lf_va = y_lfp_va.astype(np.float32)
        yhat_lf_te = y_lfp_te.astype(np.float32)

        feat_dim = int(args.student_feat_dim)
        feat_tr = np.zeros((int(x_hf_tr.shape[0]), feat_dim), dtype=np.float32)
        feat_va = np.zeros((int(x_hf_va.shape[0]), feat_dim), dtype=np.float32)
        feat_te = np.zeros((int(x_hf_te.shape[0]), feat_dim), dtype=np.float32)

        stage1_rmse_paired_tr = 0.0
        stage1_rmse_paired_va = 0.0
        stage1_rmse_paired_te = 0.0

        # dummy meta
        stu_info = {
            "meta": {"best_val_mse": 0.0, "epochs_ran": 0},
            "note": "skip_student=1 (A4): Stage-I bypassed",
            "y_scaler_fit_on": y_fit_on,
            "y_scaler_path": str(out_dir / "student_scaler_y.pkl") if sy_student is not None else "",
        }
        with open(out_dir / "student_meta.json", "w", encoding="utf-8") as f:
            json.dump(stu_info, f, ensure_ascii=False, indent=2)

        print("[STAGE-I][RMSE][mix]   train/val/test = 0 / 0 / 0")
        print("[STAGE-I][RMSE][paired] train/val/test = 0 / 0 / 0")

        y_lf_tr = y_lfp_tr.astype(np.float32)
        y_lf_va = y_lfp_va.astype(np.float32)
        y_lf_te = y_lfp_te.astype(np.float32)

        dbg_student_metrics = dbg_student_on_hf_errors(
            yhat_lf_va=yhat_lf_va, y_lf_va=y_lf_va,
            yhat_lf_te=yhat_lf_te, y_lf_te=y_lf_te
        )

    else:
        # --- student train/val set selection
        # Train can be 'mix' (paired+unpaired) or 'paired' (paired-only)
        student_train_set = str(args.student_train_set).lower().strip()
        if student_train_set == "paired":
            X_stu_train = _student_x(x_lfp_tr)
            y_stu_train = _student_y_fwd(y_lfp_tr)
            print("[STAGE-I] student_train_set=paired | train on lf_paired/train only")
        elif student_train_set == "mix":
            X_stu_train = X_lf_tr_all_s
            y_stu_train = y_lf_tr_all_y
            print("[STAGE-I] student_train_set=mix | train on (lf_paired+lf_unpaired)/train")
        else:
            raise ValueError(f"Unknown --student_train_set={args.student_train_set}. Use paired|mix")

        # Val can be 'paired' (recommended) or 'mix'
        student_val_set = str(args.student_val_set).lower().strip()
        if student_val_set == "paired":
            X_stu_val = _student_x(x_lfp_va)  # paired val only
            y_stu_val = _student_y_fwd(y_lfp_va)
            print("[STAGE-I] student_val_set=paired | early-stop on lf_paired/val only")
        elif student_val_set == "mix":
            X_stu_val = X_lf_va_all_s
            y_stu_val = y_lf_va_all_y
            print("[STAGE-I] student_val_set=mix | early-stop on (lf_paired+lf_unpaired)/val")
        else:
            raise ValueError(f"Unknown --student_val_set={args.student_val_set}. Use paired|mix")

        
        out_dim_student = (2 * K) if int(getattr(args, "lf_prob", 0)) == 1 else K
        student = FeatureMLP(
            in_dim=xdim,
            out_dim=out_dim_student,
            hidden=tuple(int(x) for x in args.student_hidden),
            feat_dim=int(args.student_feat_dim),
            act=str(args.student_act),
            dropout=float(args.student_dropout),
            feat_act=str(args.student_feat_act),
            leaky_slope=float(args.student_feat_leaky_slope),
        ).to(device)

        print(f"[INFO] student arch: in_dim={xdim} hidden={tuple(int(x) for x in args.student_hidden)} "
              f"feat_dim={int(args.student_feat_dim)} out_dim={out_dim_student} act={args.student_act} feat_act={args.student_feat_act} "
              f"lf_prob={int(getattr(args,'lf_prob',0))}")

        if int(getattr(args, "lf_prob", 0)) == 1:
            student, stu_info = train_feature_mlp_prob(
                student,
                X_train=X_stu_train, y_train=y_stu_train,
                X_val=X_stu_val, y_val=y_stu_val,
                lr=float(args.student_lr),
                batch_size=int(args.student_bs),
                weight_decay=float(args.student_wd),
                max_epochs=int(args.student_epochs),
                patience=int(args.student_patience),
                min_delta=float(args.student_min_delta),
                device=device,
                print_every=int(args.student_print_every),
                K=int(K),
                logvar_clip=float(getattr(args, "lf_logvar_clip", 10.0)),
            )
            # compatibility keys (some downstream code expects best_val_mse)
            if "meta" in stu_info and "best_val_nll" in stu_info["meta"]:
                stu_info["meta"]["best_val_mse"] = float(stu_info["meta"]["best_val_nll"])
        else:
            student, stu_info = train_feature_mlp(
                student,
                X_train=X_stu_train, y_train=y_stu_train,
                X_val=X_stu_val, y_val=y_stu_val,
                lr=float(args.student_lr),
                batch_size=int(args.student_bs),
                weight_decay=float(args.student_wd),
                max_epochs=int(args.student_epochs),
                patience=int(args.student_patience),
                min_delta=float(args.student_min_delta),
                device=device,
                print_every=int(args.student_print_every),
            )

        torch.save(student.state_dict(), out_dir / "student_feature_mlp.pth")
        stu_info["y_scaler_fit_on"] = (str(y_fit_on) if (sy_student is not None) else "")
        stu_info["y_scaler_path"] = (str(out_dir / "student_scaler_y.pkl") if (sy_student is not None) else "")
        stu_info["lf_prob"] = int(getattr(args, "lf_prob", 0))
        stu_info["mc_lf_samples"] = int(getattr(args, "mc_lf_samples", 1))
        stu_info["lf_logvar_clip"] = float(getattr(args, "lf_logvar_clip", 10.0))

        with open(out_dir / "student_meta.json", "w", encoding="utf-8") as f:
            json.dump(stu_info, f, ensure_ascii=False, indent=2)

        if int(getattr(args, "lf_prob", 0)) == 1:
            print(f"[STAGE-I] best_val_nll={stu_info['meta'].get('best_val_nll', float('nan')):.6g} | epochs={stu_info['meta'].get('epochs_ran', -1)}")
        else:
            print(f"[STAGE-I] best_val_mse={stu_info['meta']['best_val_mse']:.6g} | epochs={stu_info['meta']['epochs_ran']}")

        # Stage-I eval
        student = student.to(device).eval()

        if int(getattr(args, "lf_prob", 0)) == 1:
            yhat_mix_tr_s, yvar_mix_tr_s, _ = mlp_predict_mu_logvar_and_features(
                student, X_lf_tr_all_s, device=device, batch_size=4096, K=int(K),
                logvar_clip=float(getattr(args, "lf_logvar_clip", 10.0)),
            )
            yhat_mix_va_s, yvar_mix_va_s, _ = mlp_predict_mu_logvar_and_features(
                student, X_lf_va_all_s, device=device, batch_size=4096, K=int(K),
                logvar_clip=float(getattr(args, "lf_logvar_clip", 10.0)),
            )
            yhat_mix_te_s, yvar_mix_te_s, _ = mlp_predict_mu_logvar_and_features(
                student, X_lf_te_all_s, device=device, batch_size=4096, K=int(K),
                logvar_clip=float(getattr(args, "lf_logvar_clip", 10.0)),
            )
        else:
            yhat_mix_tr_s, _ = mlp_predict_and_features(student, X_lf_tr_all_s, device=device, batch_size=4096)
            yhat_mix_va_s, _ = mlp_predict_and_features(student, X_lf_va_all_s, device=device, batch_size=4096)
            yhat_mix_te_s, _ = mlp_predict_and_features(student, X_lf_te_all_s, device=device, batch_size=4096)
            yvar_mix_tr_s = None
            yvar_mix_va_s = None
            yvar_mix_te_s = None

        yhat_mix_tr = _student_y_inv(yhat_mix_tr_s)
        yhat_mix_va = _student_y_inv(yhat_mix_va_s)
        yhat_mix_te = _student_y_inv(yhat_mix_te_s)

        stage1_rmse_mix_tr = rmse(yhat_mix_tr, y_lf_tr_all)
        stage1_rmse_mix_va = rmse(yhat_mix_va, y_lf_va_all)
        stage1_rmse_mix_te = rmse(yhat_mix_te, y_lf_te_all)

        print(f"[STAGE-I][RMSE][mix]   train/val/test = "
              f"{stage1_rmse_mix_tr:.6g} / {stage1_rmse_mix_va:.6g} / {stage1_rmse_mix_te:.6g}")

        x_hf_tr_s_for_student = _student_x(x_hf_tr)
        x_hf_va_s_for_student = _student_x(x_hf_va)
        x_hf_te_s_for_student = _student_x(x_hf_te)

        if int(getattr(args, "lf_prob", 0)) == 1:
            yhat_lf_tr_s, yvar_lf_tr_s, feat_tr = mlp_predict_mu_logvar_and_features(
                student, x_hf_tr_s_for_student, device=device, batch_size=4096, K=int(K),
                logvar_clip=float(getattr(args, "lf_logvar_clip", 10.0)),
            )
            yhat_lf_va_s, yvar_lf_va_s, feat_va = mlp_predict_mu_logvar_and_features(
                student, x_hf_va_s_for_student, device=device, batch_size=4096, K=int(K),
                logvar_clip=float(getattr(args, "lf_logvar_clip", 10.0)),
            )
            yhat_lf_te_s, yvar_lf_te_s, feat_te = mlp_predict_mu_logvar_and_features(
                student, x_hf_te_s_for_student, device=device, batch_size=4096, K=int(K),
                logvar_clip=float(getattr(args, "lf_logvar_clip", 10.0)),
            )
        else:
            yhat_lf_tr_s, feat_tr = mlp_predict_and_features(student, x_hf_tr_s_for_student, device=device, batch_size=4096)
            yhat_lf_va_s, feat_va = mlp_predict_and_features(student, x_hf_va_s_for_student, device=device, batch_size=4096)
            yhat_lf_te_s, feat_te = mlp_predict_and_features(student, x_hf_te_s_for_student, device=device, batch_size=4096)
            yvar_lf_tr_s = None
            yvar_lf_va_s = None
            yvar_lf_te_s = None

        yhat_lf_tr = _student_y_inv(yhat_lf_tr_s)
        yhat_lf_va = _student_y_inv(yhat_lf_va_s)
        yhat_lf_te = _student_y_inv(yhat_lf_te_s)
        _collapse_guard("hf/train", yhat_lf_tr, feat_tr, n_expect=int(x_hf_tr.shape[0]))
        _collapse_guard("hf/val", yhat_lf_va, feat_va, n_expect=int(x_hf_va.shape[0]))
        _collapse_guard("hf/test", yhat_lf_te, feat_te, n_expect=int(x_hf_te.shape[0]))

        stage1_rmse_paired_tr = rmse(yhat_lf_tr, y_lfp_tr)
        stage1_rmse_paired_va = rmse(yhat_lf_va, y_lfp_va)
        stage1_rmse_paired_te = rmse(yhat_lf_te, y_lfp_te)

        print(f"[STAGE-I][RMSE][paired] train/val/test = "
              f"{stage1_rmse_paired_tr:.6g} / {stage1_rmse_paired_va:.6g} / {stage1_rmse_paired_te:.6g}")

        y_lf_tr = y_lfp_tr.astype(np.float32)
        y_lf_va = y_lfp_va.astype(np.float32)
        y_lf_te = y_lfp_te.astype(np.float32)

        dbg_student_metrics = dbg_student_on_hf_errors(
            yhat_lf_va=yhat_lf_va, y_lf_va=y_lf_va,
            yhat_lf_te=yhat_lf_te, y_lf_te=y_lf_te
        )

    # -------------------------
    # Stage-II: targets & LF representations
    # -------------------------
    dim_reduce = args.dim_reduce.lower().strip()

    # shared
    idx_k: Optional[np.ndarray] = None
    W_full_from_sub: Optional[np.ndarray] = None

    # fpca path artifacts
    scaler_y: Optional[StandardScaler] = None       # first scaler on y(K)
    fpca: Optional[FPCA] = None
    scaler_z: Optional[StandardScaler] = None       # second scaler on z(R)  <-- NEW
    fpca_dim_effective: Optional[int] = None
    fpca_evr_sum: Optional[float] = None
    fpca_recon_rmse_hftr: Optional[float] = None
    fpca_recon_rmse_hfval: Optional[float] = None

    if dim_reduce == "fpca":
        print("[FPCA] Fit scaler_y (HF-train) then FPCA on scaled HF-train y_hf (K)")
        print("[FPCA] Then fit scaler_z (HF-train) on FPCA scores z(R) for SVGP target space")

        # 1) scale BEFORE FPCA (HF-train only)
        scaler_y = StandardScaler(with_mean=True, with_std=True)
        y_hf_tr_n = scaler_y.fit_transform(y_hf_tr.astype(np.float32)).astype(np.float32)

        # 2) FPCA fit (auto dim by default)
        fpca = FPCA(
            n_components=int(args.fpca_dim),
            var_ratio=float(args.fpca_var_ratio),
            max_components=int(args.fpca_max_dim),
            ridge=float(args.fpca_ridge),
            random_state=int(args.seed),
        )
        z_hf_tr = fpca.fit_transform(y_hf_tr_n).astype(np.float32)   # unscaled z
        R = int(fpca.n_components_)

        def to_fpca_z_unscaled(y: np.ndarray) -> np.ndarray:
            assert scaler_y is not None and fpca is not None
            y_n = scaler_y.transform(y.astype(np.float32)).astype(np.float32)
            return fpca.transform(y_n).astype(np.float32)

        z_hf_va = to_fpca_z_unscaled(y_hf_va)
        z_hf_te = to_fpca_z_unscaled(y_hf_te)

        z_lf_tr_repr = to_fpca_z_unscaled(y_lf_tr)
        z_lf_va_repr = to_fpca_z_unscaled(y_lf_va)
        z_lf_te_repr = to_fpca_z_unscaled(y_lf_te)

        z_hat_tr_repr = to_fpca_z_unscaled(yhat_lf_tr)
        z_hat_va_repr = to_fpca_z_unscaled(yhat_lf_va)
        z_hat_te_repr = to_fpca_z_unscaled(yhat_lf_te)

        # save first-layer artifacts
        save_pickle(out_dir / "scaler_y.pkl", scaler_y)
        save_pickle(out_dir / "fpca.pkl", fpca)

        fpca_dim_effective = int(R)
        fpca_evr_sum = float(np.sum(fpca.explained_variance_ratio_)) if fpca.explained_variance_ratio_ is not None else None
        print(f"[FPCA] R={R} | EVR_sum={fpca_evr_sum:.6f}")

        # recon RMSE via existing helper (fpca has transform/inverse_transform)
        fpca_recon_rmse_hftr = pca_recon_rmse(y_hf_tr, scaler_y, fpca)   # helper name kept but fpca-compatible
        fpca_recon_rmse_hfval = pca_recon_rmse(y_hf_va, scaler_y, fpca)
        print(f"[FPCA] recon_rmse HF-tr={fpca_recon_rmse_hftr:.6g} | HF-val={fpca_recon_rmse_hfval:.6g}")

        # 3) SECOND LAYER scaler on z (HF-train only)  <-- NEW
        scaler_z = StandardScaler(with_mean=True, with_std=True)
        Y_tr = scaler_z.fit_transform(z_hf_tr).astype(np.float32)
        Y_va = scaler_z.transform(z_hf_va).astype(np.float32)
        Y_te = scaler_z.transform(z_hf_te).astype(np.float32)

        # scale LF representations in the SAME z-space
        z_lf_tr_repr = scaler_z.transform(z_lf_tr_repr).astype(np.float32)
        z_lf_va_repr = scaler_z.transform(z_lf_va_repr).astype(np.float32)
        z_lf_te_repr = scaler_z.transform(z_lf_te_repr).astype(np.float32)

        z_hat_tr_repr = scaler_z.transform(z_hat_tr_repr).astype(np.float32)
        z_hat_va_repr = scaler_z.transform(z_hat_va_repr).astype(np.float32)
        z_hat_te_repr = scaler_z.transform(z_hat_te_repr).astype(np.float32)
        # MF calibration removed (no-op)

        save_pickle(out_dir / "scaler_z.pkl", scaler_z)
        print("[FPCA] saved scaler_z.pkl (second-layer target scaler)")

        def inv_target_to_y_full(mu_target_scaled: np.ndarray) -> np.ndarray:
            """
            mu_target_scaled: predicted mean in z_scaled space (after scaler_z)
            -> unscale to z_unscaled -> FPCA inverse to y_n -> inverse scaler_y -> y
            """
            assert scaler_y is not None and fpca is not None and scaler_z is not None
            mu_target_scaled = mu_target_scaled.astype(np.float32)
            z_unscaled = scaler_z.inverse_transform(mu_target_scaled).astype(np.float32)
            y_n = fpca.inverse_transform(z_unscaled)
            y = scaler_y.inverse_transform(y_n.astype(np.float32)).astype(np.float32)
            return y

        inv_target_to_y_full_fn = inv_target_to_y_full
        target_te_ref = Y_te  # target space reference is z_scaled

    elif dim_reduce == "subsample":
        Ks_int = int(args.subsample_K)
        idx_k = make_even_subsample_indices(Kb=K, Ks=Ks_int)
        wl_sub = wl[idx_k]
        np.save(out_dir / "idx_subsample_k.npy", idx_k.astype(np.int64))
        np.save(out_dir / "wavelengths_subsample.npy", wl_sub.astype(np.float32))

        W_full_from_sub = build_linear_interp_weights(x_full=wl, x_sub=wl_sub)
        save_pickle(out_dir / "interp_weights_full_from_sub.pkl", W_full_from_sub)
        print(f"[SUB] Ks={Ks_int} | wl_sub=[{float(wl_sub.min()):.6g},{float(wl_sub.max()):.6g}]")
        print(f"[DEBUG][SUB] K_band={K} -> Ks={Ks_int} | idx_k={idx_k.tolist()}")
        h_sub, t_sub = _dbg_head_tail(wl_sub, 5)
        print(f"[DEBUG][SUB] wl_sub head5={h_sub} tail5={t_sub}")

        y_hf_tr_sub = pick_y_sub(y_hf_tr, idx_k)
        y_hf_va_sub = pick_y_sub(y_hf_va, idx_k)
        y_hf_te_sub = pick_y_sub(y_hf_te, idx_k)

        # subsample branch already has "target scaler" on reduced y_sub (kept)
        scaler_hf_target = StandardScaler(with_mean=True, with_std=True)
        Y_tr = scaler_hf_target.fit_transform(y_hf_tr_sub).astype(np.float32)
        Y_va = scaler_hf_target.transform(y_hf_va_sub).astype(np.float32)
        Y_te = scaler_hf_target.transform(y_hf_te_sub).astype(np.float32)
        save_pickle(out_dir / "scaler_hf_y_sub.pkl", scaler_hf_target)

        z_lf_tr_repr = pick_y_sub(y_lf_tr, idx_k)
        z_lf_va_repr = pick_y_sub(y_lf_va, idx_k)
        z_lf_te_repr = pick_y_sub(y_lf_te, idx_k)

        z_hat_tr_repr = pick_y_sub(yhat_lf_tr, idx_k)
        z_hat_va_repr = pick_y_sub(yhat_lf_va, idx_k)
        z_hat_te_repr = pick_y_sub(yhat_lf_te, idx_k)

        def inv_target_to_y_full(mu_target_scaled: np.ndarray) -> np.ndarray:
            assert idx_k is not None
            y_sub = scaler_hf_target.inverse_transform(mu_target_scaled.astype(np.float32)).astype(np.float32)
            return upsample_y_sub_to_full(y_sub, idx_k=idx_k, wl_full=wl)

        inv_target_to_y_full_fn = inv_target_to_y_full
        target_te_ref = Y_te

    else:
        raise ValueError(f"Unknown dim_reduce: {dim_reduce}")

    # -------------------------
    # Ablation hook (A4): choose LF source used by MF-student in Stage-II
    #   - 'student': use Stage-I predicted LF_hat (default)
    #   - 'oracle' : use true paired LF y_lf_* (bypasses student influence)
    # This affects both U_st (LF representation) and, for delta mode, the base term.
    # -------------------------
    mf_student_lf_source = str(getattr(args, "mf_student_lf_source", "student")).lower().strip()
    if mf_student_lf_source not in ("student", "oracle"):
        mf_student_lf_source = "student"
    if mf_student_lf_source == "oracle":
        z_hat_tr_repr = np.array(z_lf_tr_repr, copy=True)
        z_hat_va_repr = np.array(z_lf_va_repr, copy=True)
        z_hat_te_repr = np.array(z_lf_te_repr, copy=True)
        print("[STAGE-II] mf_student_lf_source=oracle | use true paired LF as LF_hat (A4)")
    else:
        print("[STAGE-II] mf_student_lf_source=student | use Stage-I predicted LF_hat")

    # -------------------------
    # Build inputs & standardization for GP
    # -------------------------
    sx = StandardScaler(with_mean=True, with_std=True).fit(x_hf_tr.astype(np.float32))
    X_hf_tr_s = sx.transform(x_hf_tr.astype(np.float32)).astype(np.float32)
    X_hf_va_s = sx.transform(x_hf_va.astype(np.float32)).astype(np.float32)
    X_hf_te_s = sx.transform(x_hf_te.astype(np.float32)).astype(np.float32)
    save_pickle(out_dir / "scaler_x_hf.pkl", sx)

    use_feat_or = False  # feature not used in Stage-II (simplified)
    use_feat_st = False  # feature not used in Stage-II (simplified)
    feat_dim = int(args.student_feat_dim)  # kept for student MLP architecture
    # ----------------------------
    # Stage-II MF inputs (u): allow lf / xlf / flf / xflf
    #
    # Pieces (all in FPCA-coef space unless noted):
    #   lf : z_lf_*_repr   (oracle uses true LF; student uses predicted LF_hat)
    #   f  : feat_*        (Stage-I student last-layer feature)
    #   x  : X_hf_*_s      (scaled raw x)
    #
    # Ordering for block-kernel compatibility:
    #   u = [f(if used), x(if used), lf]
    #
    # Semantics:
    #   - "lf"   : u = lf                (or [f, lf] if --use_feature_* = 1)
    #   - "xlf"  : u = [x, lf]           (or [f, x, lf] if --use_feature_* = 1)
    #   - "flf"  : u = [f, lf]           (feature forced-in, regardless of --use_feature_*)
    #   - "xflf" : u = [f, x, lf]        (feature forced-in, x included)
    #
    # This design lets你直接做“x vs feature”的对照：
    #   --use_feature_oracle 0 --use_feature_student 0
    #   + (a) --mf_u_mode xlf   (只加 x)
    #   + (b) --mf_u_mode flf   (只加 feature)
    #   + (c) --mf_u_mode xflf  (x + feature)
    # ----------------------------
    lf_or_tr = z_lf_tr_repr
    lf_or_va = z_lf_va_repr
    lf_or_te = z_lf_te_repr

    lf_st_tr = z_hat_tr_repr
    lf_st_va = z_hat_va_repr
    lf_st_te = z_hat_te_repr

    mode_u = str(args.mf_u_mode).lower()
    if mode_u not in ("lf", "xlf", "flf", "x", "xhf"):
        raise ValueError(f"Invalid --mf_u_mode={mode_u}. Choose from: lf/xlf/flf/x/xhf")

    if mode_u == "xhf":
        mode_u = "x"

    use_x = (mode_u in ("xlf", "x"))
    use_lf = (mode_u in ("lf", "xlf", "flf"))

    # only-enable feature for flf (feature replaces x)
    use_feat_or = (mode_u == "flf")
    use_feat_st = (mode_u == "flf")

    def _build_u(x_s, lf_repr, feat):
        pieces = []
        if use_feat_st:
            pieces.append(feat.astype(np.float32))
        if use_x:
            pieces.append(x_s.astype(np.float32))
        if use_lf:
            pieces.append(lf_repr.astype(np.float32))
        return np.concatenate(pieces, axis=1).astype(np.float32)

    U_or_tr = _build_u(X_hf_tr_s, lf_or_tr, feat_tr)
    U_or_va = _build_u(X_hf_va_s, lf_or_va, feat_va)
    U_or_te = _build_u(X_hf_te_s, lf_or_te, feat_te)

    U_st_tr = _build_u(X_hf_tr_s, lf_st_tr, feat_tr)
    U_st_va = _build_u(X_hf_va_s, lf_st_va, feat_va)
    U_st_te = _build_u(X_hf_te_s, lf_st_te, feat_te)


    print(f"[MF][u] U_or_tr={U_or_tr.shape} U_or_va={U_or_va.shape} U_or_te={U_or_te.shape}")
    print(f"[MF][u] U_st_tr={U_st_tr.shape} U_st_va={U_st_va.shape} U_st_te={U_st_te.shape}")
# standardize u separately for oracle/student (keeps previous behavior)
    su_or = StandardScaler(with_mean=True, with_std=True).fit(U_or_tr)
    U_or_tr_s = su_or.transform(U_or_tr).astype(np.float32)
    U_or_va_s = su_or.transform(U_or_va).astype(np.float32)
    U_or_te_s = su_or.transform(U_or_te).astype(np.float32)
    save_pickle(out_dir / "scaler_u_oracle.pkl", su_or)

    su_st = StandardScaler(with_mean=True, with_std=True).fit(U_st_tr)
    U_st_tr_s = su_st.transform(U_st_tr).astype(np.float32)
    U_st_va_s = su_st.transform(U_st_va).astype(np.float32)
    U_st_te_s = su_st.transform(U_st_te).astype(np.float32)
    save_pickle(out_dir / "scaler_u_student.pkl", su_st)

    # --------- delta-student: fit affine rho on paired HF-train in *target space*
    student_mode = args.student_mode.lower().strip()
    rho_fit_source = args.rho_fit_source.lower().strip()
    rho_intercept = bool(int(args.rho_intercept))

    rho_a = None
    rho_b = None
    base_hat_tr = None
    base_hat_va = None
    base_hat_te = None

    if student_mode == "delta":
        # Map LF representations into the SAME *scaled target space* as Y_tr.
        #   - FPCA:   Y_* is z_scaled (after scaler_z). z_*_repr are already z_scaled => DO NOT transform again.
        #   - SUB:    Y_* is y_sub_scaled (after scaler_hf_target). z_*_repr are y_sub_unscaled => transform once.
        if dim_reduce == "fpca":
            lf_or_tr_t = z_lf_tr_repr.astype(np.float32)
            lf_or_va_t = z_lf_va_repr.astype(np.float32)
            lf_or_te_t = z_lf_te_repr.astype(np.float32)

            lf_hat_tr_t = z_hat_tr_repr.astype(np.float32)
            lf_hat_va_t = z_hat_va_repr.astype(np.float32)
            lf_hat_te_t = z_hat_te_repr.astype(np.float32)

        elif dim_reduce == "subsample":
            import pickle as _pickle
            with open(out_dir / "scaler_hf_y_sub.pkl", "rb") as f:
                scaler_hf_target = _pickle.load(f)

            lf_or_tr_t = scaler_hf_target.transform(z_lf_tr_repr.astype(np.float32)).astype(np.float32)
            lf_or_va_t = scaler_hf_target.transform(z_lf_va_repr.astype(np.float32)).astype(np.float32)
            lf_or_te_t = scaler_hf_target.transform(z_lf_te_repr.astype(np.float32)).astype(np.float32)

            lf_hat_tr_t = scaler_hf_target.transform(z_hat_tr_repr.astype(np.float32)).astype(np.float32)
            lf_hat_va_t = scaler_hf_target.transform(z_hat_va_repr.astype(np.float32)).astype(np.float32)
            lf_hat_te_t = scaler_hf_target.transform(z_hat_te_repr.astype(np.float32)).astype(np.float32)
        else:
            raise ValueError(f"Unknown dim_reduce for delta-student: {dim_reduce}")

        lf_fit = lf_or_tr_t if rho_fit_source == "oracle" else lf_hat_tr_t
        rho_a, rho_b = fit_affine_rho(
            lf_fit, Y_tr,
            ridge=float(args.rho_ridge),
            use_intercept=rho_intercept,
        )

        np.save(out_dir / "rho_a.npy", rho_a.astype(np.float32))
        np.save(out_dir / "rho_b.npy", rho_b.astype(np.float32))

        print(
            f"[DELTA-STUDENT][rho] fit_source={rho_fit_source} intercept={rho_intercept} ridge={float(args.rho_ridge):.3g} | "
            f"rho_a mean={float(np.mean(rho_a)):.4g} std={float(np.std(rho_a)):.4g} min={float(np.min(rho_a)):.4g} max={float(np.max(rho_a)):.4g} | "
            f"rho_b mean={float(np.mean(rho_b)):.4g} std={float(np.std(rho_b)):.4g} min={float(np.min(rho_b)):.4g} max={float(np.max(rho_b)):.4g}"
        )

        # base term for MF-student: affine(rho) * LF_hat  (all in scaled target space)
        base_hat_tr = (lf_hat_tr_t * rho_a[None, :] + rho_b[None, :]).astype(np.float32)
        base_hat_va = (lf_hat_va_t * rho_a[None, :] + rho_b[None, :]).astype(np.float32)
        base_hat_te = (lf_hat_te_t * rho_a[None, :] + rho_b[None, :]).astype(np.float32)

    dbg_block_stats("z_lf_tr_repr", z_lf_tr_repr)
    dbg_block_stats("z_hat_tr_repr", z_hat_tr_repr)
    dbg_block_stats("U_or_tr", U_or_tr)
    dbg_block_stats("U_st_tr", U_st_tr)
    dbg_block_stats("X_hf_tr_s", X_hf_tr_s)
    dbg_block_stats("U_or_tr_s", U_or_tr_s)
    dbg_block_stats("U_st_tr_s", U_st_tr_s)

    kernel_struct_raw = args.kernel_struct.lower().strip()
    kernel_struct = kernel_struct_raw
    ard = bool(int(args.gp_ard))

    # Effective feature split point for block kernels (passed to mf_utils.train_svgp_per_dim)
    feat_dim_or: Optional[int] = None
    feat_dim_st: Optional[int] = None

    # Determine effective kernel struct per branch
    kernel_struct_or = kernel_struct
    kernel_struct_st = kernel_struct

    if kernel_struct_raw == "xlf_block":
        # True x|lf block kernel: u = [x, lf] and kernel = k_x(x,x') + k_lf(lf,lf')
        # This requires that u begins with x and then lf (which is how we build u under mf_u_mode='xlf').
        if not (use_x and use_lf):
            print("[WARN] kernel_struct=xlf_block but mf_u_mode is not xlf-compatible (need use_x and use_lf). Falling back to full kernel.")
            kernel_struct_or = "full"
            kernel_struct_st = "full"
        else:
            kernel_struct_or = "block"
            kernel_struct_st = "block"
            x_dim = int(X_hf_tr_s.shape[1])
            feat_dim_or = x_dim
            feat_dim_st = x_dim
            print(f"[INFO] kernel_struct=xlf_block enabled: using additive block kernel split at x_dim={x_dim} (u=[x|lf])")
    else:
        # Legacy 'block' behavior in this script: only valid when using Stage-I feature as the first block.
        if (kernel_struct == "block") and (not use_feat_or):
            kernel_struct_or = "full"
            print("[WARN] kernel_struct=block but oracle has no feature => MF-oracle forced to kernel_struct=full")
        elif kernel_struct == "block":
            feat_dim_or = int(feat_dim)

        if (kernel_struct == "block") and (not use_feat_st):
            kernel_struct_st = "full"
            print("[WARN] kernel_struct=block but student has no feature => MF-student forced to kernel_struct=full")
        elif kernel_struct == "block":
            feat_dim_st = int(feat_dim)

    csv_run_meta: Dict[str, Any] = {
        "run_id": run_id,
        "run_name": run_name,
        "timestamp": run_id,
        "dim_reduce": dim_reduce,
        "kernel": args.kernel,
        "matern_nu": float(args.matern_nu),
        "kernel_struct": args.kernel_struct,
        "gp_ard": int(args.gp_ard),
        "mf_calib": str("none"),
        "mf_calib_apply": str("both"),
        "mf_u_mode": str(args.mf_u_mode),
        "student_mode": str(student_mode),
        "rho_fit_source": str(rho_fit_source),
        "rho_intercept": int(rho_intercept),
        "rho_ridge": float(args.rho_ridge),
        "use_feature_student": int(use_feat_st),
        "use_feature_oracle": int(use_feat_or),
        "svgp_M": int(args.svgp_M),
        "svgp_steps": int(args.svgp_steps),
        "svgp_lr": float(args.svgp_lr),
        "seed": int(args.seed),
    }
    if dim_reduce == "fpca":
        csv_run_meta.update({
            "fpca_dim": int(args.fpca_dim),
            "fpca_var_ratio": float(args.fpca_var_ratio),
            "fpca_max_dim": int(args.fpca_max_dim),
            "fpca_ridge": float(args.fpca_ridge),
            "fpca_dim_effective": int(fpca_dim_effective) if fpca_dim_effective is not None else "",
            "second_scaler_z": 1,
        })
    else:
        csv_run_meta.update({
            "subsample_K": int(args.subsample_K),
            "second_scaler_z": 0,
        })

    # -------------------------
    # Stage-II: Train SVGPs
    # -------------------------
    dbg_print_stage2_inputs(
        X_hf_tr_s, X_hf_va_s, X_hf_te_s,
        U_or_tr_s, U_or_va_s, U_or_te_s,
        U_st_tr_s, U_st_va_s, U_st_te_s,
        dim_reduce=dim_reduce, fpca_dim_effective=fpca_dim_effective if "fpca_dim_effective" in globals() else None,
    )
    print(f"[DEBUG][StageII][KERNEL_CFG] HF-only : kernel_struct=full ard={ard} kernel={args.kernel} matern_nu={args.matern_nu}")
    print(f"[DEBUG][StageII][KERNEL_CFG] MF-or   : kernel_struct={kernel_struct_or} ard={ard} kernel={args.kernel} matern_nu={args.matern_nu} feat_dim={(feat_dim_or if (kernel_struct_or=='block') else None)}")
    print(f"[DEBUG][StageII][KERNEL_CFG] MF-st   : kernel_struct={kernel_struct_st} ard={ard} kernel={args.kernel} matern_nu={args.matern_nu} feat_dim={(feat_dim_st if (kernel_struct_st=='block') else None)}")

    svgp_hf = None
    if int(args.run_hf_only) == 1:
        print("[STAGE-II] Train HF-only SVGPs ...")
        svgp_hf = train_svgp_per_dim(
            Xtr=X_hf_tr_s, Ytr=Y_tr,
            device=device, inducing_M=int(args.svgp_M),
            steps=int(args.svgp_steps), lr=float(args.svgp_lr),
            ard=ard, kernel_struct="full",
            kernel_name=str(args.kernel), matern_nu=float(args.matern_nu),
            feat_dim=None, print_every=int(args.print_every),
            tag="HF", csv_logger=csv_logger, csv_run_meta=csv_run_meta,
        )
        save_svgp_bundle(out_dir, "svgp_hf_only", svgp_hf)
        dbg_print_kernel_effective(svgp_hf, "HF")

    svgp_or = None
    if int(args.run_oracle) == 1:
        print("[STAGE-II] Train MF-oracle SVGPs ...")
        svgp_or = train_svgp_per_dim(
            Xtr=U_or_tr_s, Ytr=Y_tr,
            device=device, inducing_M=int(args.svgp_M),
            steps=int(args.svgp_steps), lr=float(args.svgp_lr),
            ard=ard, kernel_struct=kernel_struct_or,
            kernel_name=str(args.kernel), matern_nu=float(args.matern_nu),
            feat_dim=(feat_dim_or if (kernel_struct_or == "block") else None),
            print_every=int(args.print_every),
            tag="OR", csv_logger=csv_logger, csv_run_meta=csv_run_meta,
        )
        save_svgp_bundle(out_dir, "svgp_mf_oracle", svgp_or)
        dbg_print_kernel_effective(svgp_or, "OR")

    svgp_st = None
    if int(args.run_student) == 1:
        print("[STAGE-II] Train MF-student SVGPs ...")

        # delta-student: learn residual in target space
        if student_mode == "delta":
            if base_hat_tr is None:
                raise RuntimeError("student_mode=delta but base_hat_tr is None")
            Y_tr_st2 = (Y_tr - base_hat_tr).astype(np.float32)
        else:
            Y_tr_st2 = Y_tr
        svgp_st = train_svgp_per_dim(
            Xtr=U_st_tr_s, Ytr=Y_tr_st2,
            device=device, inducing_M=int(args.svgp_M),
            steps=int(args.svgp_steps), lr=float(args.svgp_lr),
            ard=ard, kernel_struct=kernel_struct_st,
            kernel_name=str(args.kernel), matern_nu=float(args.matern_nu),
            feat_dim=(feat_dim_st if (kernel_struct_st == "block") else None),
            print_every=int(args.print_every),
            tag="ST", csv_logger=csv_logger, csv_run_meta=csv_run_meta,
        )
        save_svgp_bundle(out_dir, "svgp_mf_student", svgp_st)
        dbg_print_kernel_effective(svgp_st, "ST")

    # -------------------------
    # Predict VAL + TEST in target space: mean + var (target space is Y_* i.e., z_scaled if FPCA)
    # -------------------------
    R = int(Y_tr.shape[1])

    mc_lf_enabled = False  # will be set True if MC marginalization is used for MF-student predictions


    if svgp_hf is None:
        mu_hf_va = np.full((int(Y_va.shape[0]), R), np.nan, dtype=np.float32)
        var_hf_va = np.full((int(Y_va.shape[0]), R), np.nan, dtype=np.float32)
        mu_hf_te = np.full((int(Y_te.shape[0]), R), np.nan, dtype=np.float32)
        var_hf_te = np.full((int(Y_te.shape[0]), R), np.nan, dtype=np.float32)
    else:
        mu_hf_va, var_hf_va = predict_svgp_per_dim(svgp_hf, X_hf_va_s, device=device)
        mu_hf_te, var_hf_te = predict_svgp_per_dim(svgp_hf, X_hf_te_s, device=device)

    if svgp_or is None:
        mu_or_va = np.full((int(Y_va.shape[0]), R), np.nan, dtype=np.float32)
        var_or_va = np.full((int(Y_va.shape[0]), R), np.nan, dtype=np.float32)
        mu_or_te = np.full((int(Y_te.shape[0]), R), np.nan, dtype=np.float32)
        var_or_te = np.full((int(Y_te.shape[0]), R), np.nan, dtype=np.float32)
    else:
        mu_or_va, var_or_va = predict_svgp_per_dim(svgp_or, U_or_va_s, device=device)
        mu_or_te, var_or_te = predict_svgp_per_dim(svgp_or, U_or_te_s, device=device)

    if svgp_st is None:
        mu_st_va = np.full((int(Y_va.shape[0]), R), np.nan, dtype=np.float32)
        var_st_va = np.full((int(Y_va.shape[0]), R), np.nan, dtype=np.float32)
        mu_st_te = np.full((int(Y_te.shape[0]), R), np.nan, dtype=np.float32)
        var_st_te = np.full((int(Y_te.shape[0]), R), np.nan, dtype=np.float32)
    else:
        # Optional: Monte Carlo marginalization over LF uncertainty for MF-student (prediction only; SVGP training unchanged).
        do_mc = (int(getattr(args, "lf_prob", 0)) == 1) and (int(getattr(args, "mc_lf_samples", 1)) > 1)                 and (str(getattr(args, "mf_student_lf_source", "student")).lower().strip() == "student")                 and (not bool(int(getattr(args, "skip_student", 0))))
        if do_mc:
            S = int(getattr(args, "mc_lf_samples", 1))
            rng = np.random.RandomState(int(args.seed) + 777)

            def _lf_repr_from_yraw(y_raw: np.ndarray) -> np.ndarray:
                if dim_reduce == "fpca":
                    assert fpca is not None and scaler_y is not None and scaler_z is not None
                    y_n = scaler_y.transform(y_raw.astype(np.float32)).astype(np.float32)
                    z = fpca.transform(y_n).astype(np.float32)
                    z_s = scaler_z.transform(z).astype(np.float32)
                    return z_s
                else:
                    assert idx_k is not None
                    return pick_y_sub(y_raw.astype(np.float32), idx_k).astype(np.float32)

            def _lf_to_target_space(lf_repr: np.ndarray) -> np.ndarray:
                # for delta base term: map LF repr into the SAME *scaled target space* as Y_tr
                if dim_reduce == "fpca":
                    return lf_repr.astype(np.float32)  # already z_scaled
                else:
                    import pickle as _pickle
                    with open(out_dir / "scaler_hf_y_sub.pkl", "rb") as f:
                        scaler_hf_target = _pickle.load(f)
                    return scaler_hf_target.transform(lf_repr.astype(np.float32)).astype(np.float32)

            def _mc_predict_one_split(X_s: np.ndarray, feat: np.ndarray, mu_y_s: np.ndarray, var_y_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                # mu_y_s / var_y_s are in Stage-I output space (scaled y if --student_yscale=1)
                mu_list = []
                var_list = []
                for _ in range(S):
                    eps = rng.randn(*mu_y_s.shape).astype(np.float32)
                    y_samp_s = mu_y_s.astype(np.float32) + eps * np.sqrt(np.maximum(var_y_s.astype(np.float32), 1e-12))
                    y_samp_raw = _student_y_inv(y_samp_s)

                    lf_repr = _lf_repr_from_yraw(y_samp_raw)
                    U_samp = _build_u(X_s, lf_repr, feat)
                    U_samp_s = su_st.transform(U_samp).astype(np.float32)
                    mu_res, var_res = predict_svgp_per_dim(svgp_st, U_samp_s, device=device)

                    if student_mode == "delta":
                        lf_t = _lf_to_target_space(lf_repr)
                        base = (lf_t * rho_a[None, :].astype(np.float32) + rho_b[None, :].astype(np.float32)).astype(np.float32)
                        mu_res = (mu_res + base).astype(np.float32)

                    mu_list.append(mu_res.astype(np.float32))
                    var_list.append(var_res.astype(np.float32))

                mu_stack = np.stack(mu_list, axis=0)   # (S,N,R)
                var_stack = np.stack(var_list, axis=0) # (S,N,R)
                mu = np.mean(mu_stack, axis=0)
                # law of total variance: E[var] + Var[mean]
                second = np.mean(var_stack + mu_stack * mu_stack, axis=0)
                var = np.maximum(second - mu * mu, 0.0).astype(np.float32)
                return mu.astype(np.float32), var

            if (yvar_lf_va_s is None) or (yvar_lf_te_s is None):
                raise RuntimeError("--mc_lf_samples>1 requires Stage-I to provide variance (enable --lf_prob=1 and do not --skip_student).")

            mu_st_va, var_st_va = _mc_predict_one_split(X_hf_va_s, feat_va, yhat_lf_va_s, yvar_lf_va_s)
            mu_st_te, var_st_te = _mc_predict_one_split(X_hf_te_s, feat_te, yhat_lf_te_s, yvar_lf_te_s)
            mc_lf_enabled = (student_mode == "delta")
            print(f"[MC-LF] enabled: S={S} | MF predictions marginalized over LF Gaussian head uncertainty p(y_l|x) (VAL/TEST).")
        else:
            mu_st_va, var_st_va = predict_svgp_per_dim(svgp_st, U_st_va_s, device=device)
            mu_st_te, var_st_te = predict_svgp_per_dim(svgp_st, U_st_te_s, device=device)

    if student_mode == "delta":
        if base_hat_va is None or base_hat_te is None:
            raise RuntimeError("student_mode=delta but base_hat_va/base_hat_te is None")
        # If MC LF marginalization already added the base term, don't add it again.
        if not mc_lf_enabled:
            mu_st_va = (mu_st_va + base_hat_va).astype(np.float32)
            mu_st_te = (mu_st_te + base_hat_te).astype(np.float32)
        # var_st_* remains the residual GP variance (base treated deterministic unless MC enabled)

    target_rmse_hf = rmse(mu_hf_te, target_te_ref)
    target_rmse_or = rmse(mu_or_te, target_te_ref)
    target_rmse_st = rmse(mu_st_te, target_te_ref)

    # -------------------------
    # Inverse mean to full y (VAL + TEST)
    # -------------------------
    ypred_hf_va = inv_target_to_y_full_fn(mu_hf_va)
    ypred_or_va = inv_target_to_y_full_fn(mu_or_va)
    ypred_st_va = inv_target_to_y_full_fn(mu_st_va)

    ypred_hf_te = inv_target_to_y_full_fn(mu_hf_te)
    ypred_or_te = inv_target_to_y_full_fn(mu_or_te)
    ypred_st_te = inv_target_to_y_full_fn(mu_st_te)

    # -------------------------
    # Propagate variance to full y variance (VAL + TEST)
    # -------------------------
    if dim_reduce == "fpca":
        assert fpca is not None and scaler_y is not None and scaler_z is not None

        # var_* currently is in z_scaled space (after scaler_z). Convert to z_unscaled variance:
        scale2_z = (np.asarray(scaler_z.scale_, dtype=np.float32) ** 2)[None, :]  # (1, R)
        var_hf_va_z = var_hf_va.astype(np.float32) * scale2_z
        var_or_va_z = var_or_va.astype(np.float32) * scale2_z
        var_st_va_z = var_st_va.astype(np.float32) * scale2_z

        var_hf_te_z = var_hf_te.astype(np.float32) * scale2_z
        var_or_te_z = var_or_te.astype(np.float32) * scale2_z
        var_st_te_z = var_st_te.astype(np.float32) * scale2_z

        var_hf_va_y = fpca_propagate_var_to_y(var_hf_va_z, fpca, scaler_y)
        var_or_va_y = fpca_propagate_var_to_y(var_or_va_z, fpca, scaler_y)
        var_st_va_y = fpca_propagate_var_to_y(var_st_va_z, fpca, scaler_y)

        var_hf_te_y = fpca_propagate_var_to_y(var_hf_te_z, fpca, scaler_y)
        var_or_te_y = fpca_propagate_var_to_y(var_or_te_z, fpca, scaler_y)
        var_st_te_y = fpca_propagate_var_to_y(var_st_te_z, fpca, scaler_y)

    else:
        assert W_full_from_sub is not None
        import pickle as _pickle
        with open(out_dir / "scaler_hf_y_sub.pkl", "rb") as f:
            scaler_hf_target = _pickle.load(f)

        var_hf_va_y = propagate_subsample_var_to_full_y_var(var_hf_va, scaler_hf_target, W_full_from_sub)
        var_or_va_y = propagate_subsample_var_to_full_y_var(var_or_va, scaler_hf_target, W_full_from_sub)
        var_st_va_y = propagate_subsample_var_to_full_y_var(var_st_va, scaler_hf_target, W_full_from_sub)

        var_hf_te_y = propagate_subsample_var_to_full_y_var(var_hf_te, scaler_hf_target, W_full_from_sub)
        var_or_te_y = propagate_subsample_var_to_full_y_var(var_or_te, scaler_hf_target, W_full_from_sub)
        var_st_te_y = propagate_subsample_var_to_full_y_var(var_st_te, scaler_hf_target, W_full_from_sub)

    std_hf_va_raw = np.sqrt(np.maximum(var_hf_va_y, 0.0)).astype(np.float32)
    std_or_va_raw = np.sqrt(np.maximum(var_or_va_y, 0.0)).astype(np.float32)
    std_st_va_raw = np.sqrt(np.maximum(var_st_va_y, 0.0)).astype(np.float32)

    std_hf_te_raw = np.sqrt(np.maximum(var_hf_te_y, 0.0)).astype(np.float32)
    std_or_te_raw = np.sqrt(np.maximum(var_or_te_y, 0.0)).astype(np.float32)
    std_st_te_raw = np.sqrt(np.maximum(var_st_te_y, 0.0)).astype(np.float32)

    # -------------------------
    # Calibration on VAL (scale-only)
    # -------------------------
    ci_lvl = float(args.ci_level)
    do_cal = bool(int(args.ci_calibrate))

    if do_cal:
        alpha_hf = calibrate_sigma_scale(y_hf_va, ypred_hf_va, std_hf_va_raw, ci_lvl)
        alpha_or = calibrate_sigma_scale(y_hf_va, ypred_or_va, std_or_va_raw, ci_lvl)
        alpha_st = calibrate_sigma_scale(y_hf_va, ypred_st_va, std_st_va_raw, ci_lvl)
    else:
        alpha_hf = 1.0
        alpha_or = 1.0
        alpha_st = 1.0

    std_hf_va_cal = (alpha_hf * std_hf_va_raw).astype(np.float32)
    std_or_va_cal = (alpha_or * std_or_va_raw).astype(np.float32)
    std_st_va_cal = (alpha_st * std_st_va_raw).astype(np.float32)

    std_hf_te_cal = (alpha_hf * std_hf_te_raw).astype(np.float32)
    std_or_te_cal = (alpha_or * std_or_te_raw).astype(np.float32)
    std_st_te_cal = (alpha_st * std_st_te_raw).astype(np.float32)

    # -------------------------
    # Optional: save prediction/uncertainty arrays (VAL/TEST) for downstream plotting
    # -------------------------
    if int(getattr(args, "save_pred_arrays", 0)) == 1:
        pred_root = out_dir / "pred_arrays"
        pred_root.mkdir(parents=True, exist_ok=True)

        # axis
        np.save(pred_root / "axis.npy", wl.astype(np.float32))
        with open(pred_root / "axis_name.txt", "w", encoding="utf-8") as f:
            f.write("axis")

        def _rmse_per_sample(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
            # RMSE over spectrum points, per sample (N,)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            y_true = np.asarray(y_true, dtype=np.float64)
            return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=1)).astype(np.float32)

        def _save_split(split: str,
                        y_true: np.ndarray, y_lf_: np.ndarray,
                        y_pred_hf_: np.ndarray, y_pred_or_: np.ndarray, y_pred_st_: np.ndarray,
                        std_hf_raw_: np.ndarray, std_or_raw_: np.ndarray, std_st_raw_: np.ndarray,
                        std_hf_cal_: np.ndarray, std_or_cal_: np.ndarray, std_st_cal_: np.ndarray) -> None:
            d = pred_root / split
            d.mkdir(parents=True, exist_ok=True)

            np.save(d / "y_true.npy", np.asarray(y_true, dtype=np.float32))
            np.save(d / "y_lf.npy", np.asarray(y_lf_, dtype=np.float32))

            np.save(d / "y_pred__hf_only.npy", np.asarray(y_pred_hf_, dtype=np.float32))
            np.save(d / "y_pred__mf_oracle.npy", np.asarray(y_pred_or_, dtype=np.float32))
            np.save(d / "y_pred__mf_student.npy", np.asarray(y_pred_st_, dtype=np.float32))

            np.save(d / "std_raw__hf_only.npy", np.asarray(std_hf_raw_, dtype=np.float32))
            np.save(d / "std_raw__mf_oracle.npy", np.asarray(std_or_raw_, dtype=np.float32))
            np.save(d / "std_raw__mf_student.npy", np.asarray(std_st_raw_, dtype=np.float32))

            np.save(d / "std_cal__hf_only.npy", np.asarray(std_hf_cal_, dtype=np.float32))
            np.save(d / "std_cal__mf_oracle.npy", np.asarray(std_or_cal_, dtype=np.float32))
            np.save(d / "std_cal__mf_student.npy", np.asarray(std_st_cal_, dtype=np.float32))

            np.save(d / "rmse_sample__hf_only.npy", _rmse_per_sample(y_pred_hf_, y_true))
            np.save(d / "rmse_sample__mf_oracle.npy", _rmse_per_sample(y_pred_or_, y_true))
            np.save(d / "rmse_sample__mf_student.npy", _rmse_per_sample(y_pred_st_, y_true))
            np.save(d / "rmse_sample__lf.npy", _rmse_per_sample(y_lf_, y_true))

            # also save per-sample RMSE as CSV (short filenames; avoids path/filename-too-long issues)
            r_hf = _rmse_per_sample(y_pred_hf_, y_true)
            r_or = _rmse_per_sample(y_pred_or_, y_true)
            r_st = _rmse_per_sample(y_pred_st_, y_true)
            r_lf = _rmse_per_sample(y_lf_,      y_true)

            csv_path = d / "rmse_sample__summary.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["idx", "rmse_hf_only", "rmse_mf_oracle", "rmse_mf_student", "rmse_lf"])
                for i in range(int(r_hf.shape[0])):
                    w.writerow([i, float(r_hf[i]), float(r_or[i]), float(r_st[i]), float(r_lf[i])])

        _save_split(
            "val",
            y_true=y_hf_va, y_lf_=y_lf_va,
            y_pred_hf_=ypred_hf_va, y_pred_or_=ypred_or_va, y_pred_st_=ypred_st_va,
            std_hf_raw_=std_hf_va_raw, std_or_raw_=std_or_va_raw, std_st_raw_=std_st_va_raw,
            std_hf_cal_=std_hf_va_cal, std_or_cal_=std_or_va_cal, std_st_cal_=std_st_va_cal,
        )
        _save_split(
            "test",
            y_true=y_hf_te, y_lf_=y_lf_te,
            y_pred_hf_=ypred_hf_te, y_pred_or_=ypred_or_te, y_pred_st_=ypred_st_te,
            std_hf_raw_=std_hf_te_raw, std_or_raw_=std_or_te_raw, std_st_raw_=std_st_te_raw,
            std_hf_cal_=std_hf_te_cal, std_or_cal_=std_or_te_cal, std_st_cal_=std_st_te_cal,
        )
        print(f"[PRED] saved pred/unc arrays -> {pred_root}")


    # -------------------------
    # UQ metrics
    # -------------------------
    cov_val_raw_hf = ci_coverage_y(y_hf_va, ypred_hf_va, std_hf_va_raw, ci_lvl)
    cov_val_raw_or = ci_coverage_y(y_hf_va, ypred_or_va, std_or_va_raw, ci_lvl)
    cov_val_raw_st = ci_coverage_y(y_hf_va, ypred_st_va, std_st_va_raw, ci_lvl)

    cov_val_cal_hf = ci_coverage_y(y_hf_va, ypred_hf_va, std_hf_va_cal, ci_lvl)
    cov_val_cal_or = ci_coverage_y(y_hf_va, ypred_or_va, std_or_va_cal, ci_lvl)
    cov_val_cal_st = ci_coverage_y(y_hf_va, ypred_st_va, std_st_va_cal, ci_lvl)

    wid_val_raw_hf = ci_width_y(std_hf_va_raw, ci_lvl)
    wid_val_raw_or = ci_width_y(std_or_va_raw, ci_lvl)
    wid_val_raw_st = ci_width_y(std_st_va_raw, ci_lvl)

    wid_val_cal_hf = ci_width_y(std_hf_va_cal, ci_lvl)
    wid_val_cal_or = ci_width_y(std_or_va_cal, ci_lvl)
    wid_val_cal_st = ci_width_y(std_st_va_cal, ci_lvl)

    cov_test_raw_hf = ci_coverage_y(y_hf_te, ypred_hf_te, std_hf_te_raw, ci_lvl)
    cov_test_raw_or = ci_coverage_y(y_hf_te, ypred_or_te, std_or_te_raw, ci_lvl)
    cov_test_raw_st = ci_coverage_y(y_hf_te, ypred_st_te, std_st_te_raw, ci_lvl)

    cov_test_cal_hf = ci_coverage_y(y_hf_te, ypred_hf_te, std_hf_te_cal, ci_lvl)
    cov_test_cal_or = ci_coverage_y(y_hf_te, ypred_or_te, std_or_te_cal, ci_lvl)
    cov_test_cal_st = ci_coverage_y(y_hf_te, ypred_st_te, std_st_te_cal, ci_lvl)

    wid_test_raw_hf = ci_width_y(std_hf_te_raw, ci_lvl)
    wid_test_raw_or = ci_width_y(std_or_te_raw, ci_lvl)
    wid_test_raw_st = ci_width_y(std_st_te_raw, ci_lvl)

    wid_test_cal_hf = ci_width_y(std_hf_te_cal, ci_lvl)
    wid_test_cal_or = ci_width_y(std_or_te_cal, ci_lvl)
    wid_test_cal_st = ci_width_y(std_st_te_cal, ci_lvl)

    print("[UQ] alpha_hf/or/st =", alpha_hf, alpha_or, alpha_st)
    print("[UQ] VAL coverage raw  hf/or/st =", cov_val_raw_hf, cov_val_raw_or, cov_val_raw_st)
    print("[UQ] VAL coverage cal  hf/or/st =", cov_val_cal_hf, cov_val_cal_or, cov_val_cal_st)
    print("[UQ] TEST coverage raw hf/or/st =", cov_test_raw_hf, cov_test_raw_or, cov_test_raw_st)
    print("[UQ] TEST coverage cal hf/or/st =", cov_test_cal_hf, cov_test_cal_or, cov_test_cal_st)

    # -------------------------
    # Point metrics
    # -------------------------
    y_rmse_hf = rmse(ypred_hf_te, y_hf_te)
    y_rmse_or = rmse(ypred_or_te, y_hf_te)
    y_rmse_st = rmse(ypred_st_te, y_hf_te)

    # -------------------------
    # Save per-point RMSE curves (for ablation spectrum plots)
    # -------------------------
    rmse_curves_dir = out_dir / "rmse_curves"
    rmse_curves_dir.mkdir(parents=True, exist_ok=True)
    np.save(rmse_curves_dir / "axis.npy", wl.astype(np.float32))
    with open(rmse_curves_dir / "axis_name.txt", "w", encoding="utf-8") as f:
        f.write("axis")

    rmse_curve_manifest: Dict[str, Any] = {
        "axis_npy": str((rmse_curves_dir / "axis.npy").relative_to(out_dir)),
        "split": {},
    }

    def _save_curve(split_name: str, method: str, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        c = np.sqrt(np.mean((y_pred.astype(np.float64) - y_true.astype(np.float64)) ** 2, axis=0)).astype(np.float32)
        fn = rmse_curves_dir / f"rmse_{split_name}__{method}.npy"
        np.save(fn, c)
        rmse_curve_manifest["split"].setdefault(split_name, {})[method] = str(fn.relative_to(out_dir))


    # Residual RMS curves for delta-student diagnostics (used by runner to compare A0 vs A1)
    # residual := y_hf - base_hat, where base_hat is the affine-calibrated LF_hat (student/oracle depending on config)
    if student_mode == "delta" and (base_hat_va is not None) and (base_hat_te is not None):
        def _save_residual_curve(split_name: str, y_true: np.ndarray, base_hat: np.ndarray) -> None:
            # NOTE: base_hat is in Stage-II target space (R dims: FPCA z_scaled or subsample y_sub),
            # while y_true is full-spectrum y space (K dims). Convert base_hat -> full y first.
            base_hat_full = inv_target_to_y_full_fn(base_hat)
            c = np.sqrt(
                np.mean(
                    (y_true.astype(np.float64) - base_hat_full.astype(np.float64)) ** 2,
                    axis=0,
                )
            ).astype(np.float32)
            fn = rmse_curves_dir / f"rmse_{split_name}__mf_student_residual.npy"
            np.save(fn, c)
            rmse_curve_manifest["split"].setdefault(split_name, {})["mf_student_residual"] = str(fn.relative_to(out_dir))

        _save_residual_curve("val", y_hf_va, base_hat_va)
        _save_residual_curve("test", y_hf_te, base_hat_te)

    # VAL curves
    _save_curve("val", "hf_only", ypred_hf_va, y_hf_va)
    _save_curve("val", "mf_oracle", ypred_or_va, y_hf_va)
    _save_curve("val", "mf_student", ypred_st_va, y_hf_va)
    _save_curve("val", "lf", y_lf_va, y_hf_va)

    # TEST curves
    _save_curve("test", "hf_only", ypred_hf_te, y_hf_te)
    _save_curve("test", "mf_oracle", ypred_or_te, y_hf_te)
    _save_curve("test", "mf_student", ypred_st_te, y_hf_te)
    _save_curve("test", "lf", y_lf_te, y_hf_te)

    print(f"[CURVE] saved per-point RMSE curves -> {rmse_curves_dir}")

    # -------------------------
    # Additional metrics: R2 and (Gaussian) NLL/NLPD
    # -------------------------
    y_r2_val_hf = r2_score(y_hf_va, ypred_hf_va)
    y_r2_val_or = r2_score(y_hf_va, ypred_or_va)
    y_r2_val_st = r2_score(y_hf_va, ypred_st_va)

    y_r2_test_hf = r2_score(y_hf_te, ypred_hf_te)
    y_r2_test_or = r2_score(y_hf_te, ypred_or_te)
    y_r2_test_st = r2_score(y_hf_te, ypred_st_te)

    # Gaussian predictive density (element-wise independent normals).
    # NOTE: For a Gaussian, NLPD == NLL of the predictive density.
    nll_val_raw_hf = gaussian_nll(y_hf_va, ypred_hf_va, std=std_hf_va_raw)
    nll_val_raw_or = gaussian_nll(y_hf_va, ypred_or_va, std=std_or_va_raw)
    nll_val_raw_st = gaussian_nll(y_hf_va, ypred_st_va, std=std_st_va_raw)

    nll_val_cal_hf = gaussian_nll(y_hf_va, ypred_hf_va, std=std_hf_va_cal)
    nll_val_cal_or = gaussian_nll(y_hf_va, ypred_or_va, std=std_or_va_cal)
    nll_val_cal_st = gaussian_nll(y_hf_va, ypred_st_va, std=std_st_va_cal)

    nll_test_raw_hf = gaussian_nll(y_hf_te, ypred_hf_te, std=std_hf_te_raw)
    nll_test_raw_or = gaussian_nll(y_hf_te, ypred_or_te, std=std_or_te_raw)
    nll_test_raw_st = gaussian_nll(y_hf_te, ypred_st_te, std=std_st_te_raw)

    nll_test_cal_hf = gaussian_nll(y_hf_te, ypred_hf_te, std=std_hf_te_cal)
    nll_test_cal_or = gaussian_nll(y_hf_te, ypred_or_te, std=std_or_te_cal)
    nll_test_cal_st = gaussian_nll(y_hf_te, ypred_st_te, std=std_st_te_cal)

    # aliases (kept explicit for reporting)
    nlpd_val_raw_hf, nlpd_val_raw_or, nlpd_val_raw_st = nll_val_raw_hf, nll_val_raw_or, nll_val_raw_st
    nlpd_val_cal_hf, nlpd_val_cal_or, nlpd_val_cal_st = nll_val_cal_hf, nll_val_cal_or, nll_val_cal_st
    nlpd_test_raw_hf, nlpd_test_raw_or, nlpd_test_raw_st = nll_test_raw_hf, nll_test_raw_or, nll_test_raw_st
    nlpd_test_cal_hf, nlpd_test_cal_or, nlpd_test_cal_st = nll_test_cal_hf, nll_test_cal_or, nll_test_cal_st

    print("[METRIC] R2 VAL  hf/or/st =", y_r2_val_hf, y_r2_val_or, y_r2_val_st)
    print("[METRIC] R2 TEST hf/or/st =", y_r2_test_hf, y_r2_test_or, y_r2_test_st)

    print("[METRIC] NLL  VAL raw  hf/or/st =", nll_val_raw_hf, nll_val_raw_or, nll_val_raw_st)
    print("[METRIC] NLL  VAL cal  hf/or/st =", nll_val_cal_hf, nll_val_cal_or, nll_val_cal_st)
    print("[METRIC] NLL  TEST raw hf/or/st =", nll_test_raw_hf, nll_test_raw_or, nll_test_raw_st)
    print("[METRIC] NLL  TEST cal hf/or/st =", nll_test_cal_hf, nll_test_cal_or, nll_test_cal_st)


    # -------------------------
    # Plotting
    # -------------------------
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(int(args.seed))
    n_plot = int(args.n_plot)
    if n_plot < 0:
        raise ValueError("--n_plot must be >= 0")

    def choose_indices(n: int, k: int) -> np.ndarray:
        if k == 0:
            return np.array([], dtype=np.int64)
        if n <= 0:
            raise ValueError("Empty split, cannot plot")
        k = min(k, n)
        return rng.choice(np.arange(n, dtype=np.int64), size=k, replace=False)

    def plot_split(
        split_name: str,
        wl: np.ndarray,
        y_hf_gt: np.ndarray,
        y_lf: np.ndarray,
        y_pred_hf: np.ndarray,
        y_pred_or: np.ndarray,
        y_pred_st: np.ndarray,
        std_hf_raw: np.ndarray,
        std_or_raw: np.ndarray,
        std_st_raw: np.ndarray,
        std_hf_cal: np.ndarray,
        std_or_cal: np.ndarray,
        std_st_cal: np.ndarray,
        idx_pick: np.ndarray,
    ) -> dict:
        files = []
        for j, ii in enumerate(idx_pick.tolist()):
            base = f"{split_name}_case{j:02d}_idx{ii:04d}"

            f_noci = plots_dir / f"{base}__noCI.png"
            f_raw  = plots_dir / f"{base}__ciRaw.png"
            f_cal  = plots_dir / f"{base}__ciCal.png"

            title_base = (
                f"{split_name.upper()} case#{j} (i={ii}) | "
                f"dim={dim_reduce} | kernel={args.kernel}/{args.kernel_struct}"
            )

            plot_case_3curves_spectrum(
                save_path=f_noci,
                wl=wl,
                y_hf_gt=y_hf_gt[ii],
                y_lf=y_lf[ii],
                y_mf=y_pred_st[ii],
                title=title_base + " | no CI",
                mf_band=None,
            )


            if int(args.plot_ci) == 1:
                ci_bands_raw = {
                    "hf": make_ci_bands_for_curve(y_pred_hf[ii], std_hf_raw[ii], ci_lvl),
                    "or": make_ci_bands_for_curve(y_pred_or[ii], std_or_raw[ii], ci_lvl),
                    "st": make_ci_bands_for_curve(y_pred_st[ii], std_st_raw[ii], ci_lvl),
                }
                ci_bands_cal = {
                    "hf": make_ci_bands_for_curve(y_pred_hf[ii], std_hf_cal[ii], ci_lvl),
                    "or": make_ci_bands_for_curve(y_pred_or[ii], std_or_cal[ii], ci_lvl),
                    "st": make_ci_bands_for_curve(y_pred_st[ii], std_st_cal[ii], ci_lvl),
                }
                mf_band_raw = make_ci_bands_for_curve(y_pred_st[ii], std_st_raw[ii], ci_lvl)

                plot_case_3curves_spectrum(
                    save_path=f_raw,
                    wl=wl,
                    y_hf_gt=y_hf_gt[ii],
                    y_lf=y_lf[ii],
                    y_mf=y_pred_st[ii],
                    title=title_base + " | CI raw",
                    mf_band=mf_band_raw,
                )

                mf_band_cal = make_ci_bands_for_curve(y_pred_st[ii], std_st_cal[ii], ci_lvl)

                plot_case_3curves_spectrum(
                    save_path=f_cal,
                    wl=wl,
                    y_hf_gt=y_hf_gt[ii],
                    y_lf=y_lf[ii],
                    y_mf=y_pred_st[ii],
                    # title=title_base + " | CI calibrated",
                    title="MF-inferred response spectrum",
                    mf_band=mf_band_cal,
                )
                # plot_case_5curves_spectrum(
                #     wl=wl,
                #     y_hf_gt=y_hf_gt[ii],
                #     y_lf=y_lf[ii],
                #     y_pred_hf=y_pred_hf[ii],
                #     y_pred_or=y_pred_or[ii],
                #     y_pred_st=y_pred_st[ii],
                #     title=title_base + " | CI raw",
                #     save_path=f_raw,
                #     ci_bands=ci_bands_raw,
                # )

                # plot_case_5curves_spectrum(
                #     wl=wl,
                #     y_hf_gt=y_hf_gt[ii],
                #     y_lf=y_lf[ii],
                #     y_pred_hf=y_pred_hf[ii],
                #     y_pred_or=y_pred_or[ii],
                #     y_pred_st=y_pred_st[ii],
                #     title=title_base + " | CI calibrated",
                #     save_path=f_cal,
                #     ci_bands=ci_bands_cal,
                # )

            files.append({
                "split": split_name,
                "i": int(ii),
                "no_ci": str(f_noci.relative_to(out_dir)),
                "ci_raw": (str(f_raw.relative_to(out_dir)) if int(args.plot_ci) == 1 else ""),
                "ci_cal": (str(f_cal.relative_to(out_dir)) if int(args.plot_ci) == 1 else ""),
            })
        return {"split": split_name, "n_plot": int(len(idx_pick)), "files": files}

    plot_manifest = {
        "plots_dir": str(plots_dir.relative_to(out_dir)),
        "ci_level": float(ci_lvl),
        "plot_ci": int(args.plot_ci),
        "n_plot_requested": int(args.n_plot),
        "items": []
    }

    idx_val = choose_indices(n=int(y_hf_va.shape[0]), k=n_plot)
    if idx_val.size > 0:
        plot_manifest["items"].append(plot_split(
            "val",
            wl=wl,
            y_hf_gt=y_hf_va,
            y_lf=y_lf_va,
            y_pred_hf=ypred_hf_va,
            y_pred_or=ypred_or_va,
            y_pred_st=ypred_st_va,
            std_hf_raw=std_hf_va_raw,
            std_or_raw=std_or_va_raw,
            std_st_raw=std_st_va_raw,
            std_hf_cal=std_hf_va_cal,
            std_or_cal=std_or_va_cal,
            std_st_cal=std_st_va_cal,
            idx_pick=idx_val,
        ))
        print(f"[PLOT] VAL saved n={idx_val.size} -> {plots_dir}")

    idx_te = choose_indices(n=int(y_hf_te.shape[0]), k=n_plot)
    if idx_te.size > 0:
        plot_manifest["items"].append(plot_split(
            "test",
            wl=wl,
            y_hf_gt=y_hf_te,
            y_lf=y_lf_te,
            y_pred_hf=ypred_hf_te,
            y_pred_or=ypred_or_te,
            y_pred_st=ypred_st_te,
            std_hf_raw=std_hf_te_raw,
            std_or_raw=std_or_te_raw,
            std_st_raw=std_st_te_raw,
            std_hf_cal=std_hf_te_cal,
            std_or_cal=std_or_te_cal,
            std_st_cal=std_st_te_cal,
            idx_pick=idx_te,
        ))
        print(f"[PLOT] TEST saved n={idx_te.size} -> {plots_dir}")

    with open(plots_dir / "plot_manifest.json", "w", encoding="utf-8") as f:
        json.dump(plot_manifest, f, ensure_ascii=False, indent=2)

    # -------------------------
    # report.json
    # -------------------------
    report: Dict[str, Any] = {
        "run_id": run_id,
        "run_name": run_name,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "device": str(device),
        "K": int(K),
        "xdim": int(xdim),
        "dim_reduce": dim_reduce,
        "student": {
            "best_val_mse": float(stu_info["meta"]["best_val_mse"]),
            "epochs_ran": int(stu_info["meta"]["epochs_ran"]),
            "feat_dim": int(args.student_feat_dim),
            "x_scaled": True,
            "y_scaled": bool(int(args.student_yscale)),
            "y_scaler_fit_on": (str(y_fit_on) if (sy_student is not None) else ""),
            "y_scaler_fit_mode": (str(getattr(args, "student_y_scaler_fit", "mix")) if (sy_student is not None) else ""),
            "y_scaler_fit_N": (int(y_fit_n) if (sy_student is not None) else 0),
            "y_scaler_path": (str(out_dir / "student_scaler_y.pkl") if (sy_student is not None) else ""),
            "val_set": str(args.student_val_set),
            "feat_act": str(args.student_feat_act),
            **dbg_student_metrics,
        },
        "metrics": {
            "stage1_rmse": {
                "mix": {
                    "train": float(stage1_rmse_mix_tr),
                    "val": float(stage1_rmse_mix_va),
                    "test": float(stage1_rmse_mix_te),
                },
                "paired": {
                    "train": float(stage1_rmse_paired_tr),
                    "val": float(stage1_rmse_paired_va),
                    "test": float(stage1_rmse_paired_te),
                },
            },
            "target_rmse": {
                "hf_only": float(target_rmse_hf),
                "mf_oracle": float(target_rmse_or),
                "mf_student": float(target_rmse_st),
            },
            "y_rmse": {"hf_only": float(y_rmse_hf), "mf_oracle": float(y_rmse_or), "mf_student": float(y_rmse_st)},
            "y_r2": {
                "val": {"hf_only": float(y_r2_val_hf), "mf_oracle": float(y_r2_val_or), "mf_student": float(y_r2_val_st)},
                "test": {"hf_only": float(y_r2_test_hf), "mf_oracle": float(y_r2_test_or), "mf_student": float(y_r2_test_st)},
            },
            "nll": {
                "val": {
                    "raw": {"hf_only": float(nll_val_raw_hf), "mf_oracle": float(nll_val_raw_or), "mf_student": float(nll_val_raw_st)},
                    "cal": {"hf_only": float(nll_val_cal_hf), "mf_oracle": float(nll_val_cal_or), "mf_student": float(nll_val_cal_st)},
                },
                "test": {
                    "raw": {"hf_only": float(nll_test_raw_hf), "mf_oracle": float(nll_test_raw_or), "mf_student": float(nll_test_raw_st)},
                    "cal": {"hf_only": float(nll_test_cal_hf), "mf_oracle": float(nll_test_cal_or), "mf_student": float(nll_test_cal_st)},
                },
            },
            "nlpd": {
                "val": {
                    "raw": {"hf_only": float(nlpd_val_raw_hf), "mf_oracle": float(nlpd_val_raw_or), "mf_student": float(nlpd_val_raw_st)},
                    "cal": {"hf_only": float(nlpd_val_cal_hf), "mf_oracle": float(nlpd_val_cal_or), "mf_student": float(nlpd_val_cal_st)},
                },
                "test": {
                    "raw": {"hf_only": float(nlpd_test_raw_hf), "mf_oracle": float(nlpd_test_raw_or), "mf_student": float(nlpd_test_raw_st)},
                    "cal": {"hf_only": float(nlpd_test_cal_hf), "mf_oracle": float(nlpd_test_cal_or), "mf_student": float(nlpd_test_cal_st)},
                },
            },

            "fpca_diag": (
                {
                    "fpca_dim_effective": int(fpca_dim_effective) if fpca_dim_effective is not None else None,
                    "fpca_evr_sum": float(fpca_evr_sum) if fpca_evr_sum is not None else None,
                    "fpca_recon_rmse_hftr": float(fpca_recon_rmse_hftr) if fpca_recon_rmse_hftr is not None else None,
                    "fpca_recon_rmse_hfval": float(fpca_recon_rmse_hfval) if fpca_recon_rmse_hfval is not None else None,
                    "fpca_dim": int(args.fpca_dim),
                    "fpca_var_ratio": float(args.fpca_var_ratio),
                    "fpca_max_dim": int(args.fpca_max_dim),
                    "fpca_ridge": float(args.fpca_ridge),
                    "second_scaler_z": True,
                } if dim_reduce == "fpca" else {}
            ),
            "delta_student_diag": {
                "student_mode": str(student_mode),
                "rho_fit_source": str(rho_fit_source),
                "rho_intercept": bool(rho_intercept),
                "rho_ridge": float(args.rho_ridge),
                "rho_a": None if rho_a is None else {
                    "mean": float(np.mean(rho_a)),
                    "std": float(np.std(rho_a)),
                    "min": float(np.min(rho_a)),
                    "max": float(np.max(rho_a)),
                },
                "rho_b": None if rho_b is None else {
                    "mean": float(np.mean(rho_b)),
                    "std": float(np.std(rho_b)),
                    "min": float(np.min(rho_b)),
                    "max": float(np.max(rho_b)),
                },
            },

        },
        "uncertainty": {
            "ci_level": float(ci_lvl),
            "calibrate": bool(do_cal),
            "alpha": {"hf_only": float(alpha_hf), "mf_oracle": float(alpha_or), "mf_student": float(alpha_st)},
            "val": {
                "coverage_raw": {"hf_only": float(cov_val_raw_hf), "mf_oracle": float(cov_val_raw_or), "mf_student": float(cov_val_raw_st)},
                "coverage_cal": {"hf_only": float(cov_val_cal_hf), "mf_oracle": float(cov_val_cal_or), "mf_student": float(cov_val_cal_st)},
                "width_raw": {"hf_only": float(wid_val_raw_hf), "mf_oracle": float(wid_val_raw_or), "mf_student": float(wid_val_raw_st)},
                "width_cal": {"hf_only": float(wid_val_cal_hf), "mf_oracle": float(wid_val_cal_or), "mf_student": float(wid_val_cal_st)},
            },
            "test": {
                "coverage_raw": {"hf_only": float(cov_test_raw_hf), "mf_oracle": float(cov_test_raw_or), "mf_student": float(cov_test_raw_st)},
                "coverage_cal": {"hf_only": float(cov_test_cal_hf), "mf_oracle": float(cov_test_cal_or), "mf_student": float(cov_test_cal_st)},
                "width_raw": {"hf_only": float(wid_test_raw_hf), "mf_oracle": float(wid_test_raw_or), "mf_student": float(wid_test_raw_st)},
                "width_cal": {"hf_only": float(wid_test_cal_hf), "mf_oracle": float(wid_test_cal_or), "mf_student": float(wid_test_cal_st)},
            },
        },
        "rmse_curves": rmse_curve_manifest,
        "plots": plot_manifest,
    }

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n[DONE] metrics:")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))

    # -------------------------
    # results.csv
    # -------------------------
    res_row = {
        "run_id": run_id,
        "run_name": run_name,
        "timestamp": run_id,
        "data_dir": str(data_dir),
        "dim_reduce": dim_reduce,
        "kernel": str(args.kernel),
        "matern_nu": float(args.matern_nu),
        "kernel_struct": str(args.kernel_struct),
        "gp_ard": int(args.gp_ard),
        "mf_calib": str("none"),
        "mf_calib_apply": str("both"),
        "mf_u_mode": str(args.mf_u_mode),
        "use_feature_student": 0,
        "use_feature_oracle": 0,
        "seed": int(args.seed),
        "K": int(K),
        "xdim": int(xdim),
        "n_hf_train": int(x_hf_tr.shape[0]),
        "n_hf_val": int(x_hf_va.shape[0]),
        "n_hf_test": int(x_hf_te.shape[0]),
        "n_lf_train_total": int(X_lf_tr_all.shape[0]),
        "n_lf_val_total": int(X_lf_va_all.shape[0]),

        # Stage-I metrics (added)
        "stage1_rmse_mix_tr": float(stage1_rmse_mix_tr),
        "stage1_rmse_mix_va": float(stage1_rmse_mix_va),
        "stage1_rmse_mix_te": float(stage1_rmse_mix_te),
        "stage1_rmse_paired_tr": float(stage1_rmse_paired_tr),
        "stage1_rmse_paired_va": float(stage1_rmse_paired_va),
        "stage1_rmse_paired_te": float(stage1_rmse_paired_te),

        "student_best_val_mse": float(stu_info["meta"]["best_val_mse"]),
        "student_epochs_ran": int(stu_info["meta"]["epochs_ran"]),
        "student_y_scaled": int(bool(int(args.student_yscale))),
        "student_val_set": str(args.student_val_set),
        "student_mse_on_hfval": float(dbg_student_metrics["student_mse_on_hfval"]),
        "student_rmse_on_hfval": float(dbg_student_metrics["student_rmse_on_hfval"]),
        "student_mse_on_hftest": float(dbg_student_metrics["student_mse_on_hftest"]),
        "student_rmse_on_hftest": float(dbg_student_metrics["student_rmse_on_hftest"]),

        "target_rmse_hf": float(target_rmse_hf),
        "target_rmse_or": float(target_rmse_or),
        "target_rmse_st": float(target_rmse_st),

        "y_rmse_hf": float(y_rmse_hf),
        "y_rmse_or": float(y_rmse_or),
        "y_rmse_st": float(y_rmse_st),

        "y_r2_val_hf": float(y_r2_val_hf),
        "y_r2_val_or": float(y_r2_val_or),
        "y_r2_val_st": float(y_r2_val_st),
        "y_r2_test_hf": float(y_r2_test_hf),
        "y_r2_test_or": float(y_r2_test_or),
        "y_r2_test_st": float(y_r2_test_st),

        "nll_val_raw_hf": float(nll_val_raw_hf),
        "nll_val_raw_or": float(nll_val_raw_or),
        "nll_val_raw_st": float(nll_val_raw_st),
        "nll_val_cal_hf": float(nll_val_cal_hf),
        "nll_val_cal_or": float(nll_val_cal_or),
        "nll_val_cal_st": float(nll_val_cal_st),
        "nll_test_raw_hf": float(nll_test_raw_hf),
        "nll_test_raw_or": float(nll_test_raw_or),
        "nll_test_raw_st": float(nll_test_raw_st),
        "nll_test_cal_hf": float(nll_test_cal_hf),
        "nll_test_cal_or": float(nll_test_cal_or),
        "nll_test_cal_st": float(nll_test_cal_st),

        "nlpd_val_raw_hf": float(nlpd_val_raw_hf),
        "nlpd_val_raw_or": float(nlpd_val_raw_or),
        "nlpd_val_raw_st": float(nlpd_val_raw_st),
        "nlpd_val_cal_hf": float(nlpd_val_cal_hf),
        "nlpd_val_cal_or": float(nlpd_val_cal_or),
        "nlpd_val_cal_st": float(nlpd_val_cal_st),
        "nlpd_test_raw_hf": float(nlpd_test_raw_hf),
        "nlpd_test_raw_or": float(nlpd_test_raw_or),
        "nlpd_test_raw_st": float(nlpd_test_raw_st),
        "nlpd_test_cal_hf": float(nlpd_test_cal_hf),
        "nlpd_test_cal_or": float(nlpd_test_cal_or),
        "nlpd_test_cal_st": float(nlpd_test_cal_st),


        "ci_level": float(ci_lvl),
        "ci_calibrate": int(do_cal),
        "ci_alpha_hf": float(alpha_hf),
        "ci_alpha_or": float(alpha_or),
        "ci_alpha_st": float(alpha_st),

        "ci_cov_val_raw_hf": float(cov_val_raw_hf),
        "ci_cov_val_raw_or": float(cov_val_raw_or),
        "ci_cov_val_raw_st": float(cov_val_raw_st),
        "ci_cov_val_cal_hf": float(cov_val_cal_hf),
        "ci_cov_val_cal_or": float(cov_val_cal_or),
        "ci_cov_val_cal_st": float(cov_val_cal_st),

        "ci_wid_val_raw_hf": float(wid_val_raw_hf),
        "ci_wid_val_raw_or": float(wid_val_raw_or),
        "ci_wid_val_raw_st": float(wid_val_raw_st),
        "ci_wid_val_cal_hf": float(wid_val_cal_hf),
        "ci_wid_val_cal_or": float(wid_val_cal_or),
        "ci_wid_val_cal_st": float(wid_val_cal_st),

        "ci_cov_test_raw_hf": float(cov_test_raw_hf),
        "ci_cov_test_raw_or": float(cov_test_raw_or),
        "ci_cov_test_raw_st": float(cov_test_raw_st),
        "ci_cov_test_cal_hf": float(cov_test_cal_hf),
        "ci_cov_test_cal_or": float(cov_test_cal_or),
        "ci_cov_test_cal_st": float(cov_test_cal_st),

        "ci_wid_test_raw_hf": float(wid_test_raw_hf),
        "ci_wid_test_raw_or": float(wid_test_raw_or),
        "ci_wid_test_raw_st": float(wid_test_raw_st),
        "ci_wid_test_cal_hf": float(wid_test_cal_hf),
        "ci_wid_test_cal_or": float(wid_test_cal_or),
        "ci_wid_test_cal_st": float(wid_test_cal_st),
    }

    if dim_reduce == "fpca":
        res_row.update({
            "fpca_dim": int(args.fpca_dim),
            "fpca_var_ratio": float(args.fpca_var_ratio),
            "fpca_max_dim": int(args.fpca_max_dim),
            "fpca_ridge": float(args.fpca_ridge),
            "fpca_dim_effective": int(fpca_dim_effective) if fpca_dim_effective is not None else "",
            "fpca_evr_sum": float(fpca_evr_sum) if fpca_evr_sum is not None else "",
            "fpca_recon_rmse_hftr": float(fpca_recon_rmse_hftr) if fpca_recon_rmse_hftr is not None else "",
            "fpca_recon_rmse_hfval": float(fpca_recon_rmse_hfval) if fpca_recon_rmse_hfval is not None else "",
            "second_scaler_z": 1,
        })
    else:
        res_row.update({
            "subsample_K": int(args.subsample_K),
            "second_scaler_z": 0,
        })

    csv_logger.write_result(res_row)
    csv_logger.close()


if __name__ == "__main__":
    main()
