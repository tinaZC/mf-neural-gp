from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import pywt
except Exception:
    pywt = None

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


def ensure_2d_float32(x, name="array"):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={x.shape}")
    return x


def _safe_int(value, default=0):
    try:
        if value is None:
            return int(default)
        if isinstance(value, str) and value.strip() == "":
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and value.strip() == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _resolve_auto_cap(args, explicit_name: str, max_name: str, fallback_name: str = "fpca_max_dim", default_cap: int = 64):
    explicit = _safe_int(getattr(args, explicit_name, 0), 0)
    cap = _safe_int(getattr(args, max_name, 0), 0)
    if cap <= 0:
        cap = _safe_int(getattr(args, fallback_name, default_cap), default_cap)
    cap = max(1, int(cap))
    if explicit > 0:
        explicit = min(int(explicit), cap)
    return explicit, cap


class FPCA:
    def __init__(self, n_components: int = 0, var_ratio: float = 0.999, max_components: int = 64, ridge: float = 1e-8):
        self.n_components = int(n_components)
        self.var_ratio = float(var_ratio)
        self.max_components = int(max_components)
        self.ridge = float(ridge)
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None

    def fit(self, X):
        X = ensure_2d_float32(X, "FPCA.fit(X)")
        n, K = X.shape
        self.mean_ = X.mean(axis=0).astype(np.float32)
        Xc = (X - self.mean_).astype(np.float32)
        C = (Xc.T @ Xc) / float(max(1, n - 1))
        if self.ridge > 0:
            C += self.ridge * np.eye(K, dtype=np.float32)
        eigvals, eigvecs = np.linalg.eigh(C.astype(np.float64))
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(np.real(eigvals[order]), 0.0)
        eigvecs = np.real(eigvecs[:, order]).astype(np.float32)
        total = float(eigvals.sum())
        rank_cap = min(K, max(1, n - 1))
        if self.max_components > 0:
            rank_cap = min(rank_cap, self.max_components)
        if self.n_components > 0:
            R = min(self.n_components, rank_cap)
        else:
            evr = eigvals / (total + 1e-12)
            cum = np.cumsum(evr)
            R = int(np.searchsorted(cum, self.var_ratio) + 1)
            R = max(1, min(R, rank_cap))
        self.components_ = eigvecs[:, :R].T.copy()
        self.n_components_ = R
        self.explained_variance_ratio_ = (eigvals[:R] / (total + 1e-12)).astype(np.float32)
        return self

    def transform(self, X):
        X = ensure_2d_float32(X, "FPCA.transform(X)")
        return ((X - self.mean_) @ self.components_.T).astype(np.float32)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z):
        Z = ensure_2d_float32(Z, "FPCA.inverse_transform(Z)")
        return (Z @ self.components_ + self.mean_).astype(np.float32)


class BaseReducer:
    method_name = "base"

    def fit(self, Y: np.ndarray):
        raise NotImplementedError

    def transform(self, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, Y: np.ndarray) -> np.ndarray:
        self.fit(Y)
        return self.transform(Y)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def propagate_var_to_y(self, var_z: np.ndarray, z_mean: Optional[np.ndarray] = None) -> np.ndarray:
        var_z = ensure_2d_float32(var_z, f"{self.method_name}.propagate_var_to_y(var_z)")
        if z_mean is None:
            z_mean = np.zeros_like(var_z, dtype=np.float32)
        z_mean = ensure_2d_float32(z_mean, f"{self.method_name}.propagate_var_to_y(z_mean)")
        n, r = var_z.shape
        y0 = self.inverse_transform(z_mean)
        k = y0.shape[1]
        out = np.zeros((n, k), dtype=np.float32)
        eps_base = 1e-3
        for i in range(n):
            z = z_mean[i:i+1].astype(np.float32)
            J2 = np.zeros((r, k), dtype=np.float32)
            for j in range(r):
                dz = np.zeros_like(z)
                step = eps_base * max(1.0, float(abs(z[0, j])))
                dz[0, j] = step
                yp = self.inverse_transform(z + dz)
                ym = self.inverse_transform(z - dz)
                g = ((yp - ym) / max(2.0 * step, 1e-8)).reshape(-1)
                J2[j] = np.square(g).astype(np.float32)
            out[i] = var_z[i] @ J2
        return np.maximum(out, 0.0).astype(np.float32)


class FPCAReducer(BaseReducer):
    method_name = "fpca"

    def __init__(self, n_components=0, var_ratio=0.999, max_components=64, ridge=1e-8):
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        self.fpca = FPCA(n_components=n_components, var_ratio=var_ratio, max_components=max_components, ridge=ridge)
        self.latent_dim_ = None
        self.explained_variance_ratio_sum_ = None

    def fit(self, Y):
        Y = ensure_2d_float32(Y, "FPCAReducer.fit(Y)")
        Yn = self.scaler_y.fit_transform(Y).astype(np.float32)
        self.fpca.fit(Yn)
        self.latent_dim_ = int(self.fpca.n_components_)
        self.explained_variance_ratio_sum_ = float(np.sum(self.fpca.explained_variance_ratio_))
        return self

    def transform(self, Y):
        Y = ensure_2d_float32(Y, "FPCAReducer.transform(Y)")
        Yn = self.scaler_y.transform(Y).astype(np.float32)
        return self.fpca.transform(Yn).astype(np.float32)

    def inverse_transform(self, Z):
        Z = ensure_2d_float32(Z, "FPCAReducer.inverse_transform(Z)")
        Yn = self.fpca.inverse_transform(Z).astype(np.float32)
        return self.scaler_y.inverse_transform(Yn).astype(np.float32)

    def propagate_var_to_y(self, var_z, z_mean=None):
        comp2 = np.square(self.fpca.components_.astype(np.float32))
        var_yn = ensure_2d_float32(var_z, "FPCAReducer.var_z") @ comp2
        scale2 = np.square(np.asarray(self.scaler_y.scale_, dtype=np.float32))[None, :]
        return np.maximum(var_yn * scale2, 0.0).astype(np.float32)


class PhysicsGuidedFPCAReducer(BaseReducer):
    method_name = "pgfpca"

    def __init__(self, axis: np.ndarray, n_components=0, var_ratio=0.999, max_components=64, ridge=1e-8,
                 alpha_grad=0.1, beta_curv=0.01):
        self.axis = np.asarray(axis, dtype=np.float32).reshape(-1)
        self.alpha_grad = float(alpha_grad)
        self.beta_curv = float(beta_curv)
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        self.fpca = FPCA(n_components=n_components, var_ratio=var_ratio, max_components=max_components, ridge=ridge)
        self.latent_dim_ = None
        self.explained_variance_ratio_sum_ = None
        self.K_ = None

    def _augment(self, Yn):
        dy = np.gradient(Yn, self.axis, axis=1).astype(np.float32)
        d2y = np.gradient(dy, self.axis, axis=1).astype(np.float32)
        parts = [Yn]
        if self.alpha_grad > 0:
            parts.append(np.sqrt(self.alpha_grad).astype(np.float32) * dy if isinstance(np.sqrt(self.alpha_grad), np.ndarray) else np.sqrt(self.alpha_grad) * dy)
        if self.beta_curv > 0:
            parts.append(np.sqrt(self.beta_curv) * d2y)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit(self, Y):
        Y = ensure_2d_float32(Y, "PhysicsGuidedFPCAReducer.fit(Y)")
        self.K_ = int(Y.shape[1])
        Yn = self.scaler_y.fit_transform(Y).astype(np.float32)
        X = self._augment(Yn)
        self.fpca.fit(X)
        self.latent_dim_ = int(self.fpca.n_components_)
        self.explained_variance_ratio_sum_ = float(np.sum(self.fpca.explained_variance_ratio_))
        return self

    def transform(self, Y):
        Y = ensure_2d_float32(Y, "PhysicsGuidedFPCAReducer.transform(Y)")
        Yn = self.scaler_y.transform(Y).astype(np.float32)
        X = self._augment(Yn)
        return self.fpca.transform(X).astype(np.float32)

    def inverse_transform(self, Z):
        Z = ensure_2d_float32(Z, "PhysicsGuidedFPCAReducer.inverse_transform(Z)")
        Xhat = self.fpca.inverse_transform(Z).astype(np.float32)
        Ynhat = Xhat[:, :self.K_]
        return self.scaler_y.inverse_transform(Ynhat).astype(np.float32)


class FPCABTWCore:
    def __init__(self, latent_dim, var_ratio=0.999, max_components=64, wavelet="db4", level=None, global_ratio=0.7, threshold_rel=0.01, max_total_dim=None):
        if pywt is None:
            raise ImportError("pywt is required for BTW reducer")
        self.latent_dim = int(latent_dim)
        self.var_ratio = float(var_ratio)
        self.max_components = int(max_components)
        self.wavelet = str(wavelet)
        self.level = None if level in (None, "None", "") else int(level)
        self.global_ratio = float(global_ratio)
        self.threshold_rel = float(threshold_rel)
        self.max_total_dim = None if max_total_dim in (None, "", 0, "0") else int(max_total_dim)
        self.fpca_global = None
        self.fpca_local = None
        self.coeff_slices_ = None
        self.coeff_arr_shape_ = None
        self.signal_len_ = None
        self.q_global_ = None
        self.q_local_ = None
        self.total_dim_ = None

    def _coeff_array(self, y):
        coeffs = pywt.wavedec(y, self.wavelet, level=self.level, mode="symmetric")
        arr, slices = pywt.coeffs_to_array(coeffs)
        if self.coeff_slices_ is None:
            self.coeff_slices_ = slices
            self.coeff_arr_shape_ = arr.shape
        return arr.astype(np.float32)

    def _array_to_signal(self, arr):
        coeffs = pywt.array_to_coeffs(arr.reshape(self.coeff_arr_shape_), self.coeff_slices_, output_format="wavedec")
        rec = pywt.waverec(coeffs, self.wavelet, mode="symmetric")
        return rec[: self.signal_len_].astype(np.float32)

    def _resolve_total_dim(self, Y):
        rank_cap = min(int(Y.shape[1]), max(1, int(Y.shape[0]) - 1))
        if self.max_components > 0:
            rank_cap = min(rank_cap, int(self.max_components))
        if self.max_total_dim is not None and self.max_total_dim > 0:
            rank_cap = min(rank_cap, int(self.max_total_dim))
        rank_cap = max(1, int(rank_cap))
        if self.latent_dim > 0:
            return min(int(self.latent_dim), rank_cap)
        auto_fpca = FPCA(n_components=0, var_ratio=self.var_ratio, max_components=rank_cap)
        auto_fpca.fit(Y)
        return max(1, min(int(auto_fpca.n_components_), rank_cap))

    def fit(self, Y):
        Y = ensure_2d_float32(Y, "FPCABTWCore.fit(Y)")
        self.signal_len_ = Y.shape[1]
        total_dim = self._resolve_total_dim(Y)
        q_global = max(1, min(total_dim - 1, int(round(total_dim * self.global_ratio)))) if total_dim > 1 else 1
        q_local = max(0, total_dim - q_global)
        if q_local <= 0:
            q_global = 1
            q_local = 0
            total_dim = 1
        self.total_dim_ = int(total_dim)
        self.q_global_ = int(q_global)
        self.q_local_ = int(q_local)
        self.fpca_global = FPCA(n_components=self.q_global_, var_ratio=self.var_ratio, max_components=self.q_global_)
        Zg = self.fpca_global.fit_transform(Y)
        Yg = self.fpca_global.inverse_transform(Zg)
        R = (Y - Yg).astype(np.float32)
        if self.q_local_ > 0:
            coeff_mat = []
            for r in R:
                arr = self._coeff_array(r)
                thr = self.threshold_rel * max(float(np.max(np.abs(arr))), 1e-8)
                arr = pywt.threshold(arr, thr, mode="soft")
                coeff_mat.append(arr.astype(np.float32))
            coeff_mat = np.asarray(coeff_mat, dtype=np.float32)
            self.fpca_local = FPCA(n_components=self.q_local_, var_ratio=self.var_ratio, max_components=self.q_local_)
            self.fpca_local.fit(coeff_mat)
        else:
            self.fpca_local = None
        return self

    def transform(self, Y):
        Y = ensure_2d_float32(Y, "FPCABTWCore.transform(Y)")
        Zg = self.fpca_global.transform(Y)
        if self.q_local_ <= 0 or self.fpca_local is None:
            return Zg.astype(np.float32)
        Yg = self.fpca_global.inverse_transform(Zg)
        R = (Y - Yg).astype(np.float32)
        coeff_mat = []
        for r in R:
            arr = self._coeff_array(r)
            thr = self.threshold_rel * max(float(np.max(np.abs(arr))), 1e-8)
            arr = pywt.threshold(arr, thr, mode="soft")
            coeff_mat.append(arr.astype(np.float32))
        Zl = self.fpca_local.transform(np.asarray(coeff_mat, dtype=np.float32))
        return np.concatenate([Zg, Zl], axis=1).astype(np.float32)

    def inverse_transform(self, Z):
        Z = ensure_2d_float32(Z, "FPCABTWCore.inverse_transform(Z)")
        Zg = Z[:, :self.q_global_]
        Yg = self.fpca_global.inverse_transform(Zg)
        if self.q_local_ <= 0 or self.fpca_local is None:
            return Yg.astype(np.float32)
        Zl = Z[:, self.q_global_: self.q_global_ + self.q_local_]
        coeff_hat = self.fpca_local.inverse_transform(Zl)
        Yl = np.stack([self._array_to_signal(c) for c in coeff_hat], axis=0).astype(np.float32)
        return (Yg + Yl).astype(np.float32)


class BTWReducer(BaseReducer):
    method_name = "btw"

    def __init__(self, latent_dim=0, var_ratio=0.999, max_components=64, ridge=1e-8, wavelet="db4", level=3, global_ratio=0.7, threshold_rel=0.01, max_total_dim=None):
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        self.core = FPCABTWCore(latent_dim=latent_dim, var_ratio=var_ratio, max_components=max_components, wavelet=wavelet, level=level, global_ratio=global_ratio, threshold_rel=threshold_rel, max_total_dim=max_total_dim)
        self.latent_dim_ = None
        self.explained_variance_ratio_sum_ = None

    def fit(self, Y):
        Y = ensure_2d_float32(Y, "BTWReducer.fit(Y)")
        Yn = self.scaler_y.fit_transform(Y).astype(np.float32)
        self.core.fit(Yn)
        self.latent_dim_ = int(self.core.q_global_ + self.core.q_local_)
        self.explained_variance_ratio_sum_ = None
        return self

    def transform(self, Y):
        Y = ensure_2d_float32(Y, "BTWReducer.transform(Y)")
        Yn = self.scaler_y.transform(Y).astype(np.float32)
        return self.core.transform(Yn).astype(np.float32)

    def inverse_transform(self, Z):
        Yn = self.core.inverse_transform(Z).astype(np.float32)
        return self.scaler_y.inverse_transform(Yn).astype(np.float32)


class _FAENet(nn.Module):
    def __init__(self, M, proj_dim, latent_dim, basis_dim, hidden_dim, B):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(proj_dim, M, dtype=torch.float32) * 0.02)
        self.encoder = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, basis_dim),
        )
        self.register_buffer("B", torch.tensor(B, dtype=torch.float32))

    def forward(self, y):
        h = torch.einsum("bm,pm->bp", y, self.proj)
        z = self.encoder(h)
        a = self.decoder(z)
        yhat = a @ self.B.T
        return z, yhat, a


def make_rbf_basis(axis: np.ndarray, n_basis: int) -> np.ndarray:
    x = np.asarray(axis, dtype=np.float32).reshape(-1)
    centers = np.linspace(float(x.min()), float(x.max()), int(n_basis), dtype=np.float32)
    if len(centers) > 1:
        sigma = float(np.median(np.diff(centers))) * 1.5
    else:
        sigma = 1.0
    B = np.exp(-0.5 * ((x[:, None] - centers[None, :]) / max(sigma, 1e-6)) ** 2).astype(np.float32)
    B = B / np.maximum(B.sum(axis=1, keepdims=True), 1e-8)
    return B


class FunctionalAEReducer(BaseReducer):
    method_name = "fae"

    def __init__(self, axis: np.ndarray, latent_dim=0, max_latent_dim=16, var_ratio=0.999, proj_dim=64, basis_dim=64, hidden_dim=128,
                 epochs=250, lr=1e-3, batch_size=64, lambda_z=1e-4, lambda_smooth=1e-4, device="cpu"):
        if torch is None:
            raise ImportError("torch is required for FAE reducer")
        self.axis = np.asarray(axis, dtype=np.float32).reshape(-1)
        self.latent_dim = int(latent_dim)
        self.max_latent_dim = max(1, int(max_latent_dim))
        self.var_ratio = float(var_ratio)
        self.proj_dim = int(proj_dim)
        self.basis_dim = int(basis_dim)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.lambda_z = float(lambda_z)
        self.lambda_smooth = float(lambda_smooth)
        self.device = str(device)
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        self.net = None
        self.train_latent_dim_ = None
        self.latent_dim_ = None
        self.explained_variance_ratio_sum_ = None
        self._R = None
        self.active_idx_ = None
        self.latent_var_ = None

    def _make_smooth_penalty(self):
        B = make_rbf_basis(self.axis, self.basis_dim)
        d2 = np.diff(B, n=2, axis=0)
        R = d2.T @ d2
        return B, R.astype(np.float32)

    def _select_active_dims(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        train_dim = int(Z.shape[1])
        var = np.var(Z, axis=0).astype(np.float32)
        self.latent_var_ = var
        if self.latent_dim > 0:
            R = max(1, min(int(self.latent_dim), train_dim))
            active = np.arange(R, dtype=np.int64)
            evr_sum = float(var[:R].sum() / (var.sum() + 1e-12))
            return active, R, evr_sum
        order = np.argsort(var)[::-1]
        sorted_var = var[order]
        total = float(sorted_var.sum())
        if total <= 1e-12:
            R = 1
        else:
            cum = np.cumsum(sorted_var / (total + 1e-12))
            R = int(np.searchsorted(cum, self.var_ratio) + 1)
        R = max(1, min(R, train_dim, self.max_latent_dim))
        active = order[:R].astype(np.int64)
        evr_sum = float(sorted_var[:R].sum() / (total + 1e-12)) if total > 1e-12 else 1.0
        return active, R, evr_sum

    def fit(self, Y):
        Y = ensure_2d_float32(Y, "FunctionalAEReducer.fit(Y)")
        Yn = self.scaler_y.fit_transform(Y).astype(np.float32)
        B, R = self._make_smooth_penalty()
        self._R = torch.tensor(R, dtype=torch.float32, device=self.device)
        train_latent_dim = max(1, min(int(self.latent_dim) if self.latent_dim > 0 else int(self.max_latent_dim), int(self.max_latent_dim)))
        self.train_latent_dim_ = int(train_latent_dim)
        self.net = _FAENet(Y.shape[1], self.proj_dim, self.train_latent_dim_, self.basis_dim, self.hidden_dim, B).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        X = torch.tensor(Yn, dtype=torch.float32, device=self.device)
        n = X.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i+self.batch_size]
                xb = X[idx]
                z, yhat, a = self.net(xb)
                rec = ((xb - yhat) ** 2).mean()
                dz = (z ** 2).mean()
                smooth = torch.einsum("bk,kl,bl->b", a, self._R, a).mean()
                loss = rec + self.lambda_z * dz + self.lambda_smooth * smooth
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            z_all, _, _ = self.net(X)
        Z = z_all.detach().cpu().numpy().astype(np.float32)
        active_idx, eff_dim, evr_sum = self._select_active_dims(Z)
        self.active_idx_ = np.asarray(active_idx, dtype=np.int64)
        self.latent_dim_ = int(eff_dim)
        self.explained_variance_ratio_sum_ = float(evr_sum)
        return self

    def transform(self, Y):
        Y = ensure_2d_float32(Y, "FunctionalAEReducer.transform(Y)")
        Yn = self.scaler_y.transform(Y).astype(np.float32)
        with torch.no_grad():
            z, _, _ = self.net(torch.tensor(Yn, dtype=torch.float32, device=self.device))
        z_np = z.detach().cpu().numpy().astype(np.float32)
        return z_np[:, self.active_idx_].astype(np.float32)

    def inverse_transform(self, Z):
        Z = ensure_2d_float32(Z, "FunctionalAEReducer.inverse_transform(Z)")
        full = np.zeros((Z.shape[0], int(self.train_latent_dim_)), dtype=np.float32)
        full[:, self.active_idx_] = Z.astype(np.float32)
        zt = torch.tensor(full, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.net.decoder(zt)
            yhat = a @ self.net.B.T
        Yn = yhat.detach().cpu().numpy().astype(np.float32)
        return self.scaler_y.inverse_transform(Yn).astype(np.float32)


class ElasticShiftReducer(BaseReducer):
    method_name = "elastic"

    def __init__(self, axis: np.ndarray, amp_dim=0, max_components=64, shift_max_frac=0.08, max_total_dim=None, var_ratio=0.999):
        self.axis = np.asarray(axis, dtype=np.float32).reshape(-1)
        self.amp_dim = int(amp_dim)
        self.max_components = int(max_components)
        self.max_total_dim = None if max_total_dim in (None, "", 0, "0") else int(max_total_dim)
        self.var_ratio = float(var_ratio)
        self.shift_max_frac = float(shift_max_frac)
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        self.fpca_amp = None
        self.template_ = None
        self.shift_scale_ = None
        self.latent_dim_ = None
        self.explained_variance_ratio_sum_ = None
        self.amp_dim_effective_ = None

    def _shift_curve(self, y, shift):
        x = self.axis
        xq = np.clip(x - float(shift), float(x.min()), float(x.max()))
        return np.interp(xq, x, y).astype(np.float32)

    def _estimate_shift(self, y, template):
        span = float(self.axis.max() - self.axis.min())
        max_shift = self.shift_max_frac * span
        grid = np.linspace(-max_shift, max_shift, 33, dtype=np.float32)
        best_s = 0.0
        best_obj = -1e30
        tc = template - template.mean()
        for s in grid:
            ys = self._shift_curve(y, s)
            obj = float(np.dot(ys - ys.mean(), tc))
            if obj > best_obj:
                best_obj = obj
                best_s = float(s)
        return best_s

    def _resolve_amp_cap(self, aligned):
        rank_cap = min(int(aligned.shape[1]), max(1, int(aligned.shape[0]) - 1))
        if self.max_components > 0:
            rank_cap = min(rank_cap, int(self.max_components))
        if self.max_total_dim is not None and self.max_total_dim > 0:
            rank_cap = min(rank_cap, max(1, int(self.max_total_dim) - 1))
        return max(1, int(rank_cap))

    def fit(self, Y):
        Y = ensure_2d_float32(Y, "ElasticShiftReducer.fit(Y)")
        Yn = self.scaler_y.fit_transform(Y).astype(np.float32)
        template = Yn.mean(axis=0).astype(np.float32)
        shifts = []
        aligned = []
        for y in Yn:
            s = self._estimate_shift(y, template)
            shifts.append(s)
            aligned.append(self._shift_curve(y, s))
        aligned = np.asarray(aligned, dtype=np.float32)
        self.template_ = aligned.mean(axis=0).astype(np.float32)
        self.shift_scale_ = max(float(np.std(shifts)), 1e-6)
        amp_cap = self._resolve_amp_cap(aligned)
        n_comp = min(int(self.amp_dim), amp_cap) if self.amp_dim > 0 else 0
        self.fpca_amp = FPCA(n_components=n_comp, var_ratio=self.var_ratio, max_components=amp_cap)
        self.fpca_amp.fit(aligned)
        self.amp_dim_effective_ = int(self.fpca_amp.n_components_)
        self.latent_dim_ = int(self.amp_dim_effective_) + 1
        self.explained_variance_ratio_sum_ = float(np.sum(self.fpca_amp.explained_variance_ratio_))
        return self

    def transform(self, Y):
        Y = ensure_2d_float32(Y, "ElasticShiftReducer.transform(Y)")
        Yn = self.scaler_y.transform(Y).astype(np.float32)
        shifts = []
        aligned = []
        for y in Yn:
            s = self._estimate_shift(y, self.template_)
            shifts.append(s)
            aligned.append(self._shift_curve(y, s))
        aligned = np.asarray(aligned, dtype=np.float32)
        Za = self.fpca_amp.transform(aligned).astype(np.float32)
        zs = (np.asarray(shifts, dtype=np.float32)[:, None] / self.shift_scale_).astype(np.float32)
        return np.concatenate([Za, zs], axis=1).astype(np.float32)

    def inverse_transform(self, Z):
        Z = ensure_2d_float32(Z, "ElasticShiftReducer.inverse_transform(Z)")
        Za = Z[:, :-1]
        shifts = Z[:, -1] * self.shift_scale_
        aligned = self.fpca_amp.inverse_transform(Za).astype(np.float32)
        Yn = np.stack([self._shift_curve(aligned[i], -float(shifts[i])) for i in range(aligned.shape[0])], axis=0).astype(np.float32)
        return self.scaler_y.inverse_transform(Yn).astype(np.float32)


class ComplexChannelReducer(BaseReducer):
    def __init__(self, reducer_real: BaseReducer, reducer_imag: BaseReducer, half_dim: int):
        self.reducer_real = reducer_real
        self.reducer_imag = reducer_imag
        self.half_dim = int(half_dim)
        self.method_name = reducer_real.method_name
        self.latent_dim_ = None
        self.explained_variance_ratio_sum_ = None

    def fit(self, Y):
        Y = ensure_2d_float32(Y, "ComplexChannelReducer.fit(Y)")
        Yr = Y[:, :self.half_dim]
        Yi = Y[:, self.half_dim:]
        self.reducer_real.fit(Yr)
        self.reducer_imag.fit(Yi)
        self.latent_dim_ = int(self.reducer_real.latent_dim_ + self.reducer_imag.latent_dim_)
        evr_r = getattr(self.reducer_real, "explained_variance_ratio_sum_", 0.0) or 0.0
        evr_i = getattr(self.reducer_imag, "explained_variance_ratio_sum_", 0.0) or 0.0
        self.explained_variance_ratio_sum_ = float(evr_r + evr_i)
        return self

    def transform(self, Y):
        Y = ensure_2d_float32(Y, "ComplexChannelReducer.transform(Y)")
        Yr = self.reducer_real.transform(Y[:, :self.half_dim])
        Yi = self.reducer_imag.transform(Y[:, self.half_dim:])
        return np.concatenate([Yr, Yi], axis=1).astype(np.float32)

    def inverse_transform(self, Z):
        Z = ensure_2d_float32(Z, "ComplexChannelReducer.inverse_transform(Z)")
        rdim = int(self.reducer_real.latent_dim_)
        Yr = self.reducer_real.inverse_transform(Z[:, :rdim])
        Yi = self.reducer_imag.inverse_transform(Z[:, rdim:])
        return np.concatenate([Yr, Yi], axis=1).astype(np.float32)

    def propagate_var_to_y(self, var_z, z_mean=None):
        var_z = ensure_2d_float32(var_z, "ComplexChannelReducer.var_z")
        if z_mean is None:
            z_mean = np.zeros_like(var_z, dtype=np.float32)
        z_mean = ensure_2d_float32(z_mean, "ComplexChannelReducer.z_mean")
        rdim = int(self.reducer_real.latent_dim_)
        var_r = self.reducer_real.propagate_var_to_y(var_z[:, :rdim], z_mean[:, :rdim])
        var_i = self.reducer_imag.propagate_var_to_y(var_z[:, rdim:], z_mean[:, rdim:])
        return np.concatenate([var_r, var_i], axis=1).astype(np.float32)


@dataclass
class ReducerConfig:
    method: str
    axis: np.ndarray
    y_dim: int
    device: str
    args: object


def _build_single_reducer(cfg: ReducerConfig) -> BaseReducer:
    a = cfg.args
    method = cfg.method
    fpca_var_ratio = _safe_float(getattr(a, "fpca_var_ratio", 0.999), 0.999)
    fpca_max_dim = _safe_int(getattr(a, "fpca_max_dim", 64), 64)
    fpca_ridge = _safe_float(getattr(a, "fpca_ridge", 1e-8), 1e-8)
    if method == "fpca":
        return FPCAReducer(n_components=_safe_int(getattr(a, "fpca_dim", 0), 0), var_ratio=fpca_var_ratio, max_components=fpca_max_dim, ridge=fpca_ridge)
    if method == "pgfpca":
        return PhysicsGuidedFPCAReducer(axis=cfg.axis, n_components=_safe_int(getattr(a, "fpca_dim", 0), 0), var_ratio=fpca_var_ratio, max_components=fpca_max_dim, ridge=fpca_ridge, alpha_grad=_safe_float(getattr(a, "pgfpca_alpha_grad", 0.1), 0.1), beta_curv=_safe_float(getattr(a, "pgfpca_beta_curv", 0.01), 0.01))
    if method == "btw":
        latent_dim, cap = _resolve_auto_cap(a, "btw_latent_dim", "btw_max_dim", fallback_name="fpca_max_dim", default_cap=fpca_max_dim)
        return BTWReducer(latent_dim=latent_dim, var_ratio=fpca_var_ratio, max_components=min(fpca_max_dim, cap), ridge=fpca_ridge, wavelet=str(getattr(a, "btw_wavelet", "db4")), level=getattr(a, "btw_level", 3), global_ratio=_safe_float(getattr(a, "btw_global_ratio", 0.7), 0.7), threshold_rel=_safe_float(getattr(a, "btw_threshold_rel", 0.01), 0.01), max_total_dim=cap)
    if method == "fae":
        latent_dim, cap = _resolve_auto_cap(a, "fae_latent_dim", "fae_max_dim", fallback_name="fpca_max_dim", default_cap=fpca_max_dim)
        return FunctionalAEReducer(axis=cfg.axis, latent_dim=latent_dim, max_latent_dim=cap, var_ratio=fpca_var_ratio, proj_dim=_safe_int(getattr(a, "fae_proj_dim", 64), 64), basis_dim=_safe_int(getattr(a, "fae_basis_dim", 64), 64), hidden_dim=_safe_int(getattr(a, "fae_hidden_dim", 128), 128), epochs=_safe_int(getattr(a, "fae_epochs", 250), 250), lr=_safe_float(getattr(a, "fae_lr", 1e-3), 1e-3), batch_size=_safe_int(getattr(a, "fae_batch_size", 64), 64), lambda_z=_safe_float(getattr(a, "fae_lambda_z", 1e-4), 1e-4), lambda_smooth=_safe_float(getattr(a, "fae_lambda_smooth", 1e-4), 1e-4), device=cfg.device)
    if method == "elastic":
        amp_dim, total_cap = _resolve_auto_cap(a, "elastic_amp_dim", "elastic_max_dim", fallback_name="fpca_max_dim", default_cap=fpca_max_dim)
        return ElasticShiftReducer(axis=cfg.axis, amp_dim=amp_dim, max_components=min(fpca_max_dim, max(1, total_cap - 1)), shift_max_frac=_safe_float(getattr(a, "elastic_shift_max_frac", 0.08), 0.08), max_total_dim=total_cap, var_ratio=fpca_var_ratio)
    raise ValueError(f"Unknown reducer method: {method}")


def make_reducer(method: str, axis: np.ndarray, y_dim: int, args, device: str = "cpu") -> BaseReducer:
    method = str(method).lower().strip()
    axis = np.asarray(axis, dtype=np.float32).reshape(-1)
    if int(y_dim) == 2 * int(axis.size):
        r_cfg = ReducerConfig(method=method, axis=axis, y_dim=axis.size, device=device, args=args)
        return ComplexChannelReducer(_build_single_reducer(r_cfg), _build_single_reducer(r_cfg), half_dim=int(axis.size))
    return _build_single_reducer(ReducerConfig(method=method, axis=axis, y_dim=y_dim, device=device, args=args))


def reducer_run_tag_from_args(args) -> str:
    method = str(getattr(args, "reducer_method", "fpca")).lower().strip()
    if method == "fpca":
        if _safe_int(getattr(args, "fpca_dim", 0), 0) > 0:
            return f"dimfpca_fpca{_safe_int(getattr(args, 'fpca_dim', 0), 0)}"
        return f"dimfpca_fpcaAutoV{str(getattr(args, 'fpca_var_ratio', 0.999)).replace('.', 'p')}_m{_safe_int(getattr(args, 'fpca_max_dim', 50), 50)}"
    if method == "pgfpca":
        return f"dimpgfpca_V{str(getattr(args, 'fpca_var_ratio', 0.999)).replace('.', 'p')}_m{_safe_int(getattr(args, 'fpca_max_dim', 50), 50)}_ag{str(getattr(args, 'pgfpca_alpha_grad', 0.1)).replace('.', 'p')}_bc{str(getattr(args, 'pgfpca_beta_curv', 0.01)).replace('.', 'p')}"
    if method == "btw":
        latent_dim, cap = _resolve_auto_cap(args, 'btw_latent_dim', 'btw_max_dim', fallback_name='fpca_max_dim', default_cap=_safe_int(getattr(args, 'fpca_max_dim', 50), 50))
        level = getattr(args, 'btw_level', 3)
        if latent_dim > 0:
            return f"dimbtw_L{latent_dim}_{getattr(args, 'btw_wavelet', 'db4')}_lv{level}_gr{str(getattr(args, 'btw_global_ratio', 0.7)).replace('.', 'p')}_thr{str(getattr(args, 'btw_threshold_rel', 0.01)).replace('.', 'p')}"
        return f"dimbtw_AutoV{str(getattr(args, 'fpca_var_ratio', 0.999)).replace('.', 'p')}_m{cap}_{getattr(args, 'btw_wavelet', 'db4')}_lv{level}_gr{str(getattr(args, 'btw_global_ratio', 0.7)).replace('.', 'p')}_thr{str(getattr(args, 'btw_threshold_rel', 0.01)).replace('.', 'p')}"
    if method == "fae":
        latent_dim, cap = _resolve_auto_cap(args, 'fae_latent_dim', 'fae_max_dim', fallback_name='fpca_max_dim', default_cap=_safe_int(getattr(args, 'fpca_max_dim', 50), 50))
        if latent_dim > 0:
            return f"dimfae_lat{latent_dim}_proj{_safe_int(getattr(args, 'fae_proj_dim', 64), 64)}_basis{_safe_int(getattr(args, 'fae_basis_dim', 64), 64)}"
        return f"dimfae_AutoV{str(getattr(args, 'fpca_var_ratio', 0.999)).replace('.', 'p')}_m{cap}_proj{_safe_int(getattr(args, 'fae_proj_dim', 64), 64)}_basis{_safe_int(getattr(args, 'fae_basis_dim', 64), 64)}"
    if method == "elastic":
        amp_dim, cap = _resolve_auto_cap(args, 'elastic_amp_dim', 'elastic_max_dim', fallback_name='fpca_max_dim', default_cap=_safe_int(getattr(args, 'fpca_max_dim', 50), 50))
        if amp_dim > 0:
            return f"dimelastic_amp{amp_dim}_shift{str(getattr(args, 'elastic_shift_max_frac', 0.08)).replace('.', 'p')}"
        return f"dimelastic_AutoV{str(getattr(args, 'fpca_var_ratio', 0.999)).replace('.', 'p')}_m{cap}_shift{str(getattr(args, 'elastic_shift_max_frac', 0.08)).replace('.', 'p')}"
    return f"dim{method}"
