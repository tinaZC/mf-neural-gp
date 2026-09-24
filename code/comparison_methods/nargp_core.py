from dataclasses import dataclass
import time
from typing import List

import gpytorch
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from mf_train_baseline.mf_utils import SVGPModel, init_inducing_points


class NARGPKernel(gpytorch.kernels.Kernel):
    """NARGP covariance k_rho(z,z') k_x(x,x') + k_delta(x,x')."""

    def __init__(self, x_dim: int, kernel_name: str = "rbf", ard: bool = True):
        super().__init__()
        if kernel_name == "rbf":
            base = gpytorch.kernels.RBFKernel
            base_kwargs = {}
        elif kernel_name == "matern":
            base = gpytorch.kernels.MaternKernel
            base_kwargs = {"nu": 2.5}
        else:
            raise ValueError(f"Unsupported kernel_name={kernel_name!r}; use 'rbf' or 'matern'.")
        x_kwargs = {"ard_num_dims": x_dim if ard else None, **base_kwargs}
        self.kx = gpytorch.kernels.ScaleKernel(base(**x_kwargs))
        self.krho = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=1)
        )
        self.kdelta = gpytorch.kernels.ScaleKernel(base(**x_kwargs))

    def forward(self, x1, x2, diag=False, **params):
        return (
            self.krho(x1[..., -1:], x2[..., -1:], diag=diag, **params)
            * self.kx(x1[..., :-1], x2[..., :-1], diag=diag, **params)
            + self.kdelta(x1[..., :-1], x2[..., :-1], diag=diag, **params)
        )


@dataclass
class NARGPComponent:
    lf_model: object
    lf_likelihood: object
    hf_model: object
    hf_likelihood: object


@dataclass
class NARGPBundle:
    components: List[NARGPComponent]
    x_scaler: StandardScaler
    seed: int
    stage1_train_s: float
    stage2_train_s: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fit(X, Y, inducing, steps, lr, device, kernel, seed):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(Y, dtype=torch.float32, device=device).reshape(-1)
    inducing_np = init_inducing_points(
        X.cpu().numpy(), min(int(inducing), len(X)), seed=int(seed)
    )
    inducing_t = torch.as_tensor(inducing_np, dtype=torch.float32, device=device)
    model = SVGPModel(inducing_t, kernel).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    noise0 = max(float(torch.var(y, unbiased=False).detach().cpu()) * 0.01, 1e-6)
    likelihood.noise = torch.tensor(noise0, dtype=torch.float32, device=device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=lr
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(X))
    for _ in range(int(steps)):
        optimizer.zero_grad()
        loss = -mll(model(X), y)
        loss.backward()
        optimizer.step()
    return model, likelihood


def train_nargp(
    X_lf,
    Z_lf,
    X_hf,
    Z_lf_paired,
    Z_hf,
    device="cpu",
    inducing=64,
    steps=100,
    lr=0.005,
    kernel_name="rbf",
    seed=42,
):
    """Fit independent latent NARGPs in one LF-training-scaled x coordinate system."""
    device = torch.device(device)
    x_scaler = StandardScaler(with_mean=True, with_std=True).fit(
        np.asarray(X_lf, dtype=np.float32)
    )
    X_lf_scaled = x_scaler.transform(X_lf).astype(np.float32)
    X_hf_scaled = x_scaler.transform(X_hf).astype(np.float32)
    components = []
    stage1_train_s = 0.0
    stage2_train_s = 0.0

    for r in range(Z_hf.shape[1]):
        lf_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=X_lf_scaled.shape[1])
        )
        _sync(device)
        started = time.perf_counter()
        lf_model, lf_likelihood = _fit(
            X_lf_scaled, Z_lf[:, r], inducing, steps, lr, device, lf_kernel, seed
        )
        _sync(device)
        stage1_train_s += time.perf_counter() - started

        hf_inputs = np.column_stack([X_hf_scaled, Z_lf_paired[:, r]]).astype(
            np.float32
        )
        _sync(device)
        started = time.perf_counter()
        hf_model, hf_likelihood = _fit(
            hf_inputs,
            Z_hf[:, r],
            inducing,
            steps,
            lr,
            device,
            NARGPKernel(X_hf_scaled.shape[1], kernel_name),
            seed,
        )
        _sync(device)
        stage2_train_s += time.perf_counter() - started
        components.append(
            NARGPComponent(lf_model, lf_likelihood, hf_model, hf_likelihood)
        )

    return NARGPBundle(
        components=components,
        x_scaler=x_scaler,
        seed=int(seed),
        stage1_train_s=float(stage1_train_s),
        stage2_train_s=float(stage2_train_s),
    )


def predict_nargp(bundle, X, mc_samples=32, device="cpu", seed=None):
    """Marginalize the LF latent posterior through each conditional HF GP by MC."""
    if int(mc_samples) < 1:
        raise ValueError(f"mc_samples must be >= 1, got {mc_samples}")
    device = torch.device(device)
    X_scaled = bundle.x_scaler.transform(np.asarray(X, dtype=np.float32)).astype(
        np.float32
    )
    X_tensor = torch.as_tensor(X_scaled, dtype=torch.float32, device=device)
    means = []
    variances = []
    prediction_seed = bundle.seed if seed is None else int(seed)

    for r, component in enumerate(bundle.components):
        component.lf_model.eval()
        component.lf_likelihood.eval()
        component.hf_model.eval()
        component.hf_likelihood.eval()
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(prediction_seed + r)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(prediction_seed + r)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # Sample latent f_L(x); LF observation noise is not an HF input.
                lf_posterior = component.lf_model(X_tensor)
                lf_std = lf_posterior.variance.clamp_min(1e-12).sqrt()
                eps = torch.randn(
                    (int(mc_samples), X_tensor.shape[0]),
                    dtype=X_tensor.dtype,
                    device=device,
                )
                lf_samples = lf_posterior.mean.unsqueeze(0) + eps * lf_std.unsqueeze(0)

                conditional_means = []
                conditional_variances = []
                for sample in lf_samples:
                    hf_input = torch.cat([X_tensor, sample[:, None]], dim=1)
                    hf_predictive = component.hf_likelihood(
                        component.hf_model(hf_input)
                    )
                    conditional_means.append(hf_predictive.mean)
                    conditional_variances.append(hf_predictive.variance)
                conditional_means = torch.stack(conditional_means, dim=0)
                conditional_variances = torch.stack(conditional_variances, dim=0)

                # Var(f_H) = E[Var(f_H | f_L)] + Var(E[f_H | f_L]).
                mean = conditional_means.mean(dim=0)
                variance = conditional_variances.mean(dim=0) + conditional_means.var(
                    dim=0, unbiased=False
                )
                means.append(mean.cpu().numpy())
                variances.append(variance.clamp_min(0.0).cpu().numpy())

    return (
        np.stack(means, axis=1).astype(np.float32),
        np.stack(variances, axis=1).astype(np.float32),
    )
