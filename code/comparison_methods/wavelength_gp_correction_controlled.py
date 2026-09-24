#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controlled Neural--GP MF wavelength-wise Stage-II ablation.

Scientific question
-------------------
Does the HF-informed FPCA latent representation help Stage-II correction,
relative to correcting the spectrum directly with one independent GP per
wavelength?

This script keeps the submitted TM Neural--GP MF pipeline unchanged through
Stage I, and changes Stage II representation only:

Submitted Neural--GP MF:
    x -> full LF spectrum -> HF scaler -> FPCA -> latent scaler
      -> U=[x_scaled, predicted LF latent vector]
      -> independent residual SVGPs in latent coordinates
      -> inverse latent transform -> HF spectrum

Controlled wavelength-wise variant:
    x -> full LF spectrum                     [UNCHANGED Stage I]
      -> HF wavelength-wise scaler            [same HF scaler, no FPCA]
      -> U=[x_scaled, predicted LF spectrum]  [full wavelength representation]
      -> independent residual SVGP per wavelength
      -> inverse HF wavelength scaler -> HF spectrum

Important:
- This is NOT Frequency-wise NARGP.
- Stage I is the same full-spectrum FeatureMLP used by the submitted method.
- The affine rho+b residual formulation is retained.
- The Stage-II GP backend/hyperparameters are retained.
- The ONLY structural change is replacing the compact HF-informed FPCA
  coordinate system by the full wavelength coordinate system.

Official TM sweep defaults are matched to the uploaded reproduction code:
  wavelengths: 380--750 nm
  Stage I: mix train, paired val, paired y-scaler
  hidden: 256,256,256; feat_dim=32; ReLU; LeakyReLU feature
  AdamW: lr=3e-4, wd=1e-4, batch=256, epochs=2000, patience=100
  Stage II: xlf, delta, rho source=oracle, intercept=True, ridge=1e-6
  SVGP: full Matern(nu=2.5), ARD, M=64, steps=2000, lr=5e-3

Typical use:
  # 1) Train/calculate Stage I once
  python -m comparison_methods.wavelength_gp_correction_controlled \
      --data_dir .../hf100_lfx10 --out_dir .../seed200 \
      --seed 200 --device cuda --prepare_stage1

  # 2) Train all wavelength GPs (or use --freq_start/--freq_end chunks)
  python -m comparison_methods.wavelength_gp_correction_controlled \
      --data_dir .../hf100_lfx10 --out_dir .../seed200 \
      --seed 200 --device cuda --resume

  # 3) Aggregate
  python -m comparison_methods.wavelength_gp_correction_controlled \
      --data_dir .../hf100_lfx10 --out_dir .../seed200 \
      --seed 200 --device cuda --aggregate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import gpytorch
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from mf_train_baseline.mf_train import FeatureMLP, fit_affine_rho
from mf_train_baseline.mf_utils import (
    SVGPModel,
    ci_coverage_y,
    ci_width_y,
    gaussian_nll,
    gaussian_nlpd,
    init_inducing_points,
    make_single_kernel,
    mlp_predict_and_features,
    r2_score,
    rmse,
    save_pickle,
    set_seed,
    train_feature_mlp,
)

from .common import load_bundle


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _atomic_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _safe_transform(scaler: StandardScaler, x: np.ndarray, n_features: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] == 0:
        return np.zeros((0, n_features), dtype=np.float32)
    return scaler.transform(x).astype(np.float32)


def _crop_bundle_y(data, wl_low: float, wl_high: float):
    wl = np.asarray(data.wavelengths, dtype=np.float32)
    lo = float(min(wl_low, wl_high))
    hi = float(max(wl_low, wl_high))
    keep = np.where((wl >= lo) & (wl <= hi))[0].astype(np.int64)
    if keep.size == 0:
        raise ValueError(f"Wavelength crop [{lo},{hi}] retains zero points.")

    spectra = {}
    for name, block in (
        ("hf", data.hf),
        ("lf_paired", data.lf_paired),
        ("lf_unpaired", data.lf_unpaired),
    ):
        spectra[name] = {}
        for split in ("train", "val", "test"):
            y = np.asarray(block[split]["y"], dtype=np.float32)
            if y.ndim != 2 or y.shape[1] != len(wl):
                raise ValueError(
                    f"{name}/{split}: expected scalar spectrum width {len(wl)}, got {y.shape}."
                )
            spectra[name][split] = y[:, keep].astype(np.float32)

    return (
        wl[keep].astype(np.float32),
        np.asarray(data.idx_wavelength, dtype=np.int64)[keep],
        spectra,
    )


def _hash_array(h, name: str, value: np.ndarray) -> None:
    a = np.ascontiguousarray(value)
    h.update(name.encode("utf-8"))
    h.update(str(a.dtype).encode("utf-8"))
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes())


def _signature(args, data, wavelengths, spectra) -> str:
    h = hashlib.sha256()
    _hash_array(h, "wavelengths", wavelengths)
    for block_name, block in (
        ("hf", data.hf),
        ("lf_paired", data.lf_paired),
        ("lf_unpaired", data.lf_unpaired),
    ):
        for split in ("train", "val", "test"):
            _hash_array(h, f"{block_name}/{split}/x", block[split]["x"])
            _hash_array(h, f"{block_name}/{split}/idx", block[split]["idx"])
            _hash_array(h, f"{block_name}/{split}/y_cropped", spectra[block_name][split])

    config = {
        "seed": int(args.seed),
        "wl_low": float(args.wl_low),
        "wl_high": float(args.wl_high),
        "student_train_set": args.student_train_set,
        "student_val_set": args.student_val_set,
        "student_yscale": int(args.student_yscale),
        "student_y_scaler_fit": args.student_y_scaler_fit,
        "student_hidden": list(args.student_hidden),
        "student_feat_dim": int(args.student_feat_dim),
        "student_act": args.student_act,
        "student_feat_act": args.student_feat_act,
        "student_feat_leaky_slope": float(args.student_feat_leaky_slope),
        "student_dropout": float(args.student_dropout),
        "student_lr": float(args.student_lr),
        "student_wd": float(args.student_wd),
        "student_bs": int(args.student_bs),
        "student_epochs": int(args.student_epochs),
        "student_patience": int(args.student_patience),
        "student_min_delta": float(args.student_min_delta),
        "rho_ridge": float(args.rho_ridge),
        "rho_intercept": int(args.rho_intercept),
        "svgp_M": int(args.svgp_M),
        "svgp_steps": int(args.svgp_steps),
        "svgp_lr": float(args.svgp_lr),
        "kernel": args.kernel,
        "matern_nu": float(args.matern_nu),
        "gp_ard": int(args.gp_ard),
    }
    h.update(json.dumps(config, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def _cache_path(out: Path) -> Path:
    return out / "stage1_stage2_common_cache.npz"


def _checkpoint_path(out: Path, k: int) -> Path:
    return out / "wavelength_checkpoints" / f"wavelength_{k:04d}.npz"


def _requested_indices(args, k_total: int) -> List[int]:
    if args.freq_indices:
        idx = [int(x.strip()) for x in args.freq_indices.split(",") if x.strip()]
        if not idx or len(idx) != len(set(idx)):
            raise ValueError("Invalid or duplicate --freq_indices.")
        if any(k < 0 or k >= k_total for k in idx):
            raise ValueError(f"--freq_indices must lie in [0,{k_total}).")
        return idx

    start = int(args.freq_start)
    end = k_total if args.freq_end is None else int(args.freq_end)
    if start < 0 or end > k_total or start >= end:
        raise ValueError(f"Invalid wavelength interval [{start},{end}) for K={k_total}.")
    return list(range(start, end))


# ---------------------------------------------------------------------
# Stage I + common Stage-II preprocessing
# ---------------------------------------------------------------------

def prepare_stage1(args) -> None:
    set_seed(int(args.seed))
    data = load_bundle(args.data_dir)
    wavelengths, idx_wavelength, spectra = _crop_bundle_y(
        data, args.wl_low, args.wl_high
    )

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    sig = _signature(args, data, wavelengths, spectra)
    cache = _cache_path(out)
    if cache.exists() and not args.force_prepare:
        with np.load(cache, allow_pickle=False) as z:
            old = str(z["signature"].item())
        if old != sig:
            raise ValueError(
                "Existing cache belongs to a different scientific configuration. "
                "Use a new output directory or --force_prepare."
            )
        print(f"[OK] Common cache already exists: {cache}")
        return

    hf = data.hf
    lp = data.lf_paired
    lu = data.lf_unpaired
    ylp = spectra["lf_paired"]
    ylu = spectra["lf_unpaired"]
    yhf = spectra["hf"]
    K = int(len(wavelengths))

    # -------------------------------------------------------------
    # Stage I -- copied semantically from submitted mf_train.py
    # -------------------------------------------------------------
    X_lf_tr_all = np.concatenate(
        [lp["train"]["x"], lu["train"]["x"]], axis=0
    ).astype(np.float32)
    y_lf_tr_all = np.concatenate(
        [ylp["train"], ylu["train"]], axis=0
    ).astype(np.float32)
    X_lf_va_all = np.concatenate(
        [lp["val"]["x"], lu["val"]["x"]], axis=0
    ).astype(np.float32)
    y_lf_va_all = np.concatenate(
        [ylp["val"], ylu["val"]], axis=0
    ).astype(np.float32)

    sx_student = StandardScaler(with_mean=True, with_std=True).fit(X_lf_tr_all)
    X_lf_tr_all_s = sx_student.transform(X_lf_tr_all).astype(np.float32)
    X_lf_va_all_s = sx_student.transform(X_lf_va_all).astype(np.float32)

    sy_student = None
    if int(args.student_yscale) == 1:
        sy_student = StandardScaler(with_mean=True, with_std=True)
        if args.student_y_scaler_fit == "paired":
            y_fit = ylp["train"]
        else:
            y_fit = y_lf_tr_all
        sy_student.fit(y_fit.astype(np.float32))
        y_lf_tr_all_y = sy_student.transform(y_lf_tr_all).astype(np.float32)
        y_lf_va_all_y = sy_student.transform(y_lf_va_all).astype(np.float32)
    else:
        y_lf_tr_all_y = y_lf_tr_all.astype(np.float32)
        y_lf_va_all_y = y_lf_va_all.astype(np.float32)

    def student_x(x):
        return sx_student.transform(np.asarray(x, dtype=np.float32)).astype(np.float32)

    def student_y_fwd(y):
        y = np.asarray(y, dtype=np.float32)
        return y if sy_student is None else sy_student.transform(y).astype(np.float32)

    def student_y_inv(y):
        y = np.asarray(y, dtype=np.float32)
        return y if sy_student is None else sy_student.inverse_transform(y).astype(np.float32)

    if args.student_train_set == "paired":
        X_stu_train = student_x(lp["train"]["x"])
        y_stu_train = student_y_fwd(ylp["train"])
    else:
        X_stu_train = X_lf_tr_all_s
        y_stu_train = y_lf_tr_all_y

    if args.student_val_set == "paired":
        X_stu_val = student_x(lp["val"]["x"])
        y_stu_val = student_y_fwd(ylp["val"])
    else:
        X_stu_val = X_lf_va_all_s
        y_stu_val = y_lf_va_all_y

    student = FeatureMLP(
        in_dim=int(hf["train"]["x"].shape[1]),
        out_dim=K,
        hidden=tuple(int(x) for x in args.student_hidden),
        feat_dim=int(args.student_feat_dim),
        act=str(args.student_act),
        dropout=float(args.student_dropout),
        feat_act=str(args.student_feat_act),
        leaky_slope=float(args.student_feat_leaky_slope),
    ).to(device)

    _sync(device)
    t0 = time.perf_counter()
    student, stu_info = train_feature_mlp(
        student,
        X_train=X_stu_train,
        y_train=y_stu_train,
        X_val=X_stu_val,
        y_val=y_stu_val,
        lr=float(args.student_lr),
        batch_size=int(args.student_bs),
        weight_decay=float(args.student_wd),
        max_epochs=int(args.student_epochs),
        patience=int(args.student_patience),
        min_delta=float(args.student_min_delta),
        device=device,
        print_every=int(args.student_print_every),
    )
    _sync(device)
    stage1_train_s = time.perf_counter() - t0

    torch.save(student.state_dict(), out / "student_feature_mlp.pth")
    save_pickle(out / "student_scaler_x.pkl", sx_student)
    if sy_student is not None:
        save_pickle(out / "student_scaler_y.pkl", sy_student)

    def predict_lf(x):
        pred_s, feat = mlp_predict_and_features(
            student, student_x(x), device=device, batch_size=4096
        )
        return student_y_inv(pred_s), feat.astype(np.float32)

    yhat_tr, feat_tr = predict_lf(hf["train"]["x"])
    yhat_va, feat_va = predict_lf(hf["val"]["x"])
    yhat_te, feat_te = predict_lf(hf["test"]["x"])

    # Basic collapse/finite guards.
    for split, arr, feat in (
        ("train", yhat_tr, feat_tr),
        ("val", yhat_va, feat_va),
        ("test", yhat_te, feat_te),
    ):
        if not np.isfinite(arr).all() or not np.isfinite(feat).all():
            raise FloatingPointError(f"Non-finite Stage-I output on {split}.")
        if np.unique(np.round(arr, 6), axis=0).shape[0] <= 1:
            raise RuntimeError(f"Stage-I prediction collapsed on {split}.")
        if np.unique(np.round(feat, 6), axis=0).shape[0] <= 1:
            raise RuntimeError(f"Stage-I feature collapsed on {split}.")

    # -------------------------------------------------------------
    # Controlled Stage-II direct-wavelength representation
    # -------------------------------------------------------------
    # This is the exact analogue of the submitted HF scaler BEFORE FPCA.
    # With FPCA removed, its K standardized wavelength coordinates become
    # the Stage-II target/representation coordinates.
    scaler_y_hf = StandardScaler(with_mean=True, with_std=True).fit(yhf["train"])
    Y_tr = scaler_y_hf.transform(yhf["train"]).astype(np.float32)
    Y_va = scaler_y_hf.transform(yhf["val"]).astype(np.float32)
    Y_te = scaler_y_hf.transform(yhf["test"]).astype(np.float32)

    lf_or_tr_t = scaler_y_hf.transform(ylp["train"]).astype(np.float32)
    lf_or_va_t = scaler_y_hf.transform(ylp["val"]).astype(np.float32)
    lf_or_te_t = scaler_y_hf.transform(ylp["test"]).astype(np.float32)

    lf_hat_tr_t = scaler_y_hf.transform(yhat_tr).astype(np.float32)
    lf_hat_va_t = scaler_y_hf.transform(yhat_va).astype(np.float32)
    lf_hat_te_t = scaler_y_hf.transform(yhat_te).astype(np.float32)

    # Same HF-x scaler as submitted Stage II.
    sx_hf = StandardScaler(with_mean=True, with_std=True).fit(hf["train"]["x"])
    X_hf_tr_s = sx_hf.transform(hf["train"]["x"]).astype(np.float32)
    X_hf_va_s = sx_hf.transform(hf["val"]["x"]).astype(np.float32)
    X_hf_te_s = sx_hf.transform(hf["test"]["x"]).astype(np.float32)

    # IMPORTANT: retain the submitted xlf semantics.
    # Main method uses U=[x_scaled, full predicted latent vector].
    # Controlled variant therefore uses U=[x_scaled, full predicted wavelength vector],
    # NOT [x_scaled, only the current wavelength scalar].
    U_tr = np.concatenate([X_hf_tr_s, lf_hat_tr_t], axis=1).astype(np.float32)
    U_va = np.concatenate([X_hf_va_s, lf_hat_va_t], axis=1).astype(np.float32)
    U_te = np.concatenate([X_hf_te_s, lf_hat_te_t], axis=1).astype(np.float32)

    su = StandardScaler(with_mean=True, with_std=True).fit(U_tr)
    U_tr_s = su.transform(U_tr).astype(np.float32)
    U_va_s = su.transform(U_va).astype(np.float32)
    U_te_s = su.transform(U_te).astype(np.float32)

    # Same submitted delta formulation: fit rho+b on oracle paired LF/HF train,
    # then construct the actual base from Stage-I-predicted LF.
    rho, bias = fit_affine_rho(
        lf_or_tr_t,
        Y_tr,
        ridge=float(args.rho_ridge),
        use_intercept=bool(int(args.rho_intercept)),
    )
    base_tr = (lf_hat_tr_t * rho[None, :] + bias[None, :]).astype(np.float32)
    base_va = (lf_hat_va_t * rho[None, :] + bias[None, :]).astype(np.float32)
    base_te = (lf_hat_te_t * rho[None, :] + bias[None, :]).astype(np.float32)
    delta_tr = (Y_tr - base_tr).astype(np.float32)

    # Exact inducing-point construction used by mf_utils.train_svgp_per_dim
    # for a student/delta tag: seed + 33.
    M_eff = int(min(int(args.svgp_M), U_tr_s.shape[0]))
    Z0 = init_inducing_points(
        U_tr_s, M=M_eff, seed=int(args.seed) + 33
    ).astype(np.float32)

    save_pickle(out / "stage2_scaler_y_hf.pkl", scaler_y_hf)
    save_pickle(out / "stage2_scaler_x_hf.pkl", sx_hf)
    save_pickle(out / "stage2_scaler_u_student.pkl", su)
    np.save(out / "rho_a.npy", rho)
    np.save(out / "rho_b.npy", bias)
    np.save(out / "stage2_inducing_points.npy", Z0)

    _atomic_npz(
        cache,
        signature=np.asarray(sig),
        wavelengths=wavelengths,
        idx_wavelength=idx_wavelength,
        U_tr_s=U_tr_s,
        U_va_s=U_va_s,
        U_te_s=U_te_s,
        Y_tr=Y_tr,
        Y_va=Y_va,
        Y_te=Y_te,
        base_va=base_va,
        base_te=base_te,
        delta_tr=delta_tr,
        target_mean=np.asarray(scaler_y_hf.mean_, dtype=np.float32),
        target_scale=np.asarray(scaler_y_hf.scale_, dtype=np.float32),
        inducing_points=Z0,
        stage1_train_s=np.asarray(stage1_train_s, dtype=np.float64),
        stage1_rmse_train=np.asarray(rmse(yhat_tr, ylp["train"]), dtype=np.float64),
        stage1_rmse_val=np.asarray(rmse(yhat_va, ylp["val"]), dtype=np.float64),
        stage1_rmse_test=np.asarray(rmse(yhat_te, ylp["test"]), dtype=np.float64),
        student_epochs_ran=np.asarray(stu_info["meta"]["epochs_ran"], dtype=np.int64),
    )

    report = {
        "method": "Wavelength-wise GP correction controlled ablation",
        "stage": "common preparation",
        "signature": sig,
        "dataset": args.dataset,
        "setting": data.root.name,
        "seed": int(args.seed),
        "device": str(device),
        "wavelength_range_nm": [float(wavelengths.min()), float(wavelengths.max())],
        "wavelength_count": K,
        "stage1": {
            "same_as_submitted_neural_gp_mf": True,
            "train_set": args.student_train_set,
            "validation_set": args.student_val_set,
            "target_scaler_fit": args.student_y_scaler_fit
            if int(args.student_yscale) == 1
            else "none",
            "hidden": list(args.student_hidden),
            "feat_dim": int(args.student_feat_dim),
            "activation": args.student_act,
            "feature_activation": args.student_feat_act,
            "optimizer": "AdamW",
            "lr": float(args.student_lr),
            "weight_decay": float(args.student_wd),
            "batch_size": int(args.student_bs),
            "max_epochs": int(args.student_epochs),
            "patience": int(args.student_patience),
            "epochs_ran": int(stu_info["meta"]["epochs_ran"]),
            "train_seconds": float(stage1_train_s),
            "paired_lf_rmse": {
                "train": float(rmse(yhat_tr, ylp["train"])),
                "val": float(rmse(yhat_va, ylp["val"])),
                "test": float(rmse(yhat_te, ylp["test"])),
            },
        },
        "stage2_control": {
            "submitted_coordinate_system": "HF-scaled -> FPCA -> HF-latent-scaled",
            "ablation_coordinate_system": "HF-scaled wavelength coordinates (FPCA removed)",
            "input": "U=[HF-x-scaled, full Stage-I-predicted LF wavelength vector], then U standardized on HF train",
            "output": "one residual SVGP per wavelength",
            "rho_fit_source": "oracle paired LF train",
            "rho_intercept": bool(int(args.rho_intercept)),
            "rho_ridge": float(args.rho_ridge),
            "kernel_struct": "full",
            "kernel": args.kernel,
            "matern_nu": float(args.matern_nu),
            "ard": bool(int(args.gp_ard)),
            "inducing_M": int(args.svgp_M),
            "steps": int(args.svgp_steps),
            "lr": float(args.svgp_lr),
        },
    }
    _atomic_json(out / "common_report.json", report)
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------
# Exact single-output SVGP training mirroring mf_utils.train_svgp_per_dim
# ---------------------------------------------------------------------

def _train_one_svgp(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    inducing_points: np.ndarray,
    args,
    device: torch.device,
    wavelength_index: int,
) -> Tuple[SVGPModel, gpytorch.likelihoods.GaussianLikelihood, float]:
    Xt = torch.from_numpy(np.asarray(Xtr, dtype=np.float32)).to(device)
    yt = torch.from_numpy(np.asarray(ytr, dtype=np.float32).reshape(-1)).to(device)
    Zt = torch.from_numpy(np.asarray(inducing_points, dtype=np.float32)).to(device)

    covar = make_single_kernel(
        x_dim=Xtr.shape[1],
        ard=bool(int(args.gp_ard)),
        kernel_name=str(args.kernel),
        matern_nu=float(args.matern_nu),
    )
    model = SVGPModel(inducing_points=Zt.clone(), covar_module=covar).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=float(args.svgp_lr),
    )
    mll = gpytorch.mlls.VariationalELBO(
        likelihood, model, num_data=Xtr.shape[0]
    )

    _sync(device)
    t0 = time.perf_counter()
    for step in range(1, int(args.svgp_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = -mll(model(Xt), yt)
        loss.backward()
        optimizer.step()

        if int(args.print_every) > 0 and (
            step == 1
            or step == int(args.svgp_steps)
            or step % int(args.print_every) == 0
        ):
            print(
                f"[SVGP][WL={wavelength_index:04d}] "
                f"step={step:5d} loss={float(loss.detach().cpu()):.6g} "
                f"noise={float(likelihood.noise.detach().cpu()):.6g}",
                flush=True,
            )
    _sync(device)
    elapsed = time.perf_counter() - t0
    return model, likelihood, float(elapsed)


@torch.no_grad()
def _predict_one_svgp(model, likelihood, X, device):
    model.eval()
    likelihood.eval()
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    with gpytorch.settings.fast_pred_var():
        pred = likelihood(model(Xt))
        mean = pred.mean.detach().cpu().float().numpy().astype(np.float32)
        var = pred.variance.detach().cpu().float().numpy().astype(np.float32)
    return mean, var


# ---------------------------------------------------------------------
# Wavelength chunk
# ---------------------------------------------------------------------

def train_chunk(args) -> None:
    set_seed(int(args.seed))
    data = load_bundle(args.data_dir)
    wavelengths, idx_wavelength, spectra = _crop_bundle_y(
        data, args.wl_low, args.wl_high
    )
    sig = _signature(args, data, wavelengths, spectra)
    out = Path(args.out_dir).expanduser().resolve()
    cache = _cache_path(out)

    if not cache.exists():
        raise FileNotFoundError(
            f"Missing {cache}. Run --prepare_stage1 first."
        )

    with np.load(cache, allow_pickle=False) as z:
        if str(z["signature"].item()) != sig:
            raise ValueError("Common cache signature mismatch.")
        U_tr_s = z["U_tr_s"].copy()
        U_va_s = z["U_va_s"].copy()
        U_te_s = z["U_te_s"].copy()
        delta_tr = z["delta_tr"].copy()
        base_va = z["base_va"].copy()
        base_te = z["base_te"].copy()
        target_mean = z["target_mean"].copy()
        target_scale = z["target_scale"].copy()
        inducing_points = z["inducing_points"].copy()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    req = _requested_indices(args, len(wavelengths))

    for k in req:
        ckpt = _checkpoint_path(out, k)
        if ckpt.exists():
            if not args.resume:
                raise FileExistsError(
                    f"Checkpoint exists for wavelength {k}; use --resume to skip it."
                )
            with np.load(ckpt, allow_pickle=False) as z:
                if str(z["signature"].item()) != sig:
                    raise ValueError(f"Checkpoint signature mismatch: {ckpt}")
            print(f"[skip] wavelength {k:04d}", flush=True)
            continue

        # Re-seeding per coordinate makes chunked/resumed execution deterministic.
        # The architecture, data, inducing set, and optimizer are unchanged.
        set_seed(int(args.seed) + 100_000 + int(k))

        model, likelihood, train_s = _train_one_svgp(
            U_tr_s,
            delta_tr[:, k],
            inducing_points,
            args,
            device,
            wavelength_index=k,
        )

        _sync(device)
        t0 = time.perf_counter()
        mu_v_res, var_v_s = _predict_one_svgp(model, likelihood, U_va_s, device)
        mu_t_res, var_t_s = _predict_one_svgp(model, likelihood, U_te_s, device)
        _sync(device)
        inf_s = time.perf_counter() - t0

        mu_v_s = (base_va[:, k] + mu_v_res).astype(np.float32)
        mu_t_s = (base_te[:, k] + mu_t_res).astype(np.float32)

        scale = float(target_scale[k])
        center = float(target_mean[k])
        mu_v = (mu_v_s * scale + center).astype(np.float32)
        mu_t = (mu_t_s * scale + center).astype(np.float32)
        var_v = (np.maximum(var_v_s, 0.0) * scale * scale).astype(np.float32)
        var_t = (np.maximum(var_t_s, 0.0) * scale * scale).astype(np.float32)

        if not all(
            np.isfinite(x).all() for x in (mu_v, mu_t, var_v, var_t)
        ):
            raise FloatingPointError(f"Non-finite result at wavelength {k}.")

        _atomic_npz(
            ckpt,
            signature=np.asarray(sig),
            k=np.asarray(k, dtype=np.int64),
            wavelength=np.asarray(wavelengths[k], dtype=np.float32),
            idx_wavelength=np.asarray(idx_wavelength[k], dtype=np.int64),
            val_mean=mu_v,
            val_variance=var_v,
            test_mean=mu_t,
            test_variance=var_t,
            stage2_train_s=np.asarray(train_s, dtype=np.float64),
            inference_s=np.asarray(inf_s, dtype=np.float64),
        )
        print(
            f"[OK] wavelength={k:04d}/{len(wavelengths)-1:04d} "
            f"train={train_s:.3f}s inference={inf_s:.3f}s",
            flush=True,
        )


# ---------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------

def _raw_metrics(y, mean, var, ci_level):
    var = np.maximum(np.asarray(var, dtype=np.float32), 0.0)
    std = np.sqrt(var)
    return {
        "rmse": float(rmse(y, mean)),
        "r2": float(r2_score(y, mean)),
        "gaussian_nll": float(gaussian_nll(y, mean, var=var)),
        "gaussian_nlpd": float(gaussian_nlpd(y, mean, var=var)),
        "coverage_raw": float(ci_coverage_y(y, mean, std, ci_level)),
        "interval_width_raw": float(ci_width_y(std, ci_level)),
    }


def aggregate(args):
    data = load_bundle(args.data_dir)
    wavelengths, idx_wavelength, spectra = _crop_bundle_y(
        data, args.wl_low, args.wl_high
    )
    sig = _signature(args, data, wavelengths, spectra)
    out = Path(args.out_dir).expanduser().resolve()
    cache = _cache_path(out)

    if not cache.exists():
        raise FileNotFoundError(f"Missing common cache: {cache}")

    with np.load(cache, allow_pickle=False) as z:
        if str(z["signature"].item()) != sig:
            raise ValueError("Common cache signature mismatch.")
        stage1_train_s = float(z["stage1_train_s"].item())
        stage1_rmse_train = float(z["stage1_rmse_train"].item())
        stage1_rmse_val = float(z["stage1_rmse_val"].item())
        stage1_rmse_test = float(z["stage1_rmse_test"].item())

    records = []
    missing = []
    for k in range(len(wavelengths)):
        ckpt = _checkpoint_path(out, k)
        if not ckpt.exists():
            missing.append(k)
            continue
        with np.load(ckpt, allow_pickle=False) as z:
            if str(z["signature"].item()) != sig or int(z["k"].item()) != k:
                raise ValueError(f"Checkpoint mismatch: {ckpt}")
            records.append({name: z[name].copy() for name in z.files})

    if missing:
        raise ValueError(
            f"Cannot aggregate: {len(missing)} wavelengths missing. "
            f"First missing: {missing[:20]}"
        )

    val_mean = np.column_stack([r["val_mean"] for r in records]).astype(np.float32)
    val_var = np.column_stack([r["val_variance"] for r in records]).astype(np.float32)
    test_mean = np.column_stack([r["test_mean"] for r in records]).astype(np.float32)
    test_var = np.column_stack([r["test_variance"] for r in records]).astype(np.float32)

    y_val = spectra["hf"]["val"]
    y_test = spectra["hf"]["test"]
    if val_mean.shape != y_val.shape or test_mean.shape != y_test.shape:
        raise ValueError(
            f"Prediction shape mismatch: val {val_mean.shape}/{y_val.shape}, "
            f"test {test_mean.shape}/{y_test.shape}"
        )

    pred = out / "pred_arrays"
    pred.mkdir(parents=True, exist_ok=True)
    np.save(pred / "val_mean.npy", val_mean)
    np.save(pred / "val_variance.npy", val_var)
    np.save(pred / "test_mean.npy", test_mean)
    np.save(pred / "test_variance.npy", test_var)
    np.save(pred / "val_indices.npy", data.hf["val"]["idx"])
    np.save(pred / "test_indices.npy", data.hf["test"]["idx"])
    np.save(pred / "wavelengths.npy", wavelengths)
    np.save(pred / "idx_wavelength.npy", idx_wavelength)

    stage2_train_s = float(sum(float(r["stage2_train_s"]) for r in records))
    inference_s = float(sum(float(r["inference_s"]) for r in records))

    report = {
        "method": "Wavelength-wise GP correction controlled ablation",
        "run_kind": args.run_kind,
        "scientific_result": bool(args.run_kind == "scientific"),
        "dataset": args.dataset,
        "setting": data.root.name,
        "seed": int(args.seed),
        "wavelength_count": int(len(wavelengths)),
        "controlled_change": (
            "Replace the HF-informed FPCA latent Stage-II representation by "
            "the full HF-standardized wavelength representation; keep the "
            "full-spectrum Neural--GP MF Stage I, xlf input semantics, affine "
            "rho+b delta formulation, and SVGP backend/hyperparameters."
        ),
        "stage1_paired_lf_rmse": {
            "train": stage1_rmse_train,
            "val": stage1_rmse_val,
            "test": stage1_rmse_test,
        },
        "metrics": {
            "val": _raw_metrics(y_val, val_mean, val_var, args.ci_level),
            "test": _raw_metrics(y_test, test_mean, test_var, args.ci_level),
        },
        "timing": {
            "stage1_train_s": stage1_train_s,
            "stage2_train_s": stage2_train_s,
            "inference_val_test_s": inference_s,
            "total_model_wall_s": stage1_train_s + stage2_train_s + inference_s,
            "excludes_em_simulation_time": True,
        },
        "configuration": {
            "wl_low": float(args.wl_low),
            "wl_high": float(args.wl_high),
            "student_train_set": args.student_train_set,
            "student_val_set": args.student_val_set,
            "student_y_scaler_fit": args.student_y_scaler_fit,
            "student_hidden": list(args.student_hidden),
            "student_feat_dim": int(args.student_feat_dim),
            "student_act": args.student_act,
            "student_feat_act": args.student_feat_act,
            "student_lr": float(args.student_lr),
            "student_wd": float(args.student_wd),
            "student_bs": int(args.student_bs),
            "student_epochs": int(args.student_epochs),
            "student_patience": int(args.student_patience),
            "rho_fit_source": "oracle",
            "rho_intercept": bool(int(args.rho_intercept)),
            "rho_ridge": float(args.rho_ridge),
            "mf_u_mode": "xlf",
            "kernel_struct": "full",
            "kernel": args.kernel,
            "matern_nu": float(args.matern_nu),
            "gp_ard": bool(int(args.gp_ard)),
            "svgp_M": int(args.svgp_M),
            "svgp_steps": int(args.svgp_steps),
            "svgp_lr": float(args.svgp_lr),
        },
        "prediction_dir": "pred_arrays",
    }

    _atomic_json(out / "report.json", report)
    _atomic_json(
        out / "timing.json",
        {
            "method": report["method"],
            "dataset": args.dataset,
            "setting": data.root.name,
            "seed": int(args.seed),
            **report["timing"],
        },
    )
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Controlled Neural--GP MF ablation: retain the submitted full-spectrum "
            "Stage I and replace only the FPCA latent Stage-II representation with "
            "full wavelength coordinates and one residual SVGP per wavelength."
        )
    )
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", choices=("TM",), default="TM")
    p.add_argument("--seed", type=int, default=200)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--run_kind", choices=("smoke", "scientific"), default="scientific")

    p.add_argument("--wl_low", type=float, default=380.0)
    p.add_argument("--wl_high", type=float, default=750.0)

    p.add_argument("--prepare_stage1", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force_prepare", action="store_true")
    p.add_argument("--freq_start", type=int, default=0)
    p.add_argument("--freq_end", type=int, default=None)
    p.add_argument("--freq_indices", default="")

    # Actual submitted Stage-I defaults propagated by mf_baseline.py
    p.add_argument("--student_train_set", choices=("paired", "mix"), default="mix")
    p.add_argument("--student_val_set", choices=("paired", "mix"), default="paired")
    p.add_argument("--student_yscale", type=int, choices=(0, 1), default=1)
    p.add_argument("--student_y_scaler_fit", choices=("mix", "paired"), default="paired")
    p.add_argument("--student_hidden", type=int, nargs="+", default=[256, 256, 256])
    p.add_argument("--student_feat_dim", type=int, default=32)
    p.add_argument("--student_act", choices=("relu", "tanh", "gelu"), default="relu")
    p.add_argument(
        "--student_feat_act",
        choices=("leakyrelu", "identity", "same"),
        default="leakyrelu",
    )
    p.add_argument("--student_feat_leaky_slope", type=float, default=0.01)
    p.add_argument("--student_dropout", type=float, default=0.0)
    p.add_argument("--student_lr", type=float, default=3e-4)
    p.add_argument("--student_wd", type=float, default=1e-4)
    p.add_argument("--student_bs", type=int, default=256)
    p.add_argument("--student_epochs", type=int, default=2000)
    p.add_argument("--student_patience", type=int, default=100)
    p.add_argument("--student_min_delta", type=float, default=1e-4)
    p.add_argument("--student_print_every", type=int, default=20)

    # Actual submitted delta/SVGP defaults
    p.add_argument("--rho_ridge", type=float, default=1e-6)
    p.add_argument("--rho_intercept", type=int, choices=(0, 1), default=1)
    p.add_argument("--kernel", choices=("rbf", "matern"), default="matern")
    p.add_argument("--matern_nu", type=float, choices=(0.5, 1.5, 2.5), default=2.5)
    p.add_argument("--gp_ard", type=int, choices=(0, 1), default=1)
    p.add_argument("--svgp_M", type=int, default=64)
    p.add_argument("--svgp_steps", type=int, default=2000)
    p.add_argument("--svgp_lr", type=float, default=5e-3)
    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--ci_level", type=float, default=0.95)

    return p.parse_args(argv)


def main():
    args = parse_args()
    if args.prepare_stage1 and args.aggregate:
        raise ValueError("Choose only one of --prepare_stage1 and --aggregate.")
    if args.prepare_stage1:
        prepare_stage1(args)
    elif args.aggregate:
        aggregate(args)
    else:
        train_chunk(args)


if __name__ == "__main__":
    main()
