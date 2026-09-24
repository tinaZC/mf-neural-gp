import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mf_train_baseline.mf_utils import (
    ci_coverage_y,
    ci_width_y,
    gaussian_nll,
    gaussian_nlpd,
    r2_score,
    rmse,
    set_seed,
)

from .common import load_bundle
from .common import HFPCATransform
from .nargp_core import predict_nargp, train_nargp


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _crop_spectra(data, wl_low, wl_high):
    mask = np.ones(data.wavelengths.shape, dtype=bool)
    if wl_low is not None:
        mask &= data.wavelengths >= wl_low
    if wl_high is not None:
        mask &= data.wavelengths <= wl_high
    keep = np.where(mask)[0]
    if keep.size < 2:
        raise ValueError(f"Wavelength crop [{wl_low}, {wl_high}] retained {keep.size} points.")

    axis_size = len(data.wavelengths)
    spectra = {}
    for fidelity_name, fidelity in (
        ("hf", data.hf),
        ("lf_paired", data.lf_paired),
        ("lf_unpaired", data.lf_unpaired),
    ):
        spectra[fidelity_name] = {}
        for split in ("train", "val", "test"):
            y = fidelity[split]["y"]
            if y.shape[1] == axis_size:
                output_keep = keep
            elif y.shape[1] == 2 * axis_size:
                # MTM stores real and imaginary channels on the same frequency axis.
                output_keep = np.concatenate([keep, keep + axis_size])
            else:
                raise ValueError(
                    f"{fidelity_name}/{split}: response width {y.shape[1]} is not "
                    f"compatible with spectral axis length {axis_size}."
                )
            spectra[fidelity_name][split] = y[:, output_keep].astype(np.float32)
    return (
        data.wavelengths[keep].astype(np.float32),
        data.idx_wavelength[keep].astype(np.int64),
        spectra,
    )

def _metrics(y_true, mean, variance, ci_level):
    std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
    return {
        "rmse": rmse(y_true, mean),
        "r2": r2_score(y_true, mean),
        "gaussian_nll": gaussian_nll(y_true, mean, var=variance),
        "gaussian_nlpd": gaussian_nlpd(y_true, mean, var=variance),
        "coverage_raw": ci_coverage_y(y_true, mean, std, ci_level),
        "interval_width_raw": ci_width_y(std, ci_level),
    }


def _save_predictions(out_dir, split, mean, variance, indices):
    pred_dir = out_dir / "pred_arrays"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / f"{split}_mean.npy", mean.astype(np.float32))
    np.save(pred_dir / f"{split}_variance.npy", variance.astype(np.float32))
    np.save(pred_dir / f"{split}_indices.npy", indices.astype(np.int64))


def _mc_convergence(bundle, x_val, fpca, device, seed):
    sample_counts = (32, 128, 256, 512, 1000)
    predictions = {}
    for count in sample_counts:
        latent_mean, latent_var = predict_nargp(
            bundle, x_val, mc_samples=count, device=device, seed=seed
        )
        mean = fpca.inverse(latent_mean)
        std = np.sqrt(np.maximum(fpca.variance_to_y(latent_var), 0.0))
        predictions[count] = (mean, std)
    reference_mean, reference_std = predictions[1000]
    return [
        {
            "mc_samples": count,
            "mean_rmse_vs_1000": rmse(reference_mean, predictions[count][0]),
            "mean_max_abs_vs_1000": float(
                np.max(np.abs(reference_mean - predictions[count][0]))
            ),
            "std_rmse_vs_1000": rmse(reference_std, predictions[count][1]),
            "std_max_abs_vs_1000": float(
                np.max(np.abs(reference_std - predictions[count][1]))
            ),
        }
        for count in sample_counts
    ]


def run(args):
    set_seed(int(args.seed))
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    data = load_bundle(args.data_dir)
    wavelengths, idx_wavelength, spectra = _crop_spectra(
        data, args.wl_low, args.wl_high
    )

    fpca = HFPCATransform(
        var_ratio=args.fpca_var_ratio,
        max_dim=args.fpca_max_dim,
        dim=args.fpca_dim,
        ridge=args.fpca_ridge,
        random_state=args.seed,
    ).fit(spectra["hf"]["train"])

    x_lf_train = np.concatenate(
        [data.lf_paired["train"]["x"], data.lf_unpaired["train"]["x"]], axis=0
    ).astype(np.float32)
    y_lf_train = np.concatenate(
        [spectra["lf_paired"]["train"], spectra["lf_unpaired"]["train"]],
        axis=0,
    ).astype(np.float32)
    z_lf_train = fpca.transform(y_lf_train)
    z_lf_paired_train = fpca.transform(spectra["lf_paired"]["train"])
    z_hf_train = fpca.transform(spectra["hf"]["train"])

    bundle = train_nargp(
        x_lf_train,
        z_lf_train,
        data.hf["train"]["x"],
        z_lf_paired_train,
        z_hf_train,
        device=device,
        inducing=args.inducing,
        steps=args.steps,
        lr=args.lr,
        kernel_name=args.kernel,
        seed=args.seed,
    )

    _sync(device)
    inference_started = time.perf_counter()
    z_mean_val, z_var_val = predict_nargp(
        bundle,
        data.hf["val"]["x"],
        mc_samples=args.mc_samples,
        device=device,
        seed=args.seed + 100_000,
    )
    z_mean_test, z_var_test = predict_nargp(
        bundle,
        data.hf["test"]["x"],
        mc_samples=args.mc_samples,
        device=device,
        seed=args.seed + 200_000,
    )
    mean_val = fpca.inverse(z_mean_val)
    variance_val = fpca.variance_to_y(z_var_val)
    mean_test = fpca.inverse(z_mean_test)
    variance_test = fpca.variance_to_y(z_var_test)
    _sync(device)
    inference_s = time.perf_counter() - inference_started

    mc_convergence = None
    mc_convergence_s = 0.0
    if args.mc_convergence:
        _sync(device)
        convergence_started = time.perf_counter()
        mc_convergence = _mc_convergence(
            bundle,
            data.hf["val"]["x"],
            fpca,
            device,
            args.seed + 300_000,
        )
        _sync(device)
        mc_convergence_s = time.perf_counter() - convergence_started

    arrays = (mean_val, variance_val, mean_test, variance_test)
    if not all(np.isfinite(array).all() for array in arrays):
        raise FloatingPointError("FPCA-NARGP produced non-finite prediction values.")
    if mean_test.shape != spectra["hf"]["test"].shape:
        raise ValueError(
            f"Test prediction shape {mean_test.shape} != target shape "
            f"{spectra['hf']['test'].shape}."
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_predictions(
        out_dir, "val", mean_val, variance_val, data.hf["val"]["idx"]
    )
    _save_predictions(
        out_dir, "test", mean_test, variance_test, data.hf["test"]["idx"]
    )
    np.save(out_dir / "pred_arrays" / "wavelengths.npy", wavelengths)
    np.save(out_dir / "pred_arrays" / "idx_wavelength.npy", idx_wavelength)
    if mc_convergence is not None:
        with (out_dir / "mc_convergence.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "split": "validation",
                    "reference_mc_samples": 1000,
                    "same_trained_model": True,
                    "rows": mc_convergence,
                },
                handle,
                indent=2,
            )

    setting = data.root.name
    total_model_wall_s = (
        bundle.stage1_train_s + bundle.stage2_train_s + inference_s
    )
    metrics = {
        "val": _metrics(
            spectra["hf"]["val"], mean_val, variance_val, args.ci_level
        ),
        "test": _metrics(
            spectra["hf"]["test"], mean_test, variance_test, args.ci_level
        ),
    }
    timing = {
        "method": "FPCA-NARGP",
        "dataset": args.dataset,
        "setting": setting,
        "seed": int(args.seed),
        "device": str(device),
        "data_dir": str(data.root.resolve()),
        "out_dir": str(out_dir),
        "stage1_train_s": bundle.stage1_train_s,
        "stage2_train_s": bundle.stage2_train_s,
        "inference_val_test_s": float(inference_s),
        "mc_convergence_validation_s": float(mc_convergence_s),
        "total_model_wall_s": float(total_model_wall_s),
        "n_hf_train": int(len(data.hf["train"]["x"])),
        "n_lf_train_total": int(len(x_lf_train)),
        "excludes_em_simulation_time": True,
    }
    report = {
        "method": "FPCA-NARGP",
        "run_kind": args.run_kind,
        "scientific_result": args.run_kind == "scientific",
        "dataset": args.dataset,
        "setting": setting,
        "data_dir": str(data.root.resolve()),
        "seed": int(args.seed),
        "device": str(device),
        "sample_counts": {
            "hf_train": int(len(data.hf["train"]["x"])),
            "hf_val": int(len(data.hf["val"]["x"])),
            "hf_test": int(len(data.hf["test"]["x"])),
            "lf_paired_train": int(len(data.lf_paired["train"]["x"])),
            "lf_unpaired_train": int(len(data.lf_unpaired["train"]["x"])),
        },
        "spectral_range": [float(wavelengths.min()), float(wavelengths.max())],
        "wavelength_count": int(len(wavelengths)),
        "output_coordinate_count": int(spectra["hf"]["train"].shape[1]),
        "output_channel_count": int(
            spectra["hf"]["train"].shape[1] // len(wavelengths)
        ),
        "fpca_effective_dim": fpca.effective_dim,
        "fpca_protocol": {
            "fit_data": "HF train only",
            "spectral_scaler": "StandardScaler(HF train)",
            "fpca_ridge": float(args.fpca_ridge),
            "fpca_random_state": int(args.seed),
            "fpca_var_ratio": float(args.fpca_var_ratio),
            "fpca_max_dim": int(args.fpca_max_dim),
            "fpca_dim": int(args.fpca_dim),
            "latent_scaler": "StandardScaler(HF-train FPCA scores)",
        },
        "nargp": {
            "backend": "sparse variational GPyTorch",
            "kernel": "k_rho(z,z') * k_x(x,x') + k_delta(x,x')",
            "ard": True,
            "likelihood": "GaussianLikelihood",
            "likelihood_noise_init": "0.01 * Var(training target)",
            "likelihood_noise_optimization": "joint from first Adam step; never frozen",
            "x_scaler_fit": "combined LF train only",
            "stage2_lf_input": "observed paired LF train FPCA score",
            "uncertainty_propagation": (
                "MC LF latent posterior; E[var_H|f_L] + Var[E(H|f_L)]"
            ),
            "mc_samples": int(args.mc_samples),
            "mc_convergence_counts": [32, 128, 256, 512, 1000],
            "inducing": int(args.inducing),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "kernel_name": args.kernel,
        },
        "uq_calibration": "none; raw posterior uncertainty only",
        "ci_level": float(args.ci_level),
        "metrics": metrics,
        "mc_convergence": mc_convergence,
        "timing_file": "timing.json",
        "prediction_dir": "pred_arrays",
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (out_dir / "timing.json").open("w", encoding="utf-8") as handle:
        json.dump(timing, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the FPCA-NARGP multi-fidelity baseline."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=("TM", "AB", "MTM"), default="TM")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--inducing", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--fpca_var_ratio", type=float, default=0.999)
    parser.add_argument("--fpca_max_dim", type=int, default=64)
    parser.add_argument("--fpca_dim", type=int, default=0)
    parser.add_argument("--fpca_ridge", type=float, default=1e-8)
    parser.add_argument("--mc_samples", type=int, default=1000)
    parser.add_argument("--mc_convergence", type=int, choices=(0, 1), default=0)
    parser.add_argument("--kernel", choices=("rbf", "matern"), default="rbf")
    parser.add_argument("--wl_low", type=float, default=None)
    parser.add_argument("--wl_high", type=float, default=None)
    parser.add_argument("--ci_level", type=float, default=0.95)
    parser.add_argument(
        "--run_kind", choices=("smoke", "scientific"), default="scientific"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
