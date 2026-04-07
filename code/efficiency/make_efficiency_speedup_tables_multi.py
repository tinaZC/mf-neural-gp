#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_efficiency_speedup_tables_multi.py

Generate, in one run, multiple per-lfx outputs:
  (1) A per-run speedup table:
        - HF speedup (pure HF calls): n_hf_equivalent / n_hf_used
        - Wall-clock speedup (simulation-time): T_hf_only_equivalent / T_mf_used
  (2) Speedup-vs-RMSE_target curve data as .npz for each requested lfx.

Compared with the old version:
  - old: one run -> one curve_lfx -> one speedup_curves.npz
  - new: one run -> many curve_lfx values -> many subdirs, each with:
        speedup_table.csv
        speedup_curves.npz
        meta.json

Inputs:
  - sweep_results.csv
  - Each dataset_dir contains:
        times_hf_all.npy
        times_lf_all.npy

Outputs:
  out_dir/
    efficiency_out_lfx05/
      speedup_table.csv
      speedup_curves.npz
      meta.json
    efficiency_out_lfx10/
      speedup_table.csv
      speedup_curves.npz
      meta.json
    efficiency_out_lfx15/
      speedup_table.csv
      speedup_curves.npz
      meta.json

Example:
  python make_efficiency_speedup_tables_multi.py \
    --csv_path ../../result_out/mf_sweep_runs_baseline_nano_tm/sweep_results.csv \
    --out_dir  ../../result_out/mf_sweep_runs_baseline_nano_tm/efficiency_out \
    --curve_lfx_list 5,10,15
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Parsing helpers
# -----------------------------

def parse_hf_lfx(dataset_name: str) -> Tuple[int, int]:
    """
    Expected dataset_name format: 'hf{HF}_lfx{LFX}' e.g., 'hf200_lfx10'.
    Returns (HF, LFX) as ints.
    """
    import re
    m = re.search(r"hf(\d+)_lfx(\d+)", str(dataset_name))
    if m is None:
        raise ValueError(
            f"Cannot parse hf/lfx from dataset_name='{dataset_name}' "
            f"(expected like 'hf200_lfx10')."
        )
    return int(m.group(1)), int(m.group(2))


def parse_int_list(s: str) -> List[int]:
    vals: List[int] = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    if not vals:
        raise ValueError("Parsed empty integer list.")
    seen = set()
    uniq: List[int] = []
    for v in vals:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq


def load_times(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load per-sample simulation times.
    Required files:
      - times_hf_all.npy
      - times_lf_all.npy
    """
    hf_path = dataset_dir / "times_hf_all.npy"
    lf_path = dataset_dir / "times_lf_all.npy"

    if not hf_path.exists():
        raise FileNotFoundError(f"Missing required file: {hf_path}")
    if not lf_path.exists():
        raise FileNotFoundError(f"Missing required file: {lf_path}")

    t_hf = np.load(hf_path)
    t_lf = np.load(lf_path)

    if t_hf.ndim != 1 or t_lf.ndim != 1:
        raise ValueError(
            f"times arrays must be 1D. "
            f"Got t_hf.shape={t_hf.shape}, t_lf.shape={t_lf.shape} in {dataset_dir}"
        )

    return t_hf.astype(np.float64), t_lf.astype(np.float64)


# -----------------------------
# Inversion via piecewise linear interpolation
# -----------------------------

@dataclass
class PiecewiseCurve:
    """
    Represents a discrete curve y(x) by points (x_i, y_i) with x sorted ascending.
    Provides inversion: given y_target, find minimal x such that y(x) <= y_target,
    using robust segment crossing logic (handles mild non-monotonicity).
    """
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("x and y must be 1D arrays.")
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length.")
        if len(self.x) < 2:
            raise ValueError("Need at least 2 points for interpolation/inversion.")
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("x must be strictly increasing.")

    def invert_min_x_for_y_leq(self, y_target: float) -> float:
        """
        Returns the minimal x* such that y(x*) <= y_target, using piecewise linear segments.
        """
        x = self.x
        y = self.y
        yt = float(y_target)

        if y[0] <= yt:
            return float(x[0])

        if np.min(y) > yt:
            return float("nan")

        candidates: List[float] = []

        for i in range(len(x) - 1):
            x1, x2 = float(x[i]), float(x[i + 1])
            y1, y2 = float(y[i]), float(y[i + 1])

            if (y1 - yt) * (y2 - yt) <= 0:
                if y1 == yt and y2 == yt:
                    candidates.append(x1)
                    continue

                if y1 == y2:
                    continue

                frac = (y1 - yt) / (y1 - y2)
                frac = max(0.0, min(1.0, frac))
                x_star = x1 + (x2 - x1) * frac
                candidates.append(x_star)

        if not candidates:
            return float("nan")

        return float(np.min(candidates))


def build_curve_from_df(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    agg: str = "mean",
) -> PiecewiseCurve:
    """
    Group by x_col, aggregate y_col (mean/median), and return PiecewiseCurve.
    """
    if agg not in ("mean", "median"):
        raise ValueError(f"Unsupported agg='{agg}'. Use 'mean' or 'median'.")

    g = df.groupby(x_col, as_index=False)[y_col]
    d = g.mean() if agg == "mean" else g.median()
    d = d.sort_values(x_col)

    x = d[x_col].to_numpy(dtype=np.float64)
    y = d[y_col].to_numpy(dtype=np.float64)

    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError(f"Non-finite values found in curve columns: {x_col}, {y_col}")

    return PiecewiseCurve(x=x, y=y)


def build_time_curve_from_df(
    df: pd.DataFrame,
    x_col: str,
    time_col: str,
    agg: str = "mean",
) -> PiecewiseCurve:
    return build_curve_from_df(df=df, x_col=x_col, y_col=time_col, agg=agg)


def interp_time(curve: PiecewiseCurve, xq: float) -> float:
    x = curve.x
    y = curve.y
    if not np.isfinite(xq):
        return float("nan")
    if xq <= x[0]:
        return float(y[0])
    if xq >= x[-1]:
        return float(y[-1])

    j = np.searchsorted(x, xq, side="right") - 1
    j = int(np.clip(j, 0, len(x) - 2))
    x1, x2 = float(x[j]), float(x[j + 1])
    y1, y2 = float(y[j]), float(y[j + 1])

    if x2 == x1:
        return float(y1)

    frac = (xq - x1) / (x2 - x1)
    frac = max(0.0, min(1.0, frac))
    return float(y1 + (y2 - y1) * frac)


def interp_time_vec(curve: PiecewiseCurve, xq: np.ndarray) -> np.ndarray:
    out = np.full_like(xq, np.nan, dtype=np.float64)
    for i, v in enumerate(xq.tolist()):
        out[i] = interp_time(curve, float(v))
    return out


def invert_curve(curve: PiecewiseCurve, targets: np.ndarray) -> np.ndarray:
    out = np.full_like(targets, np.nan, dtype=np.float64)
    for i, t in enumerate(targets.tolist()):
        out[i] = curve.invert_min_x_for_y_leq(float(t))
    return out


# -----------------------------
# Shared preprocessing
# -----------------------------

def preprocess_df(csv_path: Path, skip_times: int) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    required_cols = [
        "dataset_name",
        "dataset_dir",
        "metrics.y_rmse.hf_only",
        "metrics.y_rmse.ar1",
        "metrics.y_rmse.ours",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in {csv_path}")

    hf_list = []
    lfx_list = []
    for dn in df["dataset_name"].astype(str).tolist():
        hf, lfx = parse_hf_lfx(dn)
        hf_list.append(hf)
        lfx_list.append(lfx)
    df["hf"] = hf_list
    df["lfx"] = lfx_list

    if "status" in df.columns:
        df_ok = df[df["status"].astype(str).str.upper().eq("OK")].copy()
        if len(df_ok) == 0:
            raise ValueError("No rows with status == OK.")
        df = df_ok

    if int(skip_times) == 0:
        sim_t_hf = []
        sim_t_lf = []
        sim_t_total = []
        gamma_rows = []

        for ddir, hf, lfx in zip(
            df["dataset_dir"].astype(str).tolist(),
            df["hf"].tolist(),
            df["lfx"].tolist(),
        ):
            ddir_p = Path(ddir)
            t_hf, t_lf = load_times(ddir_p)

            sim_t_hf.append(float(np.sum(t_hf)))
            sim_t_lf.append(float(np.sum(t_lf)))
            sim_t_total.append(float(np.sum(t_hf) + np.sum(t_lf)))

            if len(t_hf) != int(hf):
                raise ValueError(f"times_hf_all length {len(t_hf)} != hf {hf} in {ddir}")
            if len(t_lf) != int(hf) * int(lfx):
                raise ValueError(f"times_lf_all length {len(t_lf)} != hf*lfx {hf * lfx} in {ddir}")

            mean_hf = float(np.mean(t_hf))
            mean_lf = float(np.mean(t_lf))
            gamma_rows.append(mean_lf / mean_hf if mean_hf > 0 else float("nan"))

        df["sim_time_hf_sec"] = sim_t_hf
        df["sim_time_lf_sec"] = sim_t_lf
        df["sim_time_total_sec"] = sim_t_total
        df["gamma_mean_lf_over_hf"] = gamma_rows

    return df


# -----------------------------
# Per-lfx builder
# -----------------------------

def build_outputs_for_one_lfx(
    *,
    df: pd.DataFrame,
    curve_lfx: int,
    out_dir: Path,
    agg: str,
    n_targets: int,
    target_eps: float,
    skip_times: int,
    csv_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_curve_hfonly = build_curve_from_df(
        df=df,
        x_col="hf",
        y_col="metrics.y_rmse.hf_only",
        agg=agg,
    )
    hf_curve_ar1 = build_curve_from_df(
        df=df,
        x_col="hf",
        y_col="metrics.y_rmse.ar1",
        agg=agg,
    )

    def inv_eq_hf(curve: PiecewiseCurve, e: float) -> float:
        return curve.invert_min_x_for_y_leq(float(e))

    df_out = df.copy()
    df_out["eq_hf_for_ours_on_hfonly"] = [
        inv_eq_hf(hf_curve_hfonly, e) for e in df_out["metrics.y_rmse.ours"].to_numpy()
    ]
    df_out["eq_hf_for_ar1_on_hfonly"] = [
        inv_eq_hf(hf_curve_hfonly, e) for e in df_out["metrics.y_rmse.ar1"].to_numpy()
    ]
    df_out["hf_speedup_ours_vs_hfonly"] = df_out["eq_hf_for_ours_on_hfonly"] / df_out["hf"].astype(float)
    df_out["hf_speedup_ar1_vs_hfonly"] = df_out["eq_hf_for_ar1_on_hfonly"] / df_out["hf"].astype(float)

    df_out["eq_hf_for_ours_on_ar1"] = [
        inv_eq_hf(hf_curve_ar1, e) for e in df_out["metrics.y_rmse.ours"].to_numpy()
    ]
    df_out["hf_speedup_ours_vs_ar1"] = df_out["eq_hf_for_ours_on_ar1"] / df_out["hf"].astype(float)

    if int(skip_times) == 0:
        time_curve_hfonly = build_time_curve_from_df(
            df=df_out,
            x_col="hf",
            time_col="sim_time_hf_sec",
            agg=agg,
        )

        df_out["eq_sim_time_hfonly_for_ours_sec"] = [
            interp_time(time_curve_hfonly, xq)
            for xq in df_out["eq_hf_for_ours_on_hfonly"].to_numpy()
        ]
        df_out["eq_sim_time_hfonly_for_ar1_sec"] = [
            interp_time(time_curve_hfonly, xq)
            for xq in df_out["eq_hf_for_ar1_on_hfonly"].to_numpy()
        ]

        df_out["sim_speedup_ours_vs_hfonly"] = (
            df_out["eq_sim_time_hfonly_for_ours_sec"] / df_out["sim_time_total_sec"]
        )
        df_out["sim_speedup_ar1_vs_hfonly"] = (
            df_out["eq_sim_time_hfonly_for_ar1_sec"] / df_out["sim_time_total_sec"]
        )

    table_path = out_dir / "speedup_table.csv"
    df_out.to_csv(table_path, index=False)

    df_lfx = df_out[df_out["lfx"].astype(int) == int(curve_lfx)].copy()
    if len(df_lfx) == 0:
        raise ValueError(f"No rows found with lfx == {curve_lfx} for curve construction.")

    curve_ours = build_curve_from_df(
        df_lfx,
        x_col="hf",
        y_col="metrics.y_rmse.ours",
        agg=agg,
    )
    curve_ar1 = build_curve_from_df(
        df_lfx,
        x_col="hf",
        y_col="metrics.y_rmse.ar1",
        agg=agg,
    )

    e_min = max(np.min(hf_curve_hfonly.y), np.min(curve_ours.y), np.min(curve_ar1.y))
    e_max = min(np.max(hf_curve_hfonly.y), np.max(curve_ours.y), np.max(curve_ar1.y))

    if not (np.isfinite(e_min) and np.isfinite(e_max) and e_min < e_max):
        raise ValueError(
            f"Invalid target RMSE range for lfx={curve_lfx}: "
            f"e_min={e_min}, e_max={e_max}. Check your results coverage."
        )

    eps = float(target_eps)
    lo = e_min + eps
    hi = e_max - eps
    if lo >= hi:
        lo, hi = float(e_min), float(e_max)

    targets = np.linspace(lo, hi, int(n_targets), dtype=np.float64)

    hfmin_hfonly = invert_curve(hf_curve_hfonly, targets)
    hfmin_ours = invert_curve(curve_ours, targets)
    hfmin_ar1 = invert_curve(curve_ar1, targets)

    hf_speedup_curve_ours_vs_hfonly = hfmin_hfonly / hfmin_ours
    hf_speedup_curve_ar1_vs_hfonly = hfmin_hfonly / hfmin_ar1
    hf_speedup_curve_ours_vs_ar1 = hfmin_ar1 / hfmin_ours

    npz_dict = dict(
        rmse_targets=targets,
        hfmin_hfonly=hfmin_hfonly,
        hfmin_ours=hfmin_ours,
        hfmin_ar1=hfmin_ar1,
        hf_speedup_ours_vs_hfonly=hf_speedup_curve_ours_vs_hfonly,
        hf_speedup_ar1_vs_hfonly=hf_speedup_curve_ar1_vs_hfonly,
        hf_speedup_ours_vs_ar1=hf_speedup_curve_ours_vs_ar1,
        curve_lfx=np.array([int(curve_lfx)], dtype=np.int64),
        agg=np.array([agg], dtype=object),
    )

    if int(skip_times) == 0:
        time_curve_hfonly = build_time_curve_from_df(
            df=df_out,
            x_col="hf",
            time_col="sim_time_hf_sec",
            agg=agg,
        )
        time_curve_total_lfx = build_time_curve_from_df(
            df=df_lfx,
            x_col="hf",
            time_col="sim_time_total_sec",
            agg=agg,
        )

        time_hfonly_at_target = interp_time_vec(time_curve_hfonly, hfmin_hfonly)
        time_ours_at_target = interp_time_vec(time_curve_total_lfx, hfmin_ours)
        time_ar1_at_target = interp_time_vec(time_curve_total_lfx, hfmin_ar1)

        sim_speedup_curve_ours_vs_hfonly = time_hfonly_at_target / time_ours_at_target
        sim_speedup_curve_ar1_vs_hfonly = time_hfonly_at_target / time_ar1_at_target
        sim_speedup_curve_ours_vs_ar1 = time_ar1_at_target / time_ours_at_target

        npz_dict.update(dict(
            time_hfonly_at_target_sec=time_hfonly_at_target,
            time_ours_at_target_sec=time_ours_at_target,
            time_ar1_at_target_sec=time_ar1_at_target,
            sim_speedup_ours_vs_hfonly=sim_speedup_curve_ours_vs_hfonly,
            sim_speedup_ar1_vs_hfonly=sim_speedup_curve_ar1_vs_hfonly,
            sim_speedup_ours_vs_ar1=sim_speedup_curve_ours_vs_ar1,
        ))

    npz_path = out_dir / "speedup_curves.npz"
    np.savez(npz_path, **npz_dict)

    meta = {
        "csv_path": str(csv_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "curve_lfx": int(curve_lfx),
        "agg": agg,
        "n_targets": int(n_targets),
        "skip_times": int(skip_times),
        "notes": {
            "hf_speedup": "eq_hf_on_hfonly / hf_used",
            "sim_speedup": "eq_sim_time_hfonly / sim_time_total_used (HF+LF)",
            "eq_hf_on_hfonly": "piecewise-linear inversion on aggregated HF-only RMSE vs HF curve",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE][lfx={curve_lfx}] wrote: {table_path}")
    print(f"[DONE][lfx={curve_lfx}] wrote: {npz_path}")
    print(f"[DONE][lfx={curve_lfx}] wrote: {out_dir / 'meta.json'}")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv_path",
        type=str,
        default="../../result_out/mf_sweep_runs_baseline_nano_tm/sweep_results.csv",
        help="Path to sweep_results.csv",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="../../result_out/mf_sweep_runs_baseline_nano_tm/efficiency_out",
        help="Output directory",
    )
    ap.add_argument(
        "--curve_lfx_list",
        type=str,
        default="5,10,15",
        help="Comma-separated lfx values to generate in one run, e.g. 5,10,15",
    )
    ap.add_argument(
        "--agg",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Aggregation across seeds/runs at same HF.",
    )
    ap.add_argument(
        "--n_targets",
        type=int,
        default=31,
        help="Number of RMSE_target points for curves.",
    )
    ap.add_argument(
        "--target_eps",
        type=float,
        default=1e-6,
        help="Small epsilon to avoid choosing exactly min/max RMSE.",
    )
    ap.add_argument(
        "--skip_times",
        type=int,
        default=0,
        help="If 1, do not read times_*.npy and only compute HF-speedup.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    curve_lfx_list = parse_int_list(args.curve_lfx_list)
    df = preprocess_df(csv_path=csv_path, skip_times=int(args.skip_times))

    for curve_lfx in curve_lfx_list:
        subdir = out_root / f"efficiency_out_lfx{int(curve_lfx):02d}"
        build_outputs_for_one_lfx(
            df=df,
            curve_lfx=int(curve_lfx),
            out_dir=subdir,
            agg=str(args.agg),
            n_targets=int(args.n_targets),
            target_eps=float(args.target_eps),
            skip_times=int(args.skip_times),
            csv_path=csv_path,
        )

    batch_meta = {
        "csv_path": str(csv_path.resolve()),
        "out_root": str(out_root.resolve()),
        "curve_lfx_list": curve_lfx_list,
        "agg": str(args.agg),
        "n_targets": int(args.n_targets),
        "skip_times": int(args.skip_times),
    }
    (out_root / "batch_meta.json").write_text(json.dumps(batch_meta, indent=2), encoding="utf-8")
    print(f"[DONE] batch manifest -> {out_root / 'batch_meta.json'}")


if __name__ == "__main__":
    main()
