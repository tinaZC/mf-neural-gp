#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the main-paper figure for retrospective HF acquisition experiments.

Figure layout (single panel):
(a) Multi-target aggregate acquisition curve

Designed for outputs produced by:
  run_retrospective_acquisition_with_baseline_tm_v6.py

Expected inputs:
- retro_dir/retro_acq_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def read_csv_rows(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# -----------------------------------------------------------------------------
# Summary parsing
# -----------------------------------------------------------------------------

def load_summary_table(summary_csv: Path, methods: List[str]) -> List[dict]:
    rows = read_csv_rows(summary_csv)
    out: List[dict] = []
    for r in rows:
        if r["method"] not in methods:
            continue
        out.append({
            "target_global_idx": int(r["target_global_idx"]),
            "target_local_id": int(r["target_local_id"]),
            "target_row_id": int(r["target_row_id"]),
            "method": str(r["method"]),
            "step": int(r["step"]),
            "n_known_hf": int(r["n_known_hf"]),
            "best_true_target_rmse": float(r["best_true_target_rmse"]),
            "headroom": float(r["headroom"]),
            "initial_best": float(r["initial_best"]),
            "oracle_pool_best": float(r["oracle_pool_best"]),
        })
    if not out:
        raise RuntimeError(f"No usable rows found in {summary_csv} for methods={methods}")
    return out


def group_curves(rows: List[dict]) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    grouped: Dict[str, Dict[int, List[Tuple[int, float]]]] = {}
    for r in rows:
        grouped.setdefault(r["method"], {}).setdefault(r["target_global_idx"], []).append(
            (r["n_known_hf"], r["best_true_target_rmse"])
        )

    out: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for method, tg_map in grouped.items():
        out[method] = {}
        for tgt, pairs in tg_map.items():
            pairs = sorted(pairs, key=lambda z: (z[0], z[1]))
            xs = np.asarray([p[0] for p in pairs], dtype=np.int64)
            ys = np.asarray([p[1] for p in pairs], dtype=np.float64)

            # keep the last occurrence for duplicated budgets
            keep = np.ones(xs.shape[0], dtype=bool)
            for i in range(xs.shape[0] - 1):
                if xs[i] == xs[i + 1]:
                    keep[i] = False

            out[method][tgt] = (xs[keep], ys[keep])
    return out


def build_common_budget_grid(curves: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]], methods: List[str]) -> np.ndarray:
    budgets: List[int] = []
    for method in methods:
        if method not in curves:
            continue
        for xs, _ in curves[method].values():
            budgets.extend(xs.tolist())
    if not budgets:
        raise RuntimeError("No budgets found while building common grid")
    return np.asarray(sorted(set(int(v) for v in budgets)), dtype=np.int64)


def carry_forward_values(xs: np.ndarray, ys: np.ndarray, query_budgets: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.int64)
    ys = np.asarray(ys, dtype=np.float64)
    query_budgets = np.asarray(query_budgets, dtype=np.int64)

    pos = np.searchsorted(xs, query_budgets, side="right") - 1
    pos = np.clip(pos, 0, len(xs) - 1)
    return ys[pos]


def aggregate_curve(
    curves: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    method: str,
    budgets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mats: List[np.ndarray] = []
    for _, (xs, ys) in sorted(curves[method].items()):
        mats.append(carry_forward_values(xs, ys, budgets))
    if not mats:
        raise RuntimeError(f"No target curves found for method={method}")
    arr = np.stack(mats, axis=0)
    med = np.median(arr, axis=0)
    q25 = np.quantile(arr, 0.25, axis=0)
    q75 = np.quantile(arr, 0.75, axis=0)
    return med, q25, q75


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def prettify_method_name(method: str) -> str:
    mp = {
        "hf_only": "HF-only",
        "ar1": "co-kriging",
        "ours_mean": "Ours",
        "random": "random",
    }
    return mp.get(method, method)


def method_style(method: str) -> dict:
    # use explicit colors so names stay consistent if the user changes method order
    styles = {
        "hf_only": {"color": "#1f77b4", "linewidth": 2.2},
        "ar1": {"color": "#ff7f0e", "linewidth": 2.2},
        "ours_mean": {"color": "#2ca02c", "linewidth": 2.2},
        "random": {"color": "#9467bd", "linewidth": 2.2},
    }
    return styles.get(method, {"linewidth": 2.2})


def plot_aggregate_curve_only(
    *,
    curves: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    budgets: np.ndarray,
    methods: List[str],
    out_path: Path,
    dpi: int,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)

    for method in methods:
        med, q25, q75 = aggregate_curve(curves, method, budgets)
        st = method_style(method)
        ax.plot(
            budgets,
            med,
            label=prettify_method_name(method),
            color=st.get("color", None),
            linewidth=st.get("linewidth", 2.2),
        )
        ax.fill_between(
            budgets,
            q25,
            q75,
            color=st.get("color", None),
            alpha=0.18,
            linewidth=0.0,
        )

    ax.set_xlabel("Number of known HF samples")
    ax.set_ylabel("Best target RMSE in known set")
    ax.set_title("(a) Aggregate acquisition curve")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False)

    if title.strip():
        fig.suptitle(title.strip())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retro_dir", type=str, default="./retro_acq_runs_tm")
    ap.add_argument("--methods", type=str, default="hf_only,ar1,ours_mean,random")
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--out_path", type=str, default="./retro_acq_runs_tm/main_paper_curve_only.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    retro_dir = Path(args.retro_dir).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]

    summary_csv = retro_dir / "retro_acq_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    rows = load_summary_table(summary_csv, methods)
    curves = group_curves(rows)
    budgets = build_common_budget_grid(curves, methods)

    print(f"[INFO] retro_dir : {retro_dir}")
    print(f"[INFO] methods  : {methods}")
    print(f"[INFO] out_path : {out_path}")

    plot_aggregate_curve_only(
        curves=curves,
        budgets=budgets,
        methods=methods,
        out_path=out_path,
        dpi=int(args.dpi),
        title=str(args.title),
    )

    print("[DONE] figure saved")
    print(f"[OUT]  {out_path}")


if __name__ == "__main__":
    main()
