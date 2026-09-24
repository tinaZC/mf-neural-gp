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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETRO_DIR = REPO_ROOT / "result_out/final_analysis/frozen_inputs/acquisition"
DEFAULT_OUT = REPO_ROOT / "result_out/final_analysis/figures/_fig_acquisition_tm.pdf"


# ---------------------------------------------------------------------
# Publication legend ordering
# Keep plotting/data order unchanged; only move the proposed method
# "Neural-GP MF" to the final legend position.
# ---------------------------------------------------------------------

# ===== unified npj-style figure settings =====

COLOR_HF = "#1f77b4"
COLOR_COK = "#ff7f0e"
COLOR_OURS = "#2ca02c"
COLOR_RANDOM = "#9467bd"

# COLOR_HF = "#0072B2"        # blue
# COLOR_COK = "#E69F00"       # orange
# COLOR_OURS = "#009E73"      # green
# COLOR_RANDOM = "#CC79A7"    # purple

def apply_npj_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 13,
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


def add_panel_note(ax, text):
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11,
    )


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


def build_common_budget_grid(
    curves: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    methods: List[str],
) -> np.ndarray:
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
        "ar1": "AR1 / co-kriging",
        "ours_mean": "Neural–GP MF",
        "random": "Random",
    }
    return mp.get(method, method)


def method_style(method: str) -> dict:
    # Colors follow the unified paper palette. Markers make each acquisition step visible.
    styles = {
        "random": {
            "color": COLOR_RANDOM,
            "linewidth": 1.5,
            "marker": "o",
            "markersize": 4.5,
            "markerfacecolor": "white",
            "markeredgewidth": 1.0,
            "line_alpha": 0.78,
            "fill_alpha": 0.04,
            "zorder": 2,
        },
        "hf_only": {
            "color": COLOR_HF,
            "linewidth": 2.0,
            "marker": "s",
            "markersize": 5.0,
            "markerfacecolor": "white",
            "markeredgewidth": 1.1,
            "fill_alpha": 0.07,
            "zorder": 3,
        },
        "ar1": {
            "color": COLOR_COK,
            "linewidth": 2.0,
            "marker": "^",
            "markersize": 5.0,
            "markerfacecolor": "white",
            "markeredgewidth": 1.1,
            "fill_alpha": 0.07,
            "zorder": 4,
        },
        "ours_mean": {
            "color": COLOR_OURS,
            "linewidth": 2.3,
            "marker": "*",
            "markersize": 9.0,
            "markerfacecolor": COLOR_OURS,
            "markeredgewidth": 0.8,
            "fill_alpha": 0.09,
            "zorder": 5,
        },
    }
    return styles.get(method, {
        "linewidth": 2.0,
        "marker": "o",
        "markersize": 4.5,
        "markerfacecolor": "white",
        "markeredgewidth": 1.0,
        "fill_alpha": 0.10,
        "zorder": 3,
    })


def plot_aggregate_curve_only(
    *,
    curves: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    budgets: np.ndarray,
    methods: List[str],
    out_path: Path,
    dpi: int,
    title: str,
) -> None:
    apply_npj_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)

    for method in methods:
        med, q25, q75 = aggregate_curve(curves, method, budgets)
        st = method_style(method)

        # Draw the IQR band first so the median line and markers remain visible.
        ax.fill_between(
            budgets,
            q25,
            q75,
            color=st.get("color", None),
            alpha=st.get("fill_alpha", 0.10),
            linewidth=0.0,
            zorder=st.get("zorder", 3) - 1,
        )

        ax.plot(
            budgets,
            med,
            label=prettify_method_name(method),
            color=st.get("color", None),
            linewidth=st.get("linewidth", 2.0),
            marker=st.get("marker", "o"),
            markersize=st.get("markersize", 4.5),
            markerfacecolor=st.get("markerfacecolor", "white"),
            markeredgecolor=st.get("color", None),
            markeredgewidth=st.get("markeredgewidth", 1.0),
            markevery=1,
            alpha=st.get("line_alpha", 1.0),
            zorder=st.get("zorder", 3),
        )

    ax.set_xlabel(r"Number of known HF samples, $N_h^{\mathrm{known}}$")
    ax.set_ylabel("Best true target-matching RMSE")

    # Acquisition budgets are discrete, so fixed ticks make the step structure clearer.
    ax.set_xlim(int(budgets.min()) - 2, int(budgets.max()) + 2)
    ax.set_xticks(np.arange(50, int(budgets.max()) + 1, 10))

    # Add a small vertical margin so the bands do not touch the frame.
    y_values = []
    for method in methods:
        med, q25, q75 = aggregate_curve(curves, method, budgets)
        y_values.extend(q25.tolist())
        y_values.extend(q75.tolist())
        y_values.extend(med.tolist())
    if y_values:
        ymin = float(np.nanmin(y_values))
        ymax = float(np.nanmax(y_values))
        pad = 0.06 * max(ymax - ymin, 1e-6)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Force publication legend order: random reference first, proposed method last.
    handles, labels = ax.get_legend_handles_labels()
    legend_order = ["Random", "HF-only", "AR1 / co-kriging", "Neural–GP MF"]

    handle_map = {lab: h for h, lab in zip(handles, labels)}
    ordered_handles = [handle_map[lab] for lab in legend_order if lab in handle_map]
    ordered_labels = [lab for lab in legend_order if lab in handle_map]

    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper right",
        frameon=False,
        handlelength=2.2,
        borderpad=0.4,
        labelspacing=0.4,
    )

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
    ap.add_argument("--retro_dir", type=str, default=str(DEFAULT_RETRO_DIR))
    ap.add_argument("--methods", type=str, default="hf_only,ar1,ours_mean,random")
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--out_path", type=str, default=str(DEFAULT_OUT))
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
