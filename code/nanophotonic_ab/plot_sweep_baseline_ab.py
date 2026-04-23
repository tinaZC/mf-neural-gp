#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_sweep_results_baseline_v4.py

Changes vs v3:
1) RMSE plots: boxplot + seed scatter, with mean±std markers, to show distribution per x point.
2) Improvement%: only ONE figure requested:
   - x-axis groups: HF budgets (default 30/50/100/200 if present)
   - two bars per group:
       * ours vs hf_only improvement%  = (rmse_hf - rmse_ours) / rmse_hf * 100
       * ours vs ar1 improvement%      = (rmse_ar1 - rmse_ours) / rmse_ar1 * 100
   - annotate bars with value text.

Data source:
- Rebuild sweep table by scanning runs_root/**/report.json (no trust in an existing sweep_results.csv).
- Each report must have metrics.y_rmse.{hf_only, ar1, ours}. Missing values mark row BAD.

Aggregation for the improvement bar chart:
- For each HF budget, we take ALL runs (all lf_mult, all seeds) and compute per-run improvements.
- Then we report the mean improvement% across those runs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ===== unified npj-style figure settings =====
COLOR_HF = "#0072B2"        # blue
COLOR_COK = "#E69F00"       # orange
COLOR_OURS = "#009E73"      # green
COLOR_RANDOM = "#CC79A7"    # purple

def apply_npj_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        # overall text
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
        fontsize=13,
    )



HF_LFM_RE = re.compile(r"hf(\d+)_lfx(\d+)", re.IGNORECASE)
SEED_RE = re.compile(r"seed(\d+)", re.IGNORECASE)


def parse_hf_lfmult_from_text(s: str) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(s, str):
        return None, None
    m = HF_LFM_RE.search(s)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def find_seed_in_path(p: Path) -> Optional[int]:
    for part in p.parts[::-1]:
        m = SEED_RE.fullmatch(part) or SEED_RE.search(part)
        if m:
            return int(m.group(1))
    return None


def find_hf_lfmult_from_path(p: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    for part in p.parts:
        hf, lf = parse_hf_lfmult_from_text(part)
        if hf is not None and lf is not None:
            return hf, lf, part
    return None, None, None


def read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def improvement_percent(rmse_base: np.ndarray, rmse_new: np.ndarray) -> np.ndarray:
    """
    percent reduction of rmse: (base - new) / base * 100
    """
    rmse_base = np.asarray(rmse_base, dtype=float)
    rmse_new = np.asarray(rmse_new, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (rmse_base - rmse_new) / rmse_base * 100.0


def build_sweep_table_from_runs(runs_root: Path) -> pd.DataFrame:
    runs_root = runs_root.expanduser().resolve()
    reports = list(runs_root.glob("**/report.json"))

    rows: List[Dict[str, Any]] = []
    for rp in reports:
        seed = find_seed_in_path(rp)
        if seed is None:
            continue

        try:
            rep = read_json(rp)
        except Exception:
            continue

        ds_name = str(rep.get("dataset_name", ""))
        hf, lf = parse_hf_lfmult_from_text(ds_name)
        tag = None
        if hf is None or lf is None:
            hf, lf, tag = find_hf_lfmult_from_path(rp)
        else:
            tag = ds_name
        if hf is None or lf is None:
            continue

        y_rmse = (rep.get("metrics", {}) or {}).get("y_rmse", {}) or {}
        r_hf = safe_float(y_rmse.get("hf_only"))
        r_ar1 = safe_float(y_rmse.get("ar1"))
        r_ours = safe_float(y_rmse.get("ours"))

        status = "OK" if (np.isfinite(r_hf) and np.isfinite(r_ar1) and np.isfinite(r_ours)) else "BAD"

        rows.append(
            {
                "report_path": str(rp),
                "run_dir": str(rp.parent),
                "seed": int(seed),
                "hf": int(hf),
                "lf_mult": int(lf),
                "dataset_name": tag if tag is not None else f"hf{hf}_lfx{lf}",
                "status": status,
                "metrics.y_rmse.hf_only": r_hf,
                "metrics.y_rmse.ar1": r_ar1,
                "metrics.y_rmse.ours": r_ours,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No valid sweep runs found under: {runs_root}")
    return df


def _jitter(x: np.ndarray, scale: float = 0.07, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return x + rng.normal(0.0, scale, size=x.shape[0])


def _finite(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    return vals[np.isfinite(vals)]


def _mean_std(vals: np.ndarray) -> Tuple[float, float]:
    v = _finite(vals)
    if v.size == 0:
        return float("nan"), float("nan")
    mu = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if v.size >= 2 else 0.0
    return mu, sd



def plot_rmse_boxplots_smallmultiples(
    df: pd.DataFrame,
    x_col: str,
    panel_col: str,
    out_png: Path,
    *,
    rmse_style: str = "box",          # "box" | "errorbar" | "boxline" | "overlay"
    connect: str = "none",           # "none" | "dashed"
    center: str = "median",          # "median" | "mean"
    q_lo: float = 0.25,
    q_hi: float = 0.75,
    show_seed_points: int = 1,
    annotate_line_mean: int = 1,
    annotate_box_mean_var: int = 1,
    annotate_line_digits: int = 4,
    annotate_sci_sig: int = 2,
    title: str,
    xlabel: str,
    ylabel: str,
    max_hf_plot: Optional[int] = None,
) -> None:
    """Small-multiples RMSE plot split by `panel_col`.

    Notes:
      - x-values are discrete (HF budgets). We DO NOT draw continuous uncertainty bands between points.
      - For rmse_style="boxline": top axis shows trend line (center statistic) on aligned x,
      - For rmse_style="overlay": ONE axis: overlay 3-method trend lines + Ours-only boxplots.
        bottom axis shows *dodged* boxplots + seed points, and annotations avoid overlap by placing
        above local maxima with a margin and a background bbox.
    """
    import matplotlib.gridspec as gridspec

    out_png.parent.mkdir(parents=True, exist_ok=True)

    rmse_style = str(rmse_style).lower().strip()
    connect = str(connect).lower().strip()
    center = str(center).lower().strip()
    if rmse_style not in {"box", "errorbar", "boxline", "overlay"}:
        raise ValueError(f"rmse_style must be box|errorbar|boxline|overlay, got {rmse_style!r}")
    if connect not in {"none", "dashed"}:
        raise ValueError(f"connect must be none|dashed, got {connect!r}")
    if center not in {"median", "mean"}:
        raise ValueError(f"center must be median|mean, got {center!r}")

    if max_hf_plot is not None and x_col == "hf":
        df = df[df["hf"] <= float(max_hf_plot)].copy()

    x_values = sorted(df[x_col].dropna().unique().tolist())
    panel_values = sorted(df[panel_col].dropna().unique().tolist())

    # categorical x positions for dodging
    x_pos_map = {xv: float(i) for i, xv in enumerate(x_values)}
    x_ticks = np.arange(len(x_values), dtype=float)
    x_ticklabels = [str(int(x)) if float(x).is_integer() else str(x) for x in x_values]

    n_panels = len(panel_values)
    if n_panels <= 3:
        nrows, ncols = 1, n_panels
        figsize = (6.2 * n_panels, 12.2 if rmse_style == "boxline" else 8.2)
    else:
        ncols = 3
        nrows = int(np.ceil(n_panels / ncols))
        figsize = (6.2 * ncols, (12.2 if rmse_style == "boxline" else 8.2) * nrows)

    # (label, column, color, dodge_dx)  -- dx in categorical-x units
    methods = [
        ("HF-only", "metrics.y_rmse.hf_only", COLOR_HF, -0.28),
        ("co-kriging",     "metrics.y_rmse.ar1",     COLOR_COK,  0.00),
        ("Ours",    "metrics.y_rmse.ours",    COLOR_OURS, +0.28),
    ]

    def _center(vals: np.ndarray) -> float:
        v = _finite(vals)
        if v.size == 0:
            return np.nan
        return float(np.median(v)) if center == "median" else float(np.mean(v))

    def _mean_var(vals: np.ndarray) -> Tuple[float, float]:
        v = _finite(vals)
        if v.size == 0:
            return (np.nan, np.nan)
        mu = float(np.mean(v))
        var = float(np.var(v, ddof=1)) if v.size >= 2 else 0.0
        return (mu, var)

    def _format_sci(x: float, sig: int = 2) -> str:
        if not np.isfinite(x):
            return "nan"
        return f"{x:.{sig}e}"

    fig = plt.figure(figsize=figsize)
    if rmse_style == "boxline":
        outer = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.16, hspace=0.22)
    else:
        outer = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.16, hspace=0.32)

    for pi, pv in enumerate(panel_values):
        r = pi // ncols
        c = pi % ncols

        sub = df[df[panel_col] == pv]

        if rmse_style == "boxline":
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[r, c],
                height_ratios=[1, 1],  # user requested
                hspace=0.06
            )
            ax_top = fig.add_subplot(inner[0, 0])
            ax_bot = fig.add_subplot(inner[1, 0], sharex=ax_top)
        elif rmse_style == "overlay":
            ax = fig.add_subplot(outer[r, c])
        else:
            ax = fig.add_subplot(outer[r, c])

        # ---- OVERLAY: one axis with 3-method trend lines + Ours-only boxplots ----
        if rmse_style == "overlay":
            # gather all values for y-range
            all_vals = []
            for _lbl, col, _color, _dx in methods:
                all_vals.append(_finite(sub[col].to_numpy(dtype=float)))
            all_v = np.concatenate([v for v in all_vals if v.size > 0], axis=0) if any(v.size > 0 for v in all_vals) else np.array([0.0, 1.0])
            y_min = float(np.min(all_v))
            y_max = float(np.max(all_v))
            y_span = max(1e-12, y_max - y_min)

            # 1) Ours boxplots (no dodge)
            ours_col = "metrics.y_rmse.ours"
            ours_color = "C2"
            data_list = []
            pos_list = []
            for xv in x_values:
                vals = _finite(sub[sub[x_col] == xv][ours_col].to_numpy(dtype=float))
                if vals.size == 0:
                    continue
                data_list.append(vals)
                pos_list.append(x_pos_map[xv])

            if len(data_list) > 0:
                bp = ax.boxplot(
                    data_list,
                    positions=pos_list,
                    widths=0.34,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(ours_color)
                    patch.set_alpha(0.18)
                    patch.set_edgecolor(ours_color)
                    patch.set_linewidth(1.8)
                for k in ("whiskers", "caps"):
                    for ln in bp[k]:
                        ln.set_color(ours_color)
                        ln.set_linewidth(1.6)
                for med in bp["medians"]:
                    med.set_color(ours_color)
                    med.set_linewidth(2.4)

            # annotate ours boxes with mean±std (scientific notation)
            if int(annotate_box_mean_var) == 1:
                for xv in x_values:
                    vals = _finite(sub[sub[x_col] == xv][ours_col].to_numpy(dtype=float))
                    if vals.size == 0:
                        continue
                    mu, sd = _mean_std(vals)
                    if not np.isfinite(mu) or not np.isfinite(sd):
                        continue
                    x0 = x_pos_map[xv]
                    y0 = float(np.max(vals))
                    dy = 0.01 * y_span
                    txt = f"{_format_sci(mu, annotate_sci_sig)}±{_format_sci(sd, annotate_sci_sig)}"

                    dx_pts = 0
                    # 每个子图的 HF=30 都右移
                    if xv == x_values[0]:
                        dx_pts += 22

                    # 每个子图的 HF=200 都左移
                    if xv == x_values[-1]:
                        dx_pts -= 22

                    # 以箱线/seed 分布的“底部”做锚点，更符合“标识放在箱线框下面”
                    y_anchor = float(np.min(vals))
                    # 所有 ours 标签都放到箱线框下面
                    y_off_pts = -8 # 建议先用 -12~-18 之间；你写 -30 会太狠
                    va = "top"

                    # 第1幅子图(pi==0) 的 HF=100 标注：稍微上移一点，避免与 HF=200 重叠
                    if (pi == 0) and (xv == 100):
                        y_off_pts += 6  # 上移 8pt（可调 6~12）

                    ax.annotate(
                        txt,
                        (x0, y_anchor),  # 注意：锚在底部
                        xytext=(dx_pts, y_off_pts),  # 统一下移
                        textcoords="offset points",
                        ha="center",
                        va=va,
                        fontsize=13,
                        color=ours_color,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.72),
                        clip_on=True,
                        zorder=5,
                    )


            # optional seed points (ours only)
            if int(show_seed_points) == 1:
                seed_jitter = 0.06
                for xv in x_values:
                    vals = _finite(sub[sub[x_col] == xv][ours_col].to_numpy(dtype=float))
                    if vals.size == 0:
                        continue
                    xs = x_pos_map[xv] + np.random.uniform(-seed_jitter, seed_jitter, size=vals.size)
                    ax.scatter(xs, vals, s=16, alpha=0.55, color=ours_color, edgecolors="none", zorder=3)

            # 2) 3-method trend lines on the same axis (aligned x)
            for mi, (label, col, color, _dx) in enumerate(methods):
                ys = []
                for xv in x_values:
                    vals = sub[sub[x_col] == xv][col].to_numpy(dtype=float)
                    ys.append(_center(vals))
                ys = np.array(ys, dtype=float)

                ls = "--" if connect == "dashed" else "-"
                ax.plot(x_ticks, ys, marker="o", linestyle=ls, label=label, color=color, linewidth=1.8, zorder=4)

                if annotate_line_mean:
                    # In overlay mode we already annotate Ours via mean±std on the boxplot;
                    # avoid duplicating text at the line points.
                    if col == ours_col:
                        continue
                    for j, xv in enumerate(x_values):
                        vals = sub[sub[x_col] == xv][col].to_numpy(dtype=float)
                        mu, _var = _mean_var(vals)
                        if not np.isfinite(mu) or not np.isfinite(ys[j]):
                            continue
                        dy = (0.02 + 0.010 * mi) * y_span
                        # --- extra lift for AR1 at rightmost HF to avoid overlap with Ours mean±std ---
                        if label == "co-kriging" and j == (len(x_values) - 1):  # 最右 HF（例如 200）
                            dy += 0.05 * y_span  # 再抬高一点（可调 0.03~0.08）

                        ax.text(
                            x_ticks[j],
                            ys[j] + dy,
                            _format_sci(mu, annotate_sci_sig),
                            ha="center",
                            va="bottom",
                            fontsize=13,
                            color=color,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.70),
                            clip_on=True,
                            zorder=5,
                        )

            add_panel_note(ax, rf"$m_{{\mathrm{{LF}}}}={int(pv)}$")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.12 * y_span)

            if pi == 0:
                ax.legend(loc="upper right", framealpha=0.9)

            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels, rotation=0)
            continue

        # ---- TOP: trend line (aligned x, no dodge) ----
        if rmse_style == "boxline":

            # y-range for spacing
            all_top_vals = []
            for _, col, _, _dx in methods:
                all_top_vals.append(_finite(sub[col].to_numpy(dtype=float)))
            all_top = np.concatenate([v for v in all_top_vals if v.size > 0], axis=0) if any(v.size > 0 for v in all_top_vals) else np.array([0.0, 1.0])
            y_min = float(np.min(all_top))
            y_max = float(np.max(all_top))
            y_span = max(1e-12, y_max - y_min)

            for mi, (label, col, color, _dx) in enumerate(methods):
                ys = []
                for xv in x_values:
                    vals = sub[sub[x_col] == xv][col].to_numpy(dtype=float)
                    ys.append(_center(vals))
                ys = np.array(ys, dtype=float)

                ls = "--" if connect == "dashed" else "-"
                ax_top.plot(x_ticks, ys, marker="o", linestyle=ls, label=label, color=color, linewidth=1.8)

                if annotate_line_mean:
                    for j, xv in enumerate(x_values):
                        vals = sub[sub[x_col] == xv][col].to_numpy(dtype=float)
                        mu, _var = _mean_var(vals)
                        if not np.isfinite(mu) or not np.isfinite(ys[j]):
                            continue
                        # place slightly above point, with method-dependent offset to reduce overlap
                        dy = (0.06 + 0.010 * mi) * y_span
                        ax_top.text(
                            x_ticks[j],
                            ys[j] + dy,
                            _format_sci(mu, annotate_sci_sig),
                            ha="center",
                            va="bottom",
                            fontsize=13,
                            color=color,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.70),
                            clip_on=True,
                        )

            add_panel_note(ax_top, rf"$m_{{\mathrm{{LF}}}}={int(pv)}$")
            ax_top.set_ylabel(ylabel)
            ax_top.grid(True, alpha=0.25)

            # legend once (best effort)
            if pi == 0:
                ax_top.legend(loc="upper right", framealpha=0.9)

            # Hide x tick labels on top
            plt.setp(ax_top.get_xticklabels(), visible=False)

            # ---- BOTTOM: dodged boxplots + seed points + annotations ----
            ax_bot.set_xlabel(xlabel)
            ax_bot.set_ylabel(ylabel)
            ax_bot.grid(True, alpha=0.25)

            # gather for ylim padding
            all_bot = []
            for _, col, _, _dx in methods:
                all_bot.append(_finite(sub[col].to_numpy(dtype=float)))
            all_bot_v = np.concatenate([v for v in all_bot if v.size > 0], axis=0) if any(v.size > 0 for v in all_bot) else np.array([0.0, 1.0])
            bmin = float(np.min(all_bot_v))
            bmax = float(np.max(all_bot_v))
            bspan = max(1e-12, bmax - bmin)
            # add headroom for text
            ax_bot.set_ylim(bmin - 0.05 * bspan, bmax + 0.42 * bspan)

            widths = 0.22  # bigger boxes
            seed_jitter = 0.045

            for mi, (label, col, color, dx) in enumerate(methods):
                # --- boxplots per xv ---
                data_list = []
                pos_list = []
                for xv in x_values:
                    vals = _finite(sub[sub[x_col] == xv][col].to_numpy(dtype=float))
                    if vals.size == 0:
                        continue
                    data_list.append(vals)
                    pos_list.append(x_pos_map[xv] + dx)

                if len(data_list) > 0:
                    bp = ax_bot.boxplot(
                        data_list,
                        positions=pos_list,
                        widths=widths,
                        patch_artist=True,
                        showfliers=False,
                        manage_ticks=False,
                    )
                    # style: visible edges, light face
                    for patch in bp["boxes"]:
                        patch.set_facecolor(color)
                        patch.set_alpha(0.18)
                        patch.set_edgecolor(color)
                        patch.set_linewidth(1.8)
                    for k in ("whiskers", "caps"):
                        for ln in bp[k]:
                            ln.set_color(color)
                            ln.set_linewidth(1.6)
                    for med in bp["medians"]:
                        med.set_color(color)
                        med.set_linewidth(2.4)

                # --- seed points ---
                if int(show_seed_points) == 1:
                    for xv in x_values:
                        vals = _finite(sub[sub[x_col] == xv][col].to_numpy(dtype=float))
                        if vals.size == 0:
                            continue
                        x0 = x_pos_map[xv] + dx
                        # deterministic-ish jitter (seeded by method index)
                        rng = np.random.default_rng(12345 + mi)
                        jit = rng.uniform(-seed_jitter, seed_jitter, size=vals.size)
                        ax_bot.scatter(
                            x0 + jit,
                            vals,
                            s=18,
                            alpha=0.75,
                            color=color,
                            edgecolors="none",
                            zorder=3,
                        )

                # --- mean ± var annotation (two lines), placed above local max ---
                if int(annotate_box_mean_var) == 1:
                    for xv in x_values:
                        vals = _finite(sub[sub[x_col] == xv][col].to_numpy(dtype=float))
                        if vals.size == 0:
                            continue
                        mu, var = _mean_var(vals)
                        if not np.isfinite(mu) or not np.isfinite(var):
                            continue
                        x0 = x_pos_map[xv] + dx
                        y0 = float(np.max(vals))
                        dy = (0.08 + 0.015 * mi) * bspan
                        txt = f"{_format_sci(mu, annotate_sci_sig)}\n±{_format_sci(np.sqrt(var), annotate_sci_sig)}"
                        ax_bot.text(
                            x0,
                            y0 + dy,
                            txt,
                            ha="center",
                            va="bottom",
                            fontsize=13,
                            color=color,
                            rotation=0,
                            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.72),
                            clip_on=True,
                            zorder=4,
                        )

            ax_bot.set_xticks(x_ticks)
            ax_bot.set_xticklabels(x_ticklabels)

            # tighten x limits to include dodged boxes
            ax_bot.set_xlim(-0.6, len(x_values) - 0.4)

        else:
            # ---- legacy single-axis styles (box / errorbar) ----
            # (kept simple; your workflow uses boxline now)
            add_panel_note(ax, rf"$m_{{\mathrm{{LF}}}}={int(pv)}$")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            # plot centers (aligned x)
            for mi, (label, col, color, dx) in enumerate(methods):
                ys = []
                for xv in x_values:
                    vals = sub[sub[x_col] == xv][col].to_numpy(dtype=float)
                    ys.append(_center(vals))
                ys = np.array(ys, dtype=float)
                ls = "--" if connect == "dashed" else "-"
                ax.plot(x_ticks, ys, marker="o", linestyle=ls, label=label, color=color, linewidth=1.8)
            if pi == 0:
                ax.legend(loc="upper right", framealpha=0.9)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels)

    if str(title).strip():
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=170, bbox_inches="tight")
    plt.close(fig)

def heatmap_impr(pivot: pd.DataFrame, title: str, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    data = pivot.values.astype(float)

    plt.figure(figsize=(8.6, 4.9))
    ax = plt.gca()
    im = ax.imshow(data, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel(r"$m_{\mathrm{LF}}$")
    ax.set_ylabel(r"HF budget $N_h$")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(x) for x in pivot.columns.tolist()])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(y) for y in pivot.index.tolist()])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=13)

    plt.colorbar(im, ax=ax, shrink=0.9, label="Improvement %")
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=170)
    plt.close()


def write_impr_summary(df: pd.DataFrame, out_csv: Path, q_lo: float = 0.25, q_hi: float = 0.75) -> None:
    df = df.copy()
    df["impr_ours_vs_hf"] = improvement_percent(df["metrics.y_rmse.hf_only"].to_numpy(), df["metrics.y_rmse.ours"].to_numpy())
    df["impr_ours_vs_ar1"] = improvement_percent(df["metrics.y_rmse.ar1"].to_numpy(), df["metrics.y_rmse.ours"].to_numpy())

    def q(v: np.ndarray, qq: float) -> float:
        v = _finite(v)
        return float(np.quantile(v, qq)) if v.size else float("nan")

    rows = []
    for (hf, lf), g in df.groupby(["hf", "lf_mult"], sort=True):
        v1 = g["impr_ours_vs_hf"].to_numpy(dtype=float)
        v2 = g["impr_ours_vs_ar1"].to_numpy(dtype=float)
        rows.append(
            {
                "hf": int(hf),
                "lf_mult": int(lf),
                "n": int(g.shape[0]),
                "ours_vs_hf_mean": float(np.nanmean(v1)),
                "ours_vs_hf_std": float(np.nanstd(v1, ddof=1)) if np.isfinite(v1).sum() >= 2 else float("nan"),
                f"ours_vs_hf_q{int(q_lo*100)}": q(v1, q_lo),
                f"ours_vs_hf_q{int(q_hi*100)}": q(v1, q_hi),
                "ours_vs_ar1_mean": float(np.nanmean(v2)),
                "ours_vs_ar1_std": float(np.nanstd(v2, ddof=1)) if np.isfinite(v2).sum() >= 2 else float("nan"),
                f"ours_vs_ar1_q{int(q_lo*100)}": q(v2, q_lo),
                f"ours_vs_ar1_q{int(q_hi*100)}": q(v2, q_hi),
            }
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")


def plot_impr_bar_by_hf(df: pd.DataFrame, hf_groups: List[int], out_png: Path) -> None:
    """
    One figure:
      - groups on x: HF budgets (hf_groups)
      - two bars per group:
          ours vs hf_only, ours vs ar1
      - values annotated on bars
    The value per HF is mean improvement across all (lf_mult, seed) runs available for that HF.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    hf_groups = [h for h in hf_groups if h <= 500]
    df = df[df["hf"].isin(hf_groups)].copy()

    df["impr_ours_vs_hf"] = improvement_percent(df["metrics.y_rmse.hf_only"].to_numpy(), df["metrics.y_rmse.ours"].to_numpy())
    df["impr_ours_vs_ar1"] = improvement_percent(df["metrics.y_rmse.ar1"].to_numpy(), df["metrics.y_rmse.ours"].to_numpy())

    vals_hf = []
    vals_ar1 = []
    ns = []
    for hf in hf_groups:
        g = df[df["hf"] == hf]
        v1 = _finite(g["impr_ours_vs_hf"].to_numpy(dtype=float))
        v2 = _finite(g["impr_ours_vs_ar1"].to_numpy(dtype=float))
        vals_hf.append(float(np.mean(v1)) if v1.size else float("nan"))
        vals_ar1.append(float(np.mean(v2)) if v2.size else float("nan"))
        ns.append(int(g.shape[0]))

    x = np.arange(len(hf_groups), dtype=float)
    w = 0.34

    plt.figure(figsize=(18.6, 4.2))
    ax = plt.gca()

    b1 = ax.bar(x - w/2, vals_hf, width=w, label="vs HF-only", color=COLOR_HF)
    b2 = ax.bar(x + w/2, vals_ar1, width=w, label="vs co-kriging", color=COLOR_COK)

    # annotate
    def _annotate(bars):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.1f}%", ha="center", va="bottom", fontsize=13)

    _annotate(b1)
    _annotate(b2)

    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in hf_groups])
    ax.set_xlabel(r"HF budget $N_h$")
    ax.set_ylabel(r"RMSE reduction (\%)")
    ax.axhline(0.0, alpha=0.3)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    # show n under tick (optional but helpful)
    for xi, n in zip(x, ns):
        ax.text(xi, ax.get_ylim()[0], f"n={n}", ha="center", va="bottom", fontsize=13, alpha=0.8)

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=170, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="../../result_out/mf_sweep_runs_baseline_nano_ab",
                    help="Root directory that contains dataset/seed subfolders and report.json files.")
    ap.add_argument("--out_dir", type=str, default="../../result_out/mf_sweep_runs_baseline_nano_ab/plot_baseline_nano_ab")
    ap.add_argument("--only_ok", type=int, default=1, choices=[0, 1])
    ap.add_argument("--q_lo", type=float, default=0.25)
    ap.add_argument("--q_hi", type=float, default=0.75)
    ap.add_argument("--rmse_style", type=str, default="box", choices=["box", "errorbar", "boxline", "overlay"],
                    help="RMSE plot style for small-multiples. 'overlay' draws 3-method trend lines + Ours boxplot on one axis. 'boxline' draws a top trend line (no errorbars) and a bottom dodged boxplot with seed points.")
    ap.add_argument("--connect", type=str, default="none", choices=["none", "dashed"],
                    help="Connect center statistic across discrete x with a dashed line (visual guide).")
    ap.add_argument("--center", type=str, default="median", choices=["median", "mean"],
                    help="Center statistic for RMSE curves: median (with quantile band) or mean (with ±std).")
    ap.add_argument("--show_seed_points", type=int, default=0, choices=[0, 1],
                    help="If 1, scatter seed points with jitter.")
    ap.add_argument("--annotate_line_mean", type=int, default=0, choices=[0, 1],
                    help="If 1, annotate mean values on the TOP trend line points (even if center=median).")
    ap.add_argument("--annotate_box_mean_var", type=int, default=0, choices=[0, 1],
                    help="If 1, annotate mean±variance (sample variance) in scientific notation near each box on the BOTTOM boxplot.")
    ap.add_argument("--annotate_line_digits", type=int, default=4,
                    help="Decimal digits for TOP mean annotations (non-scientific).")
    ap.add_argument("--annotate_sci_sig", type=int, default=2,
                    help="Significant digits for scientific notation in BOTTOM mean±variance annotations.")
    ap.add_argument("--hf_groups", type=str, default="30,50,80,100,150,200,300,400,500,",
                    help="HF budgets to show in the improvement bar chart, comma-separated.")
    ap.add_argument("--max_hf_plot", type=float, default=None,
                    help="If set (e.g., 200), filter out rows with hf > this value for plotting.")
    args = ap.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_npj_style()

    # 1) rebuild sweep table from run folders
    df = build_sweep_table_from_runs(runs_root)
    (out_dir / "sweep_results_rebuilt.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    if int(args.only_ok) == 1:
        df = df[df["status"] == "OK"].copy()

    # required rmse
    req = ["metrics.y_rmse.hf_only", "metrics.y_rmse.ar1", "metrics.y_rmse.ours"]
    df = df.dropna(subset=req).copy()

    # optional: hide large HF budgets (e.g., hf=300)
    if args.max_hf_plot is not None and "hf" in df.columns:
        try:
            max_hf = float(args.max_hf_plot)
            df = df[df["hf"].astype(float) <= max_hf].copy()
        except Exception:
            pass

    # 2) RMSE plots: configurable styles
    plot_rmse_boxplots_smallmultiples(
        df=df,
        x_col="hf",
        panel_col="lf_mult",
        out_png=out_dir / "rmse_trend_vs_hf_by_lfmult.png",
        rmse_style="overlay",
        connect=args.connect,
        center=args.center,
        q_lo=args.q_lo,
        q_hi=args.q_hi,
        show_seed_points=args.show_seed_points,
        annotate_line_mean=args.annotate_line_mean,
        annotate_box_mean_var=args.annotate_box_mean_var,
        annotate_line_digits=args.annotate_line_digits,
        annotate_sci_sig=args.annotate_sci_sig,
        # title=f"RMSE vs HF budget ({args.rmse_style}, center={args.center}, connect={args.connect}), split by LF multiplier",
        title="",
        xlabel=r"HF budget $N_h$",
        ylabel="Test RMSE",
    )

    # plot_rmse_boxplots_smallmultiples(
    #     df=df,
    #     x_col="lf_mult",
    #     panel_col="hf",
    #     out_png=out_dir / "rmse_vs_lfmult_smallmultiples_by_hf.png",
    #     rmse_style=args.rmse_style,
    #     connect=args.connect,
    #     center=args.center,
    #     q_lo=args.q_lo,
    #     q_hi=args.q_hi,
    #     show_seed_points=args.show_seed_points,
    #     annotate_line_mean=args.annotate_line_mean,
    #     annotate_box_mean_var=args.annotate_box_mean_var,
    #     annotate_line_digits=args.annotate_line_digits,
    #     annotate_sci_sig=args.annotate_sci_sig,
    #     title=f"RMSE vs LF multiplier ({args.rmse_style}, center={args.center}, connect={args.connect}), small multiples over HF budgets",
    #     xlabel="LF multiplier",
    #     ylabel="RMSE (test)",
    # )
    #
    # # 3) Heatmaps (keep)
    # # compute mean rmse per cell
    # cell = df.groupby(["hf", "lf_mult"], sort=True).agg(
    #     hf_mean=("metrics.y_rmse.hf_only", "mean"),
    #     ar1_mean=("metrics.y_rmse.ar1", "mean"),
    #     ours_mean=("metrics.y_rmse.ours", "mean"),
    #     n=("seed", "count"),
    # ).reset_index()
    #
    # cell["ar1_impr_mean"] = improvement_percent(cell["hf_mean"].to_numpy(), cell["ar1_mean"].to_numpy())
    # cell["ours_impr_mean"] = improvement_percent(cell["hf_mean"].to_numpy(), cell["ours_mean"].to_numpy())
    #
    # hf_values = sorted(cell["hf"].unique().tolist())
    # lf_values = sorted(cell["lf_mult"].unique().tolist())
    #
    # piv_ar1 = cell.pivot(index="hf", columns="lf_mult", values="ar1_impr_mean").reindex(index=hf_values, columns=lf_values)
    # piv_ours = cell.pivot(index="hf", columns="lf_mult", values="ours_impr_mean").reindex(index=hf_values, columns=lf_values)
    #
    # heatmap_impr(piv_ar1, "co-kriging Improvement% vs HF-only (mean over seeds)", out_dir / "heatmap_impr__ar1.png")
    # heatmap_impr(piv_ours, "Ours Improvement% vs HF-only (mean over seeds)", out_dir / "heatmap_impr__ours.png")

    # 4) impr_summary.csv (default)
    write_impr_summary(df, out_dir / "impr_summary.csv", q_lo=args.q_lo, q_hi=args.q_hi)

    # 5) ONE improvement bar chart by HF
    hf_groups = [int(x.strip()) for x in str(args.hf_groups).split(",") if x.strip()]
    hf_groups = [h for h in hf_groups if h in df["hf"].unique().tolist()]
    if not hf_groups:
        # fallback: choose up to 4 smallest hf budgets present
        hf_groups = sorted(df["hf"].unique().tolist())[:4]
    plot_impr_bar_by_hf(df, hf_groups, out_dir / "impr_bar_by_hf_ours_vs_baselines.png")

    print(f"[DONE] runs_root: {runs_root}")
    print(f"[DONE] saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()