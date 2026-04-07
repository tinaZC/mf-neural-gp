#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HF_LFM_RE = re.compile(r"hf(\d+)_lfx(\d+)", re.IGNORECASE)
SEED_RE = re.compile(r"seed(\d+)", re.IGNORECASE)

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"


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
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def improvement_percent(rmse_base: np.ndarray, rmse_new: np.ndarray) -> np.ndarray:
    rmse_base = np.asarray(rmse_base, dtype=float)
    rmse_new = np.asarray(rmse_new, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (rmse_base - rmse_new) / rmse_base * 100.0


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _fmt_sci(x: float) -> str:
    return f"{x:.2e}"


def mean_std_text(x: np.ndarray) -> str:
    x = _finite(x)
    if x.size == 0:
        return "nan"
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    return f"{_fmt_sci(mu)}±{_fmt_sci(sd)}"


def build_table_from_runs(runs_root: Path) -> pd.DataFrame:
    reports = list(runs_root.expanduser().resolve().glob("**/report.json"))
    rows: List[Dict[str, Any]] = []
    for rp in reports:
        try:
            obj = read_json(rp)
        except Exception:
            continue
        seed = find_seed_in_path(rp)
        hf, lfx, tag = find_hf_lfmult_from_path(rp)
        if hf is None or lfx is None:
            data_dir = str(obj.get("data_dir", "")) if isinstance(obj, dict) else ""
            hf2, lfx2 = parse_hf_lfmult_from_text(data_dir)
            if hf2 is not None and lfx2 is not None:
                hf, lfx = hf2, lfx2
                tag = f"hf{hf}_lfx{lfx}"
        if seed is None and isinstance(obj, dict):
            seed = obj.get("seed", None)
            try:
                seed = int(seed) if seed is not None else None
            except Exception:
                seed = None
        metrics = obj.get("metrics", {}) if isinstance(obj, dict) else {}
        y_rmse = metrics.get("y_rmse", {}) if isinstance(metrics, dict) else {}
        rows.append({
            "report_path": str(rp),
            "hf": hf,
            "lfx": lfx,
            "dataset_tag": tag,
            "seed": seed,
            "metrics.y_rmse.hf_only": safe_float(y_rmse.get("hf_only", np.nan)),
            "metrics.y_rmse.ar1": safe_float(y_rmse.get("ar1", np.nan)),
            "metrics.y_rmse.ours": safe_float(y_rmse.get("ours", np.nan)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    valid = (
        np.isfinite(df["metrics.y_rmse.hf_only"].to_numpy(dtype=float)) |
        np.isfinite(df["metrics.y_rmse.ar1"].to_numpy(dtype=float)) |
        np.isfinite(df["metrics.y_rmse.ours"].to_numpy(dtype=float))
    )
    return df.loc[valid].copy()


def pick_single_dataset(df: pd.DataFrame, hf: Optional[int], lfx: Optional[int]) -> pd.DataFrame:
    if df.empty:
        return df
    if hf is not None:
        df = df[df["hf"] == int(hf)].copy()
    if lfx is not None:
        df = df[df["lfx"] == int(lfx)].copy()
    return df.sort_values(["seed", "report_path"]).reset_index(drop=True)


def plot_methods_panel(df: pd.DataFrame, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    methods = [
        ("HF-only", "metrics.y_rmse.hf_only", BLUE),
        ("co-kriging", "metrics.y_rmse.ar1", ORANGE),
        ("Ours", "metrics.y_rmse.ours", GREEN),
    ]
    vals = [_finite(df[col].to_numpy(dtype=float)) for _, col, _ in methods]
    tags = [m[0] for m in methods]
    colors = [m[2] for m in methods]

    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.5))
    positions = np.array([1.0, 2.0, 3.0], dtype=float)

    bp = ax.boxplot(
        vals,
        labels=tags,
        positions=positions,
        widths=0.42,
        showfliers=True,
        patch_artist=True,
        medianprops=dict(linewidth=2.0),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        boxprops=dict(linewidth=1.8),
        flierprops=dict(markersize=4, alpha=0.6),
    )

    for i in range(len(tags)):
        c = colors[i]
        box = bp["boxes"][i]
        box.set_facecolor(c)
        box.set_alpha(0.18)
        box.set_edgecolor(c)
        box.set_linewidth(2.0)

        med = bp["medians"][i]
        med.set_color(c)
        med.set_linewidth(2.2)

        for w in bp["whiskers"][2 * i: 2 * i + 2]:
            w.set_color(c)
            w.set_linewidth(1.6)
        for cap in bp["caps"][2 * i: 2 * i + 2]:
            cap.set_color(c)
            cap.set_linewidth(1.6)
        if "fliers" in bp and i < len(bp["fliers"]):
            bp["fliers"][i].set_markeredgecolor(c)
            bp["fliers"][i].set_markerfacecolor(c)
            bp["fliers"][i].set_alpha(0.6)

    all_vals = np.array([x for vv in vals for x in vv], dtype=float)
    if all_vals.size > 0:
        y_min = float(np.min(all_vals))
        y_max = float(np.max(all_vals))
        y_rng = max(y_max - y_min, 1e-12)
    else:
        y_min, y_max, y_rng = 0.0, 1.0, 1.0

    rng = np.random.default_rng(0)
    for i, arr in enumerate(vals, start=1):
        if arr.size == 0:
            continue
        c = colors[i - 1]
        x = rng.normal(loc=i, scale=0.04, size=arr.size)
        ax.scatter(x, arr, s=14, alpha=0.55, linewidths=0, color=c, zorder=3)

    label_tops: List[float] = []
    for i, arr in enumerate(vals, start=1):
        if arr.size == 0:
            continue
        c = colors[i - 1]
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=0))
        q3 = float(np.quantile(arr, 0.75))
        arr_max = float(np.max(arr))
        span_local = max(float(np.max(arr) - np.min(arr)), 1e-12)
        if i == 3:
            y_off = max(0.012 * y_rng, 0.20 * span_local)
        else:
            y_off = max(0.020 * y_rng, 0.28 * span_local)
        y_text = max(q3 + y_off, arr_max + 0.01 * y_rng)
        label_tops.append(y_text)

        x_text = float(i)
        if i == 1:
            x_text += 0.10
        elif i == 3:
            x_text -= 0.10

        ax.text(
            x_text,
            y_text,
            f"{mu:.2e}±{sd:.2e}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=c,
            clip_on=False,
        )

    upper = max(y_max + 0.18 * y_rng, (max(label_tops) if label_tops else y_max) + 0.04 * y_rng)
    ax.set_ylim(y_min - 0.05 * y_rng, upper)
    ax.set_xlim(0.20, 3.80)
    ax.set_ylabel("Test RMSE")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title("Test RMSE across methods")
    ax.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    n_runs = int(df.shape[0])
    fig.text(0.965, 0.03, f"n={n_runs} runs", ha="right", va="bottom", fontsize=7.8)
    plt.tight_layout(rect=[0.0, 0.07, 1.0, 0.95])
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_improvement_panel(df: pd.DataFrame, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    impr_vs_hf = _finite(improvement_percent(df["metrics.y_rmse.hf_only"].to_numpy(dtype=float), df["metrics.y_rmse.ours"].to_numpy(dtype=float)))
    impr_vs_ar1 = _finite(improvement_percent(df["metrics.y_rmse.ar1"].to_numpy(dtype=float), df["metrics.y_rmse.ours"].to_numpy(dtype=float)))

    fig, ax = plt.subplots(1, 1, figsize=(4.1, 3.5))
    bar_labels = ["Ours vs HF-only", "Ours vs co-kriging"]
    bar_vals = [
        float(np.mean(impr_vs_hf)) if impr_vs_hf.size else float("nan"),
        float(np.mean(impr_vs_ar1)) if impr_vs_ar1.size else float("nan"),
    ]
    bar_colors = [BLUE, ORANGE]
    x = np.array([0.42, 0.74], dtype=float)
    ax.bar(x, bar_vals, width=0.12, color=bar_colors, linewidth=0)
    for xi, yi, c in zip(x, bar_vals, bar_colors):
        if np.isfinite(yi):
            ax.text(xi, yi + 0.08, f"{yi:.1f}%", ha="center", va="bottom", fontsize=8.2, color=c)
    ax.set_xlim(0.26, 0.90)
    ymax_bar = max(v for v in bar_vals if np.isfinite(v)) if any(np.isfinite(v) for v in bar_vals) else 1.0
    ax.set_ylim(0.0, max(13.4, ymax_bar + 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("RMSE reduction (%)")
    ax.set_title("Relative RMSE reduction of Ours")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.30)
    ax.grid(True, axis="y", alpha=0.22)
    ax.set_axisbelow(True)

    n_runs = int(df.shape[0])
    fig.text(0.965, 0.03, f"n={n_runs} runs", ha="right", va="bottom", fontsize=7.8)
    plt.tight_layout(rect=[0.0, 0.07, 1.0, 0.95])
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, col in [
        ("HF-only", "metrics.y_rmse.hf_only"),
        ("co-kriging", "metrics.y_rmse.ar1"),
        ("Ours", "metrics.y_rmse.ours"),
    ]:
        arr = _finite(df[col].to_numpy(dtype=float))
        rows.append({
            "method": name,
            "n": int(arr.size),
            "mean": float(np.mean(arr)) if arr.size else float("nan"),
            "std": float(np.std(arr, ddof=1)) if arr.size >= 2 else float("nan"),
            "min": float(np.min(arr)) if arr.size else float("nan"),
            "max": float(np.max(arr)) if arr.size else float("nan"),
        })
    impr_vs_hf = _finite(improvement_percent(df["metrics.y_rmse.hf_only"].to_numpy(dtype=float), df["metrics.y_rmse.ours"].to_numpy(dtype=float)))
    impr_vs_ar1 = _finite(improvement_percent(df["metrics.y_rmse.ar1"].to_numpy(dtype=float), df["metrics.y_rmse.ours"].to_numpy(dtype=float)))
    rows.append({"method": "Ours_vs_HF-only_impr_percent", "n": int(impr_vs_hf.size), "mean": float(np.mean(impr_vs_hf)) if impr_vs_hf.size else float("nan"), "std": float(np.std(impr_vs_hf, ddof=1)) if impr_vs_hf.size >= 2 else float("nan"), "min": float(np.min(impr_vs_hf)) if impr_vs_hf.size else float("nan"), "max": float(np.max(impr_vs_hf)) if impr_vs_hf.size else float("nan")})
    rows.append({"method": "Ours_vs_AR1_impr_percent", "n": int(impr_vs_ar1.size), "mean": float(np.mean(impr_vs_ar1)) if impr_vs_ar1.size else float("nan"), "std": float(np.std(impr_vs_ar1, ddof=1)) if impr_vs_ar1.size >= 2 else float("nan"), "min": float(np.min(impr_vs_ar1)) if impr_vs_ar1.size else float("nan"), "max": float(np.max(impr_vs_ar1)) if impr_vs_ar1.size else float("nan")})
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="../../result_out/mf_baseline_out_microwave_mtm_multi")
    ap.add_argument("--out_dir", type=str, default="../../result_out/mf_baseline_out_microwave_mtm_multi/plot_mtm_compare")
    ap.add_argument("--hf", type=int, default=50)
    ap.add_argument("--lfx", type=int, default=10)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_table_from_runs(runs_root)
    if df.empty:
        raise RuntimeError(f"No valid report.json rows found under {runs_root}")
    df1 = pick_single_dataset(df, args.hf, args.lfx)
    if df1.empty:
        raise RuntimeError(f"No rows found for hf={args.hf}, lfx={args.lfx} under {runs_root}")
    out_png_a = out_dir / "baseline_compare_methods.png"
    out_png_b = out_dir / "baseline_compare_improvement.png"
    out_csv = out_dir / "m013_baseline_compare_summary.csv"
    out_table = out_dir / "m013_baseline_compare_rows.csv"
    plot_methods_panel(df1, out_png=out_png_a)
    plot_improvement_panel(df1, out_png=out_png_b)
    save_summary_csv(df1, out_csv)
    df1.to_csv(out_table, index=False, encoding="utf-8")
    print(f"[SAVE] methods figure -> {out_png_a}")
    print(f"[SAVE] improvement figure -> {out_png_b}")
    print(f"[SAVE] summary -> {out_csv}")
    print(f"[SAVE] rows -> {out_table}")


if __name__ == "__main__":
    main()
