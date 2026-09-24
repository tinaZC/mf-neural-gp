#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild the historical TM FPCA-dimension sensitivity figure using native/raw UQ.

This is a plotting/post-processing script only. It does NOT train any model.

Historical source (read-only):
    result_out/fpca_dim_sweep_tm_outputs

Expected experiment:
    dataset/setting : TM / hf100_lfx10
    fixed FPCA dims : 2,4,6,8,10,12,16,24,32,64
    seeds per dim   : 42,55,66,77,88,99,111,222,333,555
    total leaf runs : 100

Outputs (revised project):
    result_out/final_analysis/figures/fpca_dim_sweep_native_seed_values.csv
    result_out/final_analysis/figures/fpca_dim_sweep_native_summary.csv
    result_out/final_analysis/figures/fig_fpca_dim_sweep_native.pdf
    result_out/final_analysis/figures/fig_fpca_dim_sweep_native.png

Panels:
(a) HF-validation FPCA reconstruction RMSE
(b) Neural-GP MF test RMSE
(c) Native predictive test NLL
(d) Native empirical coverage and native mean prediction-interval width

The multi-seed bands are mean ± one sample standard deviation, matching the
original dimension-sweep convention. The discrete FPCA dimensions are shown at
equal visual spacing for readability. No background grid is drawn. Each panel
is annotated with n = 10.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_DIMS = (2, 4, 6, 8, 10, 12, 16, 24, 32, 64)
EXPECTED_SEEDS = (42, 55, 66, 77, 88, 99, 111, 222, 333, 555)
EXPECTED_SETTING = "hf100_lfx10"


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_revised_project = script_path.parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the historical fixed-dimension FPCA sweep using "
            "native/raw predictive uncertainty. No training is performed."
        )
    )
    parser.add_argument(
        "--historical-project",
        type=Path,
        required=True,
        help="Required historical project root containing result_out/fpca_dim_sweep_tm_outputs (example: a separately obtained original-project checkout).",
    )
    parser.add_argument(
        "--revised-project",
        type=Path,
        default=default_revised_project,
        help="Current revised project root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<revised-project>/result_out/final_analysis/figures"
        ),
    )
    return parser.parse_args()


def finite_float(value: Any, context: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected numeric value, got {value!r}") from exc
    if not math.isfinite(x):
        raise ValueError(f"{context}: expected finite value, got {x!r}")
    return x


def nested_get(obj: Dict[str, Any], path: Sequence[str], context: str) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            dotted = ".".join(path)
            raise KeyError(f"{context}: missing required field {dotted}")
        cur = cur[key]
    return cur


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing historical summary CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def locate_report(
    historical_root: Path,
    row: Dict[str, str],
    seed: int,
    dim: int,
) -> Path:
    run_dir_text = (row.get("run_dir") or "").strip()
    if run_dir_text:
        candidate = Path(run_dir_text) / "report.json"
        if candidate.is_file():
            return candidate

    dim_dir = historical_root / f"seed_{seed:03d}" / f"fpca_dim_{dim:03d}"
    if not dim_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot resolve run directory for seed={seed}, dim={dim}: {dim_dir}"
        )

    matches = sorted(dim_dir.glob("*/report.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one leaf report for seed={seed}, dim={dim}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def validate_config(report_path: Path, seed: int, dim: int) -> None:
    config_path = report_path.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json beside report: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    args = cfg.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"{config_path}: missing config['args'] object")

    if "seed" in args and int(args["seed"]) != seed:
        raise ValueError(
            f"{config_path}: seed mismatch, expected {seed}, got {args['seed']}"
        )

    if "fpca_dim" in args and int(args["fpca_dim"]) != dim:
        raise ValueError(
            f"{config_path}: fpca_dim mismatch, expected {dim}, got {args['fpca_dim']}"
        )

    data_dir = args.get("data_dir")
    if data_dir:
        setting = Path(str(data_dir)).name
        if setting != EXPECTED_SETTING:
            raise ValueError(
                f"{config_path}: expected data setting {EXPECTED_SETTING}, got {setting}"
            )


def extract_row(
    summary_row: Dict[str, str],
    report_path: Path,
    seed: int,
    dim: int,
) -> Dict[str, Any]:
    required_summary_cols = ("recon_rmse_hfval", "y_rmse_test")
    missing = [
        key
        for key in required_summary_cols
        if key not in summary_row or summary_row[key] in ("", None)
    ]
    if missing:
        raise KeyError(
            "Historical summary CSV is missing required per-seed field(s): "
            + ", ".join(missing)
        )

    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    if "seed" in report and int(report["seed"]) != seed:
        raise ValueError(
            f"{report_path}: report seed mismatch; expected {seed}, got {report['seed']}"
        )

    nll_raw = finite_float(
        nested_get(
            report,
            ("metrics", "nll", "test", "raw", "mf_student"),
            str(report_path),
        ),
        f"{report_path} native NLL",
    )
    coverage_raw = finite_float(
        nested_get(
            report,
            ("uncertainty", "test", "coverage_raw", "mf_student"),
            str(report_path),
        ),
        f"{report_path} native coverage",
    )
    width_raw = finite_float(
        nested_get(
            report,
            ("uncertainty", "test", "width_raw", "mf_student"),
            str(report_path),
        ),
        f"{report_path} native interval width",
    )

    return {
        "seed": seed,
        "dim": dim,
        "recon_rmse_hfval": finite_float(
            summary_row["recon_rmse_hfval"],
            f"summary seed={seed} dim={dim} recon_rmse_hfval",
        ),
        "y_rmse_test": finite_float(
            summary_row["y_rmse_test"],
            f"summary seed={seed} dim={dim} y_rmse_test",
        ),
        "nll_test_raw": nll_raw,
        "coverage_test_raw": coverage_raw,
        "width_test_raw": width_raw,
        "report_path": str(report_path),
    }


def collect(historical_project: Path) -> List[Dict[str, Any]]:
    historical_root = (
        historical_project / "result_out" / "fpca_dim_sweep_tm_outputs"
    )
    summary_path = historical_root / "fpca_dim_sweep_summary.csv"

    rows = read_csv_rows(summary_path)
    if len(rows) != 100:
        raise ValueError(
            f"{summary_path}: expected exactly 100 rows, found {len(rows)}"
        )

    for col in ("seed", "dim", "run_dir"):
        if col not in rows[0]:
            raise KeyError(f"{summary_path}: missing required column {col!r}")

    observed_pairs = set()
    extracted: List[Dict[str, Any]] = []

    for row in rows:
        seed = int(row["seed"])
        dim = int(row["dim"])

        pair = (seed, dim)
        if pair in observed_pairs:
            raise ValueError(
                f"{summary_path}: duplicate seed/dim pair seed={seed}, dim={dim}"
            )
        observed_pairs.add(pair)

        report_path = locate_report(historical_root, row, seed, dim)
        validate_config(report_path, seed, dim)
        extracted.append(extract_row(row, report_path, seed, dim))

    expected_pairs = {
        (seed, dim)
        for dim in EXPECTED_DIMS
        for seed in EXPECTED_SEEDS
    }
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        raise ValueError(
            "Historical sweep identity mismatch.\n"
            f"Missing pairs: {missing}\n"
            f"Unexpected pairs: {extra}"
        )

    print(
        "PASS: historical sweep contains exactly "
        f"{len(EXPECTED_DIMS)} dims × {len(EXPECTED_SEEDS)} seeds = "
        f"{len(extracted)} runs"
    )
    print(f"PASS: dimensions = {list(EXPECTED_DIMS)}")
    print(f"PASS: seeds = {list(EXPECTED_SEEDS)}")
    print("PASS: all 100 leaf reports contain native/raw NLL, coverage, and width")

    return sorted(extracted, key=lambda r: (r["dim"], r["seed"]))


def sample_mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        raise ValueError("Need at least two runs to compute sample standard deviation")
    return float(arr.mean()), float(arr.std(ddof=1))


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["dim"])].append(row)

    metric_names = (
        "recon_rmse_hfval",
        "y_rmse_test",
        "nll_test_raw",
        "coverage_test_raw",
        "width_test_raw",
    )

    summary: List[Dict[str, Any]] = []
    for dim in EXPECTED_DIMS:
        group = grouped[dim]
        if len(group) != len(EXPECTED_SEEDS):
            raise ValueError(
                f"dim={dim}: expected {len(EXPECTED_SEEDS)} runs, found {len(group)}"
            )

        item: Dict[str, Any] = {"dim": dim, "n": len(group)}
        for metric in metric_names:
            mean, std = sample_mean_std(row[metric] for row in group)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        summary.append(item)

    return summary


def write_seed_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = (
        "seed",
        "dim",
        "recon_rmse_hfval",
        "y_rmse_test",
        "nll_test_raw",
        "coverage_test_raw",
        "width_test_raw",
        "report_path",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    metric_names = (
        "recon_rmse_hfval",
        "y_rmse_test",
        "nll_test_raw",
        "coverage_test_raw",
        "width_test_raw",
    )
    fields = ["dim", "n"]
    for metric in metric_names:
        fields.extend([f"{metric}_mean", f"{metric}_std"])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in fields}
            for row in rows
        )


def draw_mean_std(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    marker: str = "o",
    linestyle: str = "-",
    label: str | None = None,
):
    line, = ax.plot(
        x,
        mean,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.8,
        markersize=5.5,
        label=label,
        zorder=3,
    )
    color = line.get_color()
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.16,
        color=color,
        linewidth=0,
        zorder=2,
    )
    return line


def style_axis(ax: plt.Axes, panel_label: str) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=11)
    ax.text(
        0.0,
        1.03,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def make_figure(summary: List[Dict[str, Any]]) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    dims = np.asarray([row["dim"] for row in summary], dtype=int)
    x = np.arange(len(dims), dtype=float)

    def vec(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in summary], dtype=float)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.8, 8.2),
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c, ax_d = axes.flat

    draw_mean_std(
        ax_a,
        x,
        vec("recon_rmse_hfval_mean"),
        vec("recon_rmse_hfval_std"),
    )
    ax_a.set_xlabel("FPCA dimension $R$")
    ax_a.set_ylabel("HF-validation reconstruction RMSE")
    style_axis(ax_a, "(a)")

    draw_mean_std(
        ax_b,
        x,
        vec("y_rmse_test_mean"),
        vec("y_rmse_test_std"),
    )
    ax_b.set_xlabel("FPCA dimension $R$")
    ax_b.set_ylabel("Neural-GP MF test RMSE")
    style_axis(ax_b, "(b)")

    draw_mean_std(
        ax_c,
        x,
        vec("nll_test_raw_mean"),
        vec("nll_test_raw_std"),
    )
    ax_c.set_xlabel("FPCA dimension $R$")
    ax_c.set_ylabel("Native predictive NLL")
    style_axis(ax_c, "(c)")

    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(default_colors) < 2:
        raise RuntimeError("Matplotlib default color cycle must contain >= 2 colors")

    coverage_mean = vec("coverage_test_raw_mean")
    coverage_std = vec("coverage_test_raw_std")
    width_mean = vec("width_test_raw_mean")
    width_std = vec("width_test_raw_std")

    cov_line, = ax_d.plot(
        x,
        coverage_mean,
        marker="o",
        linestyle="-",
        linewidth=1.8,
        markersize=5.5,
        label="Native coverage",
        color=default_colors[0],
        zorder=3,
    )
    ax_d.fill_between(
        x,
        coverage_mean - coverage_std,
        coverage_mean + coverage_std,
        alpha=0.16,
        color=cov_line.get_color(),
        linewidth=0,
        zorder=2,
    )
    ax_d.axhline(
        0.95,
        linestyle="--",
        linewidth=1.2,
        color="0.45",
        zorder=1,
    )
    ax_d.set_xlabel("FPCA dimension $R$")
    ax_d.set_ylabel("Native empirical coverage")

    ax_d2 = ax_d.twinx()
    width_line, = ax_d2.plot(
        x,
        width_mean,
        marker="s",
        linestyle="-",
        linewidth=1.8,
        markersize=5.0,
        label="Native interval width",
        color=default_colors[1],
        zorder=3,
    )
    ax_d2.fill_between(
        x,
        width_mean - width_std,
        width_mean + width_std,
        alpha=0.16,
        color=width_line.get_color(),
        linewidth=0,
        zorder=2,
    )
    ax_d2.set_ylabel("Mean prediction interval width")
    ax_d2.grid(False)
    ax_d2.tick_params(axis="y", labelsize=11)

    ax_d.legend(
        [cov_line, width_line],
        ["Native coverage", "Native interval width"],
        loc="center",
        bbox_to_anchor=(0.62, 0.68),
        frameon=False,
        fontsize=9.5,
    )
    style_axis(ax_d, "(d)")

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in dims])
        ax.text(
            0.98,
            0.03,
            "n = 10",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )

    return fig


def print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\nNative/raw FPCA dimension sweep summary")
    print("-" * 108)
    print(
        "R   n   recon_RMSE(mean±sd)   test_RMSE(mean±sd)   "
        "NLL_raw(mean±sd)   coverage_raw(mean±sd)   width_raw(mean±sd)"
    )
    print("-" * 108)

    for row in summary:
        print(
            f"{row['dim']:>2d}  {row['n']:>2d}  "
            f"{row['recon_rmse_hfval_mean']:.6g}±"
            f"{row['recon_rmse_hfval_std']:.3g}   "
            f"{row['y_rmse_test_mean']:.6g}±"
            f"{row['y_rmse_test_std']:.3g}   "
            f"{row['nll_test_raw_mean']:.6g}±"
            f"{row['nll_test_raw_std']:.3g}   "
            f"{row['coverage_test_raw_mean']:.6g}±"
            f"{row['coverage_test_raw_std']:.3g}   "
            f"{row['width_test_raw_mean']:.6g}±"
            f"{row['width_test_raw_std']:.3g}"
        )


def main() -> None:
    args = parse_args()

    historical_project = args.historical_project.resolve()
    revised_project = args.revised_project.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else revised_project / "result_out" / "final_analysis" / "figures"
    )

    historical_root = (
        historical_project / "result_out" / "fpca_dim_sweep_tm_outputs"
    )
    if not historical_root.is_dir():
        raise FileNotFoundError(
            "Historical fixed-dimension sweep not found:\n"
            f"  {historical_root}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    print("Historical source (read-only):")
    print(f"  {historical_root}")
    print("Revised output directory:")
    print(f"  {out_dir}")
    print("No training will be run.\n")

    seed_rows = collect(historical_project)
    summary_rows = aggregate(seed_rows)

    seed_csv = out_dir / "fpca_dim_sweep_native_seed_values.csv"
    summary_csv = out_dir / "fpca_dim_sweep_native_summary.csv"
    pdf_path = out_dir / "fig_fpca_dim_sweep_native.pdf"
    png_path = out_dir / "fig_fpca_dim_sweep_native.png"

    write_seed_csv(seed_rows, seed_csv)
    write_summary_csv(summary_rows, summary_csv)
    print_summary(summary_rows)

    fig = make_figure(summary_rows)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nCreated:")
    print(f"  {seed_csv}")
    print(f"  {summary_csv}")
    print(f"  {pdf_path}")
    print(f"  {png_path}")
    print("\nDONE: post-processing/plotting only; no model training performed.")


if __name__ == "__main__":
    main()
