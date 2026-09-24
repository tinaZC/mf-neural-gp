#!/usr/bin/env python3
"""Plot formal TM native-UQ comparisons from compact data or raw reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t


PROJECT = Path(__file__).resolve().parents[2]
FINAL = PROJECT / "result_out" / "final_runs"
SUMMARY = PROJECT / "result_out" / "final_analysis" / "uq_summary.csv"
FIGURES = PROJECT / "result_out" / "final_analysis" / "figures"
FROZEN = PROJECT / "result_out" / "final_analysis" / "frozen_inputs" / "native_uq" / "native_uq_plot_data.csv"
SETTING = "hf100_lfx10"
SEEDS = (
    200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
    300, 311, 322, 333, 344, 355, 366, 377, 388, 399,
)
METRICS = ("nll", "coverage", "width")
SUMMARY_METRICS = {
    "nll": "nll_raw",
    "coverage": "coverage_raw",
    "width": "width_raw",
}
METHODS = (
    ("HF-only", "HF-only", "main baseline", "original", "hf_only", "#0072B2"),
    ("AR1/co-kriging", "AR1/co-kriging", "main baseline", "original", "ar1", "#E69F00"),
    ("FPCA-NARGP", "FPCA-NARGP", "main baseline", "fpca", None, "#D55E00"),
    (
        "Wavelength-wise Stage II",
        "Controlled wavelength-wise Stage II",
        "computational diagnostic",
        "controlled",
        None,
        "#CC79A7",
    ),
    (
        "Wavelength-wise NARGP",
        "Wavelength-wise NARGP",
        "computational diagnostic",
        "freqwise",
        None,
        "#4D4D4D",
    ),
    ("Neural-GP MF", "Neural-GP MF (ours)", "proposed method", "original", "ours", "#009E73"),
)
ROOTS = {
    "original": FINAL / "original" / "tm" / SETTING,
    "fpca": FINAL / "comparisons" / "tm" / "fpca_nargp" / SETTING,
    "freqwise": FINAL / "representation" / "freqwise_nargp" / "tm" / SETTING,
    "controlled": FINAL / "wavelength_gp" / SETTING,
}


def require_number(value, context):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{context}: expected finite numeric value, got {value!r}")
    return float(value)


def load_reports(kind):
    root = ROOTS[kind]
    if not root.is_dir():
        raise FileNotFoundError(f"Formal report root missing: {root}")

    expected = set(SEEDS[:3] if kind in ("freqwise", "controlled") else SEEDS)
    seed_dirs = list(root.iterdir())
    found = set()

    for directory in seed_dirs:
        match = re.fullmatch(r"seed_(\d+)", directory.name)
        if not directory.is_dir() or not match:
            raise ValueError(f"Unexpected entry in formal root: {directory}")
        found.add(int(match.group(1)))

    if found != expected or len(seed_dirs) != len(expected):
        raise ValueError(f"{root}: expected seeds {sorted(expected)}, found {sorted(found)}")

    reports = {}
    for seed in sorted(expected):
        directory = root / f"seed_{seed}"

        if kind == "original":
            matches = list(directory.glob("bl0r*/report.json"))
            if len(matches) != 1 or matches[0].parent.name != f"bl0r{seed}":
                raise ValueError(f"{directory}: expected one authoritative bl0r{seed}/report.json")
            path = matches[0]
        else:
            path = directory / "report.json"

        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)

        if report.get("seed") != seed:
            raise ValueError(f"{path}: seed mismatch")

        if kind == "original":
            uq_protocol = report.get("metrics", {}).get("uq", {})
            if (
                report.get("run_name") != f"bl0r{seed}"
                or uq_protocol.get("ci_level") != 0.95
                or uq_protocol.get("ci_calibrate") != 0
            ):
                raise ValueError(f"{path}: original native/raw 95% UQ protocol mismatch")
            if Path(report.get("data_dir", "")).name != SETTING:
                raise ValueError(f"{path}: setting mismatch")
        else:
            method = {
                "fpca": "FPCA-NARGP",
                "freqwise": "Frequency-wise NARGP",
                "controlled": "Wavelength-wise GP correction controlled ablation",
            }[kind]
            if (report.get("dataset"), report.get("setting"), report.get("method")) != ("TM", SETTING, method):
                raise ValueError(f"{path}: dataset/setting/method mismatch")
            if report.get("run_kind") != "scientific" or report.get("scientific_result") is not True:
                raise ValueError(f"{path}: not a formal scientific report")

            if kind == "fpca":
                if (
                    report.get("ci_level") != 0.95
                    or report.get("uq_calibration") != "none; raw posterior uncertainty only"
                ):
                    raise ValueError(f"{path}: FPCA native/raw 95% UQ protocol mismatch")
            elif kind == "freqwise" and (
                report.get("complete_spectral_axis") is not True
                or report.get("preliminary_subset_diagnostic") is not False
            ):
                raise ValueError(f"{path}: incomplete frequency-wise spectral axis")
            elif kind == "controlled" and report.get("wavelength_count") != 500:
                raise ValueError(f"{path}: controlled Stage-II wavelength count mismatch")

        reports[seed] = (report, path)

    return reports


def observation(kind, key, report):
    if kind == "original":
        metrics = report["metrics"]
        return {
            "nll": metrics["nll"]["test"]["raw"][key],
            "coverage": metrics["uq"]["test"]["coverage_raw"][key],
            "width": metrics["uq"]["test"]["width_raw"][key],
        }

    metrics = report["metrics"]["test"]
    return {
        "nll": metrics["gaussian_nll"],
        "coverage": metrics["coverage_raw"],
        "width": metrics["interval_width_raw"],
    }


def stats(values):
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    half = float(t.ppf(0.975, n - 1)) * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


def verify_summary(rows):
    with SUMMARY.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))

    lookup = {}
    for entry in summary_rows:
        if (entry["dataset"], entry["setting"], entry["split"]) != ("TM", SETTING, "test"):
            continue
        identifier = (entry["method"], entry["metric"])
        if identifier in lookup:
            raise ValueError(f"Duplicate UQ summary row: {identifier}")
        lookup[identifier] = entry

    for row in rows:
        for metric in METRICS:
            identifier = (row["summary_name"], SUMMARY_METRICS[metric])
            if identifier not in lookup:
                raise ValueError(f"Missing UQ summary row: {identifier}")

            entry = lookup[identifier]
            if int(entry["n"]) != row["n"]:
                raise ValueError(f"{identifier}: n mismatch ({row['n']} vs {entry['n']})")

            for field, summary_field in (("mean", "mean"), ("ci_low", "ci95_low"), ("ci_high", "ci95_high")):
                actual = row[f"{metric}_{field}"]
                expected = require_number(float(entry[summary_field]), f"{identifier} {summary_field}")
                if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(
                        f"{identifier} {summary_field}: reports={actual:.16g}, summary={expected:.16g}"
                    )

            print(f"PASS summary: {identifier[0]} {identifier[1]}, n/mean/95% CI")

    print(f"PASS: all {len(rows) * len(METRICS)} UQ summary n/mean/95% CI checks")


def load_compact(path):
    with path.open(newline="", encoding="utf-8") as handle:
        compact_rows = list(csv.DictReader(handle))

    expected_fields = {"method", "summary_name", "role", "n"} | {
        f"{metric}_{field}"
        for metric in METRICS
        for field in ("mean", "std", "ci_low", "ci_high")
    }
    if not compact_rows:
        raise ValueError(f"Compact native-UQ source is empty: {path}")
    missing = expected_fields - set(compact_rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    config = {
        name: (summary_name, role, color)
        for name, summary_name, role, _, _, color in METHODS
    }
    expected_order = [entry[0] for entry in METHODS]
    if [row["method"] for row in compact_rows] != expected_order:
        raise ValueError(f"{path}: expected method order {expected_order}")

    rows = []
    for entry in compact_rows:
        name = entry["method"]
        summary_name, role, color = config[name]
        if entry["summary_name"] != summary_name or entry["role"] != role:
            raise ValueError(f"{path}: method metadata mismatch for {name}")
        row = {
            "method": name,
            "summary_name": summary_name,
            "role": role,
            "n": int(entry["n"]),
            "color": color,
            "source_root": "compact frozen summary",
        }
        for metric in METRICS:
            for field in ("mean", "std", "ci_low", "ci_high"):
                key = f"{metric}_{field}"
                row[key] = require_number(float(entry[key]), f"{path} {name} {key}")
        rows.append(row)

    verify_summary(rows)
    return rows


def collect():
    reports = {kind: load_reports(kind) for kind in ROOTS}
    rows = []

    for name, summary_name, role, kind, key, color in METHODS:
        values = {metric: [] for metric in METRICS}

        for _, (report, path) in sorted(reports[kind].items()):
            try:
                metrics = observation(kind, key, report)
            except KeyError as exc:
                raise ValueError(f"{path}: missing required raw UQ field {exc}") from exc

            for metric in METRICS:
                values[metric].append(require_number(metrics[metric], f"{path} {metric}"))

        row = {
            "method": name,
            "summary_name": summary_name,
            "role": role,
            "n": len(reports[kind]),
            "color": color,
            "source_root": str(ROOTS[kind].relative_to(PROJECT)),
        }

        for metric in METRICS:
            for field, value in stats(values[metric]).items():
                row[f"{metric}_{field}"] = value

        rows.append(row)

    verify_summary(rows)
    return rows


def _errorbar_kwargs(color):
    return {
        "ecolor": color,
        "capsize": 3,
        "capthick": 1.0,
        "elinewidth": 1.0,
    }


def _draw_hollow_marker(ax, x, y, color):
    """Draw a semi-transparent white-faced circle over the CI lines."""
    ax.plot(
        x,
        y,
        marker="o",
        linestyle="None",
        markerfacecolor=(1.0, 1.0, 1.0, 0.40),
        markeredgecolor=color,
        markeredgewidth=1.2,
        markersize=4.8,
        zorder=4,
    )


def plot(rows):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax_nll, ax_trade) = plt.subplots(1, 2, figsize=(11.2, 4.4), layout="constrained")

    for i, row in enumerate(rows):
        diagnostic = row["role"] == "computational diagnostic"

        nll_yerr = [[row["nll_mean"] - row["nll_ci_low"]], [row["nll_ci_high"] - row["nll_mean"]]]
        width_xerr = [[row["width_mean"] - row["width_ci_low"]], [row["width_ci_high"] - row["width_mean"]]]
        coverage_yerr = [[row["coverage_mean"] - row["coverage_ci_low"]], [row["coverage_ci_high"] - row["coverage_mean"]]]

        if diagnostic:
            # CI first, marker second. White face hides the CI crossing at the mean,
            # so the plotted marker matches the legend marker visually.
            ax_nll.errorbar(
                i,
                row["nll_mean"],
                yerr=nll_yerr,
                fmt="none",
                zorder=2,
                **_errorbar_kwargs(row["color"]),
            )
            _draw_hollow_marker(ax_nll, i, row["nll_mean"], row["color"])

            ax_trade.errorbar(
                row["width_mean"],
                row["coverage_mean"],
                xerr=width_xerr,
                yerr=coverage_yerr,
                fmt="none",
                zorder=2,
                **_errorbar_kwargs(row["color"]),
            )
            _draw_hollow_marker(ax_trade, row["width_mean"], row["coverage_mean"], row["color"])
        else:
            marker_style = {
                "fmt": "o",
                "color": row["color"],
                "markerfacecolor": matplotlib.colors.to_rgba(row["color"], 0.65),
                "markeredgecolor": row["color"],
                "markeredgewidth": 0.9,
                "markersize": 4.6,
                "label": "_nolegend_",
                "zorder": 3,
            }

            ax_nll.errorbar(
                i,
                row["nll_mean"],
                yerr=nll_yerr,
                **marker_style,
                **_errorbar_kwargs(row["color"]),
            )

            ax_trade.errorbar(
                row["width_mean"],
                row["coverage_mean"],
                xerr=width_xerr,
                yerr=coverage_yerr,
                fmt="none",
                zorder=2,
                **_errorbar_kwargs(row["color"]),
            )
            ax_trade.plot(
                row["width_mean"],
                row["coverage_mean"],
                marker="o",
                linestyle="None",
                markerfacecolor=matplotlib.colors.to_rgba(row["color"], 0.65),
                markeredgecolor=row["color"],
                markeredgewidth=0.9,
                markersize=4.6,
                zorder=4,
            )

    ax_nll.set_xticks(
        range(len(rows)),
        (
            "HF-only",
            "AR1/\nco-kriging",
            "FPCA-\nNARGP",
            "Wavelength-wise\nStage II",
            "Wavelength-wise\nNARGP",
            "Neural-GP\nMF",
        ),
    )
    ax_nll.set_xlim(-0.45, len(rows) - 0.55)
    ax_nll.set_ylabel("Native predictive NLL")

    ax_trade.set_xlabel("Mean prediction interval width")
    ax_trade.set_ylabel("Empirical coverage")
    ax_trade.axhline(0.95, ls="--", color="#666666", linewidth=1, zorder=1)
    ax_trade.annotate(
        "Nominal coverage = 0.95",
        xy=(0.98, 0.95),
        xycoords=("axes fraction", "data"),
        xytext=(0, 11),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
    )

    # Marker-only legend: no error bars in legend.
    legend_handles = []
    for row in rows:
        diagnostic = row["role"] == "computational diagnostic"
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="white" if diagnostic else row["color"],
                markeredgecolor=row["color"],
                markeredgewidth=1.5 if diagnostic else 1.0,
                markersize=5,
                label=row["method"],
            )
        )

    ax_trade.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=False,
        handletextpad=0.7,
        borderaxespad=0.4,
    )

    for ax, label in ((ax_nll, "(a)"), (ax_trade, "(b)")):
        ax.text(0.0, 1.03, label, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.grid(False)

    return fig


def write_csv_if_missing(rows, path):
    fields = ["method", "role", "n"] + [
        f"{metric}_{field}"
        for metric in METRICS
        for field in ("mean", "std", "ci_low", "ci_high")
    ] + ["source_root"]

    if path.exists():
        print(f"Preserved existing CSV: {path}")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    print(f"Created: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("compact", "reports"),
        default="compact",
        help="compact: public frozen CSV (default); reports: validate and rebuild from raw reports",
    )
    parser.add_argument(
        "--compact-csv",
        type=Path,
        default=FROZEN,
        help=f"Compact six-method plot source (default: {FROZEN})",
    )
    args = parser.parse_args()

    if args.source == "compact":
        if not args.compact_csv.is_file():
            raise FileNotFoundError(f"Compact native-UQ source missing: {args.compact_csv}")
        rows = load_compact(args.compact_csv)
        print(f"[INFO] plot source: {args.compact_csv.resolve()}")
    else:
        rows = collect()
        print("[INFO] plot source: validated raw reports")

    for row in rows:
        print(f"{row['method']} (n={row['n']}, {row['role']}):")
        for metric in METRICS:
            print(
                f"  {metric}: {row[f'{metric}_mean']:.12g} +/- 95% CI "
                f"[{row[f'{metric}_ci_low']:.12g}, {row[f'{metric}_ci_high']:.12g}]; "
                f"sample std={row[f'{metric}_std']:.12g}"
            )

    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": FIGURES / "_fig_native_uq.pdf",
        "png": FIGURES / "_fig_native_uq.png",
        "csv": FIGURES / "native_uq_data.csv",
    }

    fig = plot(rows)

    # Preserve formal provenance CSV once created; reruns refresh only the figures.
    write_csv_if_missing(rows, paths["csv"])

    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")

    print(
        f"Figure size: {fig.get_size_inches()[0]:.1f} x "
        f"{fig.get_size_inches()[1]:.1f} inches; PNG 300 dpi"
    )
    print(f"Created: {paths['pdf']}")
    print(f"Created: {paths['png']}")

    plt.close(fig)


if __name__ == "__main__":
    main()
