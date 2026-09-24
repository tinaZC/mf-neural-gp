#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication-style AB RMSE sweep from the frozen formal 20-seed analysis.

Data source:
    result_out/final_analysis/main_accuracy_summary.csv

Protocol:
- Dataset: AB
- Split: test
- HF budgets: 50, 100, 200, 300, 400, 500
- n = 20 seeds per setting/method
- Center: mean test RMSE
- Error bars: 95% confidence interval from the frozen formal summary
- No model training and no historical-repository dependency.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Publication legend ordering
# Keep plotting/data order unchanged; only move the proposed method
# "Neural-GP MF" to the final legend position.
# ---------------------------------------------------------------------
def _move_neural_gp_last_in_legend(fig):
    proposed_label = "Neural-GP MF"

    for ax in fig.axes:
        old_legend = ax.get_legend()
        if old_legend is None:
            continue

        handles, labels = ax.get_legend_handles_labels()
        if proposed_label not in labels:
            continue

        # Remove duplicate labels while preserving their current order.
        unique_handles = []
        unique_labels = []
        seen = set()
        for h, lab in zip(handles, labels):
            if not lab or lab.startswith("_") or lab in seen:
                continue
            seen.add(lab)
            unique_handles.append(h)
            unique_labels.append(lab)

        if proposed_label not in unique_labels:
            continue

        # Preserve all existing relative ordering, but put Neural-GP MF last.
        order = [
            i for i, lab in enumerate(unique_labels)
            if lab != proposed_label
        ]
        order.append(unique_labels.index(proposed_label))

        handles_new = [unique_handles[i] for i in order]
        labels_new = [unique_labels[i] for i in order]

        # Preserve the main visual properties of the existing legend.
        kwargs = {
            "loc": getattr(old_legend, "_loc", "best"),
            "frameon": old_legend.get_frame_on(),
        }

        ncols = getattr(old_legend, "_ncols", None)
        if ncols is not None:
            kwargs["ncols"] = ncols

        texts = old_legend.get_texts()
        if texts:
            kwargs["fontsize"] = texts[0].get_fontsize()

        title = old_legend.get_title().get_text()
        if title:
            kwargs["title"] = title

        ax.legend(handles_new, labels_new, **kwargs)

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

SUMMARY_CSV = (
    REPO_ROOT
    / "result_out"
    / "final_analysis"
    / "main_accuracy_summary.csv"
)

OUT_DIR = (
    REPO_ROOT
    / "result_out"
    / "final_analysis"
    / "figures"
)

BUDGETS = [50, 100, 200, 300, 400, 500]

SETTINGS = {
    50: "hf50_lfx10",
    100: "hf100_lfx10",
    200: "hf200_lfx10",
    300: "hf300_lfx10",
    400: "hf400_lfx10",
    500: "hf500_lfx10",
}

# Explicit mapping to the metric name used by each formal result pipeline.
# The script intentionally fails if any expected row is absent.
METHODS = [
    {
        "method": "HF-only",
        "label": "HF-only",
        "metric": "target_rmse",
        "color": "#1f77b4",
        "marker": "s",
        "linewidth": 1.9,
        "markersize": 5.5,
        "zorder": 3,
    },
    {
        "method": "AR1/co-kriging",
        "label": "AR1 / co-kriging",
        "metric": "target_rmse",
        "color": "#ff7f0e",
        "marker": "^",
        "linewidth": 1.9,
        "markersize": 5.8,
        "zorder": 4,
    },
    {
        "method": "Neural-GP MF (ours)",
        "label": "Neural-GP MF",
        "metric": "target_rmse",
        "color": "#2ca02c",
        "marker": "*",
        "linewidth": 2.3,
        "markersize": 9.0,
        "zorder": 6,
    },
    {
        "method": "FPCA-NARGP",
        "label": "FPCA-NARGP",
        "metric": "rmse",
        "color": "#d62728",
        "marker": "D",
        "linewidth": 1.8,
        "markersize": 5.2,
        "zorder": 5,
    },
    {
        "method": "MF-DeepONet",
        "label": "MF-DeepONet",
        "metric": "rmse",
        "color": "#9467bd",
        "marker": "o",
        "linewidth": 1.8,
        "markersize": 5.2,
        "zorder": 2,
    },
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "figure.titlesize": 11,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def extract_formal_rows(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for spec in METHODS:
        for budget in BUDGETS:
            setting = SETTINGS[budget]

            hit = df[
                (df["dataset"] == "AB")
                & (df["setting"] == setting)
                & (df["method"] == spec["method"])
                & (df["split"] == "test")
                & (df["metric"] == spec["metric"])
            ].copy()

            if len(hit) != 1:
                raise RuntimeError(
                    "Expected exactly one formal summary row for "
                    f"method={spec['method']!r}, "
                    f"setting={setting!r}, "
                    f"metric={spec['metric']!r}, "
                    f"split='test'; found {len(hit)}."
                )

            row = hit.iloc[0]

            if int(row["n"]) != 20:
                raise RuntimeError(
                    f"{spec['method']} {setting}: expected n=20, "
                    f"found n={row['n']}"
                )

            values = [
                float(row["mean"]),
                float(row["ci95_low"]),
                float(row["ci95_high"]),
            ]

            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"{spec['method']} {setting}: non-finite formal statistics."
                )

            if not (
                float(row["ci95_low"])
                <= float(row["mean"])
                <= float(row["ci95_high"])
            ):
                raise RuntimeError(
                    f"{spec['method']} {setting}: mean is outside its CI."
                )

            records.append(
                {
                    "dataset": "AB",
                    "setting": setting,
                    "hf_budget": budget,
                    "method": spec["method"],
                    "display_method": spec["label"],
                    "metric": spec["metric"],
                    "n": int(row["n"]),
                    "mean_rmse": float(row["mean"]),
                    "std_rmse": float(row["std"]),
                    "ci95_low": float(row["ci95_low"]),
                    "ci95_high": float(row["ci95_high"]),
                }
            )

    out = pd.DataFrame(records)

    expected_rows = len(METHODS) * len(BUDGETS)
    if len(out) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} formal plot rows, got {len(out)}."
        )

    return out


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing formal summary: {SUMMARY_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_CSV)

    required_cols = {
        "dataset",
        "setting",
        "method",
        "split",
        "metric",
        "n",
        "mean",
        "std",
        "ci95_low",
        "ci95_high",
    }

    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise RuntimeError(
            f"Formal summary is missing required columns: {missing}"
        )

    plot_df = extract_formal_rows(df)

    data_csv = OUT_DIR / "fig_ab_rmse_sweep_data.csv"
    plot_df.to_csv(data_csv, index=False)

    apply_style()

    fig, ax = plt.subplots(figsize=(6.9, 4.35))

    x = np.asarray(BUDGETS, dtype=float)

    for spec in METHODS:
        sub = (
            plot_df[
                plot_df["method"] == spec["method"]
            ]
            .sort_values("hf_budget")
            .reset_index(drop=True)
        )

        y = sub["mean_rmse"].to_numpy(dtype=float)
        lo = sub["ci95_low"].to_numpy(dtype=float)
        hi = sub["ci95_high"].to_numpy(dtype=float)

        yerr = np.vstack(
            [
                y - lo,
                hi - y,
            ]
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=spec["label"],
            color=spec["color"],
            linewidth=spec["linewidth"],
            marker=spec["marker"],
            markersize=spec["markersize"],
            markerfacecolor=(
                spec["color"]
                if spec["method"] == "Neural-GP MF (ours)"
                else "white"
            ),
            markeredgecolor=spec["color"],
            markeredgewidth=1.1,
            capsize=2.8,
            capthick=0.9,
            elinewidth=0.9,
            zorder=spec["zorder"],
        )

    ax.set_xlabel(r"HF budget, $N_h$")
    ax.set_ylabel("Test RMSE")

    ax.set_xticks(BUDGETS)
    ax.set_xlim(35, 515)

    # Publication style: no background grid.
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Upper-right region is largely free because RMSE decreases with HF budget.
    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=2.2,
        labelspacing=0.45,
        borderpad=0.2,
    )

    fig.tight_layout()

    out_png = OUT_DIR / "fig_ab_rmse_sweep.png"
    out_pdf = OUT_DIR / "fig_ab_rmse_sweep.pdf"
    manuscript_png = OUT_DIR / "_fig_ab_rmse_sweep.png"
    manuscript_pdf = OUT_DIR / "_fig_ab_rmse_sweep.pdf"

    _move_neural_gp_last_in_legend(fig)
    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
    )

    _move_neural_gp_last_in_legend(fig)
    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )

    fig.savefig(manuscript_png, dpi=300, bbox_inches="tight")
    fig.savefig(manuscript_pdf, bbox_inches="tight")
    plt.close(fig)

    print("[INFO] Formal source:", SUMMARY_CSV)
    print("[INFO] Dataset      : AB")
    print("[INFO] Split        : test")
    print("[INFO] Seeds        : n=20 per method/setting")
    print("[INFO] HF budgets   :", BUDGETS)
    print("[INFO] Error bars   : 95% confidence intervals")
    print()
    print(
        plot_df[
            [
                "hf_budget",
                "display_method",
                "n",
                "mean_rmse",
                "ci95_low",
                "ci95_high",
            ]
        ].to_string(index=False)
    )
    print()
    print("[OUT]", data_csv)
    print("[OUT]", out_png)
    print("[OUT]", out_pdf)


if __name__ == "__main__":
    main()
