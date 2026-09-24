from pathlib import Path
import csv

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plot_uq_from_cache as base


# ============================================================
# Paths and fixed original-Fig.-7 configuration
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

CACHE_PATH = (
    REPO_ROOT
    / "result_out"
    / "final_analysis"
    / "frozen_inputs"
    / "conformal_uq"
    / "uq_cache_v1.npz"
)

OUT_DIR = (
    REPO_ROOT
    / "result_out"
    / "final_analysis"
    / "figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CI_GRID = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.97,
    0.98,
    0.99,
]

CI_LEVEL = 0.95

METHODS = list(base.METHODS)

LABEL_MAP = {
    "hf_only": "HF-only",
    "ar1": "co-kriging",
    "ours": "Neural-GP MF",
}


# ============================================================
# Plot style
# ============================================================

def apply_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def get_method_colors():
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {
        method: colors[i]
        for i, method in enumerate(METHODS)
    }


# ============================================================
# Numerical provenance
# ============================================================

def compute_results():
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(
            f"UQ cache not found: {CACHE_PATH}"
        )

    cache = base.load_cache(CACHE_PATH)

    curves = base.compute_reliability_curve(
        cache,
        CI_GRID,
    )

    points = base.compute_cov_width_points(
        cache,
        CI_LEVEL,
    )

    ece = {
        method: base.ece_from_curve(
            CI_GRID,
            curves[method],
        )
        for method in METHODS
    }

    return curves, points, ece


def save_summary(points, ece):
    out_csv = (
        REPO_ROOT
        / "result_out"
        / "final_analysis"
        / "uq_conformal_summary.csv"
    )

    with open(
        out_csv,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "method",
                "raw_ece",
                "raw_coverage_95",
                "raw_width_95",
                "calibrated_coverage_95",
                "calibrated_width_95",
            ]
        )

        for method in METHODS:
            raw_cov, raw_width = points[method]["raw"]
            cal_cov, cal_width = points[method]["cal"]

            writer.writerow(
                [
                    LABEL_MAP[method],
                    f"{ece[method]:.8f}",
                    f"{raw_cov:.8f}",
                    f"{raw_width:.8f}",
                    f"{cal_cov:.8f}",
                    f"{cal_width:.8f}",
                ]
            )

    print(f"[DONE] Summary: {out_csv}")


# ============================================================
# Panel (a): raw reliability curve
# ============================================================

def draw_reliability(ax, curves, ece, method_colors):
    ax.plot(
        [min(CI_GRID), max(CI_GRID)],
        [min(CI_GRID), max(CI_GRID)],
        linestyle=":",
        linewidth=1.8,
        label="Ideal",
    )

    for method in METHODS:
        ax.plot(
            CI_GRID,
            curves[method],
            color=method_colors[method],
            linewidth=2.1,
            label=LABEL_MAP[method],
        )

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")

    ax.set_xlim(
        min(CI_GRID),
        max(CI_GRID),
    )

    y_all = [
        y
        for method in METHODS
        for y in curves[method]
    ]

    y_min = min(y_all)
    y_max = max(y_all)

    span = max(1e-6, y_max - y_min)
    margin = max(0.02, 0.15 * span)

    ax.set_ylim(
        max(0.0, y_min - margin),
        min(1.0, y_max + margin),
    )

    # Reviewer-requested clean style
    ax.grid(False)

    ece_text = "\n".join(
        [
            f"{LABEL_MAP[m]}  ECE={ece[m]:.3f}"
            for m in METHODS
        ]
    )

    ax.text(
        0.03,
        0.97,
        ece_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.6",
            "linewidth": 0.8,
            "alpha": 0.9,
        },
    )

    ax.legend(
        loc="lower right",
        frameon=False,
    )


# ============================================================
# Panel (b): raw vs conformal coverage-width
# ============================================================

def draw_cov_width(ax, points, method_colors):
    ax.axvline(
        CI_LEVEL,
        linestyle=":",
        linewidth=1.8,
        color="0.35",
        label="_nolegend_",
    )

    cov_all = []

    for method in METHODS:
        raw_cov, raw_width = points[method]["raw"]
        cal_cov, cal_width = points[method]["cal"]

        cov_all.extend(
            [raw_cov, cal_cov]
        )

        color = method_colors[method]
        name = LABEL_MAP[method]

        ax.plot(
            [raw_cov, cal_cov],
            [raw_width, cal_width],
            color=color,
            linewidth=1.5,
        )

        ax.scatter(
            [raw_cov],
            [raw_width],
            marker="x",
            s=75,
            color=color,
            linewidths=2.0,
            label="_nolegend_",
            zorder=3,
        )

        ax.scatter(
            [cal_cov],
            [cal_width],
            marker="o",
            s=58,
            color=color,
            label="_nolegend_",
            zorder=3,
        )

    cmin = min(cov_all)
    cmax = max(cov_all)

    span = max(1e-6, cmax - cmin)
    margin = max(0.02, 0.25 * span)

    ax.set_xlim(
        max(0.0, cmin - margin),
        min(1.0, cmax + margin),
    )

    ax.set_xlabel("Empirical coverage")
    ax.set_ylabel(
        "Mean interval width (response units)"
    )

    ax.grid(False)

    legend_handles = [
        Line2D(
            [0], [0],
            color=method_colors["hf_only"],
            linewidth=2.0,
            label="HF-only",
        ),
        Line2D(
            [0], [0],
            color=method_colors["ar1"],
            linewidth=2.0,
            label="co-kriging",
        ),
        Line2D(
            [0], [0],
            color=method_colors["ours"],
            linewidth=2.0,
            label="Neural-GP MF",
        ),
        Line2D(
            [0], [0],
            color="0.25",
            marker="x",
            markersize=8,
            markeredgewidth=2.0,
            linestyle="None",
            label="Raw",
        ),
        Line2D(
            [0], [0],
            color="0.25",
            marker="o",
            markersize=7,
            linestyle="None",
            label="Calibrated",
        ),
        Line2D(
            [0], [0],
            color="0.35",
            linestyle=":",
            linewidth=1.8,
            label=f"Nominal {CI_LEVEL:.2f}",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=False,
        fontsize=8.5,
    )


# ============================================================
# Save individual panels
# ============================================================

def save_single_panels(
    curves,
    points,
    ece,
    method_colors,
):
    fig, ax = plt.subplots(
        figsize=(6.0, 5.0)
    )

    draw_reliability(
        ax,
        curves,
        ece,
        method_colors,
    )

    fig.tight_layout()

    png = OUT_DIR / "uq_reliability_raw.png"
    pdf = OUT_DIR / "uq_reliability_raw.pdf"

    fig.savefig(png)
    fig.savefig(pdf)
    fig.savefig(OUT_DIR / "_fig_conformal_uq.png")
    fig.savefig(OUT_DIR / "_fig_conformal_uq.pdf")

    plt.close(fig)

    print(f"[DONE] {png}")
    print(f"[DONE] {pdf}")

    fig, ax = plt.subplots(
        figsize=(6.0, 5.0)
    )

    draw_cov_width(
        ax,
        points,
        method_colors,
    )

    fig.tight_layout()

    png = OUT_DIR / "uq_coverage_width_conformal.png"
    pdf = OUT_DIR / "uq_coverage_width_conformal.pdf"

    fig.savefig(png)
    fig.savefig(pdf)

    plt.close(fig)

    print(f"[DONE] {png}")
    print(f"[DONE] {pdf}")


# ============================================================
# Final combined supplementary figure
# ============================================================

def save_combined(
    curves,
    points,
    ece,
    method_colors,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.0),
    )

    draw_reliability(
        axes[0],
        curves,
        ece,
        method_colors,
    )

    draw_cov_width(
        axes[1],
        points,
        method_colors,
    )

    # Panel labels outside axes and aligned at the same height
    fig.text(
        0.008,
        0.995,
        "(a)",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )

    fig.text(
        0.505,
        0.995,
        "(b)",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[0.01, 0.01, 1.0, 0.965]
    )

    png = OUT_DIR / "_fig_uc.png"
    pdf = OUT_DIR / "_fig_uc.pdf"

    fig.savefig(png)
    fig.savefig(pdf)

    plt.close(fig)

    print(f"[DONE] {png}")
    print(f"[DONE] {pdf}")


# ============================================================
# Main
# ============================================================

def main():
    apply_style()

    curves, points, ece = compute_results()
    method_colors = get_method_colors()

    print("===== UQ provenance =====")
    print(f"Cache: {CACHE_PATH}")
    print()

    print("===== Raw ECE =====")
    for method in METHODS:
        print(
            f"{LABEL_MAP[method]:15s} "
            f"{ece[method]:.6f}"
        )

    print()
    print("===== 95% coverage / width =====")

    for method in METHODS:
        raw_cov, raw_width = points[method]["raw"]
        cal_cov, cal_width = points[method]["cal"]

        print(LABEL_MAP[method])
        print(
            f"  raw: coverage={raw_cov:.6f}, "
            f"width={raw_width:.6f}"
        )
        print(
            f"  cal: coverage={cal_cov:.6f}, "
            f"width={cal_width:.6f}"
        )

    save_summary(points, ece)

    save_single_panels(
        curves,
        points,
        ece,
        method_colors,
    )

    save_combined(
        curves,
        points,
        ece,
        method_colors,
    )


if __name__ == "__main__":
    main()
