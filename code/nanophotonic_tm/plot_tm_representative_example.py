from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageChops


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

# Nanophotonic unit-cell schematic
GEOM_IMG = (
    REPO_ROOT
    / "result_out"
    / "final_analysis"
    / "figures"
    / "fig_nano_unit.png"
)

# Frozen HF100-LFx10 representative-example results
PRED_DIR = (
    REPO_ROOT
    / "result_out"
    / "final_analysis"
    / "frozen_inputs"
    / "tm_representative"
)

SUMMARY_CSV = PRED_DIR / "rmse_sample__summary.csv"

# Final paper-figure directory
OUT_DIR = (
    REPO_ROOT
    / "result_out"
    / "final_analysis"
    / "figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Fixed representative case
# ============================================================

TARGET_IDX = 11

WAVELENGTH_MIN_NM = 380.0
WAVELENGTH_MAX_NM = 750.0


# ============================================================
# Helpers
# ============================================================

def find_case(summary_csv: Path, target_idx: int):
    with open(summary_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Empty summary CSV: {summary_csv}")

    for row_pos, row in enumerate(rows):
        if int(float(row["idx"])) == target_idx:
            rmse_lf = float(row["rmse_lf"])
            rmse_mf = float(row["rmse_mf_student"])

            reduction_pct = (
                100.0 * (rmse_lf - rmse_mf) / rmse_lf
            )

            return row_pos, rmse_lf, rmse_mf, reduction_pct

    raise RuntimeError(
        f"idx={target_idx} not found in {summary_csv}"
    )


def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_geometry_on_white(image_path: Path):
    """
    Load the original RGBA geometry image without cropping or repositioning.
    Transparent pixels are composited onto a white background.
    """
    img = Image.open(image_path).convert("RGBA")

    white = Image.new(
        "RGBA",
        img.size,
        (255, 255, 255, 255),
    )
    white.alpha_composite(img)

    return np.asarray(white.convert("RGB"))



# ============================================================
# Main
# ============================================================

def main():
    required_files = [
        GEOM_IMG,
        SUMMARY_CSV,
        PRED_DIR / "y_true.npy",
        PRED_DIR / "y_lf.npy",
        PRED_DIR / "y_pred__mf_student.npy",
        PRED_DIR / "std_raw__mf_student.npy",
    ]

    for path in required_files:
        require_file(path)

    (
        row_pos,
        rmse_lf,
        rmse_mf,
        reduction_pct,
    ) = find_case(
        SUMMARY_CSV,
        TARGET_IDX,
    )

    print(f"[INFO] TARGET_IDX      = {TARGET_IDX}")
    print(f"[INFO] row position    = {row_pos}")
    print(f"[INFO] LF RMSE         = {rmse_lf:.8f}")
    print(f"[INFO] Neural-GP RMSE  = {rmse_mf:.8f}")
    print(f"[INFO] RMSE reduction  = {reduction_pct:.2f}%")

    # --------------------------------------------------------
    # Load frozen prediction arrays
    # --------------------------------------------------------

    y_true = np.load(PRED_DIR / "y_true.npy")
    y_lf = np.load(PRED_DIR / "y_lf.npy")
    y_mf = np.load(PRED_DIR / "y_pred__mf_student.npy")
    std_raw = np.load(PRED_DIR / "std_raw__mf_student.npy")

    print(f"[INFO] y_true shape     = {y_true.shape}")
    print(f"[INFO] y_lf shape       = {y_lf.shape}")
    print(f"[INFO] y_mf shape       = {y_mf.shape}")
    print(f"[INFO] std_raw shape    = {std_raw.shape}")

    if not (
        len(y_true)
        == len(y_lf)
        == len(y_mf)
        == len(std_raw)
    ):
        raise RuntimeError(
            "Prediction arrays have inconsistent sample counts."
        )

    if row_pos >= len(y_true):
        raise RuntimeError(
            f"row_pos={row_pos} exceeds prediction-array length "
            f"{len(y_true)}"
        )

    y_true_i = np.asarray(y_true[row_pos]).squeeze()
    y_lf_i = np.asarray(y_lf[row_pos]).squeeze()
    y_mf_i = np.asarray(y_mf[row_pos]).squeeze()
    std_i = np.asarray(std_raw[row_pos]).squeeze()

    if not (
        y_true_i.shape
        == y_lf_i.shape
        == y_mf_i.shape
        == std_i.shape
    ):
        raise RuntimeError(
            "Shape mismatch for selected case:\n"
            f"  y_true : {y_true_i.shape}\n"
            f"  y_lf   : {y_lf_i.shape}\n"
            f"  y_mf   : {y_mf_i.shape}\n"
            f"  std_raw: {std_i.shape}"
        )

    if y_true_i.ndim != 1:
        raise RuntimeError(
            f"Expected 1-D response arrays, got {y_true_i.shape}"
        )

    for name, arr in [
        ("y_true", y_true_i),
        ("y_lf", y_lf_i),
        ("y_mf", y_mf_i),
        ("std_raw", std_i),
    ]:
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(
                f"Non-finite values found in {name}"
            )

    if np.any(std_i < 0):
        raise RuntimeError(
            "Negative predictive standard deviations found."
        )

    # Same spectral range used by the TM benchmark
    wavelength_nm = np.linspace(
        WAVELENGTH_MIN_NM,
        WAVELENGTH_MAX_NM,
        y_true_i.size,
    )

    # Native Gaussian 95% predictive interval
    ci_low = y_mf_i - 1.96 * std_i
    ci_high = y_mf_i + 1.96 * std_i

    # Independent numerical check against the summary CSV
    rmse_lf_check = np.sqrt(
        np.mean((y_lf_i - y_true_i) ** 2)
    )
    rmse_mf_check = np.sqrt(
        np.mean((y_mf_i - y_true_i) ** 2)
    )

    print(
        f"[CHECK] LF RMSE from arrays        = "
        f"{rmse_lf_check:.8f}"
    )
    print(
        f"[CHECK] Neural-GP RMSE from arrays = "
        f"{rmse_mf_check:.8f}"
    )

    if not np.isclose(
        rmse_lf_check,
        rmse_lf,
        rtol=1e-5,
        atol=1e-7,
    ):
        raise RuntimeError(
            "LF RMSE reconstructed from arrays does not match "
            "rmse_sample__summary.csv."
        )

    if not np.isclose(
        rmse_mf_check,
        rmse_mf,
        rtol=1e-5,
        atol=1e-7,
    ):
        raise RuntimeError(
            "Neural-GP RMSE reconstructed from arrays does not match "
            "rmse_sample__summary.csv."
        )

    # --------------------------------------------------------
    # Publication plotting style
    # --------------------------------------------------------

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )

    # --------------------------------------------------------
    # Save the response panel independently
    # --------------------------------------------------------

    fig_panel, ax = plt.subplots(
        figsize=(5.3, 4.0)
    )

    ax.plot(
        wavelength_nm,
        y_true_i,
        linewidth=2.0,
        label="HF target",
    )

    ax.plot(
        wavelength_nm,
        y_lf_i,
        linewidth=1.8,
        linestyle="--",
        label="LF",
    )

    mf_line, = ax.plot(
        wavelength_nm,
        y_mf_i,
        linewidth=2.0,
        label="Neural-GP MF",
    )

    ax.fill_between(
        wavelength_nm,
        ci_low,
        ci_high,
        color=mf_line.get_color(),
        alpha=0.18,
        linewidth=0,
        label="95% predictive interval",
    )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission")
    ax.set_xlim(
        WAVELENGTH_MIN_NM,
        WAVELENGTH_MAX_NM,
    )

    # Reviewer-requested figure format: no background grid
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="best",
    )

    fig_panel.tight_layout()

    panel_png = (
        OUT_DIR
        / "tm_example_hf100_lfx10_idx0011_pub.png"
    )
    panel_pdf = (
        OUT_DIR
        / "tm_example_hf100_lfx10_idx0011_pub.pdf"
    )

    fig_panel.savefig(panel_png)
    fig_panel.savefig(panel_pdf)

    plt.close(fig_panel)

    # --------------------------------------------------------
    # Final two-panel paper figure
    # (a) nanophotonic unit-cell schematic
    # (b) representative HF100-LFx10 reconstruction
    # --------------------------------------------------------

    geom_img = load_geometry_on_white(GEOM_IMG)

    fig = plt.figure(
        figsize=(11.0, 4.4)
    )

    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.12, 1.0],
        wspace=0.22,
    )

    ax_geom = fig.add_subplot(gs[0, 0])
    ax_resp = fig.add_subplot(gs[0, 1])

    # Panel (a)
    ax_geom.imshow(geom_img)
    ax_geom.axis("off")


    ax_geom.text(
        -0.02,
        1.02,
        "(a)",
        transform=ax_geom.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
    )

    # Panel (b)
    ax_resp.plot(
        wavelength_nm,
        y_true_i,
        linewidth=2.0,
        label="HF target",
    )

    ax_resp.plot(
        wavelength_nm,
        y_lf_i,
        linewidth=1.8,
        linestyle="--",
        label="LF",
    )

    mf_line_combined, = ax_resp.plot(
        wavelength_nm,
        y_mf_i,
        linewidth=2.0,
        label="Neural-GP MF",
    )

    ax_resp.fill_between(
        wavelength_nm,
        ci_low,
        ci_high,
        color=mf_line_combined.get_color(),
        alpha=0.18,
        linewidth=0,
        label="95% predictive interval",
    )

    ax_resp.set_xlabel("Wavelength (nm)")
    ax_resp.set_ylabel("Transmission")

    ax_resp.set_xlim(
        WAVELENGTH_MIN_NM,
        WAVELENGTH_MAX_NM,
    )

    ax_resp.grid(False)

    ax_resp.spines["top"].set_visible(False)
    ax_resp.spines["right"].set_visible(False)

    ax_resp.legend(
        frameon=False,
        loc="best",
    )

    ax_resp.text(
        -0.02,
        1.02,
        "(b)",
        transform=ax_resp.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
    )

    # Semantic filename, independent of final manuscript figure number
    final_png = OUT_DIR / "_fig_tm_examples.png"
    final_pdf = OUT_DIR / "_fig_tm_examples.pdf"

    fig.savefig(final_png)
    fig.savefig(final_pdf)

    plt.close(fig)

    print()
    print("[DONE] Generated:")
    print(f"  {panel_png}")
    print(f"  {panel_pdf}")
    print(f"  {final_png}")
    print(f"  {final_pdf}")


if __name__ == "__main__":
    main()
