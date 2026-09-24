#!/usr/bin/env python3
"""Build the publication-facing timing table from the two frozen timing roots."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "result_out/final_timing_baseline"
MODERN_ROOT = ROOT / "result_out/final_timing"
OUT_CSV = ROOT / "result_out/final_analysis/final_timing_comparison.csv"
OUT_MD = ROOT / "result_out/final_analysis/final_timing_comparison.md"
SETTINGS = {
    "hf50_lfx10": (40, 400),
    "hf100_lfx10": (80, 800),
    "hf500_lfx10": (400, 4000),
}


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read required JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def positive(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise RuntimeError(f"{label} must be finite and positive, got {number!r}")
    return number


def validate_identity(data: dict, path: Path, setting: str) -> None:
    if data.get("dataset") not in (None, "TM"):
        raise RuntimeError(f"Non-TM dataset in {path}: {data.get('dataset')!r}")
    if data.get("setting") not in (None, setting):
        raise RuntimeError(f"Setting mismatch in {path}: {data.get('setting')!r}")
    if data.get("seed") != 42:
        raise RuntimeError(f"Seed mismatch in {path}: {data.get('seed')!r}")
    if data.get("device") not in (None, "cuda"):
        raise RuntimeError(f"Device mismatch in {path}: {data.get('device')!r}")


def row(
    setting: str,
    method: str,
    role: str,
    training_s: float,
    inference_s: float,
    source: Path,
    kpi: str,
) -> dict:
    hf_count, lf_count = SETTINGS[setting]
    return {
        "TM setting": setting,
        "HF train count": hf_count,
        "LF train count": lf_count,
        "method": method,
        "method role": role,
        "training time seconds": training_s,
        "training time minutes": training_s / 60.0,
        "inference time seconds": inference_s,
        "Neural-GP / comparator training-time ratio": "",
        "Neural-GP training-time reduction percent": "",
        "training KPI components": kpi,
        "canonical source": source.relative_to(ROOT).as_posix(),
    }


def collect_rows() -> tuple[list[dict], list[str]]:
    expected_baseline: set[Path] = set()
    expected_modern: set[Path] = set()
    rows: list[dict] = []
    checks: list[str] = []

    for setting, counts in SETTINGS.items():
        run = BASELINE_ROOT / f"{setting}_seed42/timing_tm/timing.json"
        ours = run.parent / "ours/timing.json"
        expected_baseline.update((run, ours))
        base = load_json(run)
        neural = load_json(ours)
        validate_identity(base, run, setting)
        validate_identity(neural, ours, setting)
        if (base.get("n_hf_train"), base.get("n_lf_train_total")) != counts:
            raise RuntimeError(f"Training-count mismatch in {run}")
        if (neural.get("n_hf_train"), neural.get("n_lf_train_total")) != counts:
            raise RuntimeError(f"Training-count mismatch in {ours}")

        base_stage2 = base.get("stage2_train_s", {})
        base_infer = base.get("inference_s", {})
        neural_stage2 = neural.get("stage2_train_s", {})
        neural_infer = neural.get("inference_s", {})
        embedded = base.get("ours_delegate_timing")
        if embedded != neural or Path(base.get("ours_delegate_timing_path", "")) != ours:
            raise RuntimeError(f"Embedded/delegated Neural-GP timing mismatch for {setting}")

        rows.extend(
            [
                row(
                    setting,
                    "HF-only",
                    "main baseline",
                    positive(base_stage2.get("hf_only"), f"{run}: HF-only training"),
                    positive(base_infer.get("hf_only_val_test"), f"{run}: HF-only inference"),
                    run,
                    "stage2_train_s.hf_only",
                ),
                row(
                    setting,
                    "AR1 / co-kriging",
                    "main baseline",
                    positive(base_stage2.get("ar1_total"), f"{run}: AR1 training"),
                    positive(base_infer.get("ar1_val_test"), f"{run}: AR1 inference"),
                    run,
                    "stage2_train_s.ar1_total",
                ),
                row(
                    setting,
                    "Neural-GP MF",
                    "proposed method",
                    positive(neural.get("stage1_train_s"), f"{ours}: Stage I")
                    + positive(neural_stage2.get("mf_student"), f"{ours}: Stage II mf_student"),
                    positive(neural_infer.get("mf_student_val_test"), f"{ours}: inference"),
                    ours,
                    "stage1_train_s + stage2_train_s.mf_student",
                ),
            ]
        )

        for slug, method in (("fpca_nargp", "FPCA-NARGP"), ("mf_deeponet", "MF-DeepONet")):
            path = MODERN_ROOT / f"{slug}_{setting}_seed42/timing.json"
            expected_modern.add(path)
            timing = load_json(path)
            validate_identity(timing, path, setting)
            if method == "FPCA-NARGP":
                if (timing.get("n_hf_train"), timing.get("n_lf_train_total")) != counts:
                    raise RuntimeError(f"Training-count mismatch in {path}")
                training = positive(timing.get("stage1_train_s"), f"{path}: Stage I") + positive(
                    timing.get("stage2_train_s"), f"{path}: Stage II"
                )
                inference = positive(timing.get("inference_val_test_s"), f"{path}: inference")
                kpi = "stage1_train_s + stage2_train_s"
            else:
                training = positive(
                    timing.get("stage_i_lf_training_seconds"), f"{path}: Stage I"
                ) + positive(
                    timing.get("stage_ii_hf_correction_training_seconds"), f"{path}: Stage II"
                )
                inference = positive(timing.get("inference_seconds"), f"{path}: inference")
                kpi = (
                    "stage_i_lf_training_seconds + "
                    "stage_ii_hf_correction_training_seconds"
                )
            rows.append(row(setting, method, "main baseline", training, inference, path, kpi))

    setting = "hf100_lfx10"
    for slug, method, role in (
        ("direct_latent", "Direct-latent", "representation ablation"),
        ("freqwise_nargp", "Frequency-wise NARGP", "computational diagnostic"),
    ):
        path = MODERN_ROOT / f"{slug}_{setting}_seed42/timing.json"
        expected_modern.add(path)
        timing = load_json(path)
        validate_identity(timing, path, setting)
        if method == "Frequency-wise NARGP" and timing.get("completed_wavelengths") != 500:
            raise RuntimeError(f"Frequency-wise run is incomplete in {path}")
        rows.append(
            row(
                setting,
                method,
                role,
                positive(timing.get("stage1_train_s"), f"{path}: Stage I")
                + positive(timing.get("stage2_train_s"), f"{path}: Stage II"),
                positive(timing.get("inference_val_test_s"), f"{path}: inference"),
                path,
                "stage1_train_s + stage2_train_s",
            )
        )

    actual_baseline = set(BASELINE_ROOT.rglob("timing.json"))
    actual_modern = set(MODERN_ROOT.rglob("timing.json"))
    if actual_baseline != expected_baseline:
        raise RuntimeError(
            f"Unexpected baseline timing sources: missing={expected_baseline - actual_baseline}, "
            f"extra={actual_baseline - expected_baseline}"
        )
    if actual_modern != expected_modern:
        raise RuntimeError(
            f"Unexpected modern timing sources: missing={expected_modern - actual_modern}, "
            f"extra={actual_modern - expected_modern}"
        )

    keys = [(r["TM setting"], r["method"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate setting/method timing source")

    by_key = {(r["TM setting"], r["method"]): r for r in rows}
    comparators = {"HF-only", "AR1 / co-kriging", "FPCA-NARGP", "MF-DeepONet"}
    for timing_row in rows:
        method = timing_row["method"]
        if method not in comparators:
            continue
        neural_s = float(by_key[(timing_row["TM setting"], "Neural-GP MF")]["training time seconds"])
        comparator_s = float(timing_row["training time seconds"])
        timing_row["Neural-GP / comparator training-time ratio"] = neural_s / comparator_s
        timing_row["Neural-GP training-time reduction percent"] = (1.0 - neural_s / comparator_s) * 100.0

    baseline_summary = BASELINE_ROOT / "timing_tables/timing_table_summary.csv"
    modern_summary = MODERN_ROOT / "timing_summary.csv"
    with baseline_summary.open(newline="", encoding="utf-8") as handle:
        baseline_records = list(csv.DictReader(handle))
    with modern_summary.open(newline="", encoding="utf-8") as handle:
        modern_records = list(csv.DictReader(handle))
    if len(baseline_records) != 12:
        raise RuntimeError(f"Expected 12 baseline summary rows, found {len(baseline_records)}")
    if len(modern_records) != 8 or any(r.get("status") != "success" for r in modern_records):
        raise RuntimeError("Modern timing summary must contain exactly eight successful jobs")

    for timing_row in rows:
        for field in ("training time seconds", "training time minutes", "inference time seconds"):
            positive(timing_row[field], f"output {timing_row['TM setting']}/{timing_row['method']}/{field}")

    checks.extend(
        [
            "14/14 formal timing.json files accounted for (6 baseline-root, 8 modern-root).",
            "17/17 setting-method rows have unique canonical sources and finite, positive KPI values.",
            "The 3 embedded Neural-GP timing objects exactly match their canonical nested timing.json files.",
            "Baseline timing summary has 12 expected rows; modern timing summary has 8/8 successful jobs.",
            "Frequency-wise NARGP covers all 500 wavelength coordinates.",
        ]
    )
    order = {"hf50_lfx10": 0, "hf100_lfx10": 1, "hf500_lfx10": 2}
    method_order = {
        "HF-only": 0,
        "AR1 / co-kriging": 1,
        "Neural-GP MF": 2,
        "FPCA-NARGP": 3,
        "MF-DeepONet": 4,
        "Direct-latent": 5,
        "Frequency-wise NARGP": 6,
    }
    rows.sort(key=lambda r: (order[r["TM setting"]], method_order[r["method"]]))
    return rows, checks


def fmt(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def write_outputs(rows: list[dict], checks: list[str]) -> None:
    fields = list(rows[0])
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            output = dict(item)
            for field in (
                "training time seconds",
                "training time minutes",
                "inference time seconds",
                "Neural-GP / comparator training-time ratio",
                "Neural-GP training-time reduction percent",
            ):
                if output[field] != "":
                    output[field] = f"{float(output[field]):.6f}"
            writer.writerow(output)

    main_rows = [r for r in rows if r["method role"] not in {"representation ablation", "computational diagnostic"}]
    lines = [
        "# Final Timing Comparison",
        "",
        "Publication-facing model-internal timing for TM, seed 42, CUDA. Training excludes electromagnetic simulation, orchestration, serialization, and evaluation overhead. Inference is the recorded validation-plus-test inference workload.",
        "",
        "## Main timing table",
        "",
        "| TM setting | HF train | LF train | Method | Training (s) | Training (min) | Inference (s) |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for item in main_rows:
        lines.append(
            f"| {item['TM setting']} | {item['HF train count']} | {item['LF train count']} | "
            f"{item['method']} | {fmt(item['training time seconds'])} | "
            f"{fmt(item['training time minutes'])} | {fmt(item['inference time seconds'])} |"
        )

    lines.extend(
        [
            "",
            "## Neural-GP training-time comparisons",
            "",
            "Ratio is `Neural-GP training time / comparator training time`. Reduction is `(comparator - Neural-GP) / comparator`; a negative reduction means Neural-GP is slower.",
            "",
            "| TM setting | Comparator | Neural-GP / comparator | Reduction |",
            "|---|---|---:|---:|",
        ]
    )
    for item in rows:
        if item["Neural-GP / comparator training-time ratio"] == "":
            continue
        lines.append(
            f"| {item['TM setting']} | {item['method']} | "
            f"{fmt(item['Neural-GP / comparator training-time ratio'])}x | "
            f"{fmt(item['Neural-GP training-time reduction percent'], 2)}% |"
        )

    diagnostics = [r for r in rows if r["method role"] in {"representation ablation", "computational diagnostic"}]
    lines.extend(
        [
            "",
            "## Separate diagnostic results",
            "",
            "| TM setting | HF train | LF train | Method | Role | Training (s) | Training (min) | Inference (s) |",
            "|---|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for item in diagnostics:
        lines.append(
            f"| {item['TM setting']} | {item['HF train count']} | {item['LF train count']} | "
            f"{item['method']} | {item['method role']} | {fmt(item['training time seconds'])} | "
            f"{fmt(item['training time minutes'])} | {fmt(item['inference time seconds'])} |"
        )
    lines.extend(
        [
            "",
            "Frequency-wise NARGP is a computational diagnostic with 500 independent wavelength-wise model pairs; its scale is not directly comparable to latent-space methods. Direct-latent is a representation ablation and is not a main baseline.",
            "",
            "## Validation and provenance",
            "",
        ]
    )
    lines.extend(f"- {check}" for check in checks)
    lines.extend(
        [
            "- Formal sources are restricted to `result_out/final_timing_baseline/` and `result_out/final_timing/`; `final_timing_core`, the old project, and reviewer/dev/temporary results are excluded.",
            "- Setting labels are retained from the frozen data directories. Actual post-split training counts are reported in the table (hf50/hf100/hf500 correspond to 40/80/400 HF training samples).",
            "- Each CSV row records its exact canonical timing source and KPI components.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows, checks = collect_rows()
    write_outputs(rows, checks)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    for check in checks:
        print(f"PASS: {check}")


if __name__ == "__main__":
    main()
