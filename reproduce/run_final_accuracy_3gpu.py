#!/usr/bin/env python3
"""GPU scheduler for the frozen Phase-1 accuracy matrix.

Dry-run and list modes never launch model code. Real execution requires the
explicit --run flag and uses one subprocess per physical GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import queue
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FROZEN_COMMIT = "a78c3c7a9965df4721d8111260cafcd6c8020656"
DEFAULT_GPU_IDS = (0, 1, 2)
SEEDS_TM_AB = (
    200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
    300, 311, 322, 333, 344, 355, 366, 377, 388, 399,
)
SEEDS_MTM = tuple(range(1, 21))
SEEDS_FREQWISE = (200, 210, 220)
HF_BUDGETS = (50, 100, 200, 300, 400, 500)
LF_MULTIPLIERS = (5, 10, 15)
MODERN_SETTINGS = tuple(f"hf{hf}_lfx10" for hf in HF_BUDGETS)
EXPECTED_AXIS_SIZE = 500

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = REPO_ROOT / "result_out" / "final_runs"
SCHEDULER_ROOT = FINAL_ROOT / "_scheduler"
LOG_ROOT = SCHEDULER_ROOT / "logs"
MANIFEST_CSV = SCHEDULER_ROOT / "task_manifest.csv"
MANIFEST_JSON = SCHEDULER_ROOT / "task_manifest.json"
SUMMARY_JSON = SCHEDULER_ROOT / "scheduler_summary.json"

SCIENTIFIC_PATHS = (
    "code/comparison_methods",
    "code/mf_train_baseline",
    "code/nanophotonic_tm",
    "code/nanophotonic_ab",
    "code/microwave_mtm",
    "reproduce/run_comparison_sweep_tm.sh",
    "reproduce/run_comparison_sweep_ab.sh",
    "reproduce/run_direct_latent_tm.sh",
    "reproduce/run_freqwise_nargp_tm.sh",
)


@dataclass
class Task:
    job_id: str
    phase: str
    dataset: str
    setting: str
    method: str
    seed: int
    data_dir: str
    out_dir: str
    command: list[str]
    kind: str
    freq_start: int | None = None
    freq_end: int | None = None
    status: str = "PENDING"
    depends_on: list[str] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2) + "\n")


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def require_path(path: Path, *, directory: bool | None = None) -> Path:
    resolved = path.resolve()
    if not is_within(resolved, REPO_ROOT):
        raise ValueError(f"Path escapes repository boundary: {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    if directory is True and not resolved.is_dir():
        raise NotADirectoryError(resolved)
    if directory is False and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def validate_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if not is_within(resolved, FINAL_ROOT):
        raise ValueError(f"Formal output escapes {FINAL_ROOT}: {resolved}")
    if is_within(resolved, SCHEDULER_ROOT):
        raise ValueError(f"Model output may not use scheduler namespace: {resolved}")
    return resolved


def verify_frozen_state() -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    if head != FROZEN_COMMIT:
        raise RuntimeError(f"HEAD {head} != frozen commit {FROZEN_COMMIT}")
    for cached in (False, True):
        cmd = ["git", "diff", "--quiet"]
        if cached:
            cmd.append("--cached")
        cmd.extend([FROZEN_COMMIT, "--", *SCIENTIFIC_PATHS])
        result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            area = "staged" if cached else "working-tree"
            raise RuntimeError(f"Frozen scientific paths have {area} changes")
    status = subprocess.run(
        [
            "git", "status", "--porcelain", "--untracked-files=all", "--",
            *SCIENTIFIC_PATHS,
        ],
        cwd=REPO_ROOT, check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"Frozen scientific paths contain untracked changes:\n{status}")


def dataset_is_valid(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "wavelengths.npy").is_file()
        and all((path / name).is_dir() for name in ("hf", "lf_paired", "lf_unpaired"))
    )


def discover_settings(root: Path) -> tuple[str, ...]:
    import re

    found = []
    for child in root.iterdir():
        if dataset_is_valid(child) and re.fullmatch(r"hf\d+_lfx\d+", child.name):
            found.append(child.name)
    expected = {
        f"hf{hf}_lfx{mult:02d}" if mult == 5 else f"hf{hf}_lfx{mult}"
        for hf in HF_BUDGETS for mult in LF_MULTIPLIERS
    }
    if set(found) != expected or len(found) != 18:
        raise ValueError(
            f"Dataset setting mismatch under {root}: found={sorted(found)}, "
            f"expected={sorted(expected)}"
        )

    def key(name: str) -> tuple[int, int]:
        match = re.fullmatch(r"hf(\d+)_lfx(\d+)", name)
        assert match is not None
        return int(match.group(1)), int(match.group(2))

    return tuple(sorted(found, key=key))


def python_command() -> str:
    return sys.executable


def parse_gpu_ids(value: str) -> list[int]:
    parts = value.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError("GPU_LIST must be comma-separated non-negative integers")
    try:
        gpu_ids = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU_LIST must be comma-separated non-negative integers"
        ) from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError("GPU IDs must be non-negative")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise argparse.ArgumentTypeError("GPU IDs must not contain duplicates")
    return gpu_ids


def original_command(dataset: str, data_dir: Path, out_dir: Path, seed: int) -> list[str]:
    prefix = "bl0" if dataset == "TM" else "ba0"
    return [
        python_command(), str(REPO_ROOT / "code/mf_train_baseline/mf_baseline.py"),
        "--data_dir", str(data_dir), "--out_dir", str(out_dir),
        "--seed", str(seed), "--run_prefix", prefix,
        "--ours_train_script", str(REPO_ROOT / "code/mf_train_baseline/mf_train.py"),
        "--methods", "hf_only,ar1,ours", "--device", "cuda",
        "--wl_low", "380.0", "--wl_high", "750.0",
        "--fpca_var_ratio", "0.999", "--svgp_M", "64",
        "--svgp_steps", "2000", "--gp_ard", "1",
    ]


def mtm_command(data_dir: Path, out_dir: Path, seed: int) -> list[str]:
    return [
        python_command(), str(REPO_ROOT / "code/microwave_mtm/mf_baseline_microwave_mtm.py"),
        "--data_dir", str(data_dir), "--out_dir", str(out_dir),
        "--seed", str(seed), "--device", "cuda",
        "--methods", "hf_only,ar1,ours", "--delegate_ours_to_train", "1",
        "--lf_prob", "0", "--n_plot", "10", "--plot_ci", "1",
        "--save_pred_arrays", "1", "--save_uq_cache", "1",
        "--ci_calibrate", "0",
    ]


def comparison_command(
    dataset: str, method: str, data_dir: Path, out_dir: Path, seed: int,
) -> list[str]:
    wrapper = REPO_ROOT / f"reproduce/run_comparison_sweep_{dataset.lower()}.sh"
    return [
        "bash", str(wrapper), "--run", method, str(data_dir), str(out_dir),
        "--seed", str(seed), "--device", "cuda", "--run_kind", "scientific",
    ]


def direct_command(data_dir: Path, out_dir: Path, seed: int) -> list[str]:
    return [
        "bash", str(REPO_ROOT / "reproduce/run_direct_latent_tm.sh"),
        "--run", str(data_dir), str(out_dir), "--seed", str(seed),
        "--device", "cuda", "--run_kind", "scientific",
    ]


def freqwise_command(
    data_dir: Path, out_dir: Path, seed: int, start: int | None, end: int | None,
    *, aggregate: bool,
) -> list[str]:
    command = [
        "bash", str(REPO_ROOT / "reproduce/run_freqwise_nargp_tm.sh"),
        "--run", str(data_dir), str(out_dir), "--seed", str(seed),
        "--device", "cpu" if aggregate else "cuda",
        "--run_kind", "scientific",
    ]
    if aggregate:
        command.append("--aggregate")
    else:
        assert start is not None and end is not None
        command.extend([
            "--freq_start", str(start), "--freq_end", str(end), "--resume",
        ])
    return command


def build_tasks(chunk_size: int) -> list[Task]:
    import numpy as np

    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    tm_root = require_path(REPO_ROOT / "data/mf_sweep_datasets_nano_tm", directory=True)
    ab_root = require_path(REPO_ROOT / "data/mf_sweep_datasets_nano_ab", directory=True)
    mtm_data = require_path(
        REPO_ROOT / "data/mf_dataset_mw_mtm/hf50_lfx10", directory=True
    )
    if not dataset_is_valid(mtm_data):
        raise ValueError(f"Invalid MTM dataset: {mtm_data}")
    tm_settings = discover_settings(tm_root)
    ab_settings = discover_settings(ab_root)
    for root, settings in ((tm_root, tm_settings), (ab_root, ab_settings)):
        for setting in settings:
            axis = np.load(root / setting / "wavelengths.npy", mmap_mode="r")
            if axis.shape != (EXPECTED_AXIS_SIZE,):
                raise ValueError(f"Unexpected spectral axis for {root / setting}: {axis.shape}")
    tasks: list[Task] = []

    for dataset, root, settings in (("TM", tm_root, tm_settings), ("AB", ab_root, ab_settings)):
        for setting in settings:
            data_dir = require_path(root / setting, directory=True)
            for seed in SEEDS_TM_AB:
                out_dir = validate_output_path(
                    FINAL_ROOT / "original" / dataset.lower() / setting / f"seed_{seed}"
                )
                tasks.append(Task(
                    job_id=f"original_{dataset.lower()}_{setting}_seed_{seed}",
                    phase="phase1_accuracy", dataset=dataset, setting=setting,
                    method="HF-only+AR1/co-kriging+Neural-GP MF", seed=seed,
                    data_dir=str(data_dir), out_dir=str(out_dir), kind="original_bundle",
                    command=original_command(dataset, data_dir, out_dir, seed),
                ))

    for seed in SEEDS_MTM:
        out_dir = validate_output_path(
            FINAL_ROOT / "original/mtm/hf50_lfx10" / f"seed_{seed}"
        )
        tasks.append(Task(
            job_id=f"original_mtm_hf50_lfx10_seed_{seed}",
            phase="phase1_accuracy", dataset="MTM", setting="hf50_lfx10",
            method="HF-only+AR1/co-kriging+Neural-GP MF", seed=seed,
            data_dir=str(mtm_data), out_dir=str(out_dir), kind="original_mtm_bundle",
            command=mtm_command(mtm_data, out_dir, seed),
        ))

    for dataset, root in (("TM", tm_root), ("AB", ab_root)):
        for method, method_dir in (("fpca-nargp", "fpca_nargp"), ("mf-deeponet", "mf_deeponet")):
            for setting in MODERN_SETTINGS:
                data_dir = require_path(root / setting, directory=True)
                for seed in SEEDS_TM_AB:
                    out_dir = validate_output_path(
                        FINAL_ROOT / "comparisons" / dataset.lower() / method_dir
                        / setting / f"seed_{seed}"
                    )
                    tasks.append(Task(
                        job_id=f"comparison_{dataset.lower()}_{method_dir}_{setting}_seed_{seed}",
                        phase="phase1_accuracy", dataset=dataset, setting=setting,
                        method=method, seed=seed, data_dir=str(data_dir),
                        out_dir=str(out_dir), kind="comparison",
                        command=comparison_command(dataset, method, data_dir, out_dir, seed),
                    ))

    direct_data = require_path(tm_root / "hf100_lfx10", directory=True)
    for seed in SEEDS_TM_AB:
        out_dir = validate_output_path(
            FINAL_ROOT / "representation/direct_latent/tm/hf100_lfx10" / f"seed_{seed}"
        )
        tasks.append(Task(
            job_id=f"direct_latent_tm_hf100_lfx10_seed_{seed}",
            phase="phase1_accuracy", dataset="TM", setting="hf100_lfx10",
            method="direct-latent", seed=seed, data_dir=str(direct_data),
            out_dir=str(out_dir), kind="direct_latent",
            command=direct_command(direct_data, out_dir, seed),
        ))

    for seed in SEEDS_FREQWISE:
        out_dir = validate_output_path(
            FINAL_ROOT / "representation/freqwise_nargp/tm/hf100_lfx10" / f"seed_{seed}"
        )
        dependencies = []
        for start in range(0, EXPECTED_AXIS_SIZE, chunk_size):
            end = min(start + chunk_size, EXPECTED_AXIS_SIZE)
            job_id = f"freqwise_tm_hf100_lfx10_seed_{seed}_freq_{start:04d}_{end:04d}"
            dependencies.append(job_id)
            tasks.append(Task(
                job_id=job_id, phase="phase1_accuracy", dataset="TM",
                setting="hf100_lfx10", method="frequency-wise-nargp-chunk",
                seed=seed, freq_start=start, freq_end=end,
                data_dir=str(direct_data), out_dir=str(out_dir), kind="freqwise_chunk",
                command=freqwise_command(direct_data, out_dir, seed, start, end, aggregate=False),
            ))
        tasks.append(Task(
            job_id=f"freqwise_tm_hf100_lfx10_seed_{seed}_aggregate",
            phase="phase1_accuracy", dataset="TM", setting="hf100_lfx10",
            method="frequency-wise-nargp-aggregate", seed=seed,
            data_dir=str(direct_data), out_dir=str(out_dir), kind="freqwise_aggregate",
            depends_on=dependencies,
            command=freqwise_command(direct_data, out_dir, seed, None, None, aggregate=True),
        ))

    validate_task_matrix(tasks, chunk_size)
    return tasks


def validate_task_matrix(tasks: list[Task], chunk_size: int) -> None:
    ids = [task.job_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate job IDs")
    commands = [tuple(task.command) for task in tasks]
    if len(commands) != len(set(commands)):
        raise ValueError("Duplicate commands")

    counts = count_tasks(tasks)
    expected_chunks = math.ceil(EXPECTED_AXIS_SIZE / chunk_size) * len(SEEDS_FREQWISE)
    assertions = {
        "original_tm_drivers": 360,
        "original_ab_drivers": 360,
        "original_mtm_drivers": 20,
        "modern_tm_jobs": 240,
        "modern_ab_jobs": 240,
        "direct_latent_jobs": 20,
        "freqwise_scientific_seeds": 3,
        "freqwise_chunk_tasks": expected_chunks,
        "freqwise_aggregate_tasks": 3,
    }
    for key, expected in assertions.items():
        if counts[key] != expected:
            raise AssertionError(f"{key}: {counts[key]} != {expected}")
    if chunk_size == 25 and counts["freqwise_chunk_tasks"] != 60:
        raise AssertionError("Default chunk size must create exactly 60 chunk tasks")

    original_settings = {
        "TM": discover_settings(REPO_ROOT / "data/mf_sweep_datasets_nano_tm"),
        "AB": discover_settings(REPO_ROOT / "data/mf_sweep_datasets_nano_ab"),
    }
    for dataset in ("TM", "AB"):
        actual = {
            (task.setting, task.seed) for task in tasks
            if task.kind == "original_bundle" and task.dataset == dataset
        }
        expected = {(setting, seed) for setting in original_settings[dataset] for seed in SEEDS_TM_AB}
        if actual != expected:
            raise AssertionError(f"{dataset} original setting/seed matrix mismatch")
        for method in ("fpca-nargp", "mf-deeponet"):
            actual = {
                (task.setting, task.seed) for task in tasks
                if task.kind == "comparison" and task.dataset == dataset and task.method == method
            }
            expected = {(setting, seed) for setting in MODERN_SETTINGS for seed in SEEDS_TM_AB}
            if actual != expected:
                raise AssertionError(f"{dataset} {method} setting/seed matrix mismatch")
    if {task.seed for task in tasks if task.kind == "original_mtm_bundle"} != set(SEEDS_MTM):
        raise AssertionError("MTM original seed matrix mismatch")
    if {task.seed for task in tasks if task.kind == "direct_latent"} != set(SEEDS_TM_AB):
        raise AssertionError("Direct-latent seed matrix mismatch")
    if {task.seed for task in tasks if task.kind == "freqwise_aggregate"} != set(SEEDS_FREQWISE):
        raise AssertionError("Frequency-wise seed matrix mismatch")
    for seed in SEEDS_FREQWISE:
        chunks = sorted(
            (task.freq_start, task.freq_end) for task in tasks
            if task.kind == "freqwise_chunk" and task.seed == seed
        )
        expected = [
            (start, min(start + chunk_size, EXPECTED_AXIS_SIZE))
            for start in range(0, EXPECTED_AXIS_SIZE, chunk_size)
        ]
        if chunks != expected or chunks[0][0] != 0 or chunks[-1][1] != EXPECTED_AXIS_SIZE:
            raise AssertionError(f"Frequency-wise chunk coverage mismatch for seed {seed}")

    ordinary = [task for task in tasks if not task.kind.startswith("freqwise_")]
    ordinary_out = [task.out_dir for task in ordinary]
    if len(ordinary_out) != len(set(ordinary_out)):
        raise ValueError("Non-Frequency-wise scientific output directories are not unique")
    freq_roots = {
        task.seed: task.out_dir for task in tasks if task.kind == "freqwise_aggregate"
    }
    if len(freq_roots) != 3 or len(set(freq_roots.values())) != 3:
        raise ValueError("Frequency-wise seed output directories are not unique")
    for task in tasks:
        if task.kind.startswith("freqwise_") and task.out_dir != freq_roots[task.seed]:
            raise ValueError(f"Frequency-wise chunk output mismatch: {task.job_id}")
        validate_output_path(Path(task.out_dir))
        require_path(Path(task.data_dir), directory=True)
    if any(
        task.dataset == "MTM" and task.kind in {"comparison", "direct_latent", "freqwise_chunk"}
        for task in tasks
    ):
        raise AssertionError("Additional NOT_IN_REVISION_SCOPE MTM baseline was scheduled")


def count_tasks(tasks: list[Task]) -> dict[str, int]:
    return {
        "original_tm_drivers": sum(t.kind == "original_bundle" and t.dataset == "TM" for t in tasks),
        "original_ab_drivers": sum(t.kind == "original_bundle" and t.dataset == "AB" for t in tasks),
        "original_mtm_drivers": sum(t.kind == "original_mtm_bundle" for t in tasks),
        "modern_tm_jobs": sum(t.kind == "comparison" and t.dataset == "TM" for t in tasks),
        "modern_ab_jobs": sum(t.kind == "comparison" and t.dataset == "AB" for t in tasks),
        "direct_latent_jobs": sum(t.kind == "direct_latent" for t in tasks),
        "freqwise_scientific_seeds": len({t.seed for t in tasks if t.kind == "freqwise_aggregate"}),
        "freqwise_chunk_tasks": sum(t.kind == "freqwise_chunk" for t in tasks),
        "freqwise_aggregate_tasks": sum(t.kind == "freqwise_aggregate" for t in tasks),
    }


def select_group(tasks: list[Task], group: str) -> list[Task]:
    if group == "all":
        return tasks
    if group == "original":
        return [
            task for task in tasks
            if (
                (task.kind == "original_bundle" and task.dataset in {"TM", "AB"})
                or task.kind == "original_mtm_bundle"
            )
        ]
    if group == "revision_new":
        return [
            task for task in tasks
            if task.kind in {
                "comparison", "direct_latent", "freqwise_chunk", "freqwise_aggregate",
            }
        ]
    raise ValueError(f"Unknown task group: {group}")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def valid_original_completion(task: Task) -> tuple[bool, str]:
    import numpy as np

    root = Path(task.out_dir)
    if not root.is_dir():
        return False, "output directory absent"
    valid_reports = []
    for path in root.rglob("report.json"):
        report = load_json(path)
        if not report or "hf_only" not in str(report.get("baseline", "")):
            continue
        if int(report.get("seed", -1)) != task.seed:
            continue
        try:
            if Path(str(report.get("data_dir", ""))).resolve() != Path(task.data_dir).resolve():
                continue
        except OSError:
            continue
        metrics = report.get("metrics", {})
        y_rmse = metrics.get("y_rmse", {})
        r2 = metrics.get("r2", {}).get("test", {})
        expected = ("hf_only", "ar1", "ours")
        if not all(finite_number(y_rmse.get(name)) and finite_number(r2.get(name)) for name in expected):
            continue
        run_dir = path.parent
        cache = run_dir / "cache/uq_cache_v1.npz"
        axes = (run_dir / "wavelengths.npy", run_dir / "idx_wavelength.npy")
        if not cache.is_file() or not all(axis.is_file() for axis in axes):
            continue
        try:
            with np.load(cache, allow_pickle=False) as arrays:
                keys = {
                    "y_test", "y_pred_test__hf_only", "y_pred_test__ar1",
                    "y_pred_test__ours", "std_raw_test__hf_only",
                    "std_raw_test__ar1", "std_raw_test__ours",
                }
                if not keys.issubset(arrays.files):
                    continue
                shape = arrays["y_test"].shape
                if not all(arrays[key].shape == shape for key in keys):
                    continue
                if not all(np.isfinite(arrays[key]).all() for key in keys):
                    continue
        except (OSError, ValueError, KeyError):
            continue
        valid_reports.append(path)
    if len(valid_reports) != 1:
        return False, f"expected one complete three-method report, found {len(valid_reports)}"
    return True, str(valid_reports[0])


def modern_identity(task: Task) -> str:
    return {
        "fpca-nargp": "FPCA-NARGP",
        "mf-deeponet": "MF-DeepONet",
        "direct-latent": "Direct-latent Stage-I ablation",
    }[task.method]


def valid_modern_completion(task: Task) -> tuple[bool, str]:
    import numpy as np

    root = Path(task.out_dir)
    report = load_json(root / "report.json")
    if report is None:
        return False, "missing or invalid report.json"
    expected = {
        "method": modern_identity(task), "dataset": task.dataset,
        "setting": task.setting, "seed": task.seed,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        return False, f"report identity mismatch: expected {expected}"
    if report.get("run_kind") != "scientific" or report.get("scientific_result") is not True:
        return False, "report is not scientific"
    test_metrics = report.get("metrics", {}).get("test", {})
    if not finite_number(test_metrics.get("rmse")) or not finite_number(test_metrics.get("r2")):
        return False, "non-finite or missing test metrics"
    pred = root / "pred_arrays"
    required = ("test_mean.npy", "test_indices.npy", "wavelengths.npy", "idx_wavelength.npy")
    if not all((pred / name).is_file() for name in required):
        return False, "missing prediction artifacts"
    try:
        mean = np.load(pred / "test_mean.npy", mmap_mode="r")
        indices = np.load(pred / "test_indices.npy")
        wavelengths = np.load(pred / "wavelengths.npy")
        idx_wavelength = np.load(pred / "idx_wavelength.npy")
        target = np.load(Path(task.data_dir) / "hf/y_test.npy", mmap_mode="r")
        target_indices = np.load(Path(task.data_dir) / "hf/idx_test.npy")
        stored_wavelengths = np.load(Path(task.data_dir) / "wavelengths.npy")
        stored_idx = np.load(Path(task.data_dir) / "idx_wavelength.npy")
        if mean.shape != target.shape or not np.isfinite(mean).all():
            return False, "invalid test prediction array"
        if not np.array_equal(indices, target_indices):
            return False, "test sample indices mismatch"
        if not np.array_equal(wavelengths, stored_wavelengths) or not np.array_equal(idx_wavelength, stored_idx):
            return False, "spectral axis mismatch"
        if task.method == "fpca-nargp":
            variance = np.load(pred / "test_variance.npy", mmap_mode="r")
            if variance.shape != target.shape or not np.isfinite(variance).all() or np.any(variance < 0):
                return False, "invalid test variance array"
    except (OSError, ValueError, KeyError) as exc:
        return False, f"prediction artifact error: {exc}"
    return True, str(root / "report.json")


def valid_checkpoint(task: Task, coordinate: int) -> tuple[bool, str | None]:
    import numpy as np

    path = Path(task.out_dir) / "wavelength_checkpoints" / f"wavelength_{coordinate:04d}.npz"
    if not path.is_file():
        return False, None
    try:
        with np.load(path, allow_pickle=False) as item:
            required = {
                "signature", "k", "wavelength", "idx_wavelength",
                "val_mean", "val_variance", "test_mean", "test_variance",
            }
            if not required.issubset(item.files) or int(item["k"].item()) != coordinate:
                return False, None
            signature = str(item["signature"].item())
            if not signature:
                return False, None
            wavelengths = np.load(Path(task.data_dir) / "wavelengths.npy", mmap_mode="r")
            idx_wavelength = np.load(Path(task.data_dir) / "idx_wavelength.npy", mmap_mode="r")
            if float(item["wavelength"].item()) != float(wavelengths[coordinate]):
                return False, None
            if int(item["idx_wavelength"].item()) != int(idx_wavelength[coordinate]):
                return False, None
            n_val = np.load(Path(task.data_dir) / "hf/idx_val.npy", mmap_mode="r").shape[0]
            n_test = np.load(Path(task.data_dir) / "hf/idx_test.npy", mmap_mode="r").shape[0]
            for split, size in (("val", n_val), ("test", n_test)):
                mean = item[f"{split}_mean"]
                variance = item[f"{split}_variance"]
                if mean.shape != (size,) or variance.shape != (size,):
                    return False, None
                if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance < 0):
                    return False, None
            return True, signature
    except (OSError, ValueError, KeyError, IndexError):
        return False, None


def valid_freqwise_chunk(task: Task) -> tuple[bool, str]:
    assert task.freq_start is not None and task.freq_end is not None
    signatures = set()
    for coordinate in range(task.freq_start, task.freq_end):
        valid, signature = valid_checkpoint(task, coordinate)
        if not valid or signature is None:
            return False, f"checkpoint {coordinate} missing or invalid"
        signatures.add(signature)
    if len(signatures) != 1:
        return False, "chunk checkpoint signatures disagree"
    return True, f"{task.freq_end - task.freq_start} valid checkpoints"


def valid_freqwise_aggregate(task: Task) -> tuple[bool, str]:
    import numpy as np

    signatures = set()
    for coordinate in range(EXPECTED_AXIS_SIZE):
        valid, signature = valid_checkpoint(task, coordinate)
        if not valid or signature is None:
            return False, f"checkpoint {coordinate} missing or invalid"
        signatures.add(signature)
    if len(signatures) != 1:
        return False, "full-axis checkpoint signatures disagree"
    report = load_json(Path(task.out_dir) / "report.json")
    if report is None:
        return False, "missing aggregate report.json"
    expected = {
        "method": "Frequency-wise NARGP", "dataset": "TM",
        "setting": task.setting, "seed": task.seed,
        "wavelength_count": EXPECTED_AXIS_SIZE,
        "total_wavelength_count": EXPECTED_AXIS_SIZE,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        return False, f"aggregate identity mismatch: expected {expected}"
    if report.get("run_kind") != "scientific":
        return False, "aggregate run_kind is not scientific"
    if report.get("scientific_result") is not True or report.get("complete_spectral_axis") is not True:
        return False, "aggregate does not cover the complete scientific axis"
    if report.get("requested_indices") != list(range(EXPECTED_AXIS_SIZE)):
        return False, "aggregate coordinate order mismatch"
    test = report.get("metrics", {}).get("test", {})
    if not finite_number(test.get("rmse")) or not finite_number(test.get("r2")):
        return False, "aggregate test metrics missing or non-finite"
    nargp = report.get("nargp", {})
    expected_hyper = {"inducing": 64, "steps": 2000, "lr": 0.005, "mc_samples": 512}
    if any(nargp.get(key) != value for key, value in expected_hyper.items()):
        return False, "aggregate NARGP defaults mismatch"
    progress = load_json(Path(task.out_dir) / "progress.json")
    if progress is None or progress.get("completed_global_count") != EXPECTED_AXIS_SIZE:
        return False, "canonical progress is incomplete"
    if progress.get("run_signature") not in signatures:
        return False, "canonical progress signature mismatch"
    pred = Path(task.out_dir) / "pred_arrays"
    try:
        mean = np.load(pred / "test_mean.npy", mmap_mode="r")
        variance = np.load(pred / "test_variance.npy", mmap_mode="r")
        target = np.load(Path(task.data_dir) / "hf/y_test.npy", mmap_mode="r")
        if mean.shape != target.shape or variance.shape != target.shape:
            return False, "aggregate prediction shape mismatch"
        if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance < 0):
            return False, "aggregate predictions are invalid"
    except (OSError, ValueError) as exc:
        return False, f"aggregate prediction artifact error: {exc}"
    return True, str(Path(task.out_dir) / "report.json")


def completion(task: Task) -> tuple[bool, str]:
    if task.kind in {"original_bundle", "original_mtm_bundle"}:
        return valid_original_completion(task)
    if task.kind in {"comparison", "direct_latent"}:
        return valid_modern_completion(task)
    if task.kind == "freqwise_chunk":
        return valid_freqwise_chunk(task)
    if task.kind == "freqwise_aggregate":
        return valid_freqwise_aggregate(task)
    return False, f"unknown task kind: {task.kind}"


def manifest_record(task: Task) -> dict[str, Any]:
    record = asdict(task)
    record["command_argv"] = record.pop("command")
    record["command"] = shlex.join(task.command)
    return record


def write_manifest(tasks: Iterable[Task]) -> None:
    records = [manifest_record(task) for task in tasks]
    atomic_write_json(MANIFEST_JSON, records)
    fields = [
        "job_id", "phase", "dataset", "setting", "method", "seed",
        "freq_start", "freq_end", "data_dir", "out_dir", "command",
        "status", "kind", "depends_on",
    ]
    SCHEDULER_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_CSV.with_name(f".{MANIFEST_CSV.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                row = {key: record.get(key) for key in fields}
                row["depends_on"] = "|".join(record["depends_on"])
                writer.writerow(row)
        os.replace(tmp, MANIFEST_CSV)
    finally:
        if tmp.exists():
            tmp.unlink()


def initialize_statuses(tasks: list[Task]) -> None:
    for task in tasks:
        done, _ = completion(task)
        if done:
            task.status = "COMPLETE"
        elif task.kind == "freqwise_aggregate":
            task.status = "WAITING_FOR_CHUNKS"
        else:
            task.status = "PENDING"


def print_counts(tasks: list[Task], chunk_size: int) -> None:
    counts = count_tasks(tasks)
    original_method_runs = (
        counts["original_tm_drivers"] + counts["original_ab_drivers"]
        + counts["original_mtm_drivers"]
    ) * 3
    new_method_runs = (
        counts["modern_tm_jobs"] + counts["modern_ab_jobs"]
        + counts["direct_latent_jobs"] + counts["freqwise_scientific_seeds"]
    )
    print("Phase-1 accuracy counts")
    print(f"  Original TM baseline driver jobs:  {counts['original_tm_drivers']}")
    print(f"  Original AB baseline driver jobs:  {counts['original_ab_drivers']}")
    print(f"  Original MTM baseline driver jobs: {counts['original_mtm_drivers']}")
    print(f"  Original method-level runs:         {original_method_runs}")
    print(f"  TM modern comparison jobs:          {counts['modern_tm_jobs']}")
    print(f"  AB modern comparison jobs:          {counts['modern_ab_jobs']}")
    print(f"  Direct-latent jobs:                 {counts['direct_latent_jobs']}")
    print(f"  Frequency-wise scientific seeds:    {counts['freqwise_scientific_seeds']}")
    print(f"  Frequency-wise chunk tasks:         {counts['freqwise_chunk_tasks']} (chunk_size={chunk_size})")
    print(f"  Frequency-wise aggregate tasks:     {counts['freqwise_aggregate_tasks']}")
    print(f"  New method-level accuracy runs:      {new_method_runs}")
    print(f"  Total method-level accuracy runs:    {original_method_runs + new_method_runs}")
    print(f"  Scheduler tasks incl. aggregates:    {len(tasks)}")


def write_summary(
    mode: str, tasks: list[Task], attempts: list[dict[str, Any]], gpu_ids: list[int],
    workers_per_gpu: int,
) -> None:
    statuses: dict[str, int] = {}
    for task in tasks:
        statuses[task.status] = statuses.get(task.status, 0) + 1
    atomic_write_json(SUMMARY_JSON, {
        "mode": mode, "frozen_commit": FROZEN_COMMIT,
        "updated_at": utc_now(), "gpu_ids": gpu_ids,
        "workers_per_gpu": workers_per_gpu,
        "total_gpu_worker_slots": len(gpu_ids) * workers_per_gpu,
        "counts": count_tasks(tasks), "manifest_task_count": len(tasks),
        "statuses": statuses, "attempts": attempts,
    })


def worker_main(gpu_id: int, task_queue: Any, result_queue: Any) -> None:
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            return
        start = utc_now()
        started = time.monotonic()
        return_code: int | None = None
        log_path = LOG_ROOT / f"{task.job_id}.attempt_{int(time.time())}_{os.getpid()}.log"
        result: dict[str, Any]
        try:
            already_done, detail = completion(task)
            if already_done:
                result = {
                    "job_id": task.job_id, "status": "SKIPPED_COMPLETE",
                    "return_code": 0, "gpu": gpu_id, "detail": detail,
                }
            else:
                Path(task.out_dir).mkdir(parents=True, exist_ok=True)
                LOG_ROOT.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                with log_path.open("w", encoding="utf-8") as log:
                    log.write(f"start={start}\nphysical_gpu={gpu_id}\ncommand={shlex.join(task.command)}\n\n")
                    log.flush()
                    process = subprocess.run(
                        task.command, cwd=REPO_ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT, check=False,
                    )
                    return_code = int(process.returncode)
                done, detail = completion(task)
                if return_code == 0 and done:
                    status = "SUCCEEDED"
                elif return_code != 0:
                    status = "FAILED_PROCESS"
                else:
                    status = "FAILED_COMPLETION"
                result = {
                    "job_id": task.job_id, "status": status,
                    "return_code": return_code, "gpu": gpu_id,
                    "detail": detail, "log_path": str(log_path),
                }
        except BaseException as exc:
            result = {
                "job_id": task.job_id, "status": "FAILED_EXCEPTION",
                "return_code": return_code, "gpu": gpu_id,
                "detail": f"{type(exc).__name__}: {exc}",
                "log_path": str(log_path),
            }
        result["start_time"] = start
        result["end_time"] = utc_now()
        result["duration_seconds"] = time.monotonic() - started
        result_queue.put(result)
        task_queue.task_done()


SUCCESS_STATUSES = {"COMPLETE", "SUCCEEDED", "SKIPPED_COMPLETE"}
TERMINAL_STATUSES = SUCCESS_STATUSES | {
    "FAILED_PROCESS", "FAILED_COMPLETION", "FAILED_EXCEPTION", "BLOCKED_DEPENDENCY",
}


def run_scheduler(tasks: list[Task], gpu_ids: list[int], workers_per_gpu: int) -> int:
    by_id = {task.job_id: task for task in tasks}
    ctx = mp.get_context("spawn")
    task_queue = ctx.JoinableQueue()
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(
            target=worker_main, args=(gpu, task_queue, result_queue),
            name=f"gpu-{gpu}-slot-{slot}",
        )
        for gpu in gpu_ids for slot in range(workers_per_gpu)
    ]
    for worker in workers:
        worker.start()

    attempts: list[dict[str, Any]] = []
    outstanding = 0
    for task in tasks:
        if task.kind != "freqwise_aggregate" and task.status == "PENDING":
            task.status = "QUEUED"
            task_queue.put(task)
            outstanding += 1

    def update_aggregates() -> None:
        nonlocal outstanding
        for aggregate in (task for task in tasks if task.kind == "freqwise_aggregate"):
            if aggregate.status != "WAITING_FOR_CHUNKS":
                continue
            dependency_statuses = [by_id[job_id].status for job_id in aggregate.depends_on]
            if all(status in SUCCESS_STATUSES for status in dependency_statuses):
                aggregate.status = "QUEUED"
                task_queue.put(aggregate)
                outstanding += 1
            elif all(status in TERMINAL_STATUSES for status in dependency_statuses):
                aggregate.status = "BLOCKED_DEPENDENCY"
                attempts.append({
                    "job_id": aggregate.job_id, "status": aggregate.status,
                    "return_code": None, "gpu": None, "start_time": None,
                    "end_time": utc_now(), "duration_seconds": 0.0,
                    "detail": "one or more Frequency-wise chunks failed",
                })

    update_aggregates()
    write_manifest(tasks)
    write_summary("run", tasks, attempts, gpu_ids, workers_per_gpu)

    def stop_workers(*, terminate: bool) -> None:
        if terminate:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
        else:
            for _ in workers:
                task_queue.put(None)
        for worker in workers:
            worker.join()
    try:
        while outstanding:
            try:
                result = result_queue.get(timeout=5)
            except queue.Empty:
                dead = [worker.name for worker in workers if not worker.is_alive()]
                if dead:
                    raise RuntimeError(f"GPU workers exited unexpectedly: {dead}")
                continue
            outstanding -= 1
            task = by_id[result["job_id"]]
            task.status = result["status"]
            attempts.append(result)
            print(
                f"[{result['status']}] {task.job_id} gpu={result['gpu']} "
                f"rc={result['return_code']} detail={result['detail']}",
                flush=True,
            )
            update_aggregates()
            write_manifest(tasks)
            write_summary("run", tasks, attempts, gpu_ids, workers_per_gpu)
    except KeyboardInterrupt:
        stop_workers(terminate=True)
        write_manifest(tasks)
        write_summary("interrupted", tasks, attempts, gpu_ids, workers_per_gpu)
        return 130
    except BaseException as exc:
        stop_workers(terminate=True)
        attempts.append({
            "job_id": "_scheduler", "status": "FAILED_SCHEDULER",
            "return_code": None, "gpu": None, "start_time": None,
            "end_time": utc_now(), "duration_seconds": 0.0,
            "detail": f"{type(exc).__name__}: {exc}",
        })
        write_manifest(tasks)
        write_summary("scheduler_error", tasks, attempts, gpu_ids, workers_per_gpu)
        print(f"Scheduler aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    stop_workers(terminate=False)

    write_manifest(tasks)
    write_summary("run_complete", tasks, attempts, gpu_ids, workers_per_gpu)
    unresolved = [task for task in tasks if task.status not in SUCCESS_STATUSES]
    if unresolved:
        print(f"Unresolved failed/incomplete jobs: {len(unresolved)}", file=sys.stderr)
        return 1
    return 0


def list_jobs(tasks: list[Task], limit: int) -> None:
    grouped: dict[str, int] = {}
    for task in tasks:
        grouped[task.kind] = grouped.get(task.kind, 0) + 1
    print("\nTask kinds:")
    for kind in sorted(grouped):
        print(f"  {kind}: {grouped[kind]}")
    selected = tasks if limit == 0 else tasks[:limit]
    print(f"\nJobs shown: {len(selected)}/{len(tasks)}")
    for task in selected:
        span = "" if task.freq_start is None else f" [{task.freq_start},{task.freq_end})"
        print(f"  {task.job_id} | {task.status}{span}")
    if len(selected) < len(tasks):
        print(f"  ... {len(tasks) - len(selected)} more; use --list-limit 0 for all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formal multi-GPU Phase-1 accuracy scheduler (frozen at a78c3c7)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="validate and write manifests; launch nothing")
    mode.add_argument("--list-jobs", action="store_true", help="validate and list manifest jobs; launch nothing")
    mode.add_argument("--run", action="store_true", help="execute the formal queue on selected GPUs")
    parser.add_argument("--chunk-size", type=int, default=25, help="Frequency-wise coordinates per GPU task")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="concurrent worker processes per physical GPU")
    parser.add_argument("--list-limit", type=int, default=40, help="jobs printed by --list-jobs; 0 means all")
    parser.add_argument(
        "--group", choices=("all", "original", "revision_new"), default="all",
        help="task group to dispatch (default: all)",
    )
    parser.add_argument(
        "--gpus", type=parse_gpu_ids, default=list(DEFAULT_GPU_IDS), metavar="GPU_LIST",
        help="comma-separated visible GPU IDs (default: 0,1,2)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_limit < 0:
        raise SystemExit("--list-limit must be non-negative")
    if args.workers_per_gpu <= 0:
        raise SystemExit("--workers-per-gpu must be positive")
    verify_frozen_state()
    tasks = select_group(build_tasks(args.chunk_size), args.group)
    initialize_statuses(tasks)
    write_manifest(tasks)
    print_counts(tasks, args.chunk_size)
    print(f"  gpu_ids:                        {args.gpus}")
    print(f"  workers_per_gpu:                 {args.workers_per_gpu}")
    print(f"  total GPU worker slots:          {len(args.gpus) * args.workers_per_gpu}")
    if args.list_jobs:
        list_jobs(tasks, args.list_limit)
        write_summary("list-jobs", tasks, [], args.gpus, args.workers_per_gpu)
        return 0
    if args.dry_run:
        write_summary("dry-run", tasks, [], args.gpus, args.workers_per_gpu)
        print(f"Manifest CSV:  {MANIFEST_CSV}")
        print(f"Manifest JSON: {MANIFEST_JSON}")
        print("Dry-run launched no training or GPU process.")
        return 0
    return run_scheduler(tasks, args.gpus, args.workers_per_gpu)


if __name__ == "__main__":
    raise SystemExit(main())
