#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_int_list(s: str) -> List[int]:
    vals: List[int] = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    if not vals:
        raise ValueError("Parsed empty integer list.")
    return vals


def safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        xs = x.strip()
        if xs.lower() in {"nan", "none", ""}:
            return float("nan")
        try:
            return float(xs)
        except Exception:
            return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def is_nan(x: Any) -> bool:
    try:
        return math.isnan(safe_float(x))
    except Exception:
        return True


def nested_get(d: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def try_read_report_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        return None

    try:
        rep = read_json(report_path)
    except Exception:
        return None

    # 你的 report.json 里大多数主指标在 metrics 下面
    metrics_blk = rep.get("metrics", {}) or {}
    fpca_diag = metrics_blk.get("fpca_diag", {}) or {}

    # uncertainty 一般仍在顶层
    cov_cal = nested_get(rep, ["uncertainty", "test", "coverage_cal"])
    width_cal = nested_get(rep, ["uncertainty", "test", "width_cal"])

    if isinstance(cov_cal, dict):
        cov_cal = cov_cal.get("mf_student")
    if isinstance(width_cal, dict):
        width_cal = width_cal.get("mf_student")

    out = {
        "fpca_dim_effective": safe_float(fpca_diag.get("fpca_dim_effective")),
        "evr_sum": safe_float(fpca_diag.get("fpca_evr_sum")),
        "recon_rmse_hfval": safe_float(fpca_diag.get("fpca_recon_rmse_hfval")),
        "y_rmse_test": safe_float(nested_get(metrics_blk, ["y_rmse", "mf_student"])),
        "nll_test_cal": safe_float(nested_get(metrics_blk, ["nll", "test", "cal", "mf_student"])),
        "coverage_test_cal": safe_float(cov_cal),
        "ci_width_test_cal": safe_float(width_cal),
    }

    # 至少有一项不是 NaN 才认为这个 report 可用
    if all(is_nan(v) for v in out.values()):
        return None
    return out


def try_read_results_csv_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None

    try:
        with results_csv.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None

    if not rows:
        return None

    row0 = rows[0]

    def pick(*keys: str) -> float:
        for k in keys:
            if k in row0:
                return safe_float(row0[k])
        return float("nan")

    out = {
        "fpca_dim_effective": pick("fpca_dim_effective", "fpca_dim", "latent_dim"),
        "evr_sum": pick("fpca_evr_sum", "evr_sum"),
        "recon_rmse_hfval": pick("fpca_recon_rmse_hfval", "recon_rmse_hfval"),
        "y_rmse_test": pick("y_rmse_mf_student", "y_rmse_test", "y_rmse"),
        "nll_test_cal": pick("nll_test_cal_mf_student", "nll_test_cal", "nll_cal_test"),
        "coverage_test_cal": pick("coverage_test_cal_mf_student", "coverage_test_cal"),
        "ci_width_test_cal": pick("ci_width_test_cal_mf_student", "ci_width_test_cal"),
    }

    if all(is_nan(v) for v in out.values()):
        return None
    return out


def merge_metrics(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(primary)
    for k, v in secondary.items():
        if k not in out or is_nan(out[k]):
            out[k] = v
    return out


def load_metrics_from_run_dir(run_dir: Path) -> Dict[str, Any]:
    m_report = try_read_report_metrics(run_dir)
    m_csv = try_read_results_csv_metrics(run_dir)

    if m_report is None and m_csv is None:
        raise FileNotFoundError(
            f"Neither report.json nor parseable results.csv found under: {run_dir}"
        )
    if m_report is None:
        return m_csv
    if m_csv is None:
        return m_report
    return merge_metrics(m_report, m_csv)


def is_run_dir_parseable(run_dir: Path) -> bool:
    try:
        _ = load_metrics_from_run_dir(run_dir)
        return True
    except Exception:
        return False


def find_latest_leaf_with_valid_metrics(root: Path) -> Optional[Path]:
    candidates: List[Tuple[float, Path]] = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        has_report = (p / "report.json").exists()
        has_results = (p / "results.csv").exists()
        if not (has_report or has_results):
            continue
        if not is_run_dir_parseable(p):
            continue
        try:
            mt = max(
                ((p / "report.json").stat().st_mtime if has_report else 0.0),
                ((p / "results.csv").stat().st_mtime if has_results else 0.0),
            )
        except FileNotFoundError:
            mt = p.stat().st_mtime
        candidates.append((mt, p))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def build_cmd(args: argparse.Namespace, dim: int, seed: int, run_root: Path) -> List[str]:
    cmd = [
        sys.executable if args.python_bin.strip() == "" else args.python_bin,
        str(Path(args.train_script).resolve()),
        "--data_dir", str(Path(args.data_dir).resolve()),
        "--out_dir", str(run_root.resolve()),
        "--reducer_method", "fpca",
        "--exp_name", f"fpca_dim{dim:03d}_seed{seed}",
        "--fpca_dim", str(dim),
        "--seed", str(seed),
    ]

    if args.fpca_var_ratio is not None:
        cmd += ["--fpca_var_ratio", str(args.fpca_var_ratio)]
    if args.fpca_max_dim is not None:
        cmd += ["--fpca_max_dim", str(args.fpca_max_dim)]
    if args.fpca_ridge is not None:
        cmd += ["--fpca_ridge", str(args.fpca_ridge)]

    if int(args.disable_best_config):
        cmd += ["--use_best_reducer_config", "0"]
    else:
        cmd += ["--use_best_reducer_config", "1"]
        if args.best_reducer_config_csv:
            cmd += ["--best_reducer_config_csv", args.best_reducer_config_csv]
        if args.best_reducer_task_name:
            cmd += ["--best_reducer_task_name", args.best_reducer_task_name]

    if args.extra_args.strip():
        cmd += args.extra_args.strip().split()

    return cmd


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate_seed_means(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict

    num_cols = [
        "fpca_dim_effective",
        "evr_sum",
        "recon_rmse_hfval",
        "y_rmse_test",
        "nll_test_cal",
        "coverage_test_cal",
        "ci_width_test_cal",
    ]
    groups = defaultdict(list)
    for r in rows:
        status = str(r.get("status", ""))
        if status == "ok" or status.startswith("skipped_existing"):
            groups[int(r["dim"])].append(r)

    out: List[Dict[str, Any]] = []
    for dim in sorted(groups.keys()):
        grp = groups[dim]
        rec: Dict[str, Any] = {"dim": dim, "n_seeds": len(grp)}
        for c in num_cols:
            vals = [safe_float(x.get(c)) for x in grp]
            vals = [v for v in vals if not math.isnan(v)]
            if not vals:
                rec[f"{c}_mean"] = float("nan")
                rec[f"{c}_std"] = float("nan")
            else:
                mean = sum(vals) / len(vals)
                if len(vals) == 1:
                    std = 0.0
                else:
                    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                    std = math.sqrt(max(var, 0.0))
                rec[f"{c}_mean"] = mean
                rec[f"{c}_std"] = std
        out.append(rec)
    return out


def nan_record(seed: int, dim: int, run_dir: Path, status: str) -> Dict[str, Any]:
    return {
        "seed": seed,
        "dim": dim,
        "fpca_dim_effective": float("nan"),
        "evr_sum": float("nan"),
        "recon_rmse_hfval": float("nan"),
        "y_rmse_test": float("nan"),
        "nll_test_cal": float("nan"),
        "coverage_test_cal": float("nan"),
        "ci_width_test_cal": float("nan"),
        "run_dir": str(run_dir),
        "status": status,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="FPCA dim sweep runner with seed support for nanophotonic TM.")
    ap.add_argument("--train_script", type=str, default="./mf_train_nano_tm_dim_sweep.py")
    ap.add_argument("--data_dir", type=str, default="../../data/mf_sweep_datasets_nano_tm/hf100_lfx10")
    ap.add_argument("--out_root", type=str, default="../../result_out/fpca_dim_sweep_tm_outputs")
    ap.add_argument("--dims", type=str, default="2,4,6,8,10,12,16,24,32,64")
    ap.add_argument("--seeds", type=str, default="42,55,66,77,88,99,111,222,333,555")

    ap.add_argument("--python_bin", type=str, default="")

    ap.add_argument("--fpca_var_ratio", type=float, default=None)
    ap.add_argument("--fpca_max_dim", type=int, default=None)
    ap.add_argument("--fpca_ridge", type=float, default=None)

    ap.add_argument("--disable_best_config", type=int, default=1)
    ap.add_argument("--best_reducer_config_csv", type=str, default="")
    ap.add_argument("--best_reducer_task_name", type=str, default="")

    ap.add_argument("--extra_args", type=str, default="")
    ap.add_argument("--skip_existing", type=int, default=1)
    ap.add_argument("--clean_failed_run_root", type=int, default=0)
    args = ap.parse_args()

    dims = parse_int_list(args.dims)
    seeds = parse_int_list(args.seeds)

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "train_script": str(Path(args.train_script).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "out_root": str(out_root),
        "dims": dims,
        "seeds": seeds,
        "disable_best_config": int(args.disable_best_config),
        "best_reducer_config_csv": args.best_reducer_config_csv,
        "best_reducer_task_name": args.best_reducer_task_name,
        "extra_args": args.extra_args,
    }
    with (out_root / "fpca_dim_sweep_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    all_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        for dim in dims:
            run_root = out_root / f"seed_{seed:03d}" / f"fpca_dim_{dim:03d}"
            run_root.mkdir(parents=True, exist_ok=True)

            if int(args.skip_existing):
                existing = find_latest_leaf_with_valid_metrics(run_root)
                if existing is not None:
                    try:
                        metrics = load_metrics_from_run_dir(existing)
                        all_rows.append({
                            "seed": seed,
                            "dim": dim,
                            **metrics,
                            "run_dir": str(existing),
                            "status": "skipped_existing",
                        })
                        print(
                            f"[SKIP] seed={seed} dim={dim} | "
                            f"existing={existing} | "
                            f"eff_dim={metrics.get('fpca_dim_effective')} | "
                            f"y_rmse={metrics.get('y_rmse_test')}"
                        )
                        continue
                    except Exception as e:
                        print(f"[STALE] seed={seed} dim={dim} | bad existing result: {existing} | err={e}")

            cmd = build_cmd(args, dim=dim, seed=seed, run_root=run_root)
            print("[RUN]", " ".join(cmd))

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[FAIL] seed={seed} dim={dim} | rc={e.returncode}")
                if int(args.clean_failed_run_root):
                    shutil.rmtree(run_root, ignore_errors=True)
                all_rows.append(nan_record(seed, dim, run_root, f"failed_rc_{e.returncode}"))
                continue

            leaf = find_latest_leaf_with_valid_metrics(run_root)
            if leaf is None:
                all_rows.append(nan_record(seed, dim, run_root, "missing_metrics"))
                print(f"[WARN] seed={seed} dim={dim} | missing parseable metrics under {run_root}")
                continue

            try:
                metrics = load_metrics_from_run_dir(leaf)
            except Exception as e:
                all_rows.append(nan_record(seed, dim, leaf, "unparseable_metrics"))
                print(f"[WARN] seed={seed} dim={dim} | parse failed under {leaf} | err={e}")
                continue

            all_rows.append({
                "seed": seed,
                "dim": dim,
                **metrics,
                "run_dir": str(leaf),
                "status": "ok",
            })
            print(
                f"[DONE] seed={seed} dim={dim} | "
                f"eff_dim={metrics.get('fpca_dim_effective')} | "
                f"evr_sum={metrics.get('evr_sum')} | "
                f"y_rmse={metrics.get('y_rmse_test')} | "
                f"nll_cal={metrics.get('nll_test_cal')} | "
                f"cov_cal={metrics.get('coverage_test_cal')}"
            )

    per_run_csv = out_root / "fpca_dim_sweep_summary.csv"
    fieldnames = [
        "seed",
        "dim",
        "fpca_dim_effective",
        "evr_sum",
        "recon_rmse_hfval",
        "y_rmse_test",
        "nll_test_cal",
        "coverage_test_cal",
        "ci_width_test_cal",
        "run_dir",
        "status",
    ]
    write_csv(per_run_csv, all_rows, fieldnames)

    if len(seeds) > 1:
        agg_rows = aggregate_seed_means(all_rows)
        agg_fields = ["dim", "n_seeds"]
        for base in [
            "fpca_dim_effective",
            "evr_sum",
            "recon_rmse_hfval",
            "y_rmse_test",
            "nll_test_cal",
            "coverage_test_cal",
            "ci_width_test_cal",
        ]:
            agg_fields += [f"{base}_mean", f"{base}_std"]
        write_csv(out_root / "fpca_dim_sweep_summary_seedmean.csv", agg_rows, agg_fields)

    print(f"[SAVE] {per_run_csv}")


if __name__ == "__main__":
    main()