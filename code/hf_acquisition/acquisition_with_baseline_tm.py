#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_retrospective_acquisition_with_baseline_tm.py

Pool-based retrospective HF acquisition on prebuilt nanophotonic MF sweep datasets.

Designed for the user's current workflow:
- datasets were built by build_mf_sweep_datasets_trainNested_valFresh_devTestFixed.py
- each dataset dir has:
    wavelengths.npy
    idx_wavelength.npy
    hf/{x,y,t,idx}_{train,val,test,dev}.npy
    lf_paired/{...}
    lf_unpaired/{...}
- the comparison trainer (external) is expected to compare multiple methods
  (e.g. hf_only / ar1 / ours) and save outputs either under:
    - pred_arrays/test/   (direct prediction arrays), or
    - cache/uq_cache_v1.npz (the current baseline-comparison script format)

Core idea
---------
For each target spectrum (from dev_fixed or test_fixed), and for each acquisition
policy / method state:
  1) Start from the same initial known HF set (train+val from hf50_lfx10 or hf100_lfx10)
  2) Candidate pool = hf500_lfx10 full HF pool minus initial known HF set
  3) At each round, materialize a temporary dataset where:
       - train/val are the current known HF/LF paired sets
       - test is the current candidate pool (HF truth is present only for offline eval)
       - LF-unpaired train/val/test are rebuilt from the maximum unpaired pool by prefix
  4) Run the external comparison trainer once per unique dataset state and reuse its outputs across compatible policies
  5) Read predicted candidate spectra from pred_arrays/test/
  6) Score and select top-b candidates for the current policy
  7) Reveal their true HF spectra from the pool and append them to train

This script does NOT require any optimizer; selection is purely pool-based ranking.

v6 additions
------------
- Switched default comparison trainer from the old nanophotonic_tm-specific mf_baseline_tm.py
  to the shared mf_train_baseline/mf_baseline.py, with path resolution relative to this script.
- True per-method early-stop: once a method reaches the target-specific oracle_pool_best,
  that method is skipped in all later rounds.
- Target-level early-stop: once all enabled *comparison* methods (hf_only / ar1 / ours_mean /
  ours_lcb) have stopped or exhausted their candidate pools, the script finishes the current
  target early and moves on.
- Explicit target selection:
    --target_row_ids      raw row ids inside hf/{test,dev}
    --target_global_idxs  global sample indices to map back to row ids
  These selectors bypass target_start/n_targets sequential slicing and are convenient for
  resuming a failed run from a specific target.
Defaults are chosen for the transmission benchmark and can run after path editing.
No argparse required=True is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPARE_SCRIPT = "../mf_train_baseline/mf_baseline.py"


def resolve_script_relative_path(path_str: str, script_dir: Path = SCRIPT_DIR) -> Path:
    p = Path(str(path_str)).expanduser()
    if not p.is_absolute():
        p = script_dir / p
    return p.resolve()


# -----------------------------------------------------------------------------
# Basic IO helpers
# -----------------------------------------------------------------------------

def load_split(group_dir: Path, split: str) -> Dict[str, np.ndarray]:
    return {
        "x": np.load(group_dir / f"x_{split}.npy"),
        "y": np.load(group_dir / f"y_{split}.npy"),
        "t": np.load(group_dir / f"t_{split}.npy"),
        "idx": np.load(group_dir / f"idx_{split}.npy").astype(np.int64),
    }


def save_split(group_dir: Path, split: str,
               x: np.ndarray, y: np.ndarray, t: np.ndarray, idx: np.ndarray) -> None:
    group_dir.mkdir(parents=True, exist_ok=True)
    np.save(group_dir / f"x_{split}.npy", np.asarray(x, dtype=np.float32))
    np.save(group_dir / f"y_{split}.npy", np.asarray(y, dtype=np.float32))
    np.save(group_dir / f"t_{split}.npy", np.asarray(t, dtype=np.float32))
    np.save(group_dir / f"idx_{split}.npy", np.asarray(idx, dtype=np.int64))


def concat_packs(*packs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for k in ["x", "y", "t", "idx"]:
        out[k] = np.concatenate([p[k] for p in packs], axis=0)
    return out


def build_idx_map(pack: Dict[str, np.ndarray]) -> Dict[int, int]:
    return {int(idx): i for i, idx in enumerate(pack["idx"].tolist())}


def subset_by_idx(pack: Dict[str, np.ndarray], wanted_idx: np.ndarray) -> Dict[str, np.ndarray]:
    wanted_idx = np.asarray(wanted_idx, dtype=np.int64).reshape(-1)
    mp = build_idx_map(pack)
    rows = np.asarray([mp[int(i)] for i in wanted_idx.tolist()], dtype=np.int64)
    return {
        "x": pack["x"][rows],
        "y": pack["y"][rows],
        "t": pack["t"][rows],
        "idx": pack["idx"][rows],
    }


def setdiff_idx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    bs = set(int(x) for x in np.asarray(b, dtype=np.int64).reshape(-1).tolist())
    return np.asarray([int(x) for x in np.asarray(a, dtype=np.int64).reshape(-1).tolist() if int(x) not in bs], dtype=np.int64)


def assert_disjoint(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    sa = set(int(x) for x in np.asarray(a, dtype=np.int64).reshape(-1).tolist())
    sb = set(int(x) for x in np.asarray(b, dtype=np.int64).reshape(-1).tolist())
    inter = sa.intersection(sb)
    if inter:
        raise RuntimeError(f"{name_a} and {name_b} overlap, n={len(inter)}")


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _path_has_segment(path: Path, segment: str, relative_to: Optional[Path] = None) -> bool:
    p = path.relative_to(relative_to) if relative_to is not None else path
    return str(segment) in p.parts


def _ensure_finite(name: str, arr: Optional[np.ndarray]) -> None:
    if arr is None:
        return
    a = np.asarray(arr)
    finite = int(np.isfinite(a).sum()) if a.size > 0 else 0
    total = int(a.size)
    if not np.all(np.isfinite(a)):
        raise RuntimeError(f"[{name}] non-finite values detected: finite={finite} / {total}")


def parse_int_list_arg(s: str) -> List[int]:
    s = str(s or "").strip()
    if not s:
        return []
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


# -----------------------------------------------------------------------------
# Metrics / acquisition
# -----------------------------------------------------------------------------

def rmse_rows(y: np.ndarray, target_y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    target_y = np.asarray(target_y, dtype=np.float64).reshape(1, -1)
    return np.sqrt(np.mean((y - target_y) ** 2, axis=1)).astype(np.float64)


def rms_std_rows(std: np.ndarray) -> np.ndarray:
    std = np.asarray(std, dtype=np.float64)
    return np.sqrt(np.mean(std ** 2, axis=1)).astype(np.float64)


def select_top_b(score: np.ndarray, cand_idx: np.ndarray, b: int) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    cand_idx = np.asarray(cand_idx, dtype=np.int64).reshape(-1)
    if score.shape[0] != cand_idx.shape[0]:
        raise ValueError(f"score/candidate size mismatch: {score.shape[0]} vs {cand_idx.shape[0]}")
    if cand_idx.shape[0] < b:
        raise RuntimeError(f"candidate pool too small: {cand_idx.shape[0]} < batch_size={b}")
    if not np.all(np.isfinite(score)):
        finite = int(np.isfinite(score).sum())
        raise ValueError(f"select_top_b received non-finite scores: finite={finite} / {score.size}")
    order = np.lexsort((cand_idx, score))
    return cand_idx[order[:b]].astype(np.int64)


def best_true_target_rmse(full_hf_pack: Dict[str, np.ndarray], known_idx: np.ndarray, target_y: np.ndarray) -> float:
    cur = subset_by_idx(full_hf_pack, known_idx)
    return float(np.min(rmse_rows(cur["y"], target_y)))


def scan_target_headroom(*,
                         target_pack: Dict[str, np.ndarray],
                         init_known_pack: Dict[str, np.ndarray],
                         cand_pack: Dict[str, np.ndarray]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row_id in range(int(target_pack["idx"].shape[0])):
        target_idx = int(target_pack["idx"][row_id])
        target_y = target_pack["y"][row_id]

        initial_rmse = rmse_rows(init_known_pack["y"], target_y)
        initial_pos = int(np.argmin(initial_rmse))
        initial_best = float(initial_rmse[initial_pos])
        initial_best_idx = int(init_known_pack["idx"][initial_pos])

        oracle_rmse = rmse_rows(cand_pack["y"], target_y)
        oracle_pos = int(np.argmin(oracle_rmse))
        oracle_pool_best = float(oracle_rmse[oracle_pos])
        oracle_best_idx = int(cand_pack["idx"][oracle_pos])

        headroom = float(initial_best - oracle_pool_best)
        rows.append({
            "target_row_id": int(row_id),
            "target_global_idx": int(target_idx),
            "initial_best": initial_best,
            "initial_best_idx": initial_best_idx,
            "oracle_pool_best": oracle_pool_best,
            "oracle_best_idx": oracle_best_idx,
            "headroom": headroom,
            "is_saturated": int(headroom <= 0.0),
        })
    return rows


# -----------------------------------------------------------------------------
# Dataset family loaders
# -----------------------------------------------------------------------------

def load_family(dataset_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    fam = {
        "hf_train": load_split(dataset_dir / "hf", "train"),
        "hf_val": load_split(dataset_dir / "hf", "val"),
        "hf_test": load_split(dataset_dir / "hf", "test"),
        "hf_dev": load_split(dataset_dir / "hf", "dev"),
        "lfp_train": load_split(dataset_dir / "lf_paired", "train"),
        "lfp_val": load_split(dataset_dir / "lf_paired", "val"),
        "lfp_test": load_split(dataset_dir / "lf_paired", "test"),
        "lfp_dev": load_split(dataset_dir / "lf_paired", "dev"),
        "lfu_train": load_split(dataset_dir / "lf_unpaired", "train"),
        "lfu_val": load_split(dataset_dir / "lf_unpaired", "val"),
        "lfu_test": load_split(dataset_dir / "lf_unpaired", "test"),
    }
    return fam


def parse_lf_multiplier_from_name(name: str) -> int:
    # e.g. hf50_lfx10 -> 10 ; hf100_lfx05 -> 5
    s = str(name)
    pos = s.rfind("lfx")
    if pos < 0:
        raise ValueError(f"Cannot parse LF multiplier from subdir name: {name}")
    return int(s[pos + 3:])


# -----------------------------------------------------------------------------
# External trainer orchestration
# -----------------------------------------------------------------------------

@lru_cache(maxsize=32)
def script_supports_flag(python_bin: str, script: Path, flag: str) -> bool:
    try:
        proc = subprocess.run(
            [python_bin, str(script), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
            check=False,
        )
        out = proc.stdout or ""
        return flag in out
    except Exception:
        return False


def find_run_artifact(
    root: Path,
    rel_parts: Tuple[str, ...],
    *,
    allow_recursive: bool = True,
    exclude_segments: Optional[Tuple[str, ...]] = None,
) -> Optional[Path]:
    p = root.joinpath(*rel_parts)
    if p.exists():
        return p
    if not allow_recursive:
        return None

    exclude_segments = tuple(str(x) for x in (exclude_segments or tuple()))
    cands = list(root.rglob(rel_parts[-1]))
    if not cands:
        return None

    filtered = []
    for c in cands:
        try:
            rel = c.relative_to(root)
        except Exception:
            rel = c
        if exclude_segments and any(seg in rel.parts for seg in exclude_segments):
            continue
        tail = rel.parts[-len(rel_parts):] if len(rel.parts) >= len(rel_parts) else tuple()
        if tuple(tail) == rel_parts:
            filtered.append(c)

    if not filtered:
        return None

    filtered.sort(key=lambda x: (len(x.relative_to(root).parts), str(x)))
    return filtered[0]


def run_compare_script(
    *,
    round_dataset_dir: Path,
    run_dir: Path,
    compare_script: Path,
    python_bin: str,
    seed: int,
    extra_args: List[str],
    force_save_pred_arrays: bool,
    timeout_sec: int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    cmd = [
        python_bin,
        str(compare_script),
        "--data_dir", str(round_dataset_dir),
        "--out_dir", str(run_dir),
        "--seed", str(int(seed)),
    ]

    if force_save_pred_arrays and ("--save_pred_arrays" not in extra_args):
        cmd += ["--save_pred_arrays", "1"]

    # some scripts may support --no_subdir; use it if available to keep outputs shallow
    if script_supports_flag(python_bin, compare_script, "--no_subdir") and ("--no_subdir" not in extra_args):
        cmd += ["--no_subdir", "1"]

    cmd += list(extra_args)

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("[CMD] " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        logf.flush()
        subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(run_dir),
            timeout=(int(timeout_sec) if int(timeout_sec) > 0 else None),
            check=True,
        )


# -----------------------------------------------------------------------------
# Materialize one round dataset
# -----------------------------------------------------------------------------

def empty_pack(n_feat: int, n_spec: int) -> Dict[str, np.ndarray]:
    return {
        "x": np.empty((0, n_feat), dtype=np.float32),
        "y": np.empty((0, n_spec), dtype=np.float32),
        "t": np.empty((0,), dtype=np.float32),
        "idx": np.empty((0,), dtype=np.int64),
    }


def split_prefix_pack(master: Dict[str, np.ndarray], n_total: int, ratios: Tuple[float, float, float]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    n_total = int(max(0, n_total))
    pref = {
        "x": master["x"][:n_total],
        "y": master["y"][:n_total],
        "t": master["t"][:n_total],
        "idx": master["idx"][:n_total],
    }
    n_feat = int(master["x"].shape[1]) if master["x"].ndim == 2 else 0
    n_spec = int(master["y"].shape[1]) if master["y"].ndim == 2 else 0
    if n_total == 0:
        e = empty_pack(n_feat, n_spec)
        return e, e, e

    r0, r1, r2 = ratios
    s = r0 + r1 + r2
    if s <= 0:
        raise ValueError("lf_unpaired_split ratios must sum to > 0")
    r0, r1, r2 = r0 / s, r1 / s, r2 / s

    n_tr = int(round(n_total * r0))
    n_va = int(round(n_total * r1))
    if n_tr + n_va > n_total:
        n_va = max(0, n_total - n_tr)
    n_te = n_total - n_tr - n_va

    def take(i0: int, i1: int) -> Dict[str, np.ndarray]:
        return {
            "x": pref["x"][i0:i1],
            "y": pref["y"][i0:i1],
            "t": pref["t"][i0:i1],
            "idx": pref["idx"][i0:i1],
        }

    tr = take(0, n_tr)
    va = take(n_tr, n_tr + n_va)
    te = take(n_tr + n_va, n_total)
    return tr, va, te


def materialize_round_dataset(
    *,
    round_dataset_dir: Path,
    wavelengths_src_dir: Path,
    full_hf_pack: Dict[str, np.ndarray],
    full_lfp_pack: Dict[str, np.ndarray],
    lfu_master_pack: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    candidate_idx: np.ndarray,
    lf_multiplier: int,
    lfu_split: Tuple[float, float, float],
) -> None:
    round_dataset_dir.mkdir(parents=True, exist_ok=True)

    # copy wavelength axis files
    for fn in ["wavelengths.npy", "idx_wavelength.npy"]:
        src = wavelengths_src_dir / fn
        dst = round_dataset_dir / fn
        if src.exists():
            arr = np.load(src)
            np.save(dst, arr)

    train_hf = subset_by_idx(full_hf_pack, train_idx)
    val_hf = subset_by_idx(full_hf_pack, val_idx)
    cand_hf = subset_by_idx(full_hf_pack, candidate_idx)

    train_lfp = subset_by_idx(full_lfp_pack, train_idx)
    val_lfp = subset_by_idx(full_lfp_pack, val_idx)
    cand_lfp = subset_by_idx(full_lfp_pack, candidate_idx)

    n_known_hf = int(train_idx.size + val_idx.size)
    n_lfu_total = int(max(0, (int(lf_multiplier) - 1) * n_known_hf))
    if lfu_master_pack["idx"].shape[0] < n_lfu_total:
        raise RuntimeError(f"LF-unpaired master pool too small: have {lfu_master_pack['idx'].shape[0]}, need {n_lfu_total}")
    lfu_tr, lfu_va, lfu_te = split_prefix_pack(lfu_master_pack, n_lfu_total, lfu_split)

    save_split(round_dataset_dir / "hf", "train", train_hf["x"], train_hf["y"], train_hf["t"], train_hf["idx"])
    save_split(round_dataset_dir / "hf", "val", val_hf["x"], val_hf["y"], val_hf["t"], val_hf["idx"])
    save_split(round_dataset_dir / "hf", "test", cand_hf["x"], cand_hf["y"], cand_hf["t"], cand_hf["idx"])

    save_split(round_dataset_dir / "lf_paired", "train", train_lfp["x"], train_lfp["y"], train_lfp["t"], train_lfp["idx"])
    save_split(round_dataset_dir / "lf_paired", "val", val_lfp["x"], val_lfp["y"], val_lfp["t"], val_lfp["idx"])
    save_split(round_dataset_dir / "lf_paired", "test", cand_lfp["x"], cand_lfp["y"], cand_lfp["t"], cand_lfp["idx"])

    save_split(round_dataset_dir / "lf_unpaired", "train", lfu_tr["x"], lfu_tr["y"], lfu_tr["t"], lfu_tr["idx"])
    save_split(round_dataset_dir / "lf_unpaired", "val", lfu_va["x"], lfu_va["y"], lfu_va["t"], lfu_va["idx"])
    save_split(round_dataset_dir / "lf_unpaired", "test", lfu_te["x"], lfu_te["y"], lfu_te["t"], lfu_te["idx"])

    save_json(round_dataset_dir / "round_dataset_meta.json", {
        "n_train_hf": int(train_idx.size),
        "n_val_hf": int(val_idx.size),
        "n_candidate": int(candidate_idx.size),
        "n_lf_unpaired_total": int(n_lfu_total),
        "lf_multiplier": int(lf_multiplier),
        "lfu_split": [float(x) for x in lfu_split],
    })


# -----------------------------------------------------------------------------
# Method state and scoring from trainer outputs
# -----------------------------------------------------------------------------

@dataclass
class MethodState:
    name: str
    pred_key: str
    score_mode: str  # mean_only | lcb | random | oracle
    train_idx: np.ndarray
    val_idx: np.ndarray
    candidate_idx: np.ndarray
    queries: List[int]
    best_true_rmse: List[float]
    is_stopped: bool = False
    stop_round: Optional[int] = None
    stop_reason: str = ""


def state_fingerprint(train_idx: np.ndarray, val_idx: np.ndarray, candidate_idx: np.ndarray) -> str:
    """Hash a round dataset state so identical states can share one compare-script run."""
    h = hashlib.sha1()
    for name, arr in (("train", train_idx), ("val", val_idx), ("cand", candidate_idx)):
        a = np.asarray(arr, dtype=np.int64).reshape(-1)
        h.update(name.encode("utf-8"))
        h.update(str(a.shape[0]).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def align_prediction_to_candidate_idx(
    *,
    y_pred: np.ndarray,
    std: Optional[np.ndarray],
    y_true: np.ndarray,
    cand_idx_from_pred: np.ndarray,
    candidate_idx: np.ndarray,
    label: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Align trainer outputs to the exact candidate_idx ordering used by the state."""
    cand_idx_from_pred = np.asarray(cand_idx_from_pred, dtype=np.int64).reshape(-1)
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64).reshape(-1)
    if cand_idx_from_pred.shape[0] != candidate_idx.shape[0] or not np.array_equal(cand_idx_from_pred, candidate_idx):
        order_map = build_idx_map({
            "x": np.empty((cand_idx_from_pred.shape[0], 0), dtype=np.float32),
            "y": y_pred,
            "t": np.empty((cand_idx_from_pred.shape[0],), dtype=np.float32),
            "idx": cand_idx_from_pred,
        })
        rows = np.asarray([order_map[int(i)] for i in candidate_idx.tolist()], dtype=np.int64)
        y_pred = y_pred[rows]
        y_true = y_true[rows]
        if std is not None:
            std = std[rows]
        cand_idx_from_pred = cand_idx_from_pred[rows]
    if not np.array_equal(cand_idx_from_pred, candidate_idx):
        raise RuntimeError(f"[{label}] candidate idx mismatch after reindexing")
    return y_pred, std, y_true, cand_idx_from_pred


def load_shared_state_predictions(
    *,
    run_dir: Path,
    candidate_idx: np.ndarray,
    pred_key_hf_only: str,
    pred_key_ar1: str,
    pred_key_ours: str,
    uq_cache_name: str,
) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
    """Load one compare-script run and expose aligned predictions for all model-based policies.

    Returned keys: hf_only, ar1, ours
      each maps to {"y_pred", "std", "y_true", "cand_idx"}
    """
    out: Dict[str, Dict[str, Optional[np.ndarray]]] = {}

    y_pred, std, y_true, cand_idx = load_pred_arrays(
        run_dir, pred_key_hf_only, need_std=False, uq_cache_name=uq_cache_name
    )
    y_pred, std, y_true, cand_idx = align_prediction_to_candidate_idx(
        y_pred=y_pred, std=std, y_true=y_true, cand_idx_from_pred=cand_idx,
        candidate_idx=candidate_idx, label="hf_only"
    )
    _ensure_finite("load_shared_state_predictions::hf_only::y_pred", y_pred)
    _ensure_finite("load_shared_state_predictions::hf_only::y_true", y_true)
    out["hf_only"] = {"y_pred": y_pred, "std": std, "y_true": y_true, "cand_idx": cand_idx}

    y_pred, std, y_true, cand_idx = load_pred_arrays(
        run_dir, pred_key_ar1, need_std=False, uq_cache_name=uq_cache_name
    )
    y_pred, std, y_true, cand_idx = align_prediction_to_candidate_idx(
        y_pred=y_pred, std=std, y_true=y_true, cand_idx_from_pred=cand_idx,
        candidate_idx=candidate_idx, label="ar1"
    )
    _ensure_finite("load_shared_state_predictions::ar1::y_pred", y_pred)
    _ensure_finite("load_shared_state_predictions::ar1::y_true", y_true)
    out["ar1"] = {"y_pred": y_pred, "std": std, "y_true": y_true, "cand_idx": cand_idx}

    y_pred, std, y_true, cand_idx = load_pred_arrays(
        run_dir, pred_key_ours, need_std=True, uq_cache_name=uq_cache_name
    )
    y_pred, std, y_true, cand_idx = align_prediction_to_candidate_idx(
        y_pred=y_pred, std=std, y_true=y_true, cand_idx_from_pred=cand_idx,
        candidate_idx=candidate_idx, label="ours"
    )
    _ensure_finite("load_shared_state_predictions::ours::y_pred", y_pred)
    _ensure_finite("load_shared_state_predictions::ours::y_true", y_true)
    if std is not None:
        _ensure_finite("load_shared_state_predictions::ours::std", std)
    out["ours"] = {"y_pred": y_pred, "std": std, "y_true": y_true, "cand_idx": cand_idx}

    return out


def load_pred_arrays(run_dir: Path, pred_key: str, need_std: bool, uq_cache_name: str = "uq_cache_v1") -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Load candidate predictions in one of two ways:

    1) cache/<uq_cache_name>.npz  (preferred for the baseline comparison script)
    2) pred_arrays/test/          (fallback for trainer scripts that only save pred_arrays)

    Important: when searching recursively, exclude nested delegated sub-runs such as
    run/.../ours/pred_arrays/test/*.npy, because those can contain placeholder NaN
    arrays for methods that were intentionally disabled inside the delegated run.

    Returns:
      y_pred  : (N,K)
      std     : (N,K) or None
      y_true  : (N,K)
      cand_idx: (N,)
    """
    # --- prefer UQ cache .npz from the baseline comparison script ---
    cache_npz = find_run_artifact(
        run_dir,
        ("cache", f"{uq_cache_name}.npz"),
        allow_recursive=True,
        exclude_segments=("ours",),
    )
    if cache_npz is not None:
        blob = np.load(cache_npz)
        y_pred_key = f"y_pred_test__{pred_key}"
        std_key = f"std_raw_test__{pred_key}"
        y_true_key = "y_test"

        if y_pred_key not in blob or y_true_key not in blob:
            raise KeyError(
                f"Cache file {cache_npz} missing required keys: need {y_pred_key!r} and {y_true_key!r}. "
                f"Available keys: {sorted(blob.files)}"
            )

        y_pred = np.asarray(blob[y_pred_key], dtype=np.float64)
        y_true = np.asarray(blob[y_true_key], dtype=np.float64)
        _ensure_finite(f"cache::{y_pred_key}", y_pred)
        _ensure_finite(f"cache::{y_true_key}", y_true)

        hf_test_idx = find_run_artifact(
            run_dir.parent,
            ("dataset", "hf", "idx_test.npy"),
            allow_recursive=True,
            exclude_segments=("ours",),
        )
        if hf_test_idx is None:
            raise FileNotFoundError(f"Cannot find candidate idx_test.npy for {run_dir}")
        cand_idx = np.load(hf_test_idx).astype(np.int64)

        std = None
        if need_std:
            if std_key not in blob:
                raise KeyError(
                    f"Cache file {cache_npz} missing required std key {std_key!r}. Available keys: {sorted(blob.files)}"
                )
            std = np.asarray(blob[std_key], dtype=np.float64)
            _ensure_finite(f"cache::{std_key}", std)

        return y_pred, std, y_true, cand_idx

    # --- fallback: pred_arrays/test/ from trainer-only scripts ---
    pred_root = find_run_artifact(
        run_dir,
        ("pred_arrays", "test", f"y_pred__{pred_key}.npy"),
        allow_recursive=True,
        exclude_segments=("ours",),
    )
    if pred_root is not None:
        test_dir = pred_root.parent
        y_pred = np.load(test_dir / f"y_pred__{pred_key}.npy").astype(np.float64)
        y_true = np.load(test_dir / "y_true.npy").astype(np.float64)
        _ensure_finite(f"pred_arrays::{pred_key}::y_pred", y_pred)
        _ensure_finite(f"pred_arrays::{pred_key}::y_true", y_true)

        cand_idx = np.load(test_dir / "idx_test.npy").astype(np.int64) if (test_dir / "idx_test.npy").exists() else None
        if cand_idx is None:
            hf_test_idx = find_run_artifact(
                run_dir.parent,
                ("dataset", "hf", "idx_test.npy"),
                allow_recursive=True,
                exclude_segments=("ours",),
            )
            if hf_test_idx is None:
                raise FileNotFoundError(f"Cannot find candidate idx_test.npy for {run_dir}")
            cand_idx = np.load(hf_test_idx).astype(np.int64)

        std = None
        if need_std:
            cal = test_dir / f"std_cal__{pred_key}.npy"
            raw = test_dir / f"std_raw__{pred_key}.npy"
            if cal.exists():
                std = np.load(cal).astype(np.float64)
            elif raw.exists():
                std = np.load(raw).astype(np.float64)
            else:
                raise FileNotFoundError(f"Need std for pred_key={pred_key}, but neither std_cal nor std_raw exists under {test_dir}")
            _ensure_finite(f"pred_arrays::{pred_key}::std", std)

        return y_pred, std, y_true, cand_idx

    raise FileNotFoundError(
        f"Cannot find usable predictions for pred_key={pred_key!r} under {run_dir}. "
        f"Looked for cache/{uq_cache_name}.npz first, then pred_arrays/test/y_pred__{pred_key}.npy, excluding nested delegated sub-runs such as 'ours/'."
    )


def resolve_score_from_predictions(*, y_pred: np.ndarray, std: Optional[np.ndarray], target_y: np.ndarray,
                                   score_mode: str, beta: float) -> np.ndarray:
    _ensure_finite("resolve_score_from_predictions::y_pred", y_pred)
    _ensure_finite("resolve_score_from_predictions::target_y", target_y)
    base = rmse_rows(y_pred, target_y)
    _ensure_finite("resolve_score_from_predictions::base", base)
    if score_mode == "mean_only":
        return base
    if score_mode == "lcb":
        if std is None:
            raise ValueError("lcb score_mode requires std")
        _ensure_finite("resolve_score_from_predictions::std", std)
        score = base - float(beta) * rms_std_rows(std)
        _ensure_finite("resolve_score_from_predictions::lcb_score", score)
        return score
    raise ValueError(f"Unknown score_mode={score_mode}")


# -----------------------------------------------------------------------------
# Summary writers
# -----------------------------------------------------------------------------

def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default="../../data/mf_sweep_datasets_nano_tm")
    ap.add_argument("--initial_subdir", type=str, default="hf50_lfx10")
    ap.add_argument("--max_subdir", type=str, default="hf500_lfx10")
    ap.add_argument("--target_split", type=str, default="test", choices=["test", "dev"])
    ap.add_argument("--n_targets", type=int, default=20)
    ap.add_argument("--target_start", type=int, default=0)
    ap.add_argument("--target_row_ids", type=str, default="",
                    help="Optional explicit raw row ids inside hf/{test,dev}, comma-separated. If set, bypasses target_start/n_targets sequential slicing.")
    ap.add_argument("--target_global_idxs", type=str, default="",
                    help="Optional explicit target global sample indices, comma-separated. Mapped back to row ids inside hf/{test,dev}. If set, bypasses target_start/n_targets sequential slicing.")
    ap.add_argument("--filter_targets_by_headroom", type=int, default=1, choices=[0, 1],
                    help="If 1, automatically keep only targets with headroom > min_headroom before applying target_start/n_targets.")
    ap.add_argument("--min_headroom", type=float, default=0.01,
                    help="Target filtering threshold: keep targets with initial_best - oracle_pool_best > min_headroom.")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--out_dir", type=str, default="../../result_out/retro_acq_runs_tm")

    # shared comparison trainer that compares hf_only / ar1 / ours
    # default path is resolved relative to this acquisition script, not the shell cwd
    ap.add_argument("--compare_script", type=str, default=DEFAULT_COMPARE_SCRIPT,
                    help="Path to shared baseline script (default: ../mf_train_baseline/mf_baseline.py, resolved relative to this script)")
    ap.add_argument("--python_bin", type=str, default=sys.executable)
    ap.add_argument("--extra_args", type=str, default="--wl_low 380 --wl_high 750 --fpca_var_ratio 0.999 --svgp_M 64 --svgp_steps 500 --gp_ard 1 --plot_ci 0 --n_plot 0")
    ap.add_argument("--timeout_sec", type=int, default=0)
    ap.add_argument("--force_save_pred_arrays", type=int, default=1, choices=[0, 1])

    # prediction keys under pred_arrays/test/
    ap.add_argument("--pred_key_hf_only", type=str, default="hf_only")
    ap.add_argument("--pred_key_ar1", type=str, default="ar1")
    ap.add_argument("--pred_key_ours", type=str, default="ours")
    ap.add_argument("--uq_cache_name", type=str, default="uq_cache_v1",
                    help="Cache stem used by the shared mf_baseline.py under run_dir/**/cache/<name>.npz")

    # enable/disable policies
    ap.add_argument("--run_hf_only", type=int, default=1, choices=[0, 1])
    ap.add_argument("--run_ar1", type=int, default=1, choices=[0, 1])
    ap.add_argument("--run_ours_mean", type=int, default=1, choices=[0, 1])
    ap.add_argument("--run_ours_lcb", type=int, default=1, choices=[0, 1])
    ap.add_argument("--run_random", type=int, default=1, choices=[0, 1])
    ap.add_argument("--run_oracle", type=int, default=1, choices=[0, 1])

    # early stop: once a method reaches the target-specific oracle_pool_best, stop further rounds for that method
    ap.add_argument("--early_stop_on_oracle", type=int, default=1, choices=[0, 1])
    ap.add_argument("--oracle_stop_tol", type=float, default=1e-10)

    # LF-unpaired split used when rebuilding each round dataset
    ap.add_argument("--lfu_split", type=str, default="0.8,0.1,0.1")

    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    init_dir = root / args.initial_subdir
    max_dir = root / args.max_subdir
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    compare_script = resolve_script_relative_path(args.compare_script)
    if not init_dir.exists():
        raise SystemExit(f"initial_subdir not found: {init_dir}")
    if not max_dir.exists():
        raise SystemExit(f"max_subdir not found: {max_dir}")
    if not compare_script.exists():
        raise SystemExit(f"compare_script not found: {compare_script}")

    lfu_split_vals = [float(x) for x in str(args.lfu_split).split(",") if x.strip()]
    if len(lfu_split_vals) != 3:
        raise SystemExit("--lfu_split must have exactly 3 comma-separated floats, e.g. 0.8,0.1,0.1")
    lfu_split = (lfu_split_vals[0], lfu_split_vals[1], lfu_split_vals[2])

    extra_args = shlex.split(args.extra_args.strip()) if str(args.extra_args).strip() else []

    fam_init = load_family(init_dir)
    fam_max = load_family(max_dir)

    # Full shared HF/LF-paired pool comes from the maximum budget dataset: train+val = all HF in the shared pool
    full_hf_pack = concat_packs(fam_max["hf_train"], fam_max["hf_val"])
    full_lfp_pack = concat_packs(fam_max["lfp_train"], fam_max["lfp_val"])

    # Use the full LF-unpaired master pool from the maximum dataset (train+val+test)
    lfu_master_pack = concat_packs(fam_max["lfu_train"], fam_max["lfu_val"], fam_max["lfu_test"])

    # Initial known HF = initial train + val
    init_train_idx = fam_init["hf_train"]["idx"].astype(np.int64)
    init_val_idx = fam_init["hf_val"]["idx"].astype(np.int64)
    init_known_idx = np.concatenate([init_train_idx, init_val_idx], axis=0).astype(np.int64)

    # Candidate pool = maximum HF pool minus initial known HF set
    full_pool_idx = full_hf_pack["idx"].astype(np.int64)
    candidate_pool0 = setdiff_idx(full_pool_idx, init_known_idx)

    # Targets come from fixed test/dev split in the initial dataset (disjoint from HF/LF pools by construction)
    target_pack = fam_init["hf_test"] if str(args.target_split) == "test" else fam_init["hf_dev"]
    target_idx_all = target_pack["idx"].astype(np.int64)
    assert_disjoint(candidate_pool0, target_idx_all, "candidate_pool", f"target_{args.target_split}")
    assert_disjoint(init_known_idx, target_idx_all, "initial_known_hf", f"target_{args.target_split}")

    lf_multiplier = parse_lf_multiplier_from_name(args.initial_subdir)

    init_known_pack = subset_by_idx(full_hf_pack, init_known_idx)
    cand_pack0 = subset_by_idx(full_hf_pack, candidate_pool0)
    headroom_rows = scan_target_headroom(
        target_pack=target_pack,
        init_known_pack=init_known_pack,
        cand_pack=cand_pack0,
    )
    write_summary_csv(out_root / f"target_headroom__{args.target_split}.csv", headroom_rows)
    save_json(out_root / f"target_headroom__{args.target_split}.json", {"rows": headroom_rows})

    all_target_row_ids = [int(r["target_row_id"]) for r in headroom_rows]
    if bool(int(args.filter_targets_by_headroom)):
        eligible_target_row_ids = [int(r["target_row_id"]) for r in headroom_rows if float(r["headroom"]) > float(args.min_headroom)]
    else:
        eligible_target_row_ids = all_target_row_ids

    explicit_row_ids = parse_int_list_arg(args.target_row_ids)
    explicit_global_idxs = parse_int_list_arg(args.target_global_idxs)
    if explicit_row_ids and explicit_global_idxs:
        raise SystemExit("Use only one of --target_row_ids or --target_global_idxs, not both.")

    target_row_from_global: Dict[int, int] = {int(r["target_global_idx"]): int(r["target_row_id"]) for r in headroom_rows}

    if explicit_global_idxs:
        missing = [int(g) for g in explicit_global_idxs if int(g) not in target_row_from_global]
        if missing:
            raise SystemExit(f"--target_global_idxs contains ids not present in target_pack: {missing}")
        selected_target_row_ids = [target_row_from_global[int(g)] for g in explicit_global_idxs]
        selection_mode = "explicit_global_idxs"
    elif explicit_row_ids:
        missing = [int(rid) for rid in explicit_row_ids if int(rid) not in set(all_target_row_ids)]
        if missing:
            raise SystemExit(f"--target_row_ids contains row ids not present in target_pack: {missing}")
        selected_target_row_ids = [int(rid) for rid in explicit_row_ids]
        selection_mode = "explicit_row_ids"
    else:
        eligible_after_start = eligible_target_row_ids[int(args.target_start):]
        n_targets = min(int(args.n_targets), len(eligible_after_start))
        if n_targets <= 0:
            raise SystemExit("No targets available under current filtering / target_start / n_targets")
        selected_target_row_ids = eligible_after_start[:n_targets]
        selection_mode = "sequential"

    n_targets = len(selected_target_row_ids)
    saturated_count = int(sum(float(r["headroom"]) <= 0.0 for r in headroom_rows))
    print(f"[INFO] init_dir={init_dir}")
    print(f"[INFO] max_dir={max_dir}")
    print(f"[INFO] compare_script={compare_script}")
    print(f"[INFO] initial known HF = train({init_train_idx.size}) + val({init_val_idx.size}) = {init_known_idx.size}")
    print(f"[INFO] candidate pool size = {candidate_pool0.size}")
    print(f"[INFO] target split={args.target_split} | available targets={target_pack['idx'].shape[0]}")
    print(f"[INFO] target headroom scan: saturated={saturated_count} / {len(headroom_rows)}")
    print(f"[INFO] filter_targets_by_headroom={int(args.filter_targets_by_headroom)} | min_headroom={float(args.min_headroom):.6f} | eligible={len(eligible_target_row_ids)} | run n_targets={n_targets}")
    print(f"[INFO] target selection mode={selection_mode}")
    if explicit_global_idxs:
        print(f"[INFO] selected target_global_idxs={explicit_global_idxs}")
    print(f"[INFO] selected target_row_ids={selected_target_row_ids}")
    print(f"[INFO] lf_multiplier={lf_multiplier} | lfu_split={lfu_split}")
    print(f"[INFO] early_stop_on_oracle={int(args.early_stop_on_oracle)} | oracle_stop_tol={float(args.oracle_stop_tol):.3g}")

    headroom_by_row_id = {int(r["target_row_id"]): r for r in headroom_rows}

    rng = np.random.default_rng(int(args.seed))
    summary_rows: List[Dict[str, object]] = []

    for local_tid, target_row_id in enumerate(selected_target_row_ids):
        target_row_id = int(target_row_id)
        target_y = target_pack["y"][target_row_id].astype(np.float64)
        target_global_idx = int(target_pack["idx"][target_row_id])
        headroom_row = headroom_by_row_id[target_row_id]

        print("\n" + "#" * 100)
        print(f"[TARGET] local={local_tid:03d} | global_idx={target_global_idx}")
        print("#" * 100)

        method_states: List[MethodState] = []
        if int(args.run_hf_only):
            method_states.append(MethodState("hf_only", args.pred_key_hf_only, "mean_only",
                                            init_train_idx.copy(), init_val_idx.copy(), candidate_pool0.copy(),
                                            [int(init_known_idx.size)],
                                            [best_true_target_rmse(full_hf_pack, init_known_idx, target_y)]))
        if int(args.run_ar1):
            method_states.append(MethodState("ar1", args.pred_key_ar1, "mean_only",
                                            init_train_idx.copy(), init_val_idx.copy(), candidate_pool0.copy(),
                                            [int(init_known_idx.size)],
                                            [best_true_target_rmse(full_hf_pack, init_known_idx, target_y)]))
        if int(args.run_ours_mean):
            method_states.append(MethodState("ours_mean", args.pred_key_ours, "mean_only",
                                            init_train_idx.copy(), init_val_idx.copy(), candidate_pool0.copy(),
                                            [int(init_known_idx.size)],
                                            [best_true_target_rmse(full_hf_pack, init_known_idx, target_y)]))
        if int(args.run_ours_lcb):
            method_states.append(MethodState("ours_lcb", args.pred_key_ours, "lcb",
                                            init_train_idx.copy(), init_val_idx.copy(), candidate_pool0.copy(),
                                            [int(init_known_idx.size)],
                                            [best_true_target_rmse(full_hf_pack, init_known_idx, target_y)]))
        if int(args.run_random):
            method_states.append(MethodState("random", "", "random",
                                            init_train_idx.copy(), init_val_idx.copy(), candidate_pool0.copy(),
                                            [int(init_known_idx.size)],
                                            [best_true_target_rmse(full_hf_pack, init_known_idx, target_y)]))
        if int(args.run_oracle):
            method_states.append(MethodState("oracle", "", "oracle",
                                            init_train_idx.copy(), init_val_idx.copy(), candidate_pool0.copy(),
                                            [int(init_known_idx.size)],
                                            [best_true_target_rmse(full_hf_pack, init_known_idx, target_y)]))

        if not method_states:
            raise RuntimeError("No methods enabled")

        comparison_method_names = {st.name for st in method_states if st.name in {"hf_only", "ar1", "ours_mean", "ours_lcb"}}

        for r in range(int(args.rounds)):
            # If all enabled comparison methods have already stopped (or exhausted the pool), finish this target early.
            if comparison_method_names:
                cmp_states = [st for st in method_states if st.name in comparison_method_names]
                if all(st.is_stopped or st.candidate_idx.size == 0 for st in cmp_states):
                    print(f"\n[TARGET] all comparison methods stopped or exhausted by end of previous round; finish target early at round {r:02d}")
                    break

            print(f"\n[ROUND] {r:02d}")
            # Cache compare-script outputs by unique dataset state so identical states share one run.
            state_run_cache: Dict[str, Dict[str, object]] = {}

            for st in method_states:
                if st.is_stopped:
                    print(f"  [{st.name}] early-stopped ({st.stop_reason}); skip")
                    continue
                if st.candidate_idx.size == 0:
                    print(f"  [{st.name}] candidate pool empty; skip")
                    continue

                tgt_dir = out_root / f"target_{local_tid:03d}__idx_{target_global_idx}" / st.name
                round_dir = tgt_dir / f"round_{r:02d}"
                round_dir.mkdir(parents=True, exist_ok=True)

                # Choose new HF queries
                if st.score_mode == "random":
                    chosen = rng.choice(st.candidate_idx, size=min(int(args.batch_size), st.candidate_idx.size), replace=False).astype(np.int64)
                    score_resolved = None
                    shared_state_key = None
                elif st.score_mode == "oracle":
                    cand_true = subset_by_idx(full_hf_pack, st.candidate_idx)
                    score_resolved = rmse_rows(cand_true["y"], target_y)
                    chosen = select_top_b(score_resolved, st.candidate_idx, b=min(int(args.batch_size), st.candidate_idx.size))
                    np.save(round_dir / "oracle_candidate_score.npy", score_resolved.astype(np.float32))
                    shared_state_key = None
                else:
                    shared_state_key = state_fingerprint(st.train_idx, st.val_idx, st.candidate_idx)
                    cache_hit = (shared_state_key in state_run_cache)
                    if not cache_hit:
                        shared_root = out_root / f"target_{local_tid:03d}__idx_{target_global_idx}" / "_shared_model_runs"
                        state_root = shared_root / f"round_{r:02d}__state_{shared_state_key[:12]}"
                        round_dataset_dir = state_root / "dataset"
                        round_run_dir = state_root / "run"
                        state_root.mkdir(parents=True, exist_ok=True)

                        materialize_round_dataset(
                            round_dataset_dir=round_dataset_dir,
                            wavelengths_src_dir=init_dir,
                            full_hf_pack=full_hf_pack,
                            full_lfp_pack=full_lfp_pack,
                            lfu_master_pack=lfu_master_pack,
                            train_idx=st.train_idx,
                            val_idx=st.val_idx,
                            candidate_idx=st.candidate_idx,
                            lf_multiplier=lf_multiplier,
                            lfu_split=lfu_split,
                        )

                        save_json(state_root / "target.json", {
                            "target_global_idx": int(target_global_idx),
                            "target_y_shape": list(target_y.shape),
                            "state_key": shared_state_key,
                            "train_size": int(st.train_idx.size),
                            "val_size": int(st.val_idx.size),
                            "candidate_size": int(st.candidate_idx.size),
                        })

                        run_compare_script(
                            round_dataset_dir=round_dataset_dir,
                            run_dir=round_run_dir,
                            compare_script=compare_script,
                            python_bin=str(args.python_bin),
                            seed=int(args.seed),
                            extra_args=extra_args,
                            force_save_pred_arrays=bool(int(args.force_save_pred_arrays)),
                            timeout_sec=int(args.timeout_sec),
                        )

                        pred_bundle = load_shared_state_predictions(
                            run_dir=round_run_dir,
                            candidate_idx=st.candidate_idx,
                            pred_key_hf_only=str(args.pred_key_hf_only),
                            pred_key_ar1=str(args.pred_key_ar1),
                            pred_key_ours=str(args.pred_key_ours),
                            uq_cache_name=str(args.uq_cache_name),
                        )
                        state_run_cache[shared_state_key] = {
                            "state_root": state_root,
                            "run_dir": round_run_dir,
                            "pred_bundle": pred_bundle,
                        }
                    shared_info = state_run_cache[shared_state_key]
                    pred_bundle = shared_info["pred_bundle"]
                    pred_slot = "ours" if st.pred_key == args.pred_key_ours else st.pred_key
                    slot = pred_bundle[pred_slot]
                    y_pred = slot["y_pred"]
                    std = slot["std"] if st.score_mode == "lcb" else None

                    score_resolved = resolve_score_from_predictions(
                        y_pred=y_pred,
                        std=std,
                        target_y=target_y,
                        score_mode=st.score_mode,
                        beta=float(args.beta),
                    )
                    _ensure_finite(f"round={r:02d} method={st.name} candidate_score", score_resolved)
                    np.save(round_dir / f"candidate_score__{st.name}.npy", score_resolved.astype(np.float32))
                    save_json(round_dir / "shared_state_ref.json", {
                        "state_key": shared_state_key,
                        "shared_run_dir": str(Path(shared_info["run_dir"])),
                        "pred_slot": pred_slot,
                        "cache_hit_within_round": int(cache_hit),
                    })
                    chosen = select_top_b(score_resolved, st.candidate_idx, b=min(int(args.batch_size), st.candidate_idx.size))

                chosen = np.asarray(chosen, dtype=np.int64)
                chosen_set = set(int(x) for x in chosen.tolist())
                st.train_idx = np.concatenate([st.train_idx, chosen], axis=0).astype(np.int64)
                st.candidate_idx = np.asarray([int(x) for x in st.candidate_idx.tolist() if int(x) not in chosen_set], dtype=np.int64)

                known_idx = np.concatenate([st.train_idx, st.val_idx], axis=0).astype(np.int64)
                best_now = best_true_target_rmse(full_hf_pack, known_idx, target_y)
                st.queries.append(int(known_idx.size))
                st.best_true_rmse.append(float(best_now))

                reached_oracle = False
                if bool(int(args.early_stop_on_oracle)):
                    oracle_target_best = float(headroom_row["oracle_pool_best"])
                    if float(best_now) <= oracle_target_best + float(args.oracle_stop_tol):
                        st.is_stopped = True
                        st.stop_round = int(r)
                        st.stop_reason = "reached_oracle_pool_best"
                        reached_oracle = True

                save_json(round_dir / "round_result.json", {
                    "target_global_idx": int(target_global_idx),
                    "method": st.name,
                    "round": int(r),
                    "state_key": shared_state_key,
                    "selected_idx": [int(x) for x in chosen.tolist()],
                    "n_known_hf_after": int(known_idx.size),
                    "best_true_target_rmse": float(best_now),
                    "oracle_pool_best": float(headroom_row["oracle_pool_best"]),
                    "reached_oracle_pool_best": int(reached_oracle),
                    "is_stopped_after_round": int(st.is_stopped),
                    "stop_reason": st.stop_reason,
                })
                stop_suffix = " | stop=oracle" if reached_oracle else ""
                if shared_state_key is None:
                    print(f"  [{st.name}] +{chosen.size} HF | known={known_idx.size:4d} | best_true={best_now:.6f}{stop_suffix}")
                else:
                    print(f"  [{st.name}] +{chosen.size} HF | known={known_idx.size:4d} | best_true={best_now:.6f} | state={shared_state_key[:8]}{stop_suffix}")

            if comparison_method_names:
                cmp_states = [st for st in method_states if st.name in comparison_method_names]
                if all(st.is_stopped or st.candidate_idx.size == 0 for st in cmp_states):
                    print(f"[TARGET] all comparison methods stopped or exhausted within round {r:02d}; finish target early")
                    break

        # Per-target summary
        per_target = {
            "target_local_id": int(local_tid),
            "target_row_id": int(target_row_id),
            "target_global_idx": int(target_global_idx),
            "target_selection_mode": selection_mode,
            "headroom": float(headroom_row["headroom"]),
            "initial_best": float(headroom_row["initial_best"]),
            "oracle_pool_best": float(headroom_row["oracle_pool_best"]),
            "methods": {
                st.name: {
                    "queries": [int(x) for x in st.queries],
                    "best_true_target_rmse": [float(x) for x in st.best_true_rmse],
                    "is_stopped": int(st.is_stopped),
                    "stop_round": (None if st.stop_round is None else int(st.stop_round)),
                    "stop_reason": st.stop_reason,
                }
                for st in method_states
            },
        }
        target_root = out_root / f"target_{local_tid:03d}__idx_{target_global_idx}"
        save_json(target_root / "target_summary.json", per_target)

        # Long-form summary rows for CSV
        for st in method_states:
            for step, (q, bval) in enumerate(zip(st.queries, st.best_true_rmse)):
                summary_rows.append({
                    "target_local_id": int(local_tid),
                    "target_row_id": int(target_row_id),
                    "target_global_idx": int(target_global_idx),
                    "headroom": float(headroom_row["headroom"]),
                    "initial_best": float(headroom_row["initial_best"]),
                    "oracle_pool_best": float(headroom_row["oracle_pool_best"]),
                    "method": st.name,
                    "step": int(step),
                    "n_known_hf": int(q),
                    "best_true_target_rmse": float(bval),
                    "is_stopped": int(st.is_stopped),
                    "stop_round": (None if st.stop_round is None else int(st.stop_round)),
                    "stop_reason": st.stop_reason,
                })

    write_summary_csv(out_root / "retro_acq_summary.csv", summary_rows)
    save_json(out_root / "retro_acq_summary.json", {"rows": summary_rows})
    print("\n[DONE] retrospective acquisition finished")
    print(f"[OUT] {out_root / 'retro_acq_summary.csv'}")


if __name__ == "__main__":
    main()
