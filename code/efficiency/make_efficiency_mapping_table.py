#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_efficiency_mapping_table.py  (CSV-only)

Create a compact CSV table listing 2–3 RMSE_target anchors (mid / best / worst optional)
and their equal-accuracy mapping:
  RMSE_target
  HF_ours, HF_hfonly
  T_ours, T_hfonly
  speedup_HF, speedup_time

Reads multiple speedup_curves.npz files (lfx=5/10/15) and outputs ONLY:
  - mapping_table.csv

Example:
  python make_efficiency_mapping_table.py \
    --npz_5  ./.../efficiency_out_lfx05/speedup_curves.npz \
    --npz_10 ./.../efficiency_out_lfx10/speedup_curves.npz \
    --npz_15 ./.../efficiency_out_lfx15/speedup_curves.npz \
    --out_dir ./.../efficiency_out_multi \
    --anchors mid,best
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_npz(p: Path) -> Dict[str, np.ndarray]:
    if not p.exists():
        raise FileNotFoundError(f"NPZ not found: {p.resolve()}")
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def pick_index(rmse: np.ndarray, y: np.ndarray, mode: str) -> int:
    m = np.isfinite(rmse) & np.isfinite(y)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return -1
    r = rmse[idx]
    if mode == "best":
        return int(idx[np.argmin(r)])
    if mode == "worst":
        return int(idx[np.argmax(r)])
    # mid
    med = np.median(r)
    return int(idx[np.argmin(np.abs(r - med))])



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_5", type=str, default="../../result_out/mf_sweep_runs_baseline_nano_tm/efficiency_out_lfx05/speedup_curves.npz")
    ap.add_argument("--npz_10", type=str, default="../../result_out/mf_sweep_runs_baseline_nano_tm/efficiency_out_lfx10/speedup_curves.npz")
    ap.add_argument("--npz_15", type=str, default="../../result_out/mf_sweep_runs_baseline_nano_tm/efficiency_out_lfx15/speedup_curves.npz")
    ap.add_argument("--out_dir", type=str, default="../../result_out/mf_sweep_runs_baseline_nano_tm/efficiency_out_multi")
    ap.add_argument("--anchors", type=str, default="mid,best,worst",
                    help="Comma-separated list from {mid,best,worst}. Typical: mid,best (2 rows per lfx).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items: List[Tuple[str, Path]] = []
    if args.npz_5.strip():
        items.append(("lfx=5", Path(args.npz_5)))
    if args.npz_10.strip():
        items.append(("lfx=10", Path(args.npz_10)))
    if args.npz_15.strip():
        items.append(("lfx=15", Path(args.npz_15)))
    if not items:
        raise ValueError("Provide at least one of --npz_5/--npz_10/--npz_15")

    anchor_modes = [a.strip() for a in args.anchors.split(",") if a.strip()]
    for a in anchor_modes:
        if a not in ("mid", "best", "worst"):
            raise ValueError(f"Invalid anchor: {a}")

    rows = []
    for lab, p in items:
        d = load_npz(p)
        rmse = d["rmse_targets"].astype(np.float64)

        s_hf = d.get("hf_speedup_ours_vs_hfonly", None)
        if s_hf is None:
            raise KeyError(f"{lab} missing hf_speedup_ours_vs_hfonly")

        for mode in anchor_modes:
            i = pick_index(rmse, s_hf.astype(np.float64), mode)
            if i < 0:
                continue

            e = float(rmse[i])

            hf_ours = float(d["hfmin_ours"][i]) if "hfmin_ours" in d else np.nan
            hf_hfonly = float(d["hfmin_hfonly"][i]) if "hfmin_hfonly" in d else np.nan

            t_ours = float(d["time_ours_at_target_sec"][i]) if "time_ours_at_target_sec" in d else np.nan
            t_hfonly = float(d["time_hfonly_at_target_sec"][i]) if "time_hfonly_at_target_sec" in d else np.nan

            speed_hf = float(d["hf_speedup_ours_vs_hfonly"][i]) if "hf_speedup_ours_vs_hfonly" in d else np.nan
            speed_time = float(d["sim_speedup_ours_vs_hfonly"][i]) if "sim_speedup_ours_vs_hfonly" in d else np.nan

            rows.append({
                "lfx": lab,
                "anchor": mode,
                "RMSE_target": e,
                "HF_ours": int(round(hf_ours)) if np.isfinite(hf_ours) else np.nan,
                "HF_hfonly": int(round(hf_hfonly)) if np.isfinite(hf_hfonly) else np.nan,
                "T_ours_h": (t_ours / 3600.0) if np.isfinite(t_ours) else np.nan,
                "T_hfonly_h": (t_hfonly / 3600.0) if np.isfinite(t_hfonly) else np.nan,
                "speedup_HF": speed_hf,
                "speedup_time": speed_time,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid anchor rows found (likely RMSE overlap is empty for provided curves).")

    # Sort: lfx then anchor (worst, mid, best)
    anchor_order = {"worst": 0, "mid": 1, "best": 2}
    lfx_order = {"lfx=5": 5, "lfx=10": 10, "lfx=15": 15}
    df["_a"] = df["anchor"].map(anchor_order)
    df["_l"] = df["lfx"].map(lfx_order).fillna(999)
    df = df.sort_values(["_l", "_a", "RMSE_target"]).drop(columns=["_a", "_l"])

    # Pretty formatting (still CSV)
    df_out = df.copy()
    df_out["RMSE_target"] = df_out["RMSE_target"].map(lambda x: f"{x:.4f}")
    df_out["T_ours_h"] = df_out["T_ours_h"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    df_out["T_hfonly_h"] = df_out["T_hfonly_h"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    df_out["speedup_HF"] = df_out["speedup_HF"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    df_out["speedup_time"] = df_out["speedup_time"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    out_csv = out_dir / "mapping_table.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"[DONE] wrote: {out_csv}")


if __name__ == "__main__":
    main()