#!/usr/bin/env python3
from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "result_out/final_analysis/final_timing_comparison.csv"
OUT = ROOT / "result_out/final_analysis/tables"
OUT.mkdir(parents=True, exist_ok=True)


def get_path(obj, path):
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def recursive_find_numeric(obj, target):
    vals = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == target and isinstance(v, (int, float)):
                    vals.append(float(v))
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return vals[0] if len(vals) == 1 else None


def first_numeric(obj, candidates):
    for c in candidates:
        if "." in c:
            v = get_path(obj, c)
        else:
            v = recursive_find_numeric(obj, c)
        if v is not None:
            return v
    return None


def fmt(x, nd=3):
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):.{nd}f}"


df = pd.read_csv(SRC)

# ============================================================
# MAIN TABLE
# ============================================================

methods = [
    "HF-only",
    "AR1 / co-kriging",
    "FPCA-NARGP",
    "MF-DeepONet",
    "Neural-GP MF",
]

settings = [
    "hf50_lfx10",
    "hf100_lfx10",
    "hf500_lfx10",
]

headers = {
    "hf50_lfx10": "HF50 (40 train)",
    "hf100_lfx10": "HF100 (80 train)",
    "hf500_lfx10": "HF500 (400 train)",
}

pretty_method = {
    "HF-only": "HF-only",
    "AR1 / co-kriging": "AR1/co-kriging",
    "Neural-GP MF": "Neural-GP MF",
    "FPCA-NARGP": "FPCA-NARGP",
    "MF-DeepONet": "MF-DeepONet",
    "Direct-latent": "Direct-latent",
    "Frequency-wise NARGP": "Frequency-wise NARGP",
}

main_rows = []

for method in methods:
    row = {"Method": pretty_method[method]}
    for setting in settings:
        hit = df[
            (df["method"] == method)
            & (df["TM setting"] == setting)
        ]
        if len(hit) != 1:
            raise RuntimeError(
                f"Expected one row for {method} / {setting}, got {len(hit)}"
            )
        row[headers[setting]] = float(
            hit.iloc[0]["training time seconds"]
        )
    main_rows.append(row)

main = pd.DataFrame(main_rows)

main.to_csv(
    OUT / "timing_main_table.csv",
    index=False,
)

with open(OUT / "timing_main_table.md", "w") as f:
    f.write(main.to_markdown(index=False, floatfmt=".2f"))
    f.write("\n")

latex_main = r"""
\begin{table}[t]
\centering
\caption{Internal model training time on the TM benchmark under representative HF-budget settings. The numbers in parentheses denote the actual numbers of HF samples used for model training after data splitting. Timing excludes electromagnetic simulation, process startup, file I/O, plotting, result aggregation, and inference.}
\label{tab:timing_main}
\begin{tabular}{lccc}
\toprule
Method & HF50 (40 train) & HF100 (80 train) & HF500 (400 train) \\
\midrule
"""

for _, r in main.iterrows():
    latex_main += (
        f"{r['Method']} & "
        f"{r['HF50 (40 train)']:.2f} & "
        f"{r['HF100 (80 train)']:.2f} & "
        f"{r['HF500 (400 train)']:.2f} \\\\\n"
    )

latex_main += r"""\bottomrule
\end{tabular}
\end{table}
"""

(OUT / "timing_main_table.tex").write_text(latex_main)


# ============================================================
# SUPPLEMENTARY TABLE
# ============================================================

supp_rows = []

for _, r in df.iterrows():

    source = ROOT / str(r["canonical source"])
    if not source.is_file():
        raise FileNotFoundError(source)

    timing = json.loads(source.read_text())

    method = str(r["method"])

    stage1 = None
    stage2 = None

    # Do not force conceptual Stage I/II labels onto the
    # single-stage HF-only and AR1 baselines.
    if method == "Neural-GP MF":
        stage1 = first_numeric(
            timing,
            ["stage1_train_s"],
        )
        stage2 = first_numeric(
            timing,
            ["stage2_train_s.mf_student"],
        )

    elif method == "FPCA-NARGP":
        stage1 = first_numeric(
            timing,
            ["stage1_train_s"],
        )
        stage2 = first_numeric(
            timing,
            ["stage2_train_s"],
        )

    elif method == "MF-DeepONet":
        stage1 = first_numeric(
            timing,
            ["stage_i_lf_training_seconds"],
        )
        stage2 = first_numeric(
            timing,
            ["stage_ii_hf_correction_training_seconds"],
        )

    elif method in {
        "Direct-latent",
        "Frequency-wise NARGP",
    }:
        stage1 = first_numeric(
            timing,
            ["stage1_train_s"],
        )
        stage2 = first_numeric(
            timing,
            ["stage2_train_s"],
        )

    supp_rows.append({
        "TM setting": r["TM setting"],
        "HF train": int(r["HF train count"]),
        "LF train": int(r["LF train count"]),
        "Method": pretty_method.get(method, method),
        "Stage I (s)": stage1,
        "Stage II (s)": stage2,
        "Total training (s)": float(r["training time seconds"]),
        "Inference (s)": float(r["inference time seconds"]),
    })

supp = pd.DataFrame(supp_rows)

order_map = {
    "HF-only": 0,
    "AR1/co-kriging": 1,
    "FPCA-NARGP": 2,
    "MF-DeepONet": 3,
    "Neural-GP MF": 4,
    "Direct-latent": 5,
    "Frequency-wise NARGP": 6,
}

setting_map = {
    "hf50_lfx10": 0,
    "hf100_lfx10": 1,
    "hf500_lfx10": 2,
}

supp["_s"] = supp["TM setting"].map(setting_map)
supp["_m"] = supp["Method"].map(order_map)
supp = (
    supp.sort_values(["_s", "_m"])
        .drop(columns=["_s", "_m"])
        .reset_index(drop=True)
)

supp.to_csv(
    OUT / "timing_supplementary_table.csv",
    index=False,
)

display = supp.copy()

for c in [
    "Stage I (s)",
    "Stage II (s)",
    "Total training (s)",
    "Inference (s)",
]:
    display[c] = display[c].map(lambda x: fmt(x, 3))

with open(OUT / "timing_supplementary_table.md", "w") as f:
    f.write(display.to_markdown(index=False))
    f.write("\n")

latex_supp = r"""
\begin{table*}[t]
\centering
\caption{Detailed computational-time breakdown on the TM benchmark. Stage-I and Stage-II values are reported only for methods with an explicit two-stage implementation; ``--'' denotes that the decomposition is not applicable. Timing excludes electromagnetic simulation, process startup, file I/O, plotting, and result aggregation.}
\label{tab:timing_supp}
\begin{tabular}{lrrlrrrr}
\toprule
Setting & HF train & LF train & Method &
Stage I (s) & Stage II (s) &
Total training (s) & Inference (s) \\
\midrule
"""

for _, r in supp.iterrows():
    latex_supp += (
        f"{r['TM setting']} & "
        f"{int(r['HF train'])} & "
        f"{int(r['LF train'])} & "
        f"{r['Method']} & "
        f"{fmt(r['Stage I (s)'])} & "
        f"{fmt(r['Stage II (s)'])} & "
        f"{fmt(r['Total training (s)'])} & "
        f"{fmt(r['Inference (s)'])} \\\\\n"
    )

latex_supp += r"""\bottomrule
\end{tabular}
\end{table*}
"""

(OUT / "timing_supplementary_table.tex").write_text(
    latex_supp
)

print("\n===== MAIN TABLE =====")
print(main.to_string(index=False))

print("\n===== SUPPLEMENTARY TABLE =====")
print(display.to_string(index=False))

print("\n[SAVE]", OUT / "timing_main_table.csv")
print("[SAVE]", OUT / "timing_main_table.md")
print("[SAVE]", OUT / "timing_main_table.tex")
print("[SAVE]", OUT / "timing_supplementary_table.csv")
print("[SAVE]", OUT / "timing_supplementary_table.md")
print("[SAVE]", OUT / "timing_supplementary_table.tex")
