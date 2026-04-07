# Reproducibility scripts

This directory contains shell entry points for reproducing the main experimental figures and tables in the paper.

All scripts resolve `REPO_ROOT` automatically, so they can be launched from any working directory. By default, datasets are expected under `data/` and outputs are written to `result_out/`.

## Recommended order

Run the scripts in the following order:

1. `run_baseline_sweep_tm.sh` — Fig. 2–3
2. `run_baseline_sweep_ab.sh` — Fig. 4
3. `run_efficiency_tm.sh` — Fig. 5 and Table 1
4. `run_hf_acquisition_tm.sh` — Fig. 6
5. `run_uq_tm.sh` — Fig. 7
6. `run_complexity.sh` — Fig. 8
7. `run_fpca_dim_sweep_tm.sh` — Fig. 9
8. `run_baseline_sweep_mtm.sh` — Fig. 10
9. `run_ablation_tm.sh` — Fig. 11

## Scripts

### 1. Transmission baseline sweep (`run_baseline_sweep_tm.sh`)

**Covers:** Fig. 2–3

```bash
bash reproduce/run_baseline_sweep_tm.sh
```

This script runs the nanophotonic transmission baseline sweep and then plots the aggregated baseline comparison results.

Main outputs:
- `result_out/mf_sweep_runs_baseline_nano_tm/sweep_results.csv`
- `result_out/mf_sweep_runs_baseline_nano_tm/plot_result_baseline/`

---

### 2. Absorption baseline sweep (`run_baseline_sweep_ab.sh`)

**Covers:** Fig. 4

```bash
bash reproduce/run_baseline_sweep_ab.sh
```

This script runs the nanophotonic absorption baseline sweep and then plots the aggregated baseline comparison results.

Main outputs:
- `result_out/mf_sweep_runs_baseline_nano_ab/sweep_results.csv`
- `result_out/mf_sweep_runs_baseline_nano_ab/plot_baseline_nano_ab/`

---

### 3. Efficiency analysis (`run_efficiency_tm.sh`)

**Covers:** Fig. 5 and Table 1

```bash
bash reproduce/run_efficiency_tm.sh
```

This script performs the full efficiency analysis in one run. It first builds speedup curve data for `lfx=5,10,15`, then generates the multi-LF efficiency figure, and finally writes the compact mapping table.

**Important:** this script depends on the transmission baseline sweep results already existing, especially:

- `result_out/mf_sweep_runs_baseline_nano_tm/sweep_results.csv`

So run `run_baseline_sweep_tm.sh` first.

Main outputs:
- `result_out/mf_sweep_runs_baseline_nano_tm/efficiency_out/`
- `result_out/mf_sweep_runs_baseline_nano_tm/efficiency_out_multi/efficiency_multi_lfx.png`
- `result_out/mf_sweep_runs_baseline_nano_tm/efficiency_out_multi/efficiency_multi_lfx.pdf`
- `result_out/mf_sweep_runs_baseline_nano_tm/efficiency_out_multi/mapping_table.csv`

---

### 4. Retrospective HF acquisition (`run_hf_acquisition_tm.sh`)

**Covers:** Fig. 6

```bash
bash reproduce/run_hf_acquisition_tm.sh
```

This script runs the retrospective HF acquisition experiment on the nanophotonic transmission benchmark and then plots the aggregate acquisition curve.

The reproducibility script is configured to match the main experimental setting used in the paper:

- `initial_subdir = hf50_lfx10`
- `max_subdir = hf500_lfx10`
- `target_split = test`
- `n_targets = 20`
- `rounds = 10`
- `batch_size = 5`
- `beta = 0.5`

In addition, the external transmission baseline comparison model is launched with:

```text
--wl_low 380 --wl_high 750 --fpca_var_ratio 0.999 --svgp_M 64 --svgp_steps 500 --gp_ard 1 --plot_ci 0 --n_plot 0
```

In particular, the HF acquisition experiments use `--svgp_steps 500` in the comparison model, consistent with the original experiment configuration.

Main outputs:
- `result_out/retro_acq_runs_tm/retro_acq_summary.csv`
- `result_out/retro_acq_runs_tm/_fig_acquisition.png`

If needed, the dataset root, output directory, and script paths can still be overridden via environment variables in the shell wrapper.

---

### 5. UQ figure (`run_uq_tm.sh`)

**Covers:** Fig. 7

```bash
bash reproduce/run_uq_tm.sh
```

This script generates the UQ figure from an existing cached baseline run. It does **not** retrain a model.

If you want to use a different cached run, override `BEST_CACHE_NPZ` manually, for example:

```bash
BEST_CACHE_NPZ=/path/to/uq_cache_v1.npz bash reproduce/run_uq_tm.sh
```

Main outputs:
- `result_out/figs_uq/.../`

---

### 6. Structural complexity (`run_complexity.sh`)

**Covers:** Fig. 8

```bash
bash reproduce/run_complexity.sh
```

This script computes and plots the structural complexity figure by scanning the nanophotonic absorption and transmission dataset roots.

Main output:
- `result_out/fig_structural_complexity_2panel.png`

---

### 7. FPCA latent-dimension sweep (`run_fpca_dim_sweep_tm.sh`)

**Covers:** Fig. 9

```bash
bash reproduce/run_fpca_dim_sweep_tm.sh
```

This script runs the FPCA latent-dimension sweep on the default transmission dataset `hf100_lfx10` and then plots the summary figure.

Main outputs:
- `result_out/fpca_dim_sweep_tm_outputs/fpca_dim_sweep_summary.csv`
- `result_out/fpca_dim_sweep_tm_outputs/fpca_dim_sweep_summary_seedmean.csv` if multiple seeds are used
- `result_out/fpca_dim_sweep_tm_outputs/plots/`

---

### 8. Microwave MTM baseline sweep (`run_baseline_sweep_mtm.sh`)

**Covers:** Fig. 10

```bash
bash reproduce/run_baseline_sweep_mtm.sh
```

This script runs the default MTM baseline sweep and then plots the MTM baseline comparison figure.

Main outputs:
- `result_out/mf_baseline_out_microwave_mtm_multi/hf50_lfx10/`
- `result_out/mf_baseline_out_microwave_mtm_multi/hf50_lfx10/plot_mtm_compare/`

If you change `DATA_DIR` to another MTM subset, you should also update the plotting path and the `--hf/--lfx` plotting arguments accordingly.

---

### 9. Ablation study (`run_ablation_tm.sh`)

**Covers:** Fig. 11

```bash
bash reproduce/run_ablation_tm.sh
```

This script runs the transmission ablation study on the default `hf100_lfx10` dataset and writes outputs to `result_out/ablate_runs_tm`.

Main outputs:
- `result_out/ablate_runs_tm/`

## Notes

- These shell scripts are thin wrappers around the Python drivers under `code/`.
- Most paths can be overridden by exporting environment variables before running the script.
- The scripts assume that the required datasets have already been prepared under `data/`.
- Outputs are written to `result_out/` by default.
- `run_uq_tm.sh` and `run_efficiency_tm.sh` are downstream steps on top of the transmission baseline outputs, so they should be run only after the corresponding transmission baseline results are available.
