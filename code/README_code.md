# Code organization

This directory contains the Python source code for all experiments in the paper.

The code is organized by experiment family rather than by generic software layers. In other words, each subdirectory corresponds to one benchmark, analysis module, or figure-generation pipeline used in the manuscript.

## Directory overview

### `nanophotonic_tm/`

Code for the nanophotonic **transmission** benchmark.

Main scripts:
- `mf_train_tm.py`  
  Core training script for the transmission multi-fidelity model.
- `mf_baseline_tm.py`  
  Baseline comparison script for transmission, including HF-only, co-kriging/AR1, and the proposed method.
- `run_sweep_mf_baseline_tm.py`  
  Runs the transmission baseline sweep across multiple HF budgets / LF multipliers / seeds.
- `plot_sweep_results_baseline_tm.py`  
  Rebuilds and plots the aggregated transmission baseline results from run outputs.
- `plot_uq_from_cache.py`  
  Generates uncertainty quantification figures from a cached baseline run.
- `run_ablate_tm.py`  
  Runs the transmission ablation study and produces the associated summary outputs.
- `mf_utils.py`  
  Shared utility functions for the transmission benchmark.

---

### `nanophotonic_ab/`

Code for the nanophotonic **absorption** benchmark.

Main scripts:
- `mf_train_ab.py`  
  Core training script for the absorption multi-fidelity model.
- `mf_baseline_ab.py`  
  Baseline comparison script for absorption.
- `run_sweep_mf_baseline_ab.py`  
  Runs the absorption baseline sweep.
- `plot_sweep_baseline_ab.py`  
  Rebuilds and plots the aggregated absorption baseline results.
- `mf_utils.py`  
  Shared utility functions for the absorption benchmark.

---

### `microwave_mtm/`

Code for the **microwave metamaterial** benchmark.

Main scripts:
- `mf_train_microwave_mtm.py`  
  Core training script for the microwave multi-fidelity model.
- `mf_baseline_microwave_mtm.py`  
  Baseline comparison script for the microwave benchmark.
- `run_baseline_mtm_multi_seed.py`  
  Runs the microwave baseline benchmark over multiple random seeds.
- `plot_baseline_mtm_compare.py`  
  Rebuilds and plots the aggregated microwave comparison results.
- `mf_utils.py`  
  Shared utility functions for the microwave benchmark.

---

### `efficiency/`

Code for the **equal-accuracy efficiency analysis**.

Main scripts:
- `make_efficiency_speedup_tables_multi.py`  
  Builds speedup curve data for multiple LF multipliers (`lfx=5,10,15`) in one run.
- `plot_efficiency_multi_lfx.py`  
  Plots the multi-LF efficiency figure.
- `make_efficiency_mapping_table.py`  
  Builds the compact CSV table used for the equal-accuracy mapping summary.

This module depends on the baseline transmission sweep results, especially the aggregated `sweep_results.csv`.

---

### `hf_acquisition/`

Code for the **retrospective HF acquisition** experiment.

Main scripts:
- `acquisition_with_baseline_tm.py`  
  Runs the retrospective target-oriented HF acquisition experiment on the transmission benchmark.
- `plot_retro_acq_curve.py`  
  Plots the aggregate acquisition curve from the generated acquisition summary.

---

### `fpca/`

Code for the **FPCA latent-dimension sweep** and related dimensionality-reduction analysis.

Main scripts:
- `fpca_dim_sweep_tm_with_seeds.py`  
  Runs the FPCA latent-dimension sweep over one or more seeds.
- `mf_train_nano_tm_dim_sweep.py`  
  Training backend used during the FPCA sweep.
- `plot_fpca_dim_sweep_tm.py`  
  Plots the final FPCA sweep figure.
- `mf_reducers_shared.py`  
  Shared reducer implementations and helper code for dimensionality-reduction experiments.
- `mf_utils.py`  
  Shared utility functions for the FPCA module.

---

### `complexity/`

Code for the **structural complexity analysis**.

Main scripts:
- `plot_structural_complexity.py`  
  Computes and plots the structural diagnostics, including linear mapping error and effective-rank analysis, using the nanophotonic absorption and transmission datasets.

## Suggested reading order

If you want to understand the codebase quickly, a practical order is:

1. `nanophotonic_tm/mf_baseline_tm.py`
2. `nanophotonic_tm/run_sweep_mf_baseline_tm.py`
3. `nanophotonic_tm/plot_sweep_results_baseline_tm.py`
4. `efficiency/`
5. `hf_acquisition/`
6. `fpca/`
7. `complexity/`
8. `nanophotonic_ab/`
9. `microwave_mtm/`

This order roughly follows the role of each module in the paper: main transmission benchmark first, then downstream analysis modules, then the additional benchmark families.

## Notes

- The scripts under `code/` are the implementation layer.
- The scripts under `reproduce/` are thin shell wrappers that call these Python modules in the order used for the paper figures and tables.
- Utility files named `mf_utils.py` are local helpers for their corresponding benchmark family; they are not intended as a single global utilities module.