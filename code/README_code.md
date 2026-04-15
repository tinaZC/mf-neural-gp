# Code organization

This directory contains the Python source code for all experiments in the paper.

Compared with the earlier layout, the code is no longer organized purely as separate per-benchmark training stacks.  
The recent refactor introduces a **shared training/baseline backend** under `mf_train_baseline/`, while the benchmark-specific directories keep the task-facing entry scripts, sweep scripts, plotting scripts, and experiment wrappers.

In practice, the codebase is now best understood as having two layers:

1. **Shared implementation layer**  
   Common multi-fidelity training, baseline comparison, and utility code used by multiple nanophotonic benchmarks.

2. **Benchmark / experiment layer**  
   Benchmark-specific entry scripts and downstream experiment modules for transmission, absorption, microwave, acquisition, efficiency, FPCA analysis, and complexity analysis.

---

## Directory overview

### `mf_train_baseline/`

Shared backend for the refactored nanophotonic multi-fidelity pipeline.

Main scripts:
- `mf_train.py`  
  Shared training backend for the proposed Neural--GP multi-fidelity model.
- `mf_baseline.py`  
  Shared baseline comparison backend, including HF-only, AR1/co-kriging-style baseline, and the proposed method.
- `mf_utils.py`  
  Shared utility functions used by the shared training and baseline scripts.
- `__init__.py`  
  Package marker for importing the shared module cleanly.

This directory is now the **core implementation layer** for the refactored nanophotonic training/baseline code.

---

### `nanophotonic_tm/`

Code for the nanophotonic **transmission** benchmark.

Main scripts:
- `mf_train_tm.py`  
  Transmission-facing training entry script. After the refactor, this serves as the benchmark-specific entry point while the underlying shared implementation lives in `mf_train_baseline/mf_train.py`.
- `mf_baseline_tm.py`  
  Transmission-facing baseline entry script. The underlying shared implementation is provided by `mf_train_baseline/mf_baseline.py`.
- `run_sweep_mf_baseline_tm.py`  
  Runs the transmission baseline sweep across multiple HF budgets / LF multipliers / seeds.
- `plot_sweep_results_baseline_tm.py`  
  Rebuilds and plots the aggregated transmission baseline results from run outputs.
- `plot_uq_from_cache.py`  
  Generates uncertainty-quantification figures from cached baseline runs.
- `run_ablate_tm.py`  
  Runs the transmission ablation study and produces the associated summary outputs.

In short, this directory now mainly contains **transmission-specific experiment drivers and plotting code**, while the reusable model/baseline logic is centralized under `mf_train_baseline/`.

---

### `nanophotonic_ab/`

Code for the nanophotonic **absorption** benchmark.

Main scripts:
- `mf_train_ab.py`  
  Absorption-facing training entry script built on the shared backend.
- `mf_baseline_ab.py`  
  Absorption-facing baseline entry script built on the shared backend.
- `run_sweep_mf_baseline_ab.py`  
  Runs the absorption baseline sweep.
- `plot_sweep_baseline_ab.py`  
  Rebuilds and plots the aggregated absorption baseline results.

As with `nanophotonic_tm/`, this directory now mainly provides **absorption-specific experiment wrappers and plotting utilities**, with shared core logic moved into `mf_train_baseline/`.

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
  Local utility functions used by the microwave benchmark.

At present, this benchmark remains organized as its own benchmark-specific stack.

---

### `hf_acquisition/`

Code for the **retrospective HF acquisition** experiment.

Main scripts:
- `acquisition_with_baseline_tm.py`  
  Runs the retrospective target-oriented HF acquisition experiment on the transmission benchmark.
- `acquisition_baseline_tm.py`  
  Auxiliary acquisition-related script for the transmission benchmark.
- `plot_retro_acq_curve.py`  
  Plots the aggregate acquisition curve from the generated acquisition summary.

After the refactor, this module should conceptually be understood as depending on the **shared nanophotonic baseline backend** (`mf_train_baseline/mf_baseline.py`) rather than on an older benchmark-private baseline implementation.

---

### `efficiency/`

Code for the **equal-accuracy efficiency analysis**.

Main scripts:
- `make_efficiency_speedup_tables_multi.py`  
  Builds speedup-curve data for multiple LF multipliers (`lfx=5,10,15`) in one run.
- `plot_efficiency_multi_lfx.py`  
  Plots the multi-LF efficiency figure.
- `make_efficiency_mapping_table.py`  
  Builds the compact CSV table used for the equal-accuracy mapping summary.

This module depends primarily on the aggregated sweep outputs from the transmission benchmark.

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
  Local utility functions for the FPCA module.

This directory is separate from `mf_train_baseline/` because it supports the reducer/dimension-sweep analysis rather than the main baseline-comparison pipeline.

---

### `complexity/`

Code for the **structural complexity analysis**.

Main scripts:
- `plot_structural_complexity.py`  
  Computes and plots the structural diagnostics, including linear mapping error and effective-rank analysis, using the nanophotonic absorption and transmission datasets.

---

## How to think about the refactored structure

A practical way to read the current codebase is:

- **Core reusable nanophotonic model/baseline code** lives in `mf_train_baseline/`.
- **Transmission and absorption directories** now mainly provide benchmark-specific launch points, sweeps, ablations, and plotting scripts.
- **Microwave, FPCA, efficiency, acquisition, and complexity** remain experiment-oriented modules that sit on top of either benchmark-specific outputs or the shared nanophotonic backend.

So the main architectural change is:

- **Before:** `nanophotonic_tm/` and `nanophotonic_ab/` each carried their own training/baseline implementation.
- **Now:** shared train/baseline logic is centralized in `mf_train_baseline/`, and the benchmark folders focus more on experiment orchestration.

---

## Suggested reading order

If you want to understand the codebase quickly, a practical order is:

1. `mf_train_baseline/mf_train.py`
2. `mf_train_baseline/mf_baseline.py`
3. `nanophotonic_tm/run_sweep_mf_baseline_tm.py`
4. `nanophotonic_tm/plot_sweep_results_baseline_tm.py`
5. `nanophotonic_tm/run_ablate_tm.py`
6. `hf_acquisition/`
7. `efficiency/`
8. `fpca/`
9. `complexity/`
10. `nanophotonic_ab/`
11. `microwave_mtm/`

This order reflects the current structure more accurately than the older README: shared backend first, then transmission experiments, then downstream analysis modules, and finally the additional benchmark families.

---

## Notes

- The scripts under `code/` are the implementation layer used to run the experiments and analyses.
- Benchmark-specific files such as `mf_train_tm.py`, `mf_baseline_tm.py`, `mf_train_ab.py`, and `mf_baseline_ab.py` should now be interpreted mainly as **task-facing wrappers / entry scripts**, not as the only place where the core model logic lives.
- The new `mf_train_baseline/` package is the intended place to look for the consolidated nanophotonic training and baseline implementation.
- Local `mf_utils.py` files under other directories (for example `microwave_mtm/` and `fpca/`) remain module-specific helper code rather than part of the shared nanophotonic backend.
