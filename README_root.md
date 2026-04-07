# Neural–GP Multi-Fidelity for Function-Valued Material Responses

This repository contains the code, data organization, and reproducibility scripts for the paper:

**Neural Gaussian Processes Multi-Fidelity Modeling of Function-Valued Material Responses**

The project studies uncertainty-aware multi-fidelity modeling for dense electromagnetic response functions, including nanophotonic transmission spectra, nanophotonic absorption spectra, and microwave metamaterial responses. The proposed framework combines a neural low-fidelity surrogate with Gaussian-process-based high-fidelity correction in a compact latent space.

## Repository overview

The repository is organized into three main layers:

- `code/`  
  Python source code for all benchmarks and analysis modules.
- `reproduce/`  
  Shell entry points for reproducing the main figures and tables in the paper.
- `data/`  
  Prepared datasets used by the experiments.

Outputs are written to:

- `result_out/`

## Directory layout

```text
.
├─ code/
│  ├─ nanophotonic_tm/
│  ├─ nanophotonic_ab/
│  ├─ microwave_mtm/
│  ├─ efficiency/
│  ├─ hf_acquisition/
│  ├─ fpca/
│  └─ complexity/
├─ reproduce/
├─ data/
├─ result_out/
├─ environment.yml
└─ README.md
```

## Environment setup

Create the conda environment with:

```bash
conda env create -f environment.yml
conda activate multi_fidelity
```

The environment file includes the core dependencies required for the experiments, including PyTorch, GPyTorch, BoTorch, NumPy, SciPy, pandas, matplotlib, and related packages.

## Data organization

The repository expects prepared datasets under `data/`. The main dataset roots used by the reproducibility scripts are:

```text
data/
├─ mf_sweep_datasets_nano_tm/
├─ mf_sweep_datasets_nano_ab/
└─ mf_dataset_mw_mtm/
```

These directories are used directly by the scripts in `reproduce/`.

## Reproducing the paper

The main entry points are in `reproduce/`. They are organized in the same order that the results appear in the paper.

Recommended order:

1. `run_baseline_sweep_tm.sh` — transmission benchmark
2. `run_baseline_sweep_ab.sh` — absorption benchmark
3. `run_efficiency_tm.sh` — efficiency analysis
4. `run_hf_acquisition_tm.sh` — retrospective HF acquisition
5. `run_uq_tm.sh` — uncertainty quantification
6. `run_complexity.sh` — structural complexity analysis
7. `run_fpca_dim_sweep_tm.sh` — latent-dimension sweep
8. `run_baseline_sweep_mtm.sh` — microwave benchmark
9. `run_ablation_tm.sh` — ablation study

For detailed instructions, see:

```text
reproduce/README.md
```

## Quick start

A minimal workflow is:

```bash
conda env create -f environment.yml
conda activate multi_fidelity
bash reproduce/run_baseline_sweep_tm.sh
bash reproduce/run_efficiency_tm.sh
```

This first runs the main nanophotonic transmission benchmark and then the equal-accuracy efficiency analysis built on top of that baseline output.

## Code organization

The `code/` directory is organized by experiment family:

- `nanophotonic_tm/` — transmission benchmark
- `nanophotonic_ab/` — absorption benchmark
- `microwave_mtm/` — microwave benchmark
- `efficiency/` — equal-accuracy efficiency analysis
- `hf_acquisition/` — retrospective HF acquisition
- `fpca/` — latent-dimension sweep
- `complexity/` — structural complexity analysis

For a more detailed description of the source code, see:

```text
code/README.md
```

## Main outputs

By default, experiment outputs are written under `result_out/`, for example:

```text
result_out/
├─ mf_sweep_runs_baseline_nano_tm/
├─ mf_sweep_runs_baseline_nano_ab/
├─ mf_baseline_out_microwave_mtm_multi/
├─ fpca_dim_sweep_tm_outputs/
├─ retro_acq_runs_tm/
├─ figs_uq/
└─ fig_structural_complexity_2panel.png
```

## Notes

- Some scripts are primary experiment drivers, while others are downstream analysis scripts built on previously generated outputs.
- In particular, `run_efficiency_tm.sh` and `run_uq_tm.sh` depend on transmission baseline outputs already being available.
- Most script paths can be overridden via environment variables if needed.

## Citation

If you use this repository, please cite the associated paper.

```bibtex
@article{neural_gp_mf_2026,
  title   = {Neural Gaussian Processes Multi-Fidelity Modeling of Function-Valued Material Responses},
  author  = {First Author and Second Author and Third Author},
  journal = {To be updated},
  year    = {2026}
}
```

## Contact

For questions regarding the code or data organization, please contact the corresponding author listed in the manuscript.
