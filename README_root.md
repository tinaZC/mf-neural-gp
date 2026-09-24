# Neural-GP Multi-Fidelity for Function-Valued Material Responses

This repository is the revised Engineering with Computers release for *Uncertainty-aware multi-fidelity learning of function-valued electromagnetic responses for metamaterial design*. It preserves the original submitted architecture and pipelines while adding the reviewer comparison and diagnostics required by the revised manuscript.

The original core remains under `code/mf_train_baseline/`, `code/nanophotonic_tm/`, `code/nanophotonic_ab/`, `code/microwave_mtm/`, `code/efficiency/`, `code/hf_acquisition/`, `code/fpca/`, `code/complexity/`, and `code/time/`. Revision-specific comparison methods are intentionally isolated under `code/comparison_methods/`.

The formal revised evaluation uses matched seeds and compact summaries under `result_out/final_analysis/`. Large raw runs, checkpoints, prepared datasets, and machine-specific scheduler artifacts are not part of the GitHub release; the full datasets are distributed through Zenodo.

Create the environment with `conda env create -f environment.yml` and download the prepared splits from [Zenodo](https://doi.org/10.5281/zenodo.19447099). Reproduction commands are documented in [`reproduce/README_reproduce.md`](reproduce/README_reproduce.md). Definitive figure and table provenance is in [`docs/figure_provenance.md`](docs/figure_provenance.md) and [`docs/table_provenance.md`](docs/table_provenance.md). Revision scope and module ownership are recorded in [`docs/revision_changes.md`](docs/revision_changes.md) and [`docs/revision_module_inventory.md`](docs/revision_module_inventory.md).

Do not stage, commit, or push this checkout without author review of the local release report.
