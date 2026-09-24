# Revised-manuscript reproduction

Run commands from the repository root. CPU is sufficient for compact plotting; full training and the formal timing protocol require the documented GPU environment. Raw final runs and prepared data are intentionally external/local.

## A. Environment and data

```bash
conda env create -f environment.yml
conda activate multi_fidelity
```

Download Zenodo record `10.5281/zenodo.19447099` and place prepared TM, AB, and MTM splits under `data/` as described in [`data/README_data.md`](../data/README_data.md). Do not regenerate the supplied split/index files.

## B. Full experiment reruns

These commands train models and can take hours or days. They are optional when reproducing the supplied frozen figures.

- Submitted TM/AB baselines: `bash reproduce/run_baseline_sweep_tm.sh` and `bash reproduce/run_baseline_sweep_ab.sh` (GPU; writes raw sweep trees).
- Reviewer comparisons: inspect `bash reproduce/run_comparison_sweep_tm.sh --help` and the corresponding AB wrapper. Use an explicit `--run`, method, prepared dataset, seed, and new raw output directory; no default launches training.
- Direct-latent diagnostic: `bash reproduce/run_direct_latent_tm.sh --help` (GPU; explicit run required).
- Wavelength-wise NARGP: `bash reproduce/run_freqwise_nargp_tm.sh --help` (GPU; hundreds of coordinate models; supports contiguous ranges/resume).
- Formal matched 20-seed scheduler: `python reproduce/run_final_accuracy_3gpu.py --help` (GPU; explicit `--run`; do not launch merely to plot).
- Isolated timing: `python reproduce/run_final_timing.py --help` (GPU; seed-42 protocol; separate from submitted timing).

## C. Aggregation

All aggregation reads existing reports and does not retrain.

```bash
python reproduce/aggregate_final_results.py
python reproduce/final_timing_analysis.py
python reproduce/build_final_timing_tables.py
python reproduce/result_audit.py
```

Expected outputs include `main_accuracy_summary.csv`, paired comparison CSVs, `representation_ablation_summary.csv`, `uq_summary.csv`, `final_timing_comparison.csv`, and `tables/timing_*`. The aggregator validates matched seeds and adds the completed n=3 controlled wavelength-wise Stage-II diagnostic.

## D. Plot-only reproduction from curated/frozen results

The authoritative command, output filename, run type, and external-input status for every main and supplementary item are in [`docs/figure_provenance.md`](../docs/figure_provenance.md). Key commands are:

```bash
python code/nanophotonic_tm/plot_tm_representative_example.py
python code/nanophotonic_tm/plot_tm_rmse_sweep_publication.py
python code/nanophotonic_ab/plot_ab_rmse_sweep_publication.py
python code/efficiency/plot_efficiency_multi_lfx.py
python code/hf_acquisition/plot_retro_acq_curve.py
python code/nanophotonic_tm/plot_uq_comparison.py
python code/fpca/plot_fpca_dim_sweep_native.py --basename _fig_fpca_dim_sweep_native
python code/nanophotonic_tm/plot_ablation_publication.py
python code/complexity/plot_structural_complexity.py --out_fig _fig_structural_complexity.png
python code/microwave_mtm/plot_mtm_result_publication.py
python code/nanophotonic_tm/plot_sweep_results_baseline_tm.py --sweep_csv result_out/final_analysis/supplementary_sweeps/tm_sweep_results_rebuilt.csv --out_dir result_out/final_analysis/figures/supplementary_tm
python code/nanophotonic_ab/plot_sweep_baseline_ab.py --sweep_csv result_out/final_analysis/supplementary_sweeps/ab_sweep_results_rebuilt.csv --out_dir result_out/final_analysis/figures/supplementary_ab
python code/nanophotonic_tm/plot_uq_conformal_supplement.py
```

Figures 2, 5, 6, 7-11 and Supplementary Fig. S3 use the compact inputs under `result_out/final_analysis/frozen_inputs/`. Acquisition only redraws the pre-existing experiment. The raw FPCA rebuild, raw MTM reports, and prepared structural datasets remain external for provenance but are not needed to redraw the published figures. Framework figures 1 and S4 are manual artwork.

No single `make_all` wrapper is provided because some manuscript artwork is manual and several numerical sources are intentionally external archives.
