# Data download

The full dataset used in this study is hosted on Zenodo and is not stored directly in this GitHub directory.

**Dataset title**  
Dataset for: Neural–GP Multi-Fidelity for Function-Valued Material Responses

**Zenodo record (version used in the manuscript)**  
DOI: https://doi.org/10.5281/zenodo.19447099

**Latest Zenodo record (all versions)**  
DOI: https://doi.org/10.5281/zenodo.19447098

## Notes
- This `data/` directory is provided only as a placeholder and documentation entry point.
- Please download the dataset archive from Zenodo.
- After downloading, extract the archive and place the dataset according to the paths expected by the code and reproduction scripts described in the main repository `README.md`.

## Citation
If you use this dataset, please cite the Zenodo record corresponding to the manuscript version:
`10.5281/zenodo.19447099`

## Prepared split layout
Extract the versioned Zenodo archive so the following exact roots exist:
`data/mf_sweep_datasets_nano_tm/`,
`data/mf_sweep_datasets_nano_ab/`, and
`data/mf_dataset_mw_mtm/`. TM and AB contain settings such as
`hf50_lfx05`, `hf100_lfx10` and `hf500_lfx15`; MTM contains
`hf50_lfx10/`. Each setting has `wavelengths.npy`,
`idx_wavelength.npy`, and `hf/`, `lf_paired/` and `lf_unpaired/` split
arrays. Keep the supplied index arrays and sample order unchanged.
The versioned record is the authoritative download; verify its published
archive checksums where available. No checksum manifest was added here.

The downloaded arrays are ignored by Git and retained locally. Original
training reports, scheduler manifests and bulky run artifacts also remain
local or belong in a separately checksummed result archive; the compact
release tables and figures live under `result_out/final_analysis/`.
