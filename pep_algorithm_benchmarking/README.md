# PEP algorithm benchmarking

Benchmarking of classical PEP/B-point/Q-peak extraction algorithms (the ones shipped in
`biopsykit.signals.icg.event_extraction` / `biopsykit.signals.ecg.event_extraction`) across multiple datasets,
using `pepbench`'s challenge/pipeline framework.

## Layout

- **`notebooks/challenges/`** — one notebook per dataset × event-type combination (`PEP_Benchmarking_*.ipynb`),
  each running the full `pepbench` benchmarking pipeline (EmpkinS, Guardian, ReBeatICG, TimeWindowICG datasets;
  B-point, Q-peak, or both).
- **`notebooks/analysis/`** — aggregate statistics and comparisons across algorithms/datasets, including
  inter-rater agreement analysis (`analysis/annotations/`).
- **`notebooks/demographics/`** — dataset demographics summary.
- **`notebooks/figures/`** — figure-generation notebooks (example signal/algorithm plots, the graphical abstract).
- **`notebooks/hpc_scripts/`** — scripts for running benchmarking jobs on a SLURM cluster.
- **`results/`, `exports/`** — per-dataset benchmarking results and exported statistics/plots. **Not tracked in
  git** (see `.gitignore`) - regenerate by re-running the challenge notebooks against the underlying datasets.

## Running

Each `notebooks/challenges/PEP_Benchmarking_*.ipynb` notebook expects the corresponding raw dataset to be
available locally (paths are typically read from a local `config.json`, which is git-ignored - see the notebook's
first cells for the expected format) and writes its results under `results/<dataset>_<event>/`. The
`notebooks/analysis/` and `notebooks/figures/` notebooks then read from those `results/` directories.
