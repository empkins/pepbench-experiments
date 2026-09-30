# pepbench-experiments

Research experiments, notebooks, and results built on top of [pepbench](https://github.com/empkins/pepbench)
and [biopsykit](https://github.com/mad-lab-fau/biopsykit). This repo is intentionally kept separate from the
`pepbench` library itself: it pins a released version of `pepbench` as a normal dependency (never a path or git
dependency), so the library can keep evolving and tagging releases without every experiment needing to move in
lockstep, and so `pepbench` itself doesn't have to carry experiment-only dependencies (e.g. `shap`) or multi-GB
result/model artifacts in its git history.

## Structure

- **`b_point_ml_experiments/`** — machine-learning-based B-point extraction: training data preprocessing,
  `SklearnPipelinePermuter` hyperparameter search, model evaluation/statistics, and the notebook that trained the
  pretrained model shipped via `biopsykit.signals.icg.event_extraction.get_b_point_abelstuehler2026_model`.
- **`pep_algorithm_benchmarking/`** — benchmarking of classical PEP/B-point/Q-peak extraction algorithms across
  datasets (EmpkinS, Guardian, ReBeatICG, TimeWindowICG): challenge notebooks, statistical analysis, and figures.

Each subproject has its own `README.md` with more detail.

## Setup

```bash
uv sync
```

This installs a pinned `pepbench` (which in turn pulls in `biopsykit`) plus the extra experiment-only dependencies
(`shap`, etc.) declared in this repo's `pyproject.toml`.

## Data and model artifacts

Large model/result artifacts (`models/`, `results/`, `exports/` under each subproject) are **not** tracked in git
— see each subproject's `.gitignore`'d directories. They currently live only on the machine(s) that produced them;
this repo's own `README.md`s note where to regenerate them from, and larger published artifacts (e.g. the
pretrained B-point regressor) are hosted as GitHub Release assets and fetched on demand via `pooch`, the same
pattern `pepbench.example_data` and `biopsykit`'s pretrained-model loader already use.

## Relationship to `pepbench`

This repo was split out of `pepbench`'s `experiments/` directory. At the time of the split, `pepbench/experiments/`
was left in place as a snapshot for reference; this repo is the actively maintained home for new experiment work
going forward.

## Citation

The ML-based B-point extraction work in `b_point_ml_experiments/` is described in:

> Abel L, Stühler S, Steigleder T, Ostgathe C, Rohleder N, Eskofier BM and Richer R (2026) Beat-to-beat aortic
> valve opening detection from impedance cardiography using machine learning. *Front. Digit. Health* 8:1944092.
> doi: [10.3389/fdgth.2026.1944092](https://doi.org/10.3389/fdgth.2026.1944092) (accepted, in press)
