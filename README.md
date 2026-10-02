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

## How to use

This project uses [uv](https://docs.astral.sh/uv/) to manage Python, the virtual environment, and dependencies.
All versions are pinned in `uv.lock`, so everyone runs the notebooks against the same package versions.

### 1. Install uv

If you don't have uv yet, install it (see the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/)
for other options):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the repo and install the dependencies

```bash
git clone https://github.com/empkins/pepbench-experiments.git
cd pepbench-experiments
uv sync
```

`uv sync` creates a virtual environment in `.venv/` (downloading a matching Python version if needed) and installs
a pinned `pepbench` (which in turn pulls in `biopsykit`), the extra experiment-only dependencies (`shap`, etc.)
declared in `pyproject.toml`, the `dev` dependency group (`ipykernel`, `ruff`, `poethepoet`, ...), and the small
`b_point_ml_experiments` helper package in editable mode.

You don't need to activate the environment manually: prefix commands with `uv run` to run them inside it
(e.g. `uv run python`, `uv run jupyter lab`).

### 3. Set up a Jupyter kernel

All experiments are Jupyter notebooks. To run them from an existing Jupyter installation or from an IDE
(VS Code, PyCharm), register the project's virtual environment as a Jupyter kernel:

```bash
uv run poe conf_jupyter
```

This is a shortcut for:

```bash
uv run python -m ipykernel install --user --name pepbench-experiments --display-name pepbench-experiments
```

Afterwards, select the `pepbench-experiments` kernel in your notebook. In VS Code, you can alternatively pick the
interpreter from `.venv/` directly via *Select Kernel → Python Environments*. To remove the registered kernel
again, run:

```bash
uv run poe remove_jupyter
```

### 4. Run the notebooks

Either open the notebooks in your IDE with the kernel from step 3, or start Jupyter directly from the project
environment (no kernel registration needed in this case):

```bash
uv run jupyter lab
```

See the subprojects' `README.md`s for which notebooks to run in which order and which data they expect.

### Updating or adding dependencies

```bash
uv sync                 # after pulling changes to pyproject.toml / uv.lock
uv add <package>        # add a new dependency (updates pyproject.toml and uv.lock)
uv lock --upgrade       # upgrade all dependencies within the constraints in pyproject.toml
```

Commit both `pyproject.toml` and `uv.lock` after changing dependencies. Code formatting and linting are available
via `uv run poe format` and `uv run poe lint`.

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

The `pepbench` library and the algorithm benchmarking in `pep_algorithm_benchmarking/` are described in:

> Richer R, Jorkowitz J, Stühler S, Abel L, Kurz M, Oesten M, Griesshammer SG, Albrecht NC, Küderle A, Ostgathe C,
> Kölpin A, Steigleder T, Rohleder N and Eskofier BM (2025) PEPbench – Open, Reproducible, and Systematic
> Benchmarking of Automated Pre-Ejection Period Extraction Algorithms. *Psychophysiology*, 62(11), e70176.
> https://doi.org/10.1111/psyp.70176

The ML-based B-point extraction work in `b_point_ml_experiments/` is described in:

> Abel L, Stühler S, Steigleder T, Ostgathe C, Rohleder N, Eskofier BM and Richer R (2026) Beat-to-beat aortic
> valve opening detection from impedance cardiography using machine learning. *Frontiers in Digital Health*
> (accepted, in press). Article number and DOI to be added once assigned.
