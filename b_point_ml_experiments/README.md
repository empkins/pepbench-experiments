# B-point ML experiments

Machine-learning-based B-point extraction: training-data preprocessing, `SklearnPipelinePermuter`
hyperparameter search over classical-algorithm-derived features, model evaluation, and the notebook that trained
the pretrained regressor shipped via
`biopsykit.signals.icg.event_extraction.get_b_point_abelstuehler2026_model` /
`BPointExtractionAbelStuehler2026`.

## Layout

- **`src/b_point_ml_experiments/`** — small helper package (`io.py`, `data_handling.py`, `plotting.py`) used by
  the notebooks below; ported out of `pepbench`'s core library since it's only used here (see the top of
  `pepbench-experiments/README.md`).
- **`notebooks/preprocessing/`** — builds the training data (`Preprocessing_B_Point.ipynb`).
- **`notebooks/data_handling/`** — merges permuter output with training data
  (`Build_Estimator_Results_DF.ipynb`, `Merge_Permuter.ipynb`, `Melt_ML_Results.ipynb`).
- **`notebooks/regression/`** — trains models:
  - `B_Point_Regression.ipynb` — the original 3-feature grid search over the `SklearnPipelinePermuter`.
  - `B_Point_Regression_Full_Features.ipynb` — trains on the full 13-feature set (RR-interval + all 12 classical
    B-point algorithms) with the best-performing fixed hyperparameters found by the grid search above; this is
    the notebook that produced the pretrained model biopsykit ships.
  - `B_Point_Regression_Cross_Dataset.ipynb` — cross-dataset generalization check.
  - `hpc_scripts/` — scripts for running the grid search on a SLURM cluster.
- **`notebooks/evaluation/`** — statistics and figures over the trained models (MAE/std, collinearity,
  computational cost, inter-rater agreement, etc.).
- **`models/`, `results/`** — training data, trained model artifacts (`.pkl`/`.skops`), and permuter/metric
  outputs. **Not tracked in git** (see `.gitignore`) - these are large (multiple GB) binary/CSV files produced by
  the notebooks above; regenerate them by re-running the pipeline, or ask whoever ran it for a copy.

## Training data

`notebooks/preprocessing/Preprocessing_B_Point.ipynb` builds `train_data_b_point_rr_interval_include_nan.csv`
(and variants) from labeled ICG data plus the outputs of the 12 classical B-point algorithms in `biopsykit`.

## Reproducing the pretrained model

1. Run `notebooks/preprocessing/Preprocessing_B_Point.ipynb` to (re)build the training data.
2. Run `notebooks/regression/B_Point_Regression_Full_Features.ipynb`. It fits
   `MinMaxScaler` + `RandomForestRegressor(bootstrap=True, criterion='friedman_mse', max_features=0.8,
   n_estimators=250, ccp_alpha=0.001, random_state=0)` via 5-fold `GroupKFold` cross-validation (grouped by
   participant), reporting ~8.1ms pooled out-of-fold MAE, then refits on all data and exports the final pipeline
   with `skops.io.dump` as `b_point_abelstuehler2026_full_features_{rater}.skops`.
3. Upload the resulting `.skops` file as a GitHub Release asset on this repo, update its SHA256 hash in
   `biopsykit`'s `_B_POINT_ABELSTUEHLER2026_REGISTRY` (`src/biopsykit/signals/icg/event_extraction/_pretrained_models.py`),
   and point `_B_POINT_ABELSTUEHLER2026_RELEASE_URL` at this repo's release download URL.

## Citation

This work is described in:

> Abel L, Stühler S, Steigleder T, Ostgathe C, Rohleder N, Eskofier BM and Richer R (2026) Beat-to-beat aortic
> valve opening detection from impedance cardiography using machine learning. *Front. Digit. Health* 8:1944092.
> doi: [10.3389/fdgth.2026.1944092](https://doi.org/10.3389/fdgth.2026.1944092) (accepted, in press)
