"""Loading and error-computation helpers for the B-point ML-regression experiments.

Ported verbatim from ``pepbench.io._ml_helper`` and the ``SklearnPipelinePermuter`` helpers in ``pepbench.io._io``
(``get_best_pipeline_results``, ``get_best_estimator``, ``get_pipeline_steps``) - see this package's top-level
docstring for why.
"""

from pathlib import Path
from typing import Any, Optional, TypeVar

import numpy as np
import pandas as pd
from biopsykit.classification.model_selection import SklearnPipelinePermuter

path_t = TypeVar("path_t", str, Path)  # pylint:disable=invalid-name

__all__ = [
    "compute_abs_error",
    "compute_error",
    "compute_mae_std_from_metric_summary",
    "compute_mae_std_from_permuter",
    "get_best_estimator",
    "get_best_pipeline_results",
    "get_pipeline_steps",
    "impute_missing_values",
    "load_preprocessed_training_data",
]


def load_preprocessed_training_data(
    file_path: path_t,
    algorithms: Optional[pd.MultiIndex] = None,
    index_col: Optional[list] = None,
    include_reference: Optional[bool] = True,
    event: Optional[str] = "b-point",
):
    """
    Load preprocessed training data for evaluation of the model output.

    Parameters
    ----------
    file_path: str or :class:`pathlib.Path
        The file path containing the preprocessed training data.
    algorithms: pd.MultiIndex
        The index specifying the algorihtms that should be returned for comparision.
    index_col: list
        List containing the columns that should serve as the index of the dataframe.
    include_reference: bool, optional
        ``True`` to include the reference B-Points, ``False`` to exclude the reference B-Points. Default: ``True``
    event: str, optional
        Event that is represented in the training data. Can be either ``'b-point'`` or ``'q_wave'``. Default: ``'b-point'``

    Returns
    -------
    pd.DataFrame
        The B-Points extracted by the algorithms in ms.
    """
    supported_events = ["b-point", "q_wave"]
    assert file_path.is_file(), f"File '{file_path}' does not exist!"
    assert event in supported_events, f"Event '{event}' is not supported. Supported events: {supported_events}.\n"

    data = pd.read_csv(file_path, index_col=index_col)

    if algorithms is not None:
        algos = None
        if event == "b-point":
            if "outlier_correction_algorithm" in data.columns:
                algos = algorithms.get_level_values("b_point_algorithm") + "_" + algorithms.get_level_values(
                    "outlier_correction_algorithm"
                )
            else:
                algos = algorithms.get_level_values("b_point_algorithm")
            if include_reference:
                return data[["b_point_sample_reference"] + list(algos.values)]
            else:
                return data[algos.values]
        elif event == "q_wave":
            algos = algorithms.get_level_values("q_wave_algorithm")
            if include_reference:
                return data[["q_wave_onset_sample_reference"] + list(algos.values)]
            else:
                return data[algos.values]
        else:
            raise KeyError(f"Event: '{event}' is not supported. Supported events: {supported_events}.\n")
    else:
        return data


def compute_mae_std_from_permuter(
    pipeline_permuter: SklearnPipelinePermuter,
):
    """
    Compute mae and std from permuter.
    Parameters
    ----------
    input_data: SklearnPipelinePermuter
        Pipeline permuter containing the regression results.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the mae, std, true_labels, predicted_labels, and the absolute_error of the predictions.
    """
    if pipeline_permuter is not None:
        permuter_metrics = pd.DataFrame(
            data=pipeline_permuter.metric_summary()[["true_labels", "predicted_labels"]],
            columns=["mae", "std", "true_labels", "predicted_labels"],
            index=pipeline_permuter.metric_summary().index,
        )

    permuter_metrics["absolute_error"] = np.abs(permuter_metrics["true_labels"] - permuter_metrics["predicted_labels"])

    for index in permuter_metrics.index:
        permuter_metrics.at[index, "mae"] = np.mean(permuter_metrics.loc[index]["absolute_error"])
        permuter_metrics.at[index, "std"] = np.std(permuter_metrics.loc[index]["absolute_error"])
    return permuter_metrics


def compute_mae_std_from_metric_summary(
    pipeline_permuter: pd.DataFrame,
):
    """
    Compute mae and std from permuter.
    Parameters
    ----------
    input_data: pd.DataFrame
        DataFrame containing the regression results.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the mae, std, true_labels, predicted_labels, and the absolute_error of the predictions.
    """
    permuter_metrics = pd.DataFrame(
        data=pipeline_permuter[["true_labels", "predicted_labels"]],
        columns=["mae", "std", "true_labels", "predicted_labels"],
        index=pipeline_permuter.index,
    )

    permuter_metrics["absolute_error"] = np.abs(permuter_metrics["true_labels"] - permuter_metrics["predicted_labels"])

    for index in permuter_metrics.index:
        permuter_metrics.at[index, "mae"] = np.mean(permuter_metrics.loc[index]["absolute_error"])
        permuter_metrics.at[index, "std"] = np.std(permuter_metrics.loc[index]["absolute_error"])
    return permuter_metrics


def compute_abs_error(
    predicted_labels: np.array,
    true_labels: np.array,
):
    """
    Compute the absolute error of the B-Point extraction algorithms present in the Datframe against the labeled reference data.

    Parameters
    ----------
    input data: pd.DataFrame
        Dataframe containing the automatically extracted B-Point locations and the labeled reference data
    reference: pd.Series
        Series containing the column against which the absolute error should be calculated

    Returns
    -------
    pd.DataFrame
        The absolute errors of the extracted B-Point locations against the labeled reference data
    """
    abs_error = np.abs(predicted_labels - true_labels)
    return abs_error


def compute_error(
    input_data: pd.DataFrame,
    reference: pd.Series,
):
    """
    Compute the error of the B-Point extraction algorithms present in the Datframe against the labeled reference data.

    Parameters
    ----------
    input data: pd.DataFrame
        Dataframe containing the automatically extracted B-Point locations and the labeled reference data
    reference: pd.Series
        Series containing the column against which the absolute error should be calculated

    Returns
    -------
    pd.DataFrame
        The errors of the extracted B-Point locations against the labeled reference data
    """
    if reference.name in input_data.columns:
        input_data = input_data.drop(columns=reference.name)

    error = input_data.subtract(reference, axis=0) * -1
    return error


def impute_missing_values(input_data: pd.DataFrame, mode: Optional[str] = "median"):
    """
    Impute missing values in the training data on sample- (row-) level.
    Parameters
    ----------
    input_data: pd.DataFrame
        Dataframe containing the training data that should be imputed.
    mode: str
        Technique that should be used to impute missing values. Can be ``'median'`` or ``'mean'``. Default: ``'median'``.
    Returns
    -------
    pd.DataFrame
        Dataframe containing the training data with missing values imputed.
    """
    if mode == "median":
        imputation_values = np.nanmedian(input_data, axis=1)
    elif mode == "mean":
        imputation_values = np.round(np.nanmean(input_data, axis=1))
    else:
        raise KeyError(f"Imputation mode '{mode}' is not supported. Supported modes are 'median' and 'mean'.")

    for sample in range(input_data.shape[0]):
        input_data.iloc[sample, np.isnan(input_data.iloc[sample, :].values)] = imputation_values[sample]

    return input_data


def get_best_pipeline_results(
    pipeline_permuter: SklearnPipelinePermuter,
    metric: str | None = "mean_absolute_error",
) -> pd.DataFrame:
    """
    Get the best performing algorithm from the metric summary of the SklearnPipelinePermuter.

    Parameters
    ----------
    pipeline_permuter: class biopsykit.classification.model_selection.SklearnPipelinePermuter
        The pipeline permuter object containing the prediction results and metric performances.
    metric: str, optional
        The metric that was used for scoring in the SklearnPipelinePermuter. Default ``"mean_absolute_error"``

    Returns
    -------
    pd.DataFrame
        The prediction and metric performance results of the best estimator.

    """
    if metric == "mean_absolute_error":
        sklearn_metric = "mean_test_neg_mean_absolute_error"
    else:
        raise KeyError("Specified metric is not implemented yet!")

    assert sklearn_metric in pipeline_permuter.metric_summary().columns, (
        "Specified metric is not used in the pipeline permuter!"
    )
    return (
        pipeline_permuter.metric_summary()
        .iloc[pipeline_permuter.metric_summary()["mean_test_neg_mean_absolute_error"].argmin()]
        .to_frame()
        .T
    )


def get_best_estimator(
    pipeline_permuter: SklearnPipelinePermuter,
    metric: str | None = "mean_absolute_error",
) -> tuple[Any, tuple[str, ...]]:
    """
    Get the best estimator object and its underlying pipeline combination.

    Parameters
    ----------
    pipeline_permuter: class biopsykit.classification.model_selection.SklearnPipelinePermuter
        The pipeline permuter object containing the prediction results and metric performances.
    metric: str, optional
        The metric that was used for scoring in the SklearnPipelinePermuter. Default ``"mean_absolute_error"``

    Returns
    -------
    tuple of _PipelineWrapper and tuple
        The results as a tuple containing of the biopsykit.classification.utils._PipelineWrapper object and a tuple of
        strings containing the underlying pipeline combination.
    """
    if metric == "mean_absolute_error":
        sklearn_metric = "mean_test_neg_mean_absolute_error"
    else:
        raise KeyError("Specified metric is not implemented yet!")

    assert sklearn_metric in pipeline_permuter.metric_summary().columns, (
        "Specified metric is not used in the pipeline permuter!"
    )
    best_estimator_name = (
        pipeline_permuter.metric_summary().iloc[pipeline_permuter.metric_summary()[sklearn_metric].argmin()].name
    )
    return pipeline_permuter.best_estimator_summary().loc[best_estimator_name].iloc[0], best_estimator_name


def get_pipeline_steps(
    pipeline_permuter: SklearnPipelinePermuter,
    input_data: pd.DataFrame,
    metric: str | None = "mean_absolute_error",
    step: str | None = "reduce_dim",
    scaler: bool | None = True,
) -> list[list[str]]:
    """
    Gain further insights in the components of your model. E.g. recieve the features selected by the model.

    Parameters
    ----------
    pipeline_permuter: class biopsykit.classification.model_selection.SklearnPipelinePermuter
        The pipeline permuter object containing the prediction results and metric performances.
    input_data: pd.DataFrame
        The data that was used to train the models.
    metric: str, optional
        The metric that was used for scoring in the SklearnPipelinePermuter. Default ``"mean_absolute_error"``
    step: str, optional
        The step of interest. Default ``"reduce_dim"``
    scaler: str, optional
        Specifies whether scaling was considered in the model. Default ``True``

    Results
    -------
    list
        List of selected features for each pipeline in the best_estimator object.

    """
    best_estimator, best_estimator_name = get_best_estimator(pipeline_permuter=pipeline_permuter, metric=metric)
    reduce_dim_model = ""
    selected_features = []
    selected_feature_mask = None

    if scaler:
        if step == "reduce_dim":
            reduce_dim_model = best_estimator_name[1]
        else:
            raise KeyError("Specified step is not supported yet!")
    elif step == "reduce_dim":
        reduce_dim_model = best_estimator_name[0]
    else:
        raise KeyError("Specified step is not supported yet!")

    for pipeline in best_estimator.pipeline:
        if reduce_dim_model in {"RFE", "SelectKBest"}:
            selected_feature_mask = pipeline.named_steps["reduce_dim"].get_support()
        else:
            raise KeyError("The model used in your pipeline is not supported yet!")
        selected_features.append(input_data.columns[selected_feature_mask].to_list())
    return selected_features
