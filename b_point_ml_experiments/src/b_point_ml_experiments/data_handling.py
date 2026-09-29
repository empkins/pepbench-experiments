"""Building and summarizing ML-estimator result dataframes for the B-point ML-regression experiments.

Ported verbatim from the ML-specific functions in ``pepbench.data_handling._data_handling``
(``build_ml_results_df``, ``merge_ml_result_dfs``, ``describe_ml_results_df``) - see this package's top-level
docstring for why they live here instead of in ``pepbench`` itself.
"""

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "build_ml_results_df",
    "describe_ml_results_df",
    "merge_ml_result_dfs",
]

_ml_b_point_algo_data_map = {
    "RR-Interval-Include-Nan": "rr-interval/rater_01/train_data_b_point_rr_interval_include_nan",
    "RR-Interval-Median-Imputed": "rr-interval/rater_01/train_data_b_point_rr_interval_median_imputed",
}

_ml_q_peak_algo_data_map = {
    "Without-RR-Interval": "without-rr-interval/train_data_q_peak",
    "Without-RR-Interval-Include-Nan": "without-rr-interval/train_data_q_peak_include_nan",
    "Without-RR-Interval-Median-Imputed": "without-rr-interval/train_data_q_peak_median_imputed",
    "RR-Interval": "rr-interval/train_data_q_peak_rr_interval",
    "RR-Interval-Include-Nan": "rr-interval/train_data_q_peak_rr_interval_include_nan",
    "RR-Interval-Median-Imputed": "rr-interval/train_data_q_peak_rr_interval_median_imputed",
}

_metric_dict = {
    "abs_rel_error": "MARE",
    "abs_error": "MAE",
    "rel_error": "MRE",
    "error": "ME",
}


def build_ml_results_df(data_path: Path, permuter_path: Path, event: str):
    """Add the predictions of the ML-Estimators to the corresponding training data.

    Parameters
    ----------
    data_path: Path
        Path to the directory containing the training data.
    permuter_path: Path
        Path to the directory containing the merged permuter.
    event: str
        Specifies whether the b_point or q_peak dataframe should be build.
    Returns: None
        The function does not return anything. It rather adds the predictions of the ML-Estimators to the corresponding
        training data and saves them in the same directory with the suffix _ml_results.
    -------

    """
    print(f"data path: {data_path}")
    print(f"permuter path: {permuter_path}")
    if event == "b_point":
        algo_dict = _ml_b_point_algo_data_map
    elif event == "q_peak":
        algo_dict = _ml_q_peak_algo_data_map
    else:
        raise (KeyError("event must be 'b_point' or 'q_peak'"))

    merged_permuter = pd.read_json(
        permuter_path.joinpath(f"merged_{event}_permuter_paper.json"), orient="records", lines=True
    ).set_index(["pipeline_scaler", "pipeline_reduce_dim", "pipeline_clf"])

    for key, value in algo_dict.items():
        permuter = merged_permuter[merged_permuter["Dataset"] == key]
        data = pd.read_csv(data_path.joinpath(f"{value}.csv")).drop(columns=["Unnamed: 0"])
        for index, row in permuter[["test_indices", "predicted_labels", "EstimatorID"]].iterrows():
            data[row["EstimatorID"]] = np.nan
            data.loc[row["test_indices"], row["EstimatorID"]] = row["predicted_labels"]
        data.to_csv(data_path.joinpath(f"{value}_ml_results.csv"), index=True)


def merge_ml_result_dfs(data_path: Path, master_df: pd.DataFrame, event: str):
    """Combine the ML-Predictions + training data dataframes in one dataframe.

    Parameters
    ----------
    data_path: Path
        Path to the directory containing the training data.
    master_df: Path
        Dataframe that contains the most rows (*_include_nan)
    event: str
        Specifies whether the b_point or q_peak dataframe should be build.
    Returns: pd.DataFrame
        Dataframe that contains the ML-Predictions of all experiments.
            - dropped missing values
            - median imputed missing values
            - kept missing values
    -------

    """
    if event == "b_point":
        algo_dict = _ml_b_point_algo_data_map
    elif event == "q_peak":
        algo_dict = _ml_q_peak_algo_data_map
    else:
        raise (KeyError("event must be 'b_point' or 'q_peak'"))

    old_b_point_algos = [
        "arbol2017-isoelectric-crossings",
        "arbol2017-second-derivative",
        "arbol2017-third-derivative",
        "debski1993-second-derivative",
        "drost2022",
        "forouzanfar2018",
        "lozano2007-linear-regression",
        "lozano2007-quadratic-regression",
        "sherwood1990",
        "stern1985",
        "pale2021",
        "miljkovic2022",
    ]
    old_q_peak_algos = [
        "forounzafar2018",
        "martinez2004",
        "vanlien2013-32-ms",
        "vanlien2013-34-ms",
        "vanlien2013-36-ms",
        "vanlien2013-38-ms",
        "vanlien2013-40-ms",
        "vanlien2013-42-ms",
    ]

    old_q_peak_algos_rr = old_q_peak_algos.copy()
    old_q_peak_algos_rr.append("rr_interval_ms_estimated")

    merged_df = master_df.copy().reset_index(level=0)
    for key, value in algo_dict.items():
        print(data_path.joinpath(f"{value}_ml_results.csv"))
        if event == "b_point":
            data = pd.read_csv(data_path.joinpath(f"{value}_ml_results.csv"), index_col=[0, 1, 2, 3, 4, 5]).drop(
                columns=old_b_point_algos
            )
        elif event == "q_peak":
            if value.startswith("rr-interval"):
                data = pd.read_csv(data_path.joinpath(f"{value}_ml_results.csv"), index_col=[0, 1, 2, 3, 4, 5]).drop(
                    columns=old_q_peak_algos_rr
                )
            else:
                data = pd.read_csv(data_path.joinpath(f"{value}_ml_results.csv"), index_col=[0, 1, 2, 3, 4, 5]).drop(
                    columns=old_q_peak_algos
                )
        else:
            raise (KeyError("event must be 'b_point' or 'q_peak'"))
        data = data.reset_index(level=0)
        for column in data.columns:
            if column in merged_df.columns:
                data = data.drop(columns=column)
                print("Detected duplicate column: ", column)
        merged_df = pd.merge(merged_df, data, left_index=True, right_index=True, how="left")
    return merged_df


def describe_ml_results_df(data: pd.DataFrame, ascending: bool | None = True) -> pd.DataFrame:
    """Compute mean and std on the ml results error metrics.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the error, relative error, absolute error, and absolute relative error of all Estimators per sample.
    ascending: bool, optional
        Specifies whether the data should be sorted in an ascending order.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the MAE, ME, MARE, and MRE and the corresponding std in [ms].
    """
    summarized_results = data.describe().T.drop(columns=["min", "max", "25%", "50%", "75%"])
    new_index = summarized_results.index.to_list()

    for old_suffix, new_suffix in _metric_dict.items():
        new_index = [s.replace(old_suffix, new_suffix) for s in new_index]

    summarized_results.index = new_index
    summarized_results = summarized_results.reset_index()

    pattern = r"(.*)_(ME|MRE|MAE|MARE)$"
    if "martinez2004_ME" in summarized_results["index"].unique():
        algorithm = "Q-Peak Algorithm"
        summarized_results[[algorithm, "metric_type"]] = summarized_results["index"].str.extract(pattern)
    else:
        algorithm = "B-Point Algorithm"
        summarized_results[[algorithm, "metric_type"]] = summarized_results["index"].str.extract(pattern)

    summarized_results_pivot = summarized_results.pivot(index=algorithm, columns="metric_type", values=["mean", "std"])

    relative_error_map = [("mean", "MARE"), ("mean", "MRE"), ("std", "MARE"), ("std", "MRE")]
    summarized_results_pivot[relative_error_map] = summarized_results_pivot[relative_error_map] * 100
    if "index" in summarized_results_pivot.index:
        summarized_results_pivot = summarized_results_pivot.drop(index="index")

    if ascending:
        summarized_results_pivot = summarized_results_pivot.sort_values(by=("mean", "MAE"), ascending=True)
    else:
        summarized_results_pivot = summarized_results_pivot.sort_values(by=("mean", "MAE"), ascending=False)

    return summarized_results_pivot
