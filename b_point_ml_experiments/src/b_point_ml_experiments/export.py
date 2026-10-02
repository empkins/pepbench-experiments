"""LaTeX-table export helpers for the B-point ML-regression experiments.

Ported verbatim from ``pepbench.export._latex.create_ml_algo_performance_table`` - see this package's top-level
docstring for why.
"""

from collections.abc import Sequence

import pandas as pd

__all__ = [
    "create_ml_algo_performance_table",
]


def create_ml_algo_performance_table(
    data: pd.DataFrame, algos: Sequence[str] | None = None, n_algos: int | None = None, ascending: bool | None = True
) -> pd.DataFrame:
    """Create a table with the MAE, ME, MRE, MARE and the corresponding std for all Estimators.

    Parameters
    ----------
    data: pd.DataFrame
        Output of the describe_ml_results_df function or data in similar format.
    algos: list of [str], optional
        Estimators that should be added to the table.
    n_algos: int, optional
        Amount of estimator results that should be returned.
    ascending: bool, optional
        Specifies whether the data should be sorted in an ascending order.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the specified data.
    """
    table = data.copy()
    columns = []
    for level1, level2 in data.columns:
        if level1 == "mean":
            columns.append(level2)
        elif level1 == "std":
            columns.append(f"{level2} std")
    table.columns = columns

    table = table.round(2)
    table = table.sort_values(by="MAE", ascending=bool(ascending))

    for metric in ["MAE", "ME", "MRE", "MARE"]:
        table[metric] = table.apply(lambda row, metric=metric: f"{row[metric]!s} ± {row[f'{metric} std']!s}", axis=1)
    table = table.drop(columns=["MAE std", "ME std", "MRE std", "MARE std"])
    if algos is not None:
        return table.loc[algos]
    if n_algos is not None:
        return table.head(n_algos)
    return table
