"""Helper code for the B-point ML-regression experiments.

Ported out of ``pepbench``'s core library (``pepbench.io._ml_helper``, the ``SklearnPipelinePermuter`` helpers in
``pepbench.io._io``, ``pepbench.plotting.ml_results``, ``pepbench.export._latex.create_ml_algo_performance_table``,
and the ML-specific functions in ``pepbench.data_handling._data_handling``) because it is only ever used by the
notebooks in this experiment, not by ``pepbench`` itself or by anything downstream of it. Keeping it here means the
``pepbench`` library doesn't need ``shap`` as a hard dependency just for this one experiment.
"""

from b_point_ml_experiments.data_handling import (
    build_ml_results_df,
    describe_ml_results_df,
    merge_ml_result_dfs,
)
from b_point_ml_experiments.export import create_ml_algo_performance_table
from b_point_ml_experiments.io import (
    compute_abs_error,
    compute_error,
    compute_mae_std_from_metric_summary,
    compute_mae_std_from_permuter,
    get_best_estimator,
    get_best_pipeline_results,
    get_pipeline_steps,
    impute_missing_values,
    load_preprocessed_training_data,
)

__all__ = [
    "build_ml_results_df",
    "compute_abs_error",
    "compute_error",
    "compute_mae_std_from_metric_summary",
    "compute_mae_std_from_permuter",
    "create_ml_algo_performance_table",
    "describe_ml_results_df",
    "get_best_estimator",
    "get_best_pipeline_results",
    "get_pipeline_steps",
    "impute_missing_values",
    "load_preprocessed_training_data",
    "merge_ml_result_dfs",
]
