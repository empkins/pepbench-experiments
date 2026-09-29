"""Plotting helpers for ML-estimator results.

Ported from ``pepbench.plotting.ml_results``, which was dead code in the core library (never imported by
``pepbench.plotting``'s public API). As found there, ``boxplot_algorithm_performance`` was left unfinished - its
body is a stub here (rather than a bare, un-parseable function signature as in the original) so this module can
at least be imported; treat it as a starting point to finish, not working code.
"""

from collections.abc import Callable, Sequence
from typing import Any

import biopsykit as bp
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from fau_colors import cmaps, colors_all
from matplotlib import pyplot as plt
from pepbench.data_handling import get_data_for_algo
from pepbench.data_handling._data_handling import get_performance_metric, get_reference_data
from pepbench.plotting._base_plotting import _plot_blandaltman, _plot_paired
from pepbench.plotting._utils import _get_fig_ax, _get_fig_axs, _remove_duplicate_legend_entries
from pepbench.utils._rename_maps import (
    _algo_level_mapping,
    _algorithm_mapping,
    _metric_mapping,
    _xlabel_mapping,
    _ylabel_mapping,
)

__all__ = [
    "boxplot_algorithm_performance",
]

best_ml_estimators = {
    "SScaler-SFM-SVR-RR": ("StandardScaler", "SelectFromModel", "SVR"),
    "SScaler-SKB-RFR": ("StandardScaler", "SelectKBest", "RandomForestRegressor"),
    "SScaler-SFM-RFR": ("StandardScaler", "SelectFromModel", "RandomForestRegressor"),
    "SScaler-SFM-SVR": ("StandardScaler", "SelectFromModel", "SVR"),
}


def boxplot_algorithm_performance() -> tuple[plt.Figure, plt.Axes]:
    """Plot a boxplot comparing ML-estimator performance across algorithms.

    Unfinished in the original ``pepbench`` module this was ported from - never had a parameter list or a body.
    """
    raise NotImplementedError(
        "boxplot_algorithm_performance was never finished in the original pepbench.plotting.ml_results module - "
        "this port preserves that as a clearly-marked TODO instead of silently guessing an implementation."
    )
