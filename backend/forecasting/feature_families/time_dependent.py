"""Lag and rolling-mean features (the simplest FeatureFamily).

Fold-aware: each band is computed on the prefix only, so a row's
``lag_1`` can never see the demand values from rows after the cutoff.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from forecasting.contracts import FeatureFlags

from forecasting.feature_families._protocol import (
    FeatureFamily,
    apply_family_to_fold_bands,
)


COLUMNS: tuple[str, ...] = ("lag_1", "lag_2", "rolling_mean_4")


def _compute_prefix(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("series_key", sort=False)["demand"]
    return pd.DataFrame(
        {
            "lag_1": grouped.shift(1),
            "lag_2": grouped.shift(2),
            "rolling_mean_4": grouped.transform(
                lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
            ),
        },
        index=df.index,
    )


class TimeDependentFamily:
    """Lag-1, lag-2, and rolling-4 demand features.

    Rows in the trailing "future" band (after the last fold cutoff)
    receive NaN so walk-forward validation cannot peek.
    """

    name: str = "time_dependent"
    columns: tuple[str, ...] = COLUMNS

    def enabled_by(self, flags: FeatureFlags) -> bool:
        return bool(flags.use_lag_features)

    def compute(
        self,
        df: pd.DataFrame,
        fold_cutoffs: Sequence[pd.Timestamp] | None,
    ) -> pd.DataFrame:
        return apply_family_to_fold_bands(
            df,
            fold_cutoffs,
            columns=self.columns,
            compute_prefix=_compute_prefix,
        )


# Protocol check is optional (the registry does the dispatch) but
# having it here documents the contract and surfaces breakage at
# import time.
assert isinstance(TimeDependentFamily(), FeatureFamily)
