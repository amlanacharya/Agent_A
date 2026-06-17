"""Lifecycle / cold-start family.

Three fold-aware signals describing how much history the model would
have seen for a given row, measured *as of* the fold cutoff (or the
row's own date when there is no fold):

* ``history_length`` — number of prior observations for this series
  visible to the model.
* ``days_since_first_obs`` — number of days from the first observation
  of the series to the fold cutoff (or row date).
* ``cold_start_flag`` — 1 if the row is one of the first
  ``_COLD_START_THRESHOLD`` observations for the series under the
  fold-aware view, else 0.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from forecasting.contracts import FeatureFlags

from forecasting.feature_families._protocol import (
    FeatureFamily,
    iter_fold_bands,
)


COLUMNS: tuple[str, ...] = (
    "history_length",
    "days_since_first_obs",
    "cold_start_flag",
)


_COLD_START_THRESHOLD = 4


def _lifecycle_block(frame: pd.DataFrame, out: pd.DataFrame) -> None:
    """Fill ``out`` with lifecycle stats for every series in ``frame``."""
    for _, indices in frame.groupby("series_key", sort=False).indices.items():
        sub_dates = frame["date"].iloc[indices]
        first_obs = sub_dates.iloc[0]
        history_length = pd.Series(np.arange(len(sub_dates)), index=sub_dates.index, dtype=float)
        days_since_first = (sub_dates - first_obs).dt.days.astype(float)
        cold_start = (history_length < _COLD_START_THRESHOLD).astype(float)
        out.loc[sub_dates.index, "history_length"] = history_length.to_numpy()
        out.loc[sub_dates.index, "days_since_first_obs"] = days_since_first.to_numpy()
        out.loc[sub_dates.index, "cold_start_flag"] = cold_start.to_numpy()


class LifecycleFamily:
    """Lifecycle / cold-start features; fold-aware."""

    name: str = "lifecycle"
    columns: tuple[str, ...] = COLUMNS

    def enabled_by(self, flags: FeatureFlags) -> bool:
        return bool(flags.use_lifecycle_features)

    def compute(
        self,
        df: pd.DataFrame,
        fold_cutoffs: Sequence[pd.Timestamp] | None,
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index, columns=list(self.columns), dtype=float)
        if not fold_cutoffs:
            _lifecycle_block(df, out)
            return out
        for band_mask, prefix in iter_fold_bands(df, fold_cutoffs):
            if prefix is None:
                continue
            _lifecycle_block(prefix, out)
        return out


assert isinstance(LifecycleFamily(), FeatureFamily)
