"""Intermittency family: rolling ADI, CV², trailing zero-run length.

Three fold-aware signals over a rolling 8-week window (lag-1, so the
current week is excluded):

* ``rolling_adi_8`` — Average Demand Interval, mean inter-demand
  interval (in weeks) over the prior 8 weeks, computed as
  ``window / max(weeks_with_demand, 1)``. 1.0 means the series demanded
  every week; 8.0 means the series demanded once in 8 weeks.
* ``rolling_cv2_8`` — squared coefficient of variation of demand
  values in the prior 8 weeks (variance / mean²). NaN when the mean is
  zero (the whole window is zero — the variance is zero but the CV² is
  undefined).
* ``trailing_zero_run`` — length of the current run of consecutive
  zero-demand weeks ending at the most recent observation *before* the
  current row.
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
    "rolling_adi_8",
    "rolling_cv2_8",
    "trailing_zero_run",
)


def _intermittency_block(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the intermittency block for a single series-key frame.

    Assumes the input has a single set of series, sorted by date.
    """
    out = pd.DataFrame(index=df.index, columns=list(COLUMNS), dtype=float)
    for _, indices in df.groupby("series_key", sort=False).indices.items():
        sub = df["demand"].iloc[indices]
        # Window of last 8 weeks (lag-1 so the current week is excluded).
        # min_periods=1 means we get a value as soon as there is one
        # prior observation.
        shifted = sub.shift(1)
        recent = shifted.rolling(window=8, min_periods=1)
        count = recent.count()
        nonzero = recent.apply(lambda w: float((w > 0).sum()), raw=True)
        mean = recent.mean()
        std = recent.std()
        # ADI: weeks-in-window / count-of-nonzero-weeks. NaN-safe via
        # the count == 0 branch (returns NaN).
        adi = np.where(nonzero > 0, 8.0 / nonzero, np.nan)
        # CV²: std² / mean². NaN when mean is zero (degenerate).
        cv2 = np.where(mean > 0, (std ** 2) / (mean ** 2), np.nan)
        # Trailing zero-run: the number of consecutive prior zero weeks
        # ending immediately before the current row.
        is_zero = (shifted.fillna(0) == 0).astype(int)
        # group by cumsum of non-zero: each block of zeros gets its own
        # group, so a cumcount within the block gives the run length.
        run = is_zero.groupby((is_zero == 0).cumsum()).cumcount()
        run = run.where(is_zero == 1, 0)
        out.loc[sub.index, "rolling_adi_8"] = adi
        out.loc[sub.index, "rolling_cv2_8"] = cv2
        out.loc[sub.index, "trailing_zero_run"] = run.to_numpy()
    return out


class IntermittencyFamily:
    """Rolling ADI / CV² / trailing zero-run features; fold-aware."""

    name: str = "intermittency"
    columns: tuple[str, ...] = COLUMNS

    def enabled_by(self, flags: FeatureFlags) -> bool:
        return bool(flags.use_intermittency_features)

    def compute(
        self,
        df: pd.DataFrame,
        fold_cutoffs: Sequence[pd.Timestamp] | None,
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index, columns=list(self.columns), dtype=float)
        if not fold_cutoffs:
            for column, values in _intermittency_block(df).items():
                out[column] = values
            return out
        for band_mask, prefix in iter_fold_bands(df, fold_cutoffs):
            if prefix is None:
                continue
            block = _intermittency_block(prefix)
            for column in self.columns:
                out.loc[band_mask, column] = (
                    block[column].reindex(df.index).loc[band_mask].to_numpy()
                )
        return out


assert isinstance(IntermittencyFamily(), FeatureFamily)
