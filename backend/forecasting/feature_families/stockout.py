"""Stockout / availability family.

Adds three fold-aware signals derived from the canonical
``stockout_flag`` and ``inventory_qty`` columns:

* ``stockout_rolling_count_4`` — count of stockout-flagged weeks in the
  prior 4 weeks (lag-1 window, so the current week is excluded).
* ``days_since_stockout`` — number of days since the most recent
  stockout event for the same series. NaN when no prior stockout has
  been observed.
* ``inventory_cover_ratio`` — ``inventory_qty`` divided by the prior
  4-week mean demand. NaN when there is no prior demand or when
  inventory is missing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from forecasting.contracts import FeatureFlags

from forecasting.feature_families._protocol import (
    FeatureFactoryError,
    FeatureFamily,
    apply_family_to_fold_bands,
)


COLUMNS: tuple[str, ...] = (
    "stockout_rolling_count_4",
    "days_since_stockout",
    "inventory_cover_ratio",
)


def _coerce_stockout_flag(values: pd.Series) -> pd.Series:
    """Return a 0/1 int Series for stockout_flag values (handles bool/int)."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(int)
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.fillna(0).clip(lower=0).astype(int)


def _days_since_event(
    df: pd.DataFrame, flags: pd.Series, value: int
) -> pd.Series:
    """Return the number of days since the most recent ``flags == value`` event.

    Iterates over each series in ``df["series_key"]`` and computes the
    time delta in days to the most recent prior occurrence of ``value``.
    Returns NaN when no prior occurrence exists. ``flags`` must be
    aligned with ``df`` (same index) and the result is indexed by
    ``df.index``.
    """
    flags_aligned = flags.reindex(df.index)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for _, indices in df.groupby("series_key", sort=False).indices.items():
        sub_dates = df["date"].iloc[indices]
        sub_flags = flags_aligned.iloc[indices]
        # Vectorised per-series: a "last event date" running maximum,
        # then the delta in days.
        last_event = sub_dates.where(sub_flags == value).ffill()
        delta = (sub_dates - last_event).dt.days
        result.iloc[indices] = delta.to_numpy()
    return result


def _compute_prefix(
    df: pd.DataFrame,
    stockout_int: pd.Series,
    inventory: pd.Series,
) -> pd.DataFrame:
    grouped_stockout = stockout_int.groupby(df["series_key"], sort=False)
    grouped_demand = df["demand"].groupby(df["series_key"], sort=False)
    return pd.DataFrame(
        {
            "stockout_rolling_count_4": grouped_stockout.transform(
                lambda s: s.shift(1).rolling(window=4, min_periods=1).sum()
            ),
            "days_since_stockout": _days_since_event(df, stockout_int, value=1),
            "inventory_cover_ratio": (
                inventory.reindex(df.index)
                / grouped_demand.transform(
                    lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
                ).replace(0, np.nan)
            ),
        },
        index=df.index,
    )


class StockoutFamily:
    """Stockout / availability features; fold-aware."""

    name: str = "stockout"
    columns: tuple[str, ...] = COLUMNS

    def enabled_by(self, flags: FeatureFlags) -> bool:
        return bool(flags.use_stockout_features)

    def compute(
        self,
        df: pd.DataFrame,
        fold_cutoffs: Sequence[pd.Timestamp] | None,
    ) -> pd.DataFrame:
        if "stockout_flag" not in df.columns:
            raise FeatureFactoryError(
                "stockout_flag column is required when use_stockout_features is enabled"
            )
        if "inventory_qty" not in df.columns:
            raise FeatureFactoryError(
                "inventory_qty column is required when use_stockout_features is enabled"
            )

        stockout_int = _coerce_stockout_flag(df["stockout_flag"])
        inventory = pd.to_numeric(df["inventory_qty"], errors="coerce").astype(float)
        return apply_family_to_fold_bands(
            df,
            fold_cutoffs,
            columns=self.columns,
            compute_prefix=lambda prefix: _compute_prefix(prefix, stockout_int, inventory),
        )


assert isinstance(StockoutFamily(), FeatureFamily)
