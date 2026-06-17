"""Canonical feature table generation for forecasting models.

The Feature Factory is the single entry point that turns a canonical
demand table (plus an optional set of fold cutoffs) into the wide
feature frame the forecasting harness consumes. It is intentionally a
thin orchestrator:

* the five time-dependent feature families (lag/rolling, stockout,
  hierarchy, lifecycle, intermittency) live as adapters in
  :mod:`forecasting.feature_families`;
* the time-independent promo indicator and Fourier features are
  computed inline because they do not depend on the demand history and
  are not fold-aware.

Adding a new family is one file in
:mod:`forecasting.feature_families` and one entry in its registry — no
edits to this module.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from forecasting.contracts import FeatureFlags
from forecasting.feature_families import all_families
from forecasting.feature_families._protocol import FeatureFactoryError


REQUIRED_CANONICAL_COLUMNS = ("series_key", "date", "demand")


# ``FeatureFactoryError`` is re-exported here for backward compatibility
# — it used to live in this module and the public test suite still
# imports it from ``forecasting.feature_factory``.


def validate_canonical_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_CANONICAL_COLUMNS if column not in df.columns]
    if missing:
        raise FeatureFactoryError(f"canonical data is missing required columns: {', '.join(missing)}")


def build_feature_table(
    df: pd.DataFrame,
    flags: FeatureFlags,
    fold_cutoffs: Sequence[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Build the canonical feature table.

    When ``fold_cutoffs`` is provided, time-dependent features (lags and
    rolling windows) are computed independently for each cutoff using only
    the data with ``date <= cutoff`` and then assigned to the rows in the band
    ``(prev_cutoff, current_cutoff]`` (or to all rows, when only one cutoff
    is supplied). Rows with no cutoff ``<=`` their date receive NaN for the
    time-dependent features — preventing walk-forward validation from
    peeking at future rows.

    Time-independent features (promo indicator, Fourier terms) are always
    computed from the full input frame because they do not depend on the
    demand history.
    """
    validate_canonical_columns(df)
    _validate_fold_cutoffs(fold_cutoffs)

    result = df.copy(deep=True)
    try:
        result["date"] = pd.to_datetime(result["date"], errors="raise")
    except Exception as exc:
        raise FeatureFactoryError("canonical date column contains invalid date values") from exc

    result = result.sort_values(["series_key", "date"], kind="mergesort").reset_index(drop=True)

    # Iterate the registered families. Each enabled family writes its
    # own columns into the result frame; the seam is the
    # ``FeatureFamily`` protocol, not a tuple literal.
    for family in all_families():
        if not family.enabled_by(flags):
            continue
        block = family.compute(result, fold_cutoffs)
        result[list(family.columns)] = block[list(family.columns)]

    if flags.use_promo_indicator:
        if "promo" not in result.columns:
            raise FeatureFactoryError("promo column is required when use_promo_indicator is enabled")
        result["promo_indicator"] = _promo_to_indicator(result["promo"])

    if flags.use_fourier:
        if flags.fourier_terms < 1:
            raise FeatureFactoryError("fourier_terms must be at least 1 when use_fourier is enabled")
        if flags.frequency_period is not None and flags.frequency_period < 1:
            raise FeatureFactoryError("frequency_period must be at least 1 when set")
        fourier = _add_fourier_features(result, flags.fourier_terms, flags.frequency_period)
        result[list(fourier.columns)] = fourier

    return result


def _validate_fold_cutoffs(fold_cutoffs: Sequence[pd.Timestamp] | None) -> None:
    if fold_cutoffs is None or len(fold_cutoffs) == 0:
        return
    parsed = [pd.Timestamp(c) for c in fold_cutoffs]
    for previous, current in zip(parsed, parsed[1:]):
        if current <= previous:
            raise FeatureFactoryError(
                "fold_cutoffs must be strictly ascending and contain no duplicates"
            )


def _promo_to_indicator(promo: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(promo) or pd.api.types.is_bool_dtype(promo):
        return promo.fillna(0).ne(0).astype(int)

    numeric = pd.to_numeric(promo, errors="coerce")
    text_true = promo.astype("string").fillna("").str.strip().str.lower().isin({"true", "yes", "y"})
    return (numeric.fillna(0).ne(0) | text_true).astype(int)


def _add_fourier_features(df: pd.DataFrame, terms: int, frequency_period: int | None) -> pd.DataFrame:
    row_number = df.groupby("series_key", sort=False).cumcount() + 1
    if frequency_period is None:
        period = df.groupby("series_key", sort=False)["series_key"].transform("size")
    else:
        period = frequency_period
    cols = {}
    for term in range(1, terms + 1):
        angle = 2 * np.pi * term * row_number / period
        cols[f"sin_{term}"] = np.sin(angle)
        cols[f"cos_{term}"] = np.cos(angle)
    return pd.DataFrame(cols, index=df.index)


__all__ = (
    "FeatureFactoryError",
    "build_feature_table",
    "validate_canonical_columns",
    "REQUIRED_CANONICAL_COLUMNS",
)
