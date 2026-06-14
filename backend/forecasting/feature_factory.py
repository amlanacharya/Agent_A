"""Canonical feature table generation for forecasting models."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from forecasting.contracts import FeatureFlags


REQUIRED_CANONICAL_COLUMNS = ("series_key", "date", "demand")
TIME_DEPENDENT_FEATURE_COLUMNS = ("lag_1", "lag_2", "rolling_mean_4")


class FeatureFactoryError(ValueError):
    """Raised when canonical feature generation cannot proceed."""


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

    if flags.use_lag_features:
        time_dependent = _compute_time_dependent_features(result, fold_cutoffs)
        for column in TIME_DEPENDENT_FEATURE_COLUMNS:
            result[column] = time_dependent[column]

    if flags.use_promo_indicator:
        if "promo" not in result.columns:
            raise FeatureFactoryError("promo column is required when use_promo_indicator is enabled")
        result["promo_indicator"] = _promo_to_indicator(result["promo"])

    if flags.use_fourier:
        if flags.fourier_terms < 1:
            raise FeatureFactoryError("fourier_terms must be at least 1 when use_fourier is enabled")
        if flags.frequency_period is not None and flags.frequency_period < 1:
            raise FeatureFactoryError("frequency_period must be at least 1 when set")
        _add_fourier_features(result, flags.fourier_terms, flags.frequency_period)

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


def _compute_time_dependent_features(
    df: pd.DataFrame, fold_cutoffs: Sequence[pd.Timestamp] | None
) -> pd.DataFrame:
    """Compute lag and rolling features, optionally fold-aware.

    Returns a DataFrame indexed the same way as ``df`` (after sort) with the
    three time-dependent feature columns. Rows that have no fold cutoff
    ``<=`` their date are NaN'd out (no leakage).
    """
    if not fold_cutoffs:
        grouped_demand = df.groupby("series_key", sort=False)["demand"]
        return pd.DataFrame(
            {
                "lag_1": grouped_demand.shift(1),
                "lag_2": grouped_demand.shift(2),
                "rolling_mean_4": grouped_demand.transform(
                    lambda series: series.shift(1).rolling(window=4, min_periods=1).mean()
                ),
            },
            index=df.index,
        )

    cutoffs = sorted(pd.Timestamp(c) for c in fold_cutoffs)
    bands: list[tuple[pd.Timestamp | None, pd.Timestamp]] = [(None, cutoffs[0])]
    for previous, current in zip(cutoffs, cutoffs[1:]):
        bands.append((previous, current))
    bands.append((cutoffs[-1], None))

    out = pd.DataFrame(index=df.index, columns=list(TIME_DEPENDENT_FEATURE_COLUMNS), dtype=float)
    for lower_inclusive, upper in bands:
        # Define the row positions that belong to this band.
        if upper is None:
            band_mask = df["date"] > lower_inclusive  # type: ignore[operator]
        else:
            band_mask = df["date"] <= upper
        if lower_inclusive is not None:
            band_mask &= df["date"] > lower_inclusive
        if not band_mask.any():
            continue

        # Trailing "future" band (no upper cutoff) - rows here are strictly
        # after every cutoff, so no fold-aware features are defined. Leave
        # them NaN; the forecast harness will compute inference-time features
        # for these rows out of band.
        if upper is None:
            continue

        # Compute features on the PREFIX (rows available up to and including
        # the cutoff). This is the fold-aware view: a row in this band may
        # only see demand from rows at or before ``upper``.
        prefix_mask = df["date"] <= upper
        prefix = df.loc[prefix_mask]
        prefix_grouped = prefix.groupby("series_key", sort=False)["demand"]
        prefix_features = pd.DataFrame(
            {
                "lag_1": prefix_grouped.shift(1),
                "lag_2": prefix_grouped.shift(2),
                "rolling_mean_4": prefix_grouped.transform(
                    lambda series: series.shift(1).rolling(window=4, min_periods=1).mean()
                ),
            },
            index=prefix.index,
        )

        # Reindex the prefix features onto the full df index so the band
        # rows line up; rows outside the prefix keep NaN automatically.
        aligned = prefix_features.reindex(df.index)
        out.loc[band_mask, "lag_1"] = aligned.loc[band_mask, "lag_1"].to_numpy()
        out.loc[band_mask, "lag_2"] = aligned.loc[band_mask, "lag_2"].to_numpy()
        out.loc[band_mask, "rolling_mean_4"] = aligned.loc[band_mask, "rolling_mean_4"].to_numpy()
    return out


def _promo_to_indicator(promo: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(promo) or pd.api.types.is_bool_dtype(promo):
        return promo.fillna(0).ne(0).astype(int)

    numeric = pd.to_numeric(promo, errors="coerce")
    text_true = promo.astype("string").fillna("").str.strip().str.lower().isin({"true", "yes", "y"})
    return (numeric.fillna(0).ne(0) | text_true).astype(int)


def _add_fourier_features(df: pd.DataFrame, terms: int, frequency_period: int | None) -> None:
    row_number = df.groupby("series_key", sort=False).cumcount() + 1
    if frequency_period is None:
        period = df.groupby("series_key", sort=False)["series_key"].transform("size")
    else:
        period = frequency_period

    for term in range(1, terms + 1):
        angle = 2 * np.pi * term * row_number / period
        df[f"sin_{term}"] = np.sin(angle)
        df[f"cos_{term}"] = np.cos(angle)
