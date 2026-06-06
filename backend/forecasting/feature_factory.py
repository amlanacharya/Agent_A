"""Canonical feature table generation for forecasting models."""

import numpy as np
import pandas as pd

from forecasting.contracts import FeatureFlags


REQUIRED_CANONICAL_COLUMNS = ("series_key", "date", "demand")


class FeatureFactoryError(ValueError):
    """Raised when canonical feature generation cannot proceed."""


def validate_canonical_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_CANONICAL_COLUMNS if column not in df.columns]
    if missing:
        raise FeatureFactoryError(f"canonical data is missing required columns: {', '.join(missing)}")


def build_feature_table(df: pd.DataFrame, flags: FeatureFlags) -> pd.DataFrame:
    validate_canonical_columns(df)

    result = df.copy(deep=True)
    try:
        result["date"] = pd.to_datetime(result["date"], errors="raise")
    except Exception as exc:
        raise FeatureFactoryError("canonical date column contains invalid date values") from exc

    result = result.sort_values(["series_key", "date"], kind="mergesort").reset_index(drop=True)

    if flags.use_lag_features:
        grouped_demand = result.groupby("series_key", sort=False)["demand"]
        result["lag_1"] = grouped_demand.shift(1)
        result["lag_2"] = grouped_demand.shift(2)
        result["rolling_mean_4"] = grouped_demand.transform(
            lambda series: series.shift(1).rolling(window=4, min_periods=1).mean()
        )

    if flags.use_promo_indicator:
        if "promo" not in result.columns:
            raise FeatureFactoryError("promo column is required when use_promo_indicator is enabled")
        result["promo_indicator"] = _promo_to_indicator(result["promo"])

    if flags.use_fourier:
        if flags.fourier_terms < 1:
            raise FeatureFactoryError("fourier_terms must be at least 1 when use_fourier is enabled")
        _add_fourier_features(result, flags.fourier_terms)

    return result


def _promo_to_indicator(promo: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(promo) or pd.api.types.is_bool_dtype(promo):
        return promo.fillna(0).ne(0).astype(int)

    numeric = pd.to_numeric(promo, errors="coerce")
    text_true = promo.astype("string").fillna("").str.strip().str.lower().isin({"true", "yes", "y"})
    return (numeric.fillna(0).ne(0) | text_true).astype(int)


def _add_fourier_features(df: pd.DataFrame, terms: int) -> None:
    row_number = df.groupby("series_key", sort=False).cumcount() + 1
    period = df.groupby("series_key", sort=False)["series_key"].transform("size")

    for term in range(1, terms + 1):
        angle = 2 * np.pi * term * row_number / period
        df[f"sin_{term}"] = np.sin(angle)
        df[f"cos_{term}"] = np.cos(angle)
