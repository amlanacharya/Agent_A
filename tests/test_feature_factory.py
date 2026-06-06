import math

import pandas as pd
import pytest

from forecasting.contracts import FeatureFlags
from forecasting.feature_factory import (
    FeatureFactoryError,
    build_feature_table,
    validate_canonical_columns,
)


def test_build_feature_table_sorts_rows_and_does_not_mutate_input() -> None:
    df = pd.DataFrame(
        {
            "series_key": ["B", "A", "A"],
            "date": ["2024-01-03", "2024-01-02", "2024-01-01"],
            "demand": [30, 20, 10],
        }
    )
    original = df.copy(deep=True)

    result = build_feature_table(df, FeatureFlags(use_lag_features=False))

    assert result[["series_key", "date", "demand"]].to_dict("records") == [
        {"series_key": "A", "date": pd.Timestamp("2024-01-01"), "demand": 10},
        {"series_key": "A", "date": pd.Timestamp("2024-01-02"), "demand": 20},
        {"series_key": "B", "date": pd.Timestamp("2024-01-03"), "demand": 30},
    ]
    pd.testing.assert_frame_equal(df, original)
    assert result is not df


def test_lag_features_use_only_prior_rows_per_series() -> None:
    df = pd.DataFrame(
        {
            "series_key": ["A", "A", "A", "A", "A", "B"],
            "date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-01",
            ],
            "demand": [10, 20, 30, 40, 50, 999],
        }
    )

    result = build_feature_table(df, FeatureFlags(use_lag_features=True))
    series_a = result[result["series_key"] == "A"].reset_index(drop=True)
    series_b = result[result["series_key"] == "B"].reset_index(drop=True)

    pd.testing.assert_series_equal(
        series_a["lag_1"],
        pd.Series([math.nan, 10.0, 20.0, 30.0, 40.0], name="lag_1"),
    )
    pd.testing.assert_series_equal(
        series_a["lag_2"],
        pd.Series([math.nan, math.nan, 10.0, 20.0, 30.0], name="lag_2"),
    )
    pd.testing.assert_series_equal(
        series_a["rolling_mean_4"],
        pd.Series([math.nan, 10.0, 15.0, 20.0, 25.0], name="rolling_mean_4"),
    )
    assert math.isnan(series_b.loc[0, "lag_1"])
    assert math.isnan(series_b.loc[0, "lag_2"])
    assert math.isnan(series_b.loc[0, "rolling_mean_4"])


def test_promo_indicator_requires_canonical_promo_column_when_enabled() -> None:
    df = pd.DataFrame(
        {
            "series_key": ["A", "A", "A"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "demand": [10, 20, 30],
            "promo": [True, False, 2],
        }
    )

    result = build_feature_table(df, FeatureFlags(use_lag_features=False, use_promo_indicator=True))

    assert result["promo_indicator"].tolist() == [1, 0, 1]


def test_promo_indicator_missing_column_raises_error() -> None:
    df = pd.DataFrame({"series_key": ["A"], "date": ["2024-01-01"], "demand": [10]})

    with pytest.raises(FeatureFactoryError, match="promo"):
        build_feature_table(df, FeatureFlags(use_lag_features=False, use_promo_indicator=True))


def test_fourier_features_use_row_order_per_series_and_reject_invalid_terms() -> None:
    df = pd.DataFrame(
        {
            "series_key": ["A", "A", "A", "A", "B"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-01"],
            "demand": [10, 20, 30, 40, 50],
        }
    )

    result = build_feature_table(
        df,
        FeatureFlags(use_lag_features=False, use_fourier=True, fourier_terms=2),
    )

    series_a = result[result["series_key"] == "A"].reset_index(drop=True)
    assert series_a["sin_1"].round(6).tolist() == [1.0, 0.0, -1.0, -0.0]
    assert series_a["cos_1"].round(6).tolist() == [0.0, -1.0, -0.0, 1.0]
    assert series_a["sin_2"].round(6).tolist() == [0.0, -0.0, 0.0, -0.0]
    assert series_a["cos_2"].round(6).tolist() == [-1.0, 1.0, -1.0, 1.0]

    series_b = result[result["series_key"] == "B"].reset_index(drop=True)
    assert round(series_b.loc[0, "sin_1"], 6) == 0.0
    assert round(series_b.loc[0, "cos_1"], 6) == 1.0

    with pytest.raises(FeatureFactoryError, match="fourier_terms"):
        build_feature_table(df, FeatureFlags(use_lag_features=False, use_fourier=True, fourier_terms=0))


def test_validate_canonical_columns_names_missing_required_columns() -> None:
    df = pd.DataFrame({"series_key": ["A"]})

    with pytest.raises(FeatureFactoryError) as exc_info:
        validate_canonical_columns(df)

    message = str(exc_info.value)
    assert "date" in message
    assert "demand" in message


def test_invalid_dates_raise_feature_factory_error() -> None:
    df = pd.DataFrame({"series_key": ["A"], "date": ["not-a-date"], "demand": [10]})

    with pytest.raises(FeatureFactoryError, match="date"):
        build_feature_table(df, FeatureFlags(use_lag_features=False))
