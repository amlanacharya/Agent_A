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


# ---------------------------------------------------------------------------
# Fold-aware feature generation (walk-forward validation safety)
# ---------------------------------------------------------------------------


def test_fold_cutoff_zero_keeps_existing_behavior() -> None:
    """No fold_cutoffs argument must produce identical results to the legacy path."""
    df = pd.DataFrame(
        {
            "series_key": ["A", "A", "A", "A", "A"],
            "date": pd.date_range("2024-01-01", periods=5, freq="W-MON"),
            "demand": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    baseline = build_feature_table(df, FeatureFlags(use_lag_features=True))
    with_cutoffs = build_feature_table(
        df, FeatureFlags(use_lag_features=True), fold_cutoffs=[]
    )
    pd.testing.assert_frame_equal(baseline, with_cutoffs)


def test_fold_cutoff_masks_lag_features_for_rows_after_cutoff() -> None:
    """Rows on or before the cutoff keep lag values; rows after must be NaN.

    The key guarantee: a row whose date is strictly after the cutoff must
    NOT see the demand values from rows after the cutoff in its lag/rolling
    features - the same guarantee walk-forward validation needs.
    """
    df = pd.DataFrame(
        {
            "series_key": ["A"] * 6,
            "date": pd.date_range("2024-01-01", periods=6, freq="W-MON"),
            "demand": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    # Row 0..2 are on/before the cutoff (2024-01-15). Row 3..5 are strictly
    # after and must have NaN lag/rolling features.
    cutoff = pd.Timestamp("2024-01-15")
    result = build_feature_table(
        df,
        FeatureFlags(use_lag_features=True),
        fold_cutoffs=[cutoff],
    )

    def _matches(actual: pd.Series, expected: list[float | None]) -> bool:
        if len(actual) != len(expected):
            return False
        for got, want in zip(actual.tolist(), expected):
            if want is None:
                if not pd.isna(got):
                    return False
            elif got != want:
                return False
        return True

    before = result[result["date"] <= cutoff].reset_index(drop=True)
    after = result[result["date"] > cutoff].reset_index(drop=True)

    assert _matches(before["lag_1"], [None, 10.0, 20.0])
    assert _matches(before["lag_2"], [None, None, 10.0])
    assert _matches(before["rolling_mean_4"], [None, 10.0, 15.0])
    for column in ("lag_1", "lag_2", "rolling_mean_4"):
        assert after[column].isna().all(), f"{column} leaked across fold cutoff"
    assert len(after) == 3  # rows 3, 4, 5 are strictly after the cutoff


def test_fold_cutoff_uses_latest_cutoff_not_earliest() -> None:
    """Multiple cutoffs - each row uses the latest cutoff <= its date."""
    df = pd.DataFrame(
        {
            "series_key": ["A"] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="W-MON"),
            "demand": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    # Cutoffs: 2024-01-08 and 2024-01-22.
    # Rows 0..1 fall in the (None, 2024-01-08] band; rows 2..3 fall in
    # (2024-01-08, 2024-01-22]; row 4 is strictly after both cutoffs.
    cutoffs = [pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-22")]
    result = build_feature_table(
        df,
        FeatureFlags(use_lag_features=True),
        fold_cutoffs=cutoffs,
    )

    # Row 0 (in fold 1): only its own prior rows visible (none) -> NaN.
    # Row 1 (in fold 1): sees row 0 -> lag_1 = 10.0
    # Row 2 (in fold 2): sees rows 0, 1 -> lag_1 = 20.0
    # Row 3 (in fold 2): sees rows 0, 1, 2 -> lag_1 = 30.0
    # Row 4 (post-fold): NO leakage -> NaN
    assert pd.isna(result.iloc[0]["lag_1"])
    assert result.iloc[1]["lag_1"] == 10.0
    assert result.iloc[2]["lag_1"] == 20.0
    assert result.iloc[3]["lag_1"] == 30.0
    assert pd.isna(result.iloc[4]["lag_1"]), "row after all cutoffs must not leak"


def test_fold_cutoff_rejects_unsorted_or_duplicate_cutoffs() -> None:
    df = pd.DataFrame(
        {
            "series_key": ["A", "A"],
            "date": pd.date_range("2024-01-01", periods=2, freq="W-MON"),
            "demand": [10.0, 20.0],
        }
    )

    with pytest.raises(FeatureFactoryError, match="fold_cutoffs"):
        build_feature_table(
            df,
            FeatureFlags(use_lag_features=True),
            fold_cutoffs=[pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-08")],
        )

    with pytest.raises(FeatureFactoryError, match="fold_cutoffs"):
        build_feature_table(
            df,
            FeatureFlags(use_lag_features=True),
            fold_cutoffs=[pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-15")],
        )


def test_fold_cutoff_passes_promo_indicator_through() -> None:
    """Promo indicator is row-local so it must not be NaN'd by a fold cutoff."""
    df = pd.DataFrame(
        {
            "series_key": ["A", "A", "A"],
            "date": pd.date_range("2024-01-01", periods=3, freq="W-MON"),
            "demand": [10.0, 20.0, 30.0],
            "promo": [True, False, True],
        }
    )

    result = build_feature_table(
        df,
        FeatureFlags(use_lag_features=True, use_promo_indicator=True),
        fold_cutoffs=[pd.Timestamp("2024-01-01")],
    )

    assert result["promo_indicator"].tolist() == [1, 0, 1]


def test_walk_forward_validation_has_no_future_leakage() -> None:
    """End-to-end walk-forward shape: two cutoffs, two series, demand that
    triples only after the second cutoff. Naive (non-fold-aware) features
    would let the training rows see the future surge; the fold-aware path
    must keep them blind to it.
    """
    df = pd.DataFrame(
        {
            "series_key": ["A"] * 8 + ["B"] * 8,
            "date": list(pd.date_range("2024-01-01", periods=8, freq="W-MON")) * 2,
            "demand": [10.0, 12.0, 11.0, 13.0, 200.0, 210.0, 220.0, 230.0] * 2,
        }
    )
    # Cutoffs: 2024-01-15, 2024-01-29.
    # Series A: rows 0..4 are the prefix; the surge starts at row 4 (200.0).
    # Band 1: (None, 2024-01-15] -> rows 0..1, prefix is rows 0..1.
    # Band 2: (2024-01-15, 2024-01-29] -> rows 2..3, prefix is rows 0..3.
    # Band 3 (future): (2024-01-29, None) -> rows 5..7, features must be NaN.
    # Row 4 sits exactly on the second cutoff; it belongs to band 2 (the band
    # is upper-inclusive) and may see demand up to and including itself, but
    # its lag_1 still uses prior rows only (lag = 13.0, not 200.0).
    cutoffs = [pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-29")]
    result = build_feature_table(
        df,
        FeatureFlags(use_lag_features=True),
        fold_cutoffs=cutoffs,
    )

    for series in ("A", "B"):
        sub = result[result["series_key"] == series].reset_index(drop=True)
        # Row 0 (fold 1, no prior): NaN
        assert pd.isna(sub.loc[0, "lag_1"])
        # Row 1 (fold 1, sees row 0): 10.0
        assert sub.loc[1, "lag_1"] == 10.0
        # Rows 2..3 (fold 2, prefix is rows 0..3): 12.0 then 11.0
        assert sub.loc[2, "lag_1"] == 12.0
        assert sub.loc[3, "lag_1"] == 11.0
        # Row 4 (last row of band 2; lag_1 still uses prior rows): 13.0
        assert sub.loc[4, "lag_1"] == 13.0
        # Rows 5..7 (future band): must stay NaN - no leakage of the surge.
        for i in range(5, 8):
            assert pd.isna(sub.loc[i, "lag_1"]), (
                f"series {series} row {i} leaked the post-cutoff surge"
            )


# ---------------------------------------------------------------------------
# Explicit seasonal period (Fourier fix)
# ---------------------------------------------------------------------------


def test_fourier_uses_explicit_frequency_period_when_provided() -> None:
    """When frequency_period is set, Fourier cycles over that period - not
    the row-count of the series - so it survives splits/holdouts.

    row_number is 1-indexed (groupby cumcount + 1), so with period=4 and
    term=1 the angle is 2*pi*1*row/4, producing sin values 1, 0, -1, 0
    across rows 1..4 - a complete cycle.
    """
    df = pd.DataFrame(
        {
            "series_key": ["A"] * 4,
            "date": pd.date_range("2024-01-01", periods=4, freq="W-MON"),
            "demand": [10.0, 20.0, 30.0, 40.0],
        }
    )

    result = build_feature_table(
        df,
        FeatureFlags(use_lag_features=False, use_fourier=True, fourier_terms=1, frequency_period=4),
    )

    assert result["sin_1"].round(6).tolist() == [1.0, 0.0, -1.0, 0.0]
    assert result["cos_1"].round(6).tolist() == [0.0, -1.0, 0.0, 1.0]


def test_fourier_default_falls_back_to_series_size() -> None:
    """When frequency_period is None, the existing row-count behaviour holds."""
    df = pd.DataFrame(
        {
            "series_key": ["A", "A", "A", "A"],
            "date": pd.date_range("2024-01-01", periods=4, freq="W-MON"),
            "demand": [10.0, 20.0, 30.0, 40.0],
        }
    )

    result = build_feature_table(
        df,
        FeatureFlags(use_lag_features=False, use_fourier=True, fourier_terms=1),
    )

    # Period = size of series = 4, so this should match the frequency_period=4 case.
    assert result["sin_1"].round(6).tolist() == [1.0, 0.0, -1.0, 0.0]


def test_fourier_rejects_invalid_frequency_period() -> None:
    df = pd.DataFrame(
        {
            "series_key": ["A"],
            "date": [pd.Timestamp("2024-01-01")],
            "demand": [10.0],
        }
    )

    with pytest.raises(FeatureFactoryError, match="frequency_period"):
        build_feature_table(
            df,
            FeatureFlags(use_lag_features=False, use_fourier=True, frequency_period=0),
        )
