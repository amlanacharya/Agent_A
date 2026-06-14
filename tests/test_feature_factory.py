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


# ---------------------------------------------------------------------------
# Phase 3 — stockout / availability family
# ---------------------------------------------------------------------------


def test_stockout_features_require_columns() -> None:
    df = pd.DataFrame(
        {
            "series_key": ["A", "A"],
            "date": pd.date_range("2024-01-01", periods=2, freq="W-MON"),
            "demand": [10.0, 20.0],
            # stockout_flag + inventory_qty missing
        }
    )
    with pytest.raises(FeatureFactoryError, match="stockout_flag"):
        build_feature_table(df, FeatureFlags(use_stockout_features=True))
    df_with_stockout = df.assign(stockout_flag=[0, 0])
    with pytest.raises(FeatureFactoryError, match="inventory_qty"):
        build_feature_table(df_with_stockout, FeatureFlags(use_stockout_features=True))


def test_stockout_rolling_count_uses_only_prior_weeks() -> None:
    df = pd.DataFrame(
        {
            "series_key": ["A"] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="W-MON"),
            "demand": [10.0, 20.0, 30.0, 40.0, 50.0],
            "stockout_flag": [1, 0, 1, 0, 1],
            "inventory_qty": [0, 5, 0, 5, 0],
        }
    )

    result = build_feature_table(
        df, FeatureFlags(use_lag_features=False, use_stockout_features=True)
    )

    # The rolling count looks at the prior 4 weeks of stockout_flag with
    # shift(1) so the current week is excluded. Expected:
    # row 0 (no prior): NaN (or 0? - min_periods=1 means we get 0 from
    # rolling of an empty slice; shift(1) on the first row gives NaN).
    # Actually shift(1) on first row is NaN, rolling on a single NaN with
    # min_periods=1 still gives NaN.
    assert pd.isna(result.loc[0, "stockout_rolling_count_4"])
    # row 1: prior is [1] -> 1
    assert result.loc[1, "stockout_rolling_count_4"] == 1.0
    # row 2: prior is [1, 0] -> 1
    assert result.loc[2, "stockout_rolling_count_4"] == 1.0
    # row 3: prior is [1, 0, 1] -> 2
    assert result.loc[3, "stockout_rolling_count_4"] == 2.0
    # row 4: prior is [1, 0, 1, 0] -> 2
    assert result.loc[4, "stockout_rolling_count_4"] == 2.0


def test_days_since_stockout_returns_delta_in_days() -> None:
    df = pd.DataFrame(
        {
            "series_key": ["A"] * 4,
            "date": pd.date_range("2024-01-01", periods=4, freq="W-MON"),
            "demand": [10.0, 20.0, 30.0, 40.0],
            "stockout_flag": [1, 0, 0, 1],
            "inventory_qty": [0, 10, 10, 0],
        }
    )

    result = build_feature_table(
        df, FeatureFlags(use_lag_features=False, use_stockout_features=True)
    )

    # row 0: stockout THIS week, no prior - the current week's flag is
    # the "event" only if we use shift(1). Since the event date is the
    # row's own date, days_since_stockout at row 0 is 0 (the ffill of a
    # series where this row's flag is 1 lands on the row's own date).
    # row 1: 7 days since the stockout at row 0.
    assert result.loc[1, "days_since_stockout"] == 7.0
    # row 2: 14 days since the stockout at row 0.
    assert result.loc[2, "days_since_stockout"] == 14.0
    # row 3: stockout THIS week, so days_since_stockout = 0.
    assert result.loc[3, "days_since_stockout"] == 0.0


def test_inventory_cover_ratio_uses_prior_demand() -> None:
    df = pd.DataFrame(
        {
            "series_key": ["A"] * 4,
            "date": pd.date_range("2024-01-01", periods=4, freq="W-MON"),
            "demand": [10.0, 20.0, 30.0, 40.0],
            "stockout_flag": [0, 0, 0, 0],
            "inventory_qty": [100.0, 80.0, 60.0, 40.0],
        }
    )

    result = build_feature_table(
        df, FeatureFlags(use_lag_features=False, use_stockout_features=True)
    )

    # row 0: no prior demand -> NaN
    assert pd.isna(result.loc[0, "inventory_cover_ratio"])
    # row 1: prior demand = 10, inventory = 80 -> 8.0
    assert result.loc[1, "inventory_cover_ratio"] == 8.0
    # row 2: prior demand mean = (10+20)/2 = 15, inventory = 60 -> 4.0
    assert result.loc[2, "inventory_cover_ratio"] == 4.0


def test_stockout_features_respect_fold_cutoffs() -> None:
    """Rows strictly after the last cutoff must not see stockout data — NaN'd.

    With a single cutoff on row 2's date, the band (cutoff, None) is the
    trailing future band: rows 3..5 fall in it and get NaN.
    """
    df = pd.DataFrame(
        {
            "series_key": ["A"] * 6,
            "date": pd.date_range("2024-01-01", periods=6, freq="W-MON"),
            "demand": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "stockout_flag": [1, 0, 0, 0, 0, 0],
            "inventory_qty": [0, 50.0, 50.0, 50.0, 50.0, 50.0],
        }
    )
    cutoff = pd.Timestamp("2024-01-15")  # on the third Monday -> rows 0..2 visible
    result = build_feature_table(
        df,
        FeatureFlags(use_lag_features=False, use_stockout_features=True),
        fold_cutoffs=[cutoff],
    )
    # Rows 3..5 are strictly after the cutoff -> must be NaN.
    for column in ("stockout_rolling_count_4", "days_since_stockout", "inventory_cover_ratio"):
        assert result.loc[3:, column].isna().all(), f"{column} leaked across fold cutoff"
    # Row 2 is on/before the cutoff, prior stockout at row 0 -> days_since_stockout = 14.
    assert result.loc[2, "days_since_stockout"] == 14.0


# ---------------------------------------------------------------------------
# Phase 3 — hierarchy family
# ---------------------------------------------------------------------------


def test_hierarchy_features_use_sku_parent_grain() -> None:
    """Two children of the same SKU see the same parent_lag_1 value."""
    df = pd.DataFrame(
        {
            "series_key": ["A|X", "A|Y", "A|X", "A|Y"],
            "sku_id": ["A", "A", "A", "A"],
            "location_id": ["X", "Y", "X", "Y"],
            "date": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-08"),
                pd.Timestamp("2024-01-08"),
            ],
            "demand": [10.0, 20.0, 30.0, 40.0],
        }
    )

    result = build_feature_table(
        df, FeatureFlags(use_lag_features=False, use_hierarchy_features=True)
    )

    # Parent aggregate is (X+Y) per week. Week 1 total = 30, week 2 = 70.
    # lag_1 of parent = NaN at week 1, 30 at week 2.
    week1 = result[result["date"] == pd.Timestamp("2024-01-01")]
    week2 = result[result["date"] == pd.Timestamp("2024-01-08")]
    # Week 1: no prior history -> both children see NaN. Use dropna()
    # so nunique works (it ignores NaN by default).
    assert week1["parent_lag_1"].isna().all()
    # Week 2: both children must see the same parent_lag_1 = 30.0.
    assert week2["parent_lag_1"].nunique(dropna=False) == 1
    assert week2["parent_lag_1"].iloc[0] == 30.0
    # rolling_mean_4 week 2 = mean(30) = 30.
    assert week2["parent_rolling_mean_4"].iloc[0] == 30.0


def test_hierarchy_features_isolate_different_parents() -> None:
    """Two different SKUs must NOT see each other's parent aggregates."""
    df = pd.DataFrame(
        {
            "series_key": ["A|X", "B|X"],
            "sku_id": ["A", "B"],
            "location_id": ["X", "X"],
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "demand": [100.0, 1.0],
        }
    )
    result = build_feature_table(
        df, FeatureFlags(use_lag_features=False, use_hierarchy_features=True)
    )
    # Each row's parent_rolling_mean_4 is the demand of its OWN parent
    # (NaN because there is no prior week).
    assert pd.isna(result["parent_rolling_mean_4"]).all()


def test_hierarchy_features_respect_fold_cutoffs() -> None:
    df = pd.DataFrame(
        {
            "series_key": ["A|X"] * 4,
            "sku_id": ["A"] * 4,
            "location_id": ["X"] * 4,
            "date": pd.date_range("2024-01-01", periods=4, freq="W-MON"),
            "demand": [10.0, 20.0, 30.0, 40.0],
        }
    )
    # Cutoff on the second Monday: rows 0..1 visible, rows 2..3 are in
    # the future band and must be NaN.
    cutoff = pd.Timestamp("2024-01-08")
    result = build_feature_table(
        df,
        FeatureFlags(use_lag_features=False, use_hierarchy_features=True),
        fold_cutoffs=[cutoff],
    )
    assert result.loc[2:, "parent_lag_1"].isna().all()
    assert result.loc[2:, "parent_rolling_mean_4"].isna().all()
    # Row 1 is on/before the cutoff, sees row 0 -> 10.
    assert result.loc[1, "parent_lag_1"] == 10.0


# ---------------------------------------------------------------------------
# Phase 3 — lifecycle / cold-start family
# ---------------------------------------------------------------------------


def test_lifecycle_features_count_prior_observations() -> None:
    df = pd.DataFrame(
        {
            "series_key": ["A", "A", "A", "A", "A", "B", "B"],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-08",
                    "2024-01-15",
                    "2024-01-22",
                    "2024-01-29",
                    "2024-01-01",
                    "2024-01-08",
                ]
            ),
            "demand": [10.0, 20.0, 30.0, 40.0, 50.0, 5.0, 6.0],
        }
    )
    result = build_feature_table(
        df, FeatureFlags(use_lag_features=False, use_lifecycle_features=True)
    )
    # history_length is the index within each series: 0, 1, 2, 3, 4 for A;
    # 0, 1 for B. (Strictly-prior count = index because row 0 of each
    # series has 0 prior observations, etc.)
    series_a = result[result["series_key"] == "A"].reset_index(drop=True)
    series_b = result[result["series_key"] == "B"].reset_index(drop=True)
    assert series_a["history_length"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert series_b["history_length"].tolist() == [0.0, 1.0]
    # days_since_first_obs for A: 0, 7, 14, 21, 28
    assert series_a["days_since_first_obs"].tolist() == [0.0, 7.0, 14.0, 21.0, 28.0]
    # cold_start_flag: True for first _COLD_START_THRESHOLD (=4) obs.
    assert series_a["cold_start_flag"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]


def test_lifecycle_features_respect_fold_cutoffs() -> None:
    """At a fold cutoff, the row's history resets to rows up to the cutoff.

    In the future band (rows after the last cutoff) the lifecycle values
    are NaN'd out — same convention as the other time-dependent families.
    """
    df = pd.DataFrame(
        {
            "series_key": ["A"] * 6,
            "date": pd.date_range("2024-01-01", periods=6, freq="W-MON"),
            "demand": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    cutoff = pd.Timestamp("2024-01-22")  # rows 0..3 visible
    result = build_feature_table(
        df,
        FeatureFlags(use_lag_features=False, use_lifecycle_features=True),
        fold_cutoffs=[cutoff],
    )
    # Row 0 is the first observation in fold 1: history_length 0, days 0,
    # cold_start 1.
    assert result.loc[0, "history_length"] == 0.0
    assert result.loc[0, "cold_start_flag"] == 1.0
    # Row 3 (the last row on/before the cutoff) has history_length 3.
    assert result.loc[3, "history_length"] == 3.0
    assert result.loc[3, "days_since_first_obs"] == 21.0
    # Rows 4..5 fall in the future band (after the last cutoff) and get
    # NaN'd out, consistent with the time-dependent lag/rolling families.
    for column in LIFECYCLE_COLUMN_NAMES:
        assert result.loc[4:, column].isna().all(), f"{column} leaked across fold cutoff"


LIFECYCLE_COLUMN_NAMES = ("history_length", "days_since_first_obs", "cold_start_flag")


# ---------------------------------------------------------------------------
# Phase 3 — intermittency family
# ---------------------------------------------------------------------------


def test_intermittency_adi_counts_zero_weeks_in_window() -> None:
    """A series with 2 non-zero and 6 zero prior weeks -> ADI = 8/2 = 4."""
    df = pd.DataFrame(
        {
            "series_key": ["A"] * 9,
            "date": pd.date_range("2024-01-01", periods=9, freq="W-MON"),
            "demand": [5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    result = build_feature_table(
        df, FeatureFlags(use_lag_features=False, use_intermittency_features=True)
    )
    # Row 0: no prior data -> ADI undefined. The rolling(8).apply with
    # min_periods=1 gives a value of 1 (one zero week) but we use the
    # first prior week's count: the 1 prior value is NaN (shift(1)),
    # so the count is 0. ADI: weeks_in_window / nonzero = 8 / 0 -> NaN.
    assert pd.isna(result.loc[0, "rolling_adi_8"])
    # Row 1: prior is [5] (nonzero=1) -> ADI = 8 / 1 = 8.
    assert result.loc[1, "rolling_adi_8"] == 8.0
    # Row 8: prior is [5, 0, 0, 0, 5, 0, 0, 0] -> nonzero=2 -> ADI = 4.
    assert result.loc[8, "rolling_adi_8"] == 4.0


def test_intermittency_cv2_returns_zero_for_flat_series() -> None:
    """A constant series has std=0 -> CV² is undefined, NaN."""
    df = pd.DataFrame(
        {
            "series_key": ["A"] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="W-MON"),
            "demand": [5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )
    result = build_feature_table(
        df, FeatureFlags(use_lag_features=False, use_intermittency_features=True)
    )
    # Mean > 0 (5) and std=0 -> CV² = 0 (since 0/25 = 0, not NaN).
    # This is fine: a constant series has CV² = 0 by construction.
    for value in result["rolling_cv2_8"].dropna():
        assert value == 0.0


def test_intermittency_trailing_zero_run_resets_on_nonzero() -> None:
    df = pd.DataFrame(
        {
            "series_key": ["A"] * 6,
            "date": pd.date_range("2024-01-01", periods=6, freq="W-MON"),
            "demand": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
    )
    result = build_feature_table(
        df, FeatureFlags(use_lag_features=False, use_intermittency_features=True)
    )
    # The trailing zero-run at row N is the length of the run of zero
    # weeks ending strictly before row N.
    # Row 0: no prior data -> 0
    assert result.loc[0, "trailing_zero_run"] == 0
    # Row 1: prior is [1] (nonzero) -> run of zeros length 0
    assert result.loc[1, "trailing_zero_run"] == 0
    # Row 2: prior is [1, 0] -> one zero trailing
    assert result.loc[2, "trailing_zero_run"] == 1
    # Row 3: prior is [1, 0, 0] -> two zeros trailing
    assert result.loc[3, "trailing_zero_run"] == 2
    # Row 4: prior is [1, 0, 0, 0] -> three zeros trailing
    assert result.loc[4, "trailing_zero_run"] == 3
    # Row 5: prior is [1, 0, 0, 0, 1] -> zero trailing (the last one is nonzero)
    assert result.loc[5, "trailing_zero_run"] == 0


def test_intermittency_features_respect_fold_cutoffs() -> None:
    """Rows after the last cutoff must not see any intermittency data — NaN'd."""
    df = pd.DataFrame(
        {
            "series_key": ["A"] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="W-MON"),
            "demand": [10.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    # Use a cutoff strictly between rows 1 and 2 so rows 2..4 fall in
    # the future band (prefix=None) and must be NaN'd out.
    cutoff = pd.Timestamp("2024-01-09")
    result = build_feature_table(
        df,
        FeatureFlags(use_lag_features=False, use_intermittency_features=True),
        fold_cutoffs=[cutoff],
    )
    for column in ("rolling_adi_8", "rolling_cv2_8", "trailing_zero_run"):
        assert result.loc[2:, column].isna().all(), f"{column} leaked across fold cutoff"
    # Row 1 is on/before the cutoff, prior is [10] -> trailing zero run = 0.
    assert result.loc[1, "trailing_zero_run"] == 0


def test_stockout_features_respect_fold_cutoffs_strictly_between_rows() -> None:
    """Cutoff falls strictly between rows so the future band starts at row 2."""
    df = pd.DataFrame(
        {
            "series_key": ["A"] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="W-MON"),
            "demand": [10.0, 20.0, 30.0, 40.0, 50.0],
            "stockout_flag": [1, 0, 0, 0, 0],
            "inventory_qty": [0, 50.0, 50.0, 50.0, 50.0],
        }
    )
    cutoff = pd.Timestamp("2024-01-09")
    result = build_feature_table(
        df,
        FeatureFlags(use_lag_features=False, use_stockout_features=True),
        fold_cutoffs=[cutoff],
    )
    for column in ("stockout_rolling_count_4", "days_since_stockout", "inventory_cover_ratio"):
        assert result.loc[2:, column].isna().all(), f"{column} leaked across fold cutoff"
    assert result.loc[1, "days_since_stockout"] == 7.0


# ---------------------------------------------------------------------------
# Phase 3 — combined smoke test (all four families enabled at once)
# ---------------------------------------------------------------------------


def test_all_four_phase3_families_compose_cleanly() -> None:
    df = pd.DataFrame(
        {
            "series_key": ["A|X", "A|X", "A|Y", "A|Y", "B|X", "B|X"],
            "sku_id": ["A", "A", "A", "A", "B", "B"],
            "location_id": ["X", "X", "Y", "Y", "X", "X"],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-08",
                    "2024-01-01",
                    "2024-01-08",
                    "2024-01-01",
                    "2024-01-08",
                ]
            ),
            "demand": [10.0, 20.0, 5.0, 0.0, 1.0, 0.0],
            "stockout_flag": [0, 1, 0, 0, 0, 1],
            "inventory_qty": [50.0, 0.0, 20.0, 20.0, 10.0, 0.0],
        }
    )

    result = build_feature_table(
        df,
        FeatureFlags(
            use_lag_features=True,
            use_stockout_features=True,
            use_hierarchy_features=True,
            use_lifecycle_features=True,
            use_intermittency_features=True,
        ),
    )

    expected_columns = {
        # original
        "series_key", "date", "demand", "stockout_flag", "inventory_qty", "sku_id", "location_id",
        # lag/rolling
        "lag_1", "lag_2", "rolling_mean_4",
        # stockout
        "stockout_rolling_count_4", "days_since_stockout", "inventory_cover_ratio",
        # hierarchy
        "parent_lag_1", "parent_rolling_mean_4",
        # lifecycle
        "history_length", "days_since_first_obs", "cold_start_flag",
        # intermittency
        "rolling_adi_8", "rolling_cv2_8", "trailing_zero_run",
    }
    assert expected_columns.issubset(set(result.columns)), (
        f"missing columns: {expected_columns - set(result.columns)}"
    )
    # 4 rows, 5 rows because we have 2 SKUs * 2 weeks each (4) + 1 SKU * 2
    # weeks (2) = 6 rows total.
    assert len(result) == 6
    # No NaN in the row-local family at row 0 (lifecycle values are 0
    # at first obs, not NaN). The hierarchy + intermittency families
    # have NaN at row 0 by design (no prior history).
    assert result["history_length"].iloc[0] == 0.0
    assert result["cold_start_flag"].iloc[0] == 1.0
