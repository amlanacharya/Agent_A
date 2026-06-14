import math
import warnings

import pandas as pd
import pytest

from forecasting.canonical_data import (
    CANONICAL_COLUMNS,
    CanonicalColumnMapping,
    CanonicalSchemaError,
    build_canonical_table,
    validate_canonical_table,
)
from forecasting.contracts import FeatureFlags
from forecasting.feature_factory import build_feature_table


def test_build_canonical_table_maps_source_columns_and_adds_model_aliases() -> None:
    raw = pd.DataFrame(
        {
            "item": ["sku b", "sku a", "sku a"],
            "warehouse": ["west", "east", "east"],
            "week": ["2024-W02", "2024-W01", "2024-W02"],
            "units": ["20", "10", "15"],
            "on_hand": [100, 50, 45],
            "stockout": ["no", "yes", "false"],
            "sell_price": ["9.99", "11.50", "11.25"],
            "promo": ["false", "true", "0"],
            "lt_days": [7, 14, 14],
        }
    )
    original = raw.copy(deep=True)

    result = build_canonical_table(
        raw,
        CanonicalColumnMapping(
            sku_id="item",
            location_id="warehouse",
            week_start="week",
            demand_qty="units",
            inventory_qty="on_hand",
            stockout_flag="stockout",
            price="sell_price",
            promo_flag="promo",
            lead_time="lt_days",
        ),
    )

    assert list(result.columns) == list(CANONICAL_COLUMNS) + ["series_key", "date", "demand", "promo"]
    assert result[["sku_id", "location_id", "week_start", "demand_qty"]].to_dict("records") == [
        {
            "sku_id": "sku a",
            "location_id": "east",
            "week_start": pd.Timestamp("2024-01-01"),
            "demand_qty": 10.0,
        },
        {
            "sku_id": "sku a",
            "location_id": "east",
            "week_start": pd.Timestamp("2024-01-08"),
            "demand_qty": 15.0,
        },
        {
            "sku_id": "sku b",
            "location_id": "west",
            "week_start": pd.Timestamp("2024-01-08"),
            "demand_qty": 20.0,
        },
    ]
    assert result["stockout_flag"].tolist() == [True, False, False]
    assert result["promo_flag"].tolist() == [True, False, False]
    assert result["price"].tolist() == [11.50, 11.25, 9.99]
    assert result["lead_time"].tolist() == [14.0, 14.0, 7.0]
    assert result["series_key"].tolist() == ["SKU_A|EAST", "SKU_A|EAST", "SKU_B|WEST"]
    assert result["date"].equals(result["week_start"])
    assert result["demand"].equals(result["demand_qty"])
    assert result["promo"].equals(result["promo_flag"])
    pd.testing.assert_frame_equal(raw, original)


def test_build_canonical_table_keeps_series_keys_collision_safe() -> None:
    raw = pd.DataFrame(
        {
            "sku": ["A-B", "AB", "A-B", "AB"],
            "location": ["NORTH", "NORTH", "NORTH", "NORTH"],
            "week": ["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-08"],
            "demand": [1, 100, 2, 200],
        }
    )

    result = build_canonical_table(
        raw,
        CanonicalColumnMapping(
            sku_id="sku",
            location_id="location",
            week_start="week",
            demand_qty="demand",
        ),
    )

    key_by_sku = result.groupby("sku_id")["series_key"].first().to_dict()
    assert key_by_sku["A-B"] != key_by_sku["AB"]

    features = build_feature_table(result, FeatureFlags(use_lag_features=True))
    a_dash_b = features[features["sku_id"] == "A-B"].reset_index(drop=True)
    ab = features[features["sku_id"] == "AB"].reset_index(drop=True)
    assert math.isnan(a_dash_b.loc[0, "lag_1"])
    assert a_dash_b.loc[1, "lag_1"] == 1.0
    assert math.isnan(ab.loc[0, "lag_1"])
    assert ab.loc[1, "lag_1"] == 100.0


def test_build_canonical_table_fills_missing_optional_columns() -> None:
    raw = pd.DataFrame(
        {
            "sku": ["A"],
            "location": ["NORTH"],
            "week": ["2024-01-01"],
            "demand": [5],
        }
    )

    result = build_canonical_table(
        raw,
        CanonicalColumnMapping(
            sku_id="sku",
            location_id="location",
            week_start="week",
            demand_qty="demand",
        ),
    )

    assert result.loc[0, "stockout_flag"] is False
    assert result.loc[0, "promo_flag"] is False
    assert pd.isna(result.loc[0, "inventory_qty"])
    assert pd.isna(result.loc[0, "price"])
    assert pd.isna(result.loc[0, "lead_time"])


def test_build_canonical_table_names_missing_source_columns() -> None:
    raw = pd.DataFrame({"sku": ["A"], "week": ["2024-01-01"], "demand": [1]})

    with pytest.raises(CanonicalSchemaError) as exc_info:
        build_canonical_table(
            raw,
            CanonicalColumnMapping(
                sku_id="sku",
                location_id="location",
                week_start="week",
                demand_qty="demand",
            ),
        )

    assert "location" in str(exc_info.value)


def test_build_canonical_table_rejects_invalid_dates_and_demand() -> None:
    mapping = CanonicalColumnMapping(
        sku_id="sku",
        location_id="location",
        week_start="week",
        demand_qty="demand",
    )

    with pytest.raises(CanonicalSchemaError, match="week_start"):
        build_canonical_table(
            pd.DataFrame({"sku": ["A"], "location": ["N"], "week": ["not-a-date"], "demand": [1]}),
            mapping,
        )

    with pytest.raises(CanonicalSchemaError, match="demand_qty"):
        build_canonical_table(
            pd.DataFrame({"sku": ["A"], "location": ["N"], "week": ["2024-01-01"], "demand": ["bad"]}),
            mapping,
        )


def test_build_canonical_table_rejects_non_monday_week_start() -> None:
    raw = pd.DataFrame({"sku": ["A"], "location": ["N"], "week": ["2024-01-03"], "demand": [1]})

    with pytest.raises(CanonicalSchemaError, match="week_start"):
        build_canonical_table(
            raw,
            CanonicalColumnMapping(
                sku_id="sku",
                location_id="location",
                week_start="week",
                demand_qty="demand",
            ),
        )


def test_build_canonical_table_rejects_numeric_flags_outside_binary_contract() -> None:
    raw = pd.DataFrame(
        {
            "sku": ["A"],
            "location": ["NORTH"],
            "week": ["2024-01-01"],
            "demand": [1],
            "promo": [2],
        }
    )

    with pytest.raises(CanonicalSchemaError, match="promo"):
        build_canonical_table(
            raw,
            CanonicalColumnMapping(
                sku_id="sku",
                location_id="location",
                week_start="week",
                demand_qty="demand",
                promo_flag="promo",
            ),
        )


def test_validate_canonical_table_checks_required_contract_columns() -> None:
    df = pd.DataFrame(
        {
            "sku_id": ["A"],
            "location_id": ["NORTH"],
            "week_start": ["2024-01-01"],
            "demand_qty": [1],
            "inventory_qty": [10],
            "stockout_flag": [False],
            "price": [2.5],
            "promo_flag": [False],
            "lead_time": [7],
        }
    )

    with pytest.raises(CanonicalSchemaError, match="series_key"):
        validate_canonical_table(df)

    full = df.assign(
        series_key=["A|NORTH"],
        date=pd.to_datetime(df["week_start"]),
        demand=df["demand_qty"].astype(float),
        promo=df["promo_flag"],
    )

    validate_canonical_table(full)

    with pytest.raises(CanonicalSchemaError, match="inventory_qty"):
        validate_canonical_table(df.drop(columns=["inventory_qty"]))


def test_validate_canonical_table_rejects_invalid_flag_values() -> None:
    df = pd.DataFrame(
        {
            "sku_id": ["A"],
            "location_id": ["NORTH"],
            "week_start": ["2024-01-01"],
            "demand_qty": [1],
            "inventory_qty": [10],
            "stockout_flag": ["maybe"],
            "price": [2.5],
            "promo_flag": [False],
            "lead_time": [7],
            "series_key": ["A|NORTH"],
            "date": [pd.Timestamp("2024-01-01")],
            "demand": [1.0],
            "promo": [False],
        }
    )

    with pytest.raises(CanonicalSchemaError, match="stockout_flag"):
        validate_canonical_table(df)


def test_canonical_table_feeds_feature_factory_without_extra_adapter() -> None:
    raw = pd.DataFrame(
        {
            "sku": ["A", "A"],
            "location": ["NORTH", "NORTH"],
            "week": ["2024-01-01", "2024-01-08"],
            "demand": [5, 9],
            "promo": [0, 1],
        }
    )
    canonical = build_canonical_table(
        raw,
        CanonicalColumnMapping(
            sku_id="sku",
            location_id="location",
            week_start="week",
            demand_qty="demand",
            promo_flag="promo",
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        features = build_feature_table(
            canonical,
            FeatureFlags(use_lag_features=True, use_promo_indicator=True),
        )

    assert math.isnan(features.loc[0, "lag_1"])
    assert features.loc[1, "lag_1"] == 5.0
    assert features["promo_indicator"].tolist() == [0, 1]
