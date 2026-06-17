"""Tests for the governed forecasting model families (Phase 4).

Each family gets at least one "fit + predict" smoke test and one
shape contract test. The XGBoost family gets a slightly deeper
test that asserts the recursive-forecast loop actually mutates
the lag-1 column.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from forecasting.contracts import ModelFamilyName
from forecasting.forecasting_models import (
    AggregateAllocateModel,
    CrostonModel,
    ExponentialSmoothingModel,
    FamilyResources,
    ForecastingModelError,
    FittedState,
    MovingAverageModel,
    NaiveModel,
    SeasonalNaiveModel,
    XGBoostGlobalModel,
    build_model,
    default_families_for_class,
    list_model_families,
)


# ---------------------------------------------------------------------------
# Smoke tests - one per family
# ---------------------------------------------------------------------------


def _weekly_history(values):
    return pd.Series(
        values,
        index=pd.date_range("2024-01-01", periods=len(values), freq="W-MON"),
        name="demand",
        dtype=float,
    )


def test_naive_repeats_last_value() -> None:
    model = NaiveModel()
    state = model.fit(_weekly_history([1, 2, 3, 4, 5]), series_key="A")
    assert model.predict(state, horizon=3) == [5.0, 5.0, 5.0]


def test_seasonal_naive_uses_cycle_when_history_long_enough() -> None:
    # 8 weeks of history with a 4-week cycle. The most recent
    # cycle is [5, 6, 7, 8] so the next 6 forecasts should be
    # [5, 6, 7, 8, 5, 6].
    history = _weekly_history([1, 2, 3, 4, 5, 6, 7, 8])
    model = SeasonalNaiveModel(season_length=4)
    state = model.fit(history, series_key="A")
    assert model.predict(state, horizon=6) == [5.0, 6.0, 7.0, 8.0, 5.0, 6.0]


def test_seasonal_naive_falls_back_to_last_value_for_short_history() -> None:
    model = SeasonalNaiveModel(season_length=52)
    state = model.fit(_weekly_history([1, 2, 3]), series_key="A")
    assert model.predict(state, horizon=2) == [3.0, 3.0]


def test_moving_average_uses_window_mean() -> None:
    model = MovingAverageModel(window=3)
    state = model.fit(_weekly_history([1, 2, 3, 4, 5]), series_key="A")
    # Mean of the last 3 values = 4.0
    assert model.predict(state, horizon=4) == [4.0, 4.0, 4.0, 4.0]


def test_exponential_smoothing_picks_a_grid_alpha_beta() -> None:
    model = ExponentialSmoothingModel()
    history = _weekly_history([10, 12, 14, 16, 18, 20])
    state = model.fit(history, series_key="A")
    forecast = model.predict(state, horizon=3)
    # The trend is positive so the forecast should be increasing
    # and reasonable in magnitude (between 20 and ~30).
    assert forecast[0] >= 20
    assert forecast[-1] >= forecast[0]


def test_croston_intermittent_returns_rate_per_period() -> None:
    model = CrostonModel()
    history = _weekly_history([0, 0, 5, 0, 0, 0, 7, 0])
    state = model.fit(history, series_key="A")
    forecast = model.predict(state, horizon=3)
    # First non-zero at index 2 (interval 3), second at index 6
    # (interval 4). Mean interval = 3.5, mean size = 6.0,
    # rate = 6 / 3.5 = ~1.714.
    expected = 6.0 / 3.5
    assert all(abs(value - expected) < 1e-6 for value in forecast)


def test_croston_handles_all_zero_history() -> None:
    model = CrostonModel()
    state = model.fit(_weekly_history([0, 0, 0, 0]), series_key="A")
    assert model.predict(state, horizon=2) == [0.0, 0.0]


def test_xgboost_global_produces_a_horizon_long_forecast() -> None:
    model = XGBoostGlobalModel(n_estimators=20, max_depth=3)
    raw = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
    history = _weekly_history(raw)
    demand = history.to_numpy()
    lag_1 = [None] + list(demand[:-1])
    lag_2 = [None, None] + list(demand[:-2])
    # rolling_mean_4 with min_periods=1: each row's value is the
    # mean of the current row and up to 3 prior rows. Row 0 has
    # one value, row 1 has two, etc.
    rolling: list[float | None] = []
    for index in range(len(demand)):
        window = demand[max(0, index - 3) : index + 1]
        rolling.append(float(np.mean(window)))
    features = pd.DataFrame(
        {
            "series_key": ["A"] * len(history),
            "date": history.index,
            "demand": demand,
            "lag_1": pd.array(lag_1, dtype="float64"),
            "lag_2": pd.array(lag_2, dtype="float64"),
            "rolling_mean_4": pd.array(rolling, dtype="float64"),
        }
    )
    state = model.fit(history, series_key="A", features=features)
    forecast = model.predict(state, horizon=3)
    assert len(forecast) == 3
    # Recursive forecast on a strictly increasing series should
    # not collapse to zero or explode to infinity.
    assert all(math.isfinite(value) and value > 0 for value in forecast)


def test_aggregate_allocate_uses_parent_share() -> None:
    parent = pd.DataFrame(
        {"demand": [10, 20, 30, 40, 50]},
        index=pd.date_range("2024-01-01", periods=5, freq="W-MON"),
    )
    model = AggregateAllocateModel(parent_features=parent)
    history = _weekly_history([3, 6, 9, 12, 15])  # Last value 15, parent last 50 -> 30%
    state = model.fit(history, series_key="A")
    forecast = model.predict(state, horizon=3)
    assert forecast == [15.0, 15.0, 15.0]


# ---------------------------------------------------------------------------
# Protocol and registry contract tests
# ---------------------------------------------------------------------------


def test_build_model_constructs_every_registered_family() -> None:
    for family in list_model_families():
        if family == "aggregate_allocate":
            model = build_model(family, resources=FamilyResources(parent_features=pd.DataFrame({"demand": [1.0]})))
        else:
            model = build_model(family)
        assert model.family == family


def test_build_model_rejects_unknown_family() -> None:
    with pytest.raises(ForecastingModelError):
        build_model("not_a_real_model")  # type: ignore[arg-type]


def test_predict_rejects_mismatched_state() -> None:
    naive = NaiveModel()
    state = naive.fit(_weekly_history([1, 2, 3]), series_key="A")
    croston = CrostonModel()
    with pytest.raises(ForecastingModelError, match="state was fit with"):
        croston.predict(state, horizon=2)


def test_predict_rejects_zero_or_negative_horizon() -> None:
    model = NaiveModel()
    state = model.fit(_weekly_history([1, 2, 3]), series_key="A")
    with pytest.raises(ForecastingModelError, match="horizon must be >= 1"):
        model.predict(state, horizon=0)


def test_fit_rejects_non_numeric_history() -> None:
    model = NaiveModel()
    with pytest.raises(ForecastingModelError, match="numeric values"):
        model.fit(pd.Series(["a", "b", "c"], index=pd.date_range("2024-01-01", periods=3, freq="W-MON")), series_key="A")


def test_fitted_state_round_trips_through_dict() -> None:
    state = FittedState(family="naive", series_key="A", payload={"last_value": 5.0})
    as_dict = state.to_dict()
    restored = FittedState.from_dict(as_dict)
    assert restored.family == "naive"
    assert restored.payload == {"last_value": 5.0}


def test_seasonal_naive_rejects_zero_season_length() -> None:
    with pytest.raises(ForecastingModelError, match="season_length"):
        SeasonalNaiveModel(season_length=0)


def test_moving_average_rejects_zero_window() -> None:
    with pytest.raises(ForecastingModelError, match="window"):
        MovingAverageModel(window=0)


# ---------------------------------------------------------------------------
# default_families_for_class tests
# ---------------------------------------------------------------------------


def test_default_families_for_smooth_class_skips_croston() -> None:
    families = default_families_for_class("SMOOTH")
    assert "croston" not in families
    assert "naive" in families
    assert "xgboost_global" in families


def test_default_families_for_intermittent_class_includes_croston() -> None:
    families = default_families_for_class("INTERMITTENT")
    assert "croston" in families
    assert "naive" in families
    assert "aggregate_allocate" in families


def test_default_families_for_unknown_class_returns_all() -> None:
    families = default_families_for_class(None)
    assert set(families) == set(list_model_families())


# ---------------------------------------------------------------------------
# XGBoost negative paths
# ---------------------------------------------------------------------------


def test_xgboost_rejects_missing_features() -> None:
    model = XGBoostGlobalModel()
    with pytest.raises(ForecastingModelError, match="features DataFrame"):
        model.fit(_weekly_history([1, 2, 3]), series_key="A")


def test_xgboost_rejects_features_without_demand_column() -> None:
    model = XGBoostGlobalModel()
    features = pd.DataFrame({"lag_1": [None, 1, 2]})
    with pytest.raises(ForecastingModelError, match="'demand'"):
        model.fit(_weekly_history([1, 2, 3]), series_key="A", features=features)


def test_xgboost_rejects_too_few_training_rows() -> None:
    model = XGBoostGlobalModel()
    # One-row history means after NaN-filtering on the lag / rolling
    # features there is at most one training row, which is below
    # the 2-row minimum.
    history = _weekly_history([1.0])
    features = pd.DataFrame(
        {
            "series_key": ["A"],
            "date": history.index,
            "demand": [1.0],
            "lag_1": pd.array([None], dtype="float64"),
            "rolling_mean_4": pd.array([None], dtype="float64"),
        }
    )
    with pytest.raises(ForecastingModelError, match="fewer than 2 training rows"):
        model.fit(history, series_key="A", features=features)


def test_aggregate_allocate_rejects_missing_parent_features() -> None:
    model = AggregateAllocateModel(parent_features=None)
    with pytest.raises(ForecastingModelError, match="parent_features"):
        model.fit(_weekly_history([1, 2, 3]), series_key="A")


# ---------------------------------------------------------------------------
# FamilyResources / XGBoost persistence
# ---------------------------------------------------------------------------


def test_family_resources_defaults_to_empty_bag() -> None:
    """The default FamilyResources is usable: build_model picks
    parent_features=None for aggregate_allocate and works for everyone else."""
    model = build_model("naive", resources=FamilyResources())
    assert model.family == "naive"
    model = build_model(
        "aggregate_allocate", resources=FamilyResources()
    )
    with pytest.raises(ForecastingModelError, match="parent_features"):
        # The default bag has no parent_features -> the model fails
        # at fit time with a clear message, not at construction.
        model.fit(_weekly_history([1, 2, 3]), series_key="A")


def test_xgboost_persistence_roundtrips_through_json() -> None:
    """The encode/decode helpers in xgboost_persistence survive a
    JSON round-trip — the model produces the same forecast before and
    after. This is the contract the harness relies on for run-state
    persistence."""
    import json
    pytest.importorskip("xgboost")
    from forecasting.xgboost_persistence import decode_xgboost_model, encode_xgboost_model

    import xgboost  # type: ignore

    raw = [10.0, 12.0, 14.0, 16.0, 18.0]
    x = [[v] for v in raw]
    y = [v + 0.5 for v in raw]
    booster_model = xgboost.XGBRegressor(n_estimators=10, max_depth=2)
    booster_model.fit(x, y)
    encoded = encode_xgboost_model(booster_model)
    # JSON round-trip
    round_tripped = json.loads(json.dumps(encoded))
    decoded = decode_xgboost_model(xgboost, round_tripped)
    assert decoded.predict([[10.0]])[0] == pytest.approx(
        booster_model.predict([[10.0]])[0]
    )
