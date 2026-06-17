"""Tests for ``forecasting.metrics`` (Phase 5.1 CB1).

The metric portfolio is pure functions on ``Sequence[ModelScorecard]``.
Tests use synthetic scorecards with known forecast/actual arrays
and assert the metric value. Edge cases (empty input, zero
denominator, mismatched lengths) are pinned by tests.
"""

from __future__ import annotations

import math

import pytest

from forecasting.contracts import ModelScorecard
from forecasting.metrics import (
    cumulative_mae_at_horizon,
    expected_overstock,
    expected_stockouts,
    horizon_level_error,
    interval_coverage,
    mean_bias,
    wape,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _card(
    series_key: str,
    forecast: list[float],
    actual: list[float],
    *,
    mae: float | None = None,
    bias: float = 0.0,
) -> ModelScorecard:
    """Build a ModelScorecard with sensible test defaults.

    The ``mae`` / ``rmse`` / ``mase`` fields are filled from the
    arrays so the scorecard passes Pydantic validation. Tests
    that don't care about these fields still need them set.
    """
    import numpy as np
    f = np.asarray(forecast, dtype=float)
    a = np.asarray(actual, dtype=float)
    residuals = a - f
    mae = float(np.mean(np.abs(residuals))) if len(f) else 0.0
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if len(f) else 0.0
    return ModelScorecard(
        model_family="naive",
        series_key=series_key,
        fold_cutoff="2024-01-01T00:00:00",
        horizon=len(forecast),
        forecast=forecast,
        actual=actual,
        mae=mae,
        rmse=rmse,
        mase=mae,
        bias=bias,
    )


# ---------------------------------------------------------------------------
# WAPE
# ---------------------------------------------------------------------------


def test_wape_perfect_forecast_is_zero() -> None:
    """A perfect forecast gives WAPE = 0."""
    cards = [
        _card("A", [10.0, 12.0, 14.0], [10.0, 12.0, 14.0]),
        _card("B", [20.0, 22.0], [20.0, 22.0]),
    ]
    assert wape(cards) == 0.0


def test_wape_known_value() -> None:
    """A hand-computable WAPE matches the formula.

    Two scorecards, each 3 points:
    - card A: actual [10, 20, 30], forecast [12, 18, 30]
      abs errors [2, 2, 0], abs actuals [10, 20, 30]
    - card B: actual [5, 5, 5], forecast [5, 5, 5]
      abs errors [0, 0, 0], abs actuals [5, 5, 5]
    total error 4, total actual 60+15=75, WAPE = 4/75.
    """
    cards = [
        _card("A", [12.0, 18.0, 30.0], [10.0, 20.0, 30.0]),
        _card("B", [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]),
    ]
    assert wape(cards) == pytest.approx(4.0 / 75.0)


def test_wape_empty_input_is_nan() -> None:
    assert math.isnan(wape([]))


def test_wape_zero_actuals_is_nan() -> None:
    """Total actual demand == 0 -> WAPE undefined (no division)."""
    cards = [_card("A", [0.0, 0.0], [0.0, 0.0])]
    assert math.isnan(wape(cards))


# ---------------------------------------------------------------------------
# mean_bias
# ---------------------------------------------------------------------------


def test_mean_bias_is_signed_average() -> None:
    cards = [
        _card("A", [10.0, 10.0], [12.0, 12.0], bias=2.0),
        _card("B", [10.0, 10.0], [8.0, 8.0], bias=-2.0),
    ]
    assert mean_bias(cards) == 0.0


def test_mean_bias_empty_is_nan() -> None:
    assert math.isnan(mean_bias([]))


# ---------------------------------------------------------------------------
# horizon_level_error
# ---------------------------------------------------------------------------


def test_horizon_level_error_per_step_mae() -> None:
    """Per-step MAE across scorecards sharing the same horizon."""
    # Two scorecards, horizon 3 each.
    # step 0: card A abs(|10-12|)=2, card B abs(|5-5|)=0, mean=1
    # step 1: card A abs(|20-18|)=2, card B abs(|5-5|)=0, mean=1
    # step 2: card A abs(|30-30|)=0, card B abs(|5-5|)=0, mean=0
    cards = [
        _card("A", [12.0, 18.0, 30.0], [10.0, 20.0, 30.0]),
        _card("B", [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]),
    ]
    assert horizon_level_error(cards) == pytest.approx([1.0, 1.0, 0.0])


def test_horizon_level_error_handles_mismatched_horizons() -> None:
    """A shorter card contributes only to the steps it covers."""
    cards = [
        _card("A", [10.0, 20.0, 30.0], [10.0, 20.0, 30.0]),
        _card("B", [5.0], [5.0]),
    ]
    # step 0: card A 0, card B 0 -> mean 0
    # step 1: card A 0, card B skipped -> mean 0
    # step 2: card A 0, card B skipped -> mean 0
    assert horizon_level_error(cards) == pytest.approx([0.0, 0.0, 0.0])


def test_horizon_level_error_empty_input_returns_empty_list() -> None:
    assert horizon_level_error([]) == []


# ---------------------------------------------------------------------------
# cumulative_mae_at_horizon
# ---------------------------------------------------------------------------


def test_cumulative_mae_at_horizon_sums_first_h_steps() -> None:
    cards = [
        _card("A", [10.0, 20.0, 30.0], [12.0, 18.0, 30.0]),
        _card("B", [10.0, 20.0, 30.0], [10.0, 20.0, 30.0]),
    ]
    # card A cumulative |err| at h=2 = 2+2=4
    # card B cumulative |err| at h=2 = 0+0=0
    # mean = 2
    assert cumulative_mae_at_horizon(cards, h=2) == pytest.approx(2.0)


def test_cumulative_mae_at_horizon_zero_or_negative_returns_nan() -> None:
    cards = [_card("A", [10.0], [10.0])]
    assert math.isnan(cumulative_mae_at_horizon(cards, h=0))
    assert math.isnan(cumulative_mae_at_horizon(cards, h=-1))


def test_cumulative_mae_at_horizon_empty_input_returns_nan() -> None:
    assert math.isnan(cumulative_mae_at_horizon([], h=3))


# ---------------------------------------------------------------------------
# expected_stockouts / expected_overstock
# ---------------------------------------------------------------------------


def test_expected_stockouts_under_forecast() -> None:
    """When actual > forecast, the gap is a stockout."""
    # card A: actual 12, forecast 10 -> stockout 2
    # card B: actual 8, forecast 10 -> overstock, no stockout
    # card C: actual 10, forecast 10 -> match
    cards = [
        _card("A", [10.0], [12.0]),
        _card("B", [10.0], [8.0]),
        _card("C", [10.0], [10.0]),
    ]
    # Only card A contributes (1 stockout of magnitude 2)
    assert expected_stockouts(cards) == pytest.approx(2.0)


def test_expected_overstock_over_forecast() -> None:
    """When forecast > actual, the gap is overstock."""
    cards = [
        _card("A", [10.0], [12.0]),
        _card("B", [10.0], [8.0]),
    ]
    # Only card B contributes (1 overstock of magnitude 2)
    assert expected_overstock(cards) == pytest.approx(2.0)


def test_expected_stockouts_perfect_forecast_is_zero() -> None:
    cards = [
        _card("A", [10.0], [10.0]),
        _card("B", [20.0], [20.0]),
    ]
    assert expected_stockouts(cards) == 0.0


def test_expected_overstock_perfect_forecast_is_zero() -> None:
    cards = [
        _card("A", [10.0], [10.0]),
        _card("B", [20.0], [20.0]),
    ]
    assert expected_overstock(cards) == 0.0


def test_expected_stockouts_empty_input_is_nan() -> None:
    assert math.isnan(expected_stockouts([]))


def test_expected_overstock_empty_input_is_nan() -> None:
    assert math.isnan(expected_overstock([]))


# ---------------------------------------------------------------------------
# interval_coverage
# ---------------------------------------------------------------------------


def test_interval_coverage_returns_none_until_scorecard_grows_intervals() -> None:
    """The metric is wired but the scorecard has no interval fields today."""
    cards = [_card("A", [10.0], [10.0])]
    assert interval_coverage(cards) is None


# ---------------------------------------------------------------------------
# Determinism / sanity
# ---------------------------------------------------------------------------


def test_metrics_are_pure_no_module_state() -> None:
    """Calling the same metric twice with the same input returns the same value."""
    cards = [
        _card("A", [10.0, 20.0], [12.0, 18.0]),
    ]
    first = wape(cards)
    second = wape(cards)
    assert first == second

    first_h = horizon_level_error(cards)
    second_h = horizon_level_error(cards)
    assert first_h == second_h