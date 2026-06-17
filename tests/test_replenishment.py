"""Tests for Phase 5.3 CB1: lead-time demand, safety stock, ROP.

Pure functions on a frozen dataclass config. Tests use
synthetic forecasts and known expected values; the
functions are pure and deterministic.
"""

from __future__ import annotations

import math

import pytest

from forecasting.replenishment import (
    ReplenishmentConfig,
    compute_lead_time_demand,
    compute_reorder_point,
    compute_safety_stock,
)


# ---------------------------------------------------------------------------
# ReplenishmentConfig
# ---------------------------------------------------------------------------


def test_replenishment_config_defaults() -> None:
    """FMCG-conservative defaults: 1.65 z, 7-day safety buffer, 1 MOQ/pack."""
    cfg = ReplenishmentConfig()
    assert cfg.service_level_z == 1.65
    assert cfg.safety_stock_days == 7.0
    assert cfg.moq == 1
    assert cfg.pack_size == 1
    assert cfg.approval_threshold_small == 100.0
    assert cfg.approval_threshold_large == 10000.0


def test_replenishment_config_is_frozen() -> None:
    """Defensive: a mid-call threshold change would invalidate the calculation."""
    cfg = ReplenishmentConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.service_level_z = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_lead_time_demand
# ---------------------------------------------------------------------------


def test_lead_time_demand_sums_first_n_forecast_steps() -> None:
    """Sum the forecast over the lead-time window."""
    forecast = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    # lead-time = 4 -> sum first 4 = 10+12+14+16 = 52
    assert compute_lead_time_demand(forecast, 4) == 52.0


def test_lead_time_demand_handles_forecast_shorter_than_window() -> None:
    """A forecast shorter than the lead-time window sums what's available."""
    forecast = [10.0, 20.0]
    # lead-time = 4 but only 2 forecast steps -> sum 10+20 = 30
    assert compute_lead_time_demand(forecast, 4) == 30.0


def test_lead_time_demand_empty_forecast_returns_zero() -> None:
    assert compute_lead_time_demand([], 4) == 0.0


def test_lead_time_demand_zero_or_negative_lead_time_returns_zero() -> None:
    forecast = [10.0, 20.0]
    assert compute_lead_time_demand(forecast, 0) == 0.0
    assert compute_lead_time_demand(forecast, -3) == 0.0


def test_lead_time_demand_exact_match() -> None:
    """When the forecast exactly covers the lead-time window, no partial is used."""
    forecast = [5.0, 5.0, 5.0]
    # lead-time = 3, forecast length = 3 -> sum 5+5+5 = 15
    assert compute_lead_time_demand(forecast, 3) == 15.0


# ---------------------------------------------------------------------------
# compute_safety_stock
# ---------------------------------------------------------------------------


def test_safety_stock_zero_when_forecast_std_zero() -> None:
    """A perfectly deterministic forecast has no uncertainty to buffer."""
    assert compute_safety_stock(0.0, 4.0, 1.65) == 0.0


def test_safety_stock_scales_with_sqrt_of_lead_time() -> None:
    """safety_stock = z * sigma * sqrt(L).

    Doubling lead time multiplies safety stock by sqrt(2),
    not 2 (uncertainty accumulates sub-linearly with time).
    """
    z = 1.65
    sigma = 2.0
    ss1 = compute_safety_stock(sigma, 1.0, z)
    ss4 = compute_safety_stock(sigma, 4.0, z)
    assert ss4 == pytest.approx(ss1 * math.sqrt(4.0))


def test_safety_stock_scales_linearly_with_z() -> None:
    """Doubling z doubles safety stock (linear)."""
    sigma = 2.0
    L = 4.0
    ss1 = compute_safety_stock(sigma, L, 1.0)
    ss2 = compute_safety_stock(sigma, L, 2.0)
    assert ss2 == pytest.approx(2 * ss1)


def test_safety_stock_known_value() -> None:
    """Hand-computable: z=1.65, sigma=2.0, L=4 -> 1.65 * 2 * 2 = 6.6."""
    assert compute_safety_stock(2.0, 4.0, 1.65) == pytest.approx(6.6)


def test_safety_stock_negative_inputs_return_zero() -> None:
    """Degenerate config -> 0 (don't corrupt downstream calculations)."""
    assert compute_safety_stock(-1.0, 4.0, 1.65) == 0.0
    assert compute_safety_stock(2.0, -1.0, 1.65) == 0.0
    assert compute_safety_stock(2.0, 4.0, -1.65) == 0.0


# ---------------------------------------------------------------------------
# compute_reorder_point
# ---------------------------------------------------------------------------


def test_rop_equals_lead_time_plus_safety() -> None:
    """ROP = lead-time demand + safety stock."""
    assert compute_reorder_point(50.0, 10.0) == 60.0


def test_rop_zero_when_lead_time_demand_zero() -> None:
    assert compute_reorder_point(0.0, 10.0) == 0.0


def test_rop_zero_when_safety_stock_zero() -> None:
    """A zero safety stock means the platform doesn't buffer for uncertainty.

    The ROP collapses to the lead-time demand alone — but
    the policy enforces ROP >= 0, so this degenerate case
    (negative safety stock would be caught earlier) returns
    0.
    """
    assert compute_reorder_point(50.0, 0.0) == 0.0


def test_rop_negative_inputs_return_zero() -> None:
    assert compute_reorder_point(-10.0, 5.0) == 0.0
    assert compute_reorder_point(50.0, -5.0) == 0.0


def test_rop_full_chain_example() -> None:
    """Lead-time = 4, forecast = [10, 12, 14, 16, ...], std = 2.0, z = 1.65.

    lead_time_demand = 52
    safety_stock = 1.65 * 2 * sqrt(4) = 6.6
    ROP = 58.6
    """
    forecast = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    lead_time_demand = compute_lead_time_demand(forecast, 4)
    safety_stock = compute_safety_stock(2.0, 4.0, 1.65)
    rop = compute_reorder_point(lead_time_demand, safety_stock)
    assert rop == pytest.approx(58.6)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_lead_time_demand_is_deterministic() -> None:
    forecast = [10.0, 12.0, 14.0]
    assert compute_lead_time_demand(forecast, 2) == compute_lead_time_demand(forecast, 2)


def test_safety_stock_is_deterministic() -> None:
    assert compute_safety_stock(2.0, 4.0, 1.65) == compute_safety_stock(2.0, 4.0, 1.65)


def test_rop_is_deterministic() -> None:
    assert compute_reorder_point(50.0, 10.0) == compute_reorder_point(50.0, 10.0)