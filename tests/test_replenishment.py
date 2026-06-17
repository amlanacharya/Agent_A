"""Tests for Phase 5.3 CB1: lead-time demand, safety stock, ROP.

Pure functions on a frozen dataclass config. Tests use
synthetic forecasts and known expected values; the
functions are pure and deterministic.
"""

from __future__ import annotations

import math

import pytest

from forecasting.replenishment import (
    ApprovalTier,
    InventoryState,
    ReplenishmentConfig,
    ReplenishmentRecommendation,
    classify_approval_tier,
    compute_lead_time_demand,
    compute_order_quantity,
    compute_replenishment,
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
    """Negative lead-time demand -> 0 (clamped).

    A zero lead-time demand with positive safety stock is
    *not* degenerate — the platform trusts the forecast's
    timing and only buffers for uncertainty. ROP collapses
    to the safety stock alone.
    """
    assert compute_reorder_point(0.0, 10.0) == 10.0
    assert compute_reorder_point(-5.0, 10.0) == 0.0


def test_rop_collapses_to_lead_time_when_safety_stock_zero() -> None:
    """A zero safety stock means a perfectly deterministic forecast.

    The ROP collapses to the lead-time demand alone — the
    platform trusts the forecast. Negative safety stock is
    still degenerate (clamped to 0).
    """
    assert compute_reorder_point(50.0, 0.0) == 50.0


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


# ---------------------------------------------------------------------------
# compute_order_quantity (CB2)
# ---------------------------------------------------------------------------


def _inv(current: float = 0.0, open_pos: float = 0.0) -> InventoryState:
    """Build an InventoryState with the given values."""
    return InventoryState(current_inventory=current, open_purchase_orders=open_pos)


def test_order_quantity_zero_when_inventory_above_target() -> None:
    """Current inventory alone covers the target -> no order."""
    cfg = ReplenishmentConfig()
    # Target 50, current 60 -> raw = -10 -> no order.
    assert compute_order_quantity(
        target_inventory=50.0,
        inventory=_inv(current=60.0),
        config=cfg,
    ) == 0.0


def test_order_quantity_zero_when_inventory_plus_open_pos_equals_target() -> None:
    """Inventory + open POs already match the target -> no order."""
    cfg = ReplenishmentConfig()
    # Target 50, current 30, open_pos 20 -> raw = 0 -> no order.
    assert compute_order_quantity(
        target_inventory=50.0,
        inventory=_inv(current=30.0, open_pos=20.0),
        config=cfg,
    ) == 0.0


def test_order_quantity_rounds_up_to_pack_size() -> None:
    """A raw quantity of 23 with pack_size=10 rounds up to 30."""
    cfg = ReplenishmentConfig(pack_size=10, moq=1)
    # raw 23 -> ceil(23/10)*10 = 30
    assert compute_order_quantity(
        target_inventory=23.0,
        inventory=_inv(),
        config=cfg,
    ) == 30.0


def test_order_quantity_handles_partial_pack_remainder() -> None:
    """A raw quantity exactly at a pack boundary doesn't bump up unnecessarily."""
    cfg = ReplenishmentConfig(pack_size=10, moq=1)
    # raw 20 -> ceil(20/10)*10 = 20 (no bump)
    assert compute_order_quantity(
        target_inventory=20.0,
        inventory=_inv(),
        config=cfg,
    ) == 20.0


def test_order_quantity_enforces_moq() -> None:
    """A packed quantity below MOQ is bumped up to MOQ."""
    cfg = ReplenishmentConfig(pack_size=10, moq=50)
    # raw 23 -> packed 30 (from ceil), but moq=50 -> bumped to 50.
    assert compute_order_quantity(
        target_inventory=23.0,
        inventory=_inv(),
        config=cfg,
    ) == 50.0


def test_order_quantity_default_moq_pack_size_one_passes_through() -> None:
    """Default config (pack=1, moq=1) passes the raw quantity through unchanged."""
    cfg = ReplenishmentConfig()
    # raw 23.7 -> packed = ceil(23.7/1)*1 = 23.7
    assert compute_order_quantity(
        target_inventory=23.7,
        inventory=_inv(),
        config=cfg,
    ) == 23.7


def test_order_quantity_subtracts_open_purchase_orders() -> None:
    """Open POs reduce the recommended order quantity."""
    cfg = ReplenishmentConfig(pack_size=1, moq=1)
    # Target 100, current 20, open_pos 30 -> raw = 50
    assert compute_order_quantity(
        target_inventory=100.0,
        inventory=_inv(current=20.0, open_pos=30.0),
        config=cfg,
    ) == 50.0


def test_order_quantity_zero_when_target_is_zero_or_negative() -> None:
    """Degenerate target_inventory -> no order (clamped to 0)."""
    cfg = ReplenishmentConfig()
    assert compute_order_quantity(
        target_inventory=0.0, inventory=_inv(), config=cfg,
    ) == 0.0
    assert compute_order_quantity(
        target_inventory=-10.0, inventory=_inv(), config=cfg,
    ) == 0.0


def test_order_quantity_handles_degenerate_pack_and_moq() -> None:
    """pack_size <= 0 and moq <= 0 are treated as 1 (defensive)."""
    cfg = ReplenishmentConfig(pack_size=0, moq=0)
    # Degenerate knobs become 1; raw 23.7 -> packed 23.7.
    assert compute_order_quantity(
        target_inventory=23.7,
        inventory=_inv(),
        config=cfg,
    ) == 23.7


def test_order_quantity_is_deterministic() -> None:
    """Same inputs -> same output."""
    cfg = ReplenishmentConfig(pack_size=10, moq=50)
    inv = _inv(current=20.0, open_pos=30.0)
    q1 = compute_order_quantity(
        target_inventory=100.0, inventory=inv, config=cfg
    )
    q2 = compute_order_quantity(
        target_inventory=100.0, inventory=inv, config=cfg
    )
    assert q1 == q2


def test_order_quantity_full_chain_example() -> None:
    """ROP=58.6 from CB1, current=20, open_pos=10, pack=5, moq=15.

    raw = 58.6 - 20 - 10 = 28.6 -> ceil(28.6/5)*5 = 30 -> >= moq=15 -> 30.
    """
    cfg = ReplenishmentConfig(pack_size=5, moq=15)
    assert compute_order_quantity(
        target_inventory=58.6,
        inventory=_inv(current=20.0, open_pos=10.0),
        config=cfg,
    ) == 30.0


# ---------------------------------------------------------------------------
# classify_approval_tier (CB3)
# ---------------------------------------------------------------------------


def test_zero_order_is_auto_tier() -> None:
    cfg = ReplenishmentConfig()
    assert classify_approval_tier(0.0, cfg) == "auto"


def test_negative_order_is_auto_tier() -> None:
    """Defensive: degenerate negative order -> auto (clamped at 0)."""
    cfg = ReplenishmentConfig()
    assert classify_approval_tier(-5.0, cfg) == "auto"


def test_small_order_is_small_tier() -> None:
    """An order strictly positive but at or below the small threshold -> small."""
    cfg = ReplenishmentConfig(approval_threshold_small=100.0)
    assert classify_approval_tier(50.0, cfg) == "small"


def test_at_small_threshold_inclusive() -> None:
    """An order exactly at the small threshold is still small (inclusive lower)."""
    cfg = ReplenishmentConfig(approval_threshold_small=100.0)
    assert classify_approval_tier(100.0, cfg) == "small"


def test_above_small_but_below_large_is_medium_tier() -> None:
    cfg = ReplenishmentConfig(
        approval_threshold_small=100.0,
        approval_threshold_large=10000.0,
    )
    assert classify_approval_tier(500.0, cfg) == "medium"


def test_at_large_threshold_still_medium() -> None:
    """An order exactly at the large threshold is still medium (inclusive upper)."""
    cfg = ReplenishmentConfig(
        approval_threshold_small=100.0,
        approval_threshold_large=10000.0,
    )
    assert classify_approval_tier(10000.0, cfg) == "medium"


def test_above_large_threshold_is_large_tier() -> None:
    cfg = ReplenishmentConfig(
        approval_threshold_small=100.0,
        approval_threshold_large=10000.0,
    )
    assert classify_approval_tier(50000.0, cfg) == "large"


def test_classify_is_deterministic() -> None:
    """Same inputs -> same tier."""
    cfg = ReplenishmentConfig(
        approval_threshold_small=100.0,
        approval_threshold_large=10000.0,
    )
    assert classify_approval_tier(500.0, cfg) == classify_approval_tier(500.0, cfg)


def test_classify_returns_approval_tier_literal() -> None:
    """The classifier returns one of the four closed-Literal values."""
    cfg = ReplenishmentConfig()
    valid: set[str] = {"auto", "small", "medium", "large"}
    for q in [0.0, 1.0, 100.0, 500.0, 10000.0, 50000.0]:
        assert classify_approval_tier(q, cfg) in valid


# ---------------------------------------------------------------------------
# compute_replenishment (CB4 orchestrator)
# ---------------------------------------------------------------------------


def test_compute_replenishment_returns_full_recommendation() -> None:
    """Every intermediate value is populated on the recommendation."""
    forecast = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    inv = _inv(current=20.0, open_pos=10.0)
    cfg = ReplenishmentConfig(pack_size=5, moq=15)
    rec = compute_replenishment(
        series_key="A",
        forecast=forecast,
        lead_time_days=4,
        forecast_std=2.0,
        inventory=inv,
        config=cfg,
    )
    assert isinstance(rec, ReplenishmentRecommendation)
    assert rec.series_key == "A"
    assert rec.lead_time_days == 4
    assert rec.forecast_std == 2.0
    assert rec.current_inventory == 20.0
    assert rec.open_purchase_orders == 10.0


def test_compute_replenishment_target_inventory_is_rop_plus_safety() -> None:
    """target_inventory = reorder_point + safety_stock."""
    forecast = [10.0, 12.0, 14.0, 16.0]
    cfg = ReplenishmentConfig()
    rec = compute_replenishment(
        series_key="A",
        forecast=forecast,
        lead_time_days=4,
        forecast_std=2.0,
        inventory=_inv(),
        config=cfg,
    )
    assert rec.target_inventory == rec.reorder_point + rec.safety_stock


def test_compute_replenishment_zero_order_has_auto_tier() -> None:
    """Inventory + POs already cover the target -> zero order -> auto tier.

    forecast = [10, 12, 14, 14], lead_time_days=4:
    - lead_time_demand = 50
    - safety_stock = 1.65 * 2 * sqrt(4) = 6.6
    - target_inventory = 50 + 6.6 + 6.6 = 63.2
    - inventory.current=70, open_pos=0 -> raw = 63.2 - 70 = -6.8 -> 0
    """
    forecast = [10.0, 12.0, 14.0, 14.0]
    rec = compute_replenishment(
        series_key="A",
        forecast=forecast,
        lead_time_days=4,
        forecast_std=2.0,
        inventory=_inv(current=70.0),
        config=ReplenishmentConfig(),
    )
    assert rec.order_quantity == 0.0
    assert rec.approval_tier == "auto"


def test_compute_replenishment_large_order_has_large_tier() -> None:
    """A high-quantity order is classified as large."""
    cfg = ReplenishmentConfig(
        pack_size=1,
        moq=1,
        approval_threshold_small=100.0,
        approval_threshold_large=10000.0,
    )
    # Forecast = [30000] * 5 -> lead_time_demand (4 days) = 120000
    # inventory = 0, open_pos = 0 -> raw = 120000 -> 120000 (no rounding)
    forecast = [30000.0] * 5
    rec = compute_replenishment(
        series_key="A",
        forecast=forecast,
        lead_time_days=4,
        forecast_std=1000.0,
        inventory=_inv(),
        config=cfg,
    )
    assert rec.order_quantity > 10000.0
    assert rec.approval_tier == "large"


def test_compute_replenishment_full_chain_example() -> None:
    """End-to-end: forecast + lead time + inventory + config -> recommendation.

    Inputs:
    - forecast = [10, 12, 14, 16, 18, 20]
    - lead_time_days = 4
    - forecast_std = 2.0
    - service_level_z = 1.65 (default)
    - inventory: current=20, open_pos=10
    - pack_size=5, moq=15, approval thresholds default

    Expected chain:
    - lead_time_demand = 10+12+14+16 = 52
    - safety_stock = 1.65 * 2 * sqrt(4) = 6.6
    - reorder_point = 52 + 6.6 = 58.6
    - target_inventory = 58.6 + 6.6 = 65.2
    - raw = 65.2 - 20 - 10 = 35.2 -> ceil(35.2/5)*5 = 40 -> >= moq=15 -> 40
    - order_quantity = 40 (small tier: 40 <= small_threshold=100)
    """
    cfg = ReplenishmentConfig(pack_size=5, moq=15)
    rec = compute_replenishment(
        series_key="A",
        forecast=[10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        lead_time_days=4,
        forecast_std=2.0,
        inventory=_inv(current=20.0, open_pos=10.0),
        config=cfg,
    )
    assert rec.lead_time_demand == pytest.approx(52.0)
    assert rec.safety_stock == pytest.approx(6.6)
    assert rec.reorder_point == pytest.approx(58.6)
    assert rec.target_inventory == pytest.approx(65.2)
    assert rec.order_quantity == 40.0
    assert rec.approval_tier == "small"


def test_compute_replenishment_no_order_when_inventory_sufficient() -> None:
    """Inventory + POs already cover the target -> order=0, tier=auto."""
    cfg = ReplenishmentConfig()
    rec = compute_replenishment(
        series_key="A",
        forecast=[100.0, 100.0, 100.0, 100.0],  # lead_time_demand = 400
        lead_time_days=4,
        forecast_std=10.0,
        inventory=_inv(current=500.0, open_pos=0.0),  # way above target
        config=cfg,
    )
    assert rec.order_quantity == 0.0
    assert rec.approval_tier == "auto"


def test_compute_replenishment_is_deterministic() -> None:
    """Same inputs -> same recommendation."""
    forecast = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    inv = _inv(current=20.0, open_pos=10.0)
    cfg = ReplenishmentConfig(pack_size=5, moq=15)
    rec1 = compute_replenishment(
        series_key="A",
        forecast=forecast,
        lead_time_days=4,
        forecast_std=2.0,
        inventory=inv,
        config=cfg,
    )
    rec2 = compute_replenishment(
        series_key="A",
        forecast=forecast,
        lead_time_days=4,
        forecast_std=2.0,
        inventory=inv,
        config=cfg,
    )
    assert rec1.order_quantity == rec2.order_quantity
    assert rec1.reorder_point == rec2.reorder_point
    assert rec1.approval_tier == rec2.approval_tier


# ---------------------------------------------------------------------------
# Full-chain integration tests (CB5)
# ---------------------------------------------------------------------------


def test_full_chain_three_series_independent() -> None:
    """Three series in one harness batch, each with its own inputs.

    The recommendation for one series must not contaminate
    another. (Today the function is per-series so this is
    almost tautological, but the test pins the contract:
    'call compute_replenishment separately for each series'.)
    """
    cfg = ReplenishmentConfig(pack_size=5, moq=10)
    forecasts = {
        "A": ([10.0, 12.0, 14.0, 16.0], _inv(current=20.0, open_pos=0.0)),
        "B": ([5.0, 5.0, 5.0, 5.0], _inv(current=0.0, open_pos=0.0)),
        "C": ([100.0, 100.0, 100.0, 100.0], _inv(current=500.0, open_pos=0.0)),
    }
    recommendations = {
        sk: compute_replenishment(
            series_key=sk,
            forecast=fc,
            lead_time_days=4,
            forecast_std=5.0,
            inventory=inv,
            config=cfg,
        )
        for sk, (fc, inv) in forecasts.items()
    }
    # A: lead_time_demand=52, target~70, current=20 -> order positive.
    assert recommendations["A"].order_quantity > 0
    # B: lead_time_demand=20, target~36, current=0 -> order positive.
    assert recommendations["B"].order_quantity > 0
    # C: lead_time_demand=400, target~420, current=500 -> no order.
    assert recommendations["C"].order_quantity == 0
    assert recommendations["C"].approval_tier == "auto"


def test_full_chain_with_degenerate_forecast_zero_std() -> None:
    """sigma=0 -> safety_stock=0 -> ROP = lead_time_demand only.

    The order covers exactly the lead-time demand (the
    platform trusts a perfectly deterministic forecast).
    """
    cfg = ReplenishmentConfig(pack_size=1, moq=1)
    rec = compute_replenishment(
        series_key="A",
        forecast=[10.0, 12.0, 14.0, 16.0],
        lead_time_days=4,
        forecast_std=0.0,  # perfectly deterministic
        inventory=_inv(current=0.0),
        config=cfg,
    )
    assert rec.safety_stock == 0.0
    assert rec.reorder_point == 52.0  # pure lead-time demand
    assert rec.target_inventory == 52.0
    assert rec.order_quantity == 52.0


def test_full_chain_with_open_po_just_below_target() -> None:
    """open_pos = ROP + safety - 1 -> raw = 1 -> small positive order.

    Pins: the platform surfaces a tiny order (1 unit) when
    the open POs are *just* below the target, rather than
    silently returning zero. The exact tier is config-dependent
    (1 <= small_threshold=100 -> small), so the test asserts
    the positive-order property, not the tier.
    """
    cfg = ReplenishmentConfig(pack_size=1, moq=1)
    # Build a setup where target = 100 exactly: lead_time_demand
    # = 100, sigma chosen so safety_stock = 0, ROP = 100,
    # target_inventory = 100. open_pos = 99 -> raw = 1.
    rec = compute_replenishment(
        series_key="A",
        forecast=[25.0] * 4,  # lead_time_demand = 100
        lead_time_days=4,
        forecast_std=0.0,  # safety_stock = 0
        inventory=_inv(current=0.0, open_pos=99.0),
        config=cfg,
    )
    assert rec.reorder_point == 100.0
    assert rec.target_inventory == 100.0
    assert rec.order_quantity == 1.0
    # The tier depends on the config's small_threshold; default
    # is 100, so 1 -> small. The important property is that the
    # order surfaces (not zero), not which tier it lands in.
    assert rec.order_quantity > 0


def test_full_chain_with_pack_size_larger_than_raw() -> None:
    """pack_size=100, raw=3 -> ceil(3/100)*100 = 100.

    Pins: a small raw quantity with a large pack size jumps
    up to the next pack boundary. The recommendation still
    classifies the order correctly.
    """
    cfg = ReplenishmentConfig(
        pack_size=100, moq=1,
        approval_threshold_small=50.0,
        approval_threshold_large=500.0,
    )
    rec = compute_replenishment(
        series_key="A",
        forecast=[1.0] * 4,  # lead_time_demand = 4
        lead_time_days=4,
        forecast_std=0.0,
        inventory=_inv(),
        config=cfg,
    )
    assert rec.order_quantity == 100.0  # ceil(4/100)*100
    assert rec.approval_tier == "medium"  # 100 > 50 (small) and <= 500 (large)


def test_full_chain_recommendation_is_pydantic_serialisable() -> None:
    """The recommendation round-trips through model_dump.

    The cockpit serialises the recommendation; if it can't,
    the audit log is broken.
    """
    cfg = ReplenishmentConfig(pack_size=5, moq=15)
    rec = compute_replenishment(
        series_key="A",
        forecast=[10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        lead_time_days=4,
        forecast_std=2.0,
        inventory=_inv(current=20.0, open_pos=10.0),
        config=cfg,
    )
    dumped = rec.model_dump()
    # Required keys are all present and have the right types.
    assert dumped["series_key"] == "A"
    assert isinstance(dumped["lead_time_demand"], float)
    assert isinstance(dumped["approval_tier"], str)
    # Round-trip: re-construct from the dump.
    restored = ReplenishmentRecommendation.model_validate(dumped)
    assert restored.order_quantity == rec.order_quantity
    assert restored.approval_tier == rec.approval_tier