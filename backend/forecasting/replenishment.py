"""Deterministic replenishment policy (Phase 5.3).

Turns a forecast + lead time + inventory state into a
replenishment recommendation. The whole module is pure math:
no I/O, no LLM, no harness dependency. The harness caller
passes the forecast in; this module returns the
recommendation.

Phase 5.3 sub-checkboxes:

* CB1 (this commit): lead-time demand, safety stock, reorder
  point (ROP). Pure math on a forecast array and a config.
* CB2: MOQ + pack size + inventory reconciliation (order
  quantity calc).
* CB3: Approval tiers (small / medium / large / auto).
* CB4: ``ReplenishmentRecommendation`` model + the
  ``compute_replenishment`` orchestrator that ties it all
  together.
* CB5: full-chain integration tests.

Design rules:

* **Pure functions.** No I/O, no LLM, no harness dependency.
  Tests pass synthetic forecasts + inventories; the function
  returns a typed record.
* **FROZEN dataclasses.** The config dataclass is frozen so
  a mid-call threshold change cannot invalidate the
  calculation.
* **Closed Literal surface.** Approval tiers, lead-time
  units, etc., are all closed Literals.
* **NaN-safe.** Empty or short forecasts return sensible
  defaults (0.0 for lead-time demand, 0.0 for safety stock
  when std is zero).
* **No inventory-side rounding.** Inventory can be a float
  (case-pack-partial). Order quantity is integer (whole
  cases) — handled in CB2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplenishmentConfig:
    """All thresholds for the replenishment policy.

    Defaults are FMCG-conservative. The service-level z=1.65
    corresponds to roughly a 95% service level under a normal
    forecast-error assumption. ``moq`` and ``pack_size``
    defaults are 1 (no supplier constraint); CB2 introduces
    the tests that exercise non-trivial MOQ/pack.

    The dataclass is frozen so a mid-call threshold change
    cannot invalidate the calculation. ``.env`` overrides
    are out of scope for this commit (the platform reads
    thresholds from code today; future Phase 6 hook would
    add ``ReplenishmentConfig.from_env()``).
    """

    # Service-level z-score: ~1.65 -> 95% service level under
    # the normal-distribution assumption.
    service_level_z: float = 1.65
    # Extra safety buffer in days of lead-time-equivalent demand.
    safety_stock_days: float = 7.0
    # Supplier constraint: minimum order quantity.
    moq: int = 1
    # Order in multiples of pack_size.
    pack_size: int = 1
    # Approval-tier thresholds (absolute units of the
    # recommendation's ``order_quantity``). Below the small
    # threshold -> auto; below medium -> small approval;
    # below large -> medium; above large -> large.
    approval_threshold_small: float = 100.0
    approval_threshold_large: float = 10000.0


# ---------------------------------------------------------------------------
# CB1: lead-time demand, safety stock, reorder point
# ---------------------------------------------------------------------------


def compute_lead_time_demand(
    forecast: list[float],
    lead_time_days: int,
) -> float:
    """Sum the forecast over the lead-time window.

    The forecast is the per-step (per-day, per-week, ...)
    demand projection from the harness. The lead-time window
    is the first ``lead_time_days`` entries.

    Edge cases:

    * ``forecast`` empty -> 0.0 (no projection, no order).
    * ``lead_time_days`` <= 0 -> 0.0 (no time to order).
    * ``forecast`` shorter than ``lead_time_days`` -> sum
      whatever is available (partial window). The
      recommendation surfaces this as a degraded forecast;
      the platform's policy is "order based on what we
      know" rather than blocking on missing data.

    Returns a float; rounding happens at the order-quantity
    step in CB2.
    """
    if lead_time_days <= 0 or not forecast:
        return 0.0
    n = min(lead_time_days, len(forecast))
    return float(sum(forecast[:n]))


def compute_safety_stock(
    forecast_std: float,
    lead_time_days: float,
    service_level_z: float,
) -> float:
    """z * forecast_std * sqrt(lead_time_days).

    Standard safety-stock formula under the normal-distribution
    assumption. When ``forecast_std`` is 0 (a perfectly
    deterministic forecast), the safety stock is 0 — there
    is no uncertainty to buffer against.

    NaN-safe: negative inputs return 0.0 (the caller may
    have a degenerate config; we surface as "no safety
    stock" rather than a negative number that would corrupt
    downstream calculations).
    """
    if forecast_std <= 0 or lead_time_days <= 0 or service_level_z <= 0:
        return 0.0
    return float(service_level_z * forecast_std * math.sqrt(lead_time_days))


def compute_reorder_point(
    lead_time_demand: float,
    safety_stock: float,
) -> float:
    """ROP = lead-time demand + safety stock.

    Pure addition. Negative inputs return 0.0 — the caller
    may have a degenerate config; we surface as "no ROP"
    rather than a negative number.
    """
    if lead_time_demand <= 0 or safety_stock <= 0:
        return 0.0
    return float(lead_time_demand + safety_stock)


# ---------------------------------------------------------------------------
# CB2: order quantity (MOQ + pack size + inventory reconciliation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InventoryState:
    """The current inventory posture at order time.

    ``current_inventory`` is the on-hand quantity at the
    moment the order is being computed (case-pack-partial
    units are fine — it's a float).

    ``open_purchase_orders`` is the quantity already on
    order with the supplier but not yet received. These
    reduce the recommended order because they will arrive
    before the inventory runs out (assuming the supplier
    delivers on schedule — a separate concern).

    Both fields default to 0.0 (no inventory, no open POs)
    so the dataclass can be constructed without arguments
    in tests.
    """

    current_inventory: float = 0.0
    open_purchase_orders: float = 0.0


def compute_order_quantity(
    *,
    target_inventory: float,
    inventory: InventoryState,
    config: ReplenishmentConfig,
) -> float:
    """Compute the order quantity from a target inventory level.

    Steps:

    1. ``raw_qty = target_inventory - current_inventory - open_pos``
    2. If ``raw_qty <= 0`` -> 0.0 (no order needed; inventory
       alone covers the target).
    3. Round ``raw_qty`` UP to a multiple of ``pack_size``:
       ``packed = ceil(raw_qty / pack_size) * pack_size``.
       Never round down — under-ordering leaves the platform
       short.
    4. Enforce MOQ as a floor: if ``packed < moq``, set
       ``packed = moq``. The supplier won't ship less than
       MOQ even if pack_size * 1 is below it.
    5. Return ``packed``.

    Edge cases:

    * ``target_inventory <= 0`` -> 0.0 (no positive target,
      no order).
    * ``pack_size <= 0`` is treated as ``pack_size = 1``
      (defensive — a degenerate config surface as "pack
      ignored" rather than a math error).
    * ``moq <= 0`` is treated as ``moq = 1``.
    * The order quantity is a float; the float's fractional
      part is meaningful only when pack_size is non-unit.
      With pack_size=1 and moq=1, the order is exactly
      ``raw_qty`` as a float.

    Returns 0.0 for "no order needed" — the caller branches
    on the return value to decide whether to skip the
    approval workflow entirely.
    """
    if target_inventory <= 0:
        return 0.0
    raw_qty = target_inventory - inventory.current_inventory - inventory.open_purchase_orders
    if raw_qty <= 0:
        return 0.0
    pack_size = max(int(config.pack_size), 1)
    moq = max(int(config.moq), 1)
    # Round UP to a multiple of pack_size. When pack_size=1
    # the rounding is a no-op (raw_qty is already a whole or
    # fractional unit that the platform can order in any
    # quantity). For pack_size > 1 the ceil-bump ensures the
    # order covers at least the raw need.
    if pack_size > 1:
        packed = math.ceil(raw_qty / pack_size) * pack_size
    else:
        packed = raw_qty
    if packed < moq:
        packed = moq
    return float(packed)


__all__ = (
    "ReplenishmentConfig",
    "InventoryState",
    "compute_lead_time_demand",
    "compute_safety_stock",
    "compute_reorder_point",
    "compute_order_quantity",
)