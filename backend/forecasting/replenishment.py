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


__all__ = (
    "ReplenishmentConfig",
    "compute_lead_time_demand",
    "compute_safety_stock",
    "compute_reorder_point",
)