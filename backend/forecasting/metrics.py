"""Metric portfolio for Phase 5 (CB1).

The Phase 5 plan calls for a portfolio of higher-level metrics
that roll up the per-scorecard ``ModelScorecard`` data:

* **WAPE** — Weighted Absolute Percent Error. Industry-standard
  scale-free metric; ``sum(|actual - forecast|) / sum(|actual|)``.
  Less sensitive to low-volume series than MAPE because the
  denominator is total demand, not per-point demand.
* **Bias** — already on ``ModelScorecard``; re-exported as a
  portfolio metric for completeness.
* **Horizon-level error** — error as a function of forecast step.
  The plan calls for "horizon-level error" as a separate
  metric so the cockpit can show "we're great at week 1, bad at
  week 4". Computed as per-step MAE across all scorecards
  sharing the same forecast step.
* **Segment-level error** — error aggregated per segment
  (``segment_id``). The plan calls for "segment scorecards" as
  a Phase 5.2 input; the metrics rollup is the prerequisite.
* **Interval coverage** — fraction of actuals that fall within
  a forecast interval (e.g. 80% PI). The platform does not yet
  emit quantile forecasts; the metric is wired but reports
  ``None`` when the scorecard carries no interval fields.
* **Stockout / overstock impact** — the gap between the
  forecast and the inventory posture. Computed as
  ``mean(max(0, actual - forecast))`` (expected stockouts)
  and ``mean(max(0, forecast - actual))`` (expected overstock).

All functions are pure: they consume ``Sequence[ModelScorecard]``
and return either a scalar number or a typed structure. The
typed contracts live in ``contracts.py`` alongside the
``ModelScorecard`` they consume.

Design rules:

* **Pure functions, no I/O.** No files, no network, no
  ``global`` state. Tests pass a list of synthetic scorecards
  and assert the metric value.
* **NaN-safe.** Empty input returns sensible defaults (NaN
  for averages over zero items, ``None`` for unavailable
  intervals). No division-by-zero surprises.
* **No floating-point arithmetic outside numpy/pandas.**
  Pure Python float math is fine for the simple metrics; the
  rollups use ``statistics.mean`` to keep dependencies tight.
* **Typed outputs.** Every metric is either a number
  (``float`` / ``int``) or a typed Pydantic model. Free-form
  dicts are an anti-pattern.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Literal

from forecasting.contracts import ModelScorecard


# ---------------------------------------------------------------------------
# WAPE
# ---------------------------------------------------------------------------


def wape(scorecards: Sequence[ModelScorecard]) -> float:
    """Weighted Absolute Percent Error across scorecards.

    ``sum(|actual - forecast|) / sum(|actual|)`` computed across
    all (forecast, actual) pairs in the input. Returns NaN when
    the total actual demand is zero (a degenerate case the
    platform surfaces as "the denominator is zero, WAPE is
    undefined").

    WAPE is the industry-standard scale-free metric — preferred
    over MAPE because MAPE per-point division blows up near
    zero demand, whereas WAPE weights by total demand and stays
    well-behaved.
    """
    if not scorecards:
        return float("nan")
    total_abs_error = 0.0
    total_abs_actual = 0.0
    for card in scorecards:
        actual_arr = card.actual
        forecast_arr = card.forecast
        for a, f in zip(actual_arr, forecast_arr):
            total_abs_error += abs(a - f)
            total_abs_actual += abs(a)
    if total_abs_actual == 0:
        return float("nan")
    return total_abs_error / total_abs_actual


def mean_bias(scorecards: Sequence[ModelScorecard]) -> float:
    """Mean of the per-scorecard signed bias.

    Positive = under-forecast, negative = over-forecast. NaN on
    empty input.
    """
    if not scorecards:
        return float("nan")
    return float(statistics.mean(card.bias for card in scorecards))


# ---------------------------------------------------------------------------
# Horizon-level error
# ---------------------------------------------------------------------------


def horizon_level_error(
    scorecards: Sequence[ModelScorecard],
) -> list[float]:
    """Per-step MAE across all scorecards.

    The output length is the maximum ``horizon`` across the
    input. Step ``h`` is the mean of ``|actual[h] - forecast[h]|``
    over all scorecards where ``h < len(card.actual)``. Scorecards
    shorter than ``h`` are skipped for that step.

    Returns an empty list on empty input. Returns NaN for a
    step with no contributing scorecards (defensive — the
    caller should not see this in practice because every
    scorecard has the same horizon length per fold).
    """
    if not scorecards:
        return []
    max_h = max(card.horizon for card in scorecards)
    result: list[float] = []
    for h in range(max_h):
        abs_errors: list[float] = []
        for card in scorecards:
            if h < len(card.actual) and h < len(card.forecast):
                abs_errors.append(abs(card.actual[h] - card.forecast[h]))
        if not abs_errors:
            result.append(float("nan"))
        else:
            result.append(float(statistics.mean(abs_errors)))
    return result


def cumulative_mae_at_horizon(
    scorecards: Sequence[ModelScorecard],
    h: int,
) -> float:
    """Mean cumulative absolute error over the first ``h`` forecast steps.

    For each scorecard, sum ``|actual - forecast|`` over the
    first ``h`` steps, then average across scorecards. NaN on
    empty input or ``h <= 0``.
    """
    if not scorecards or h <= 0:
        return float("nan")
    cumulatives: list[float] = []
    for card in scorecards:
        end = min(h, len(card.actual), len(card.forecast))
        if end == 0:
            continue
        cumulatives.append(
            sum(abs(a - f) for a, f in zip(card.actual[:end], card.forecast[:end]))
        )
    if not cumulatives:
        return float("nan")
    return float(statistics.mean(cumulatives))


# ---------------------------------------------------------------------------
# Stockout / overstock impact
# ---------------------------------------------------------------------------


def expected_stockouts(scorecards: Sequence[ModelScorecard]) -> float:
    """Expected stockouts per step: ``mean(max(0, actual - forecast))``.

    When ``actual > forecast`` the forecast under-supplies; the
    gap is the expected stockout. NaN on empty input.
    """
    if not scorecards:
        return float("nan")
    values: list[float] = []
    for card in scorecards:
        for a, f in zip(card.actual, card.forecast):
            if a > f:
                values.append(a - f)
    if not values:
        return 0.0
    return float(statistics.mean(values))


def expected_overstock(scorecards: Sequence[ModelScorecard]) -> float:
    """Expected overstock per step: ``mean(max(0, forecast - actual))``.

    When ``forecast > actual`` the forecast over-supplies; the
    gap is the expected overstock. NaN on empty input.
    """
    if not scorecards:
        return float("nan")
    values: list[float] = []
    for card in scorecards:
        for a, f in zip(card.actual, card.forecast):
            if f > a:
                values.append(f - a)
    if not values:
        return 0.0
    return float(statistics.mean(values))


# ---------------------------------------------------------------------------
# Interval coverage
# ---------------------------------------------------------------------------

# The interval-coverage metric is wired but only returns a number
# when the scorecard carries ``lower`` and ``upper`` arrays
# alongside ``forecast``. The current ``ModelScorecard`` does
# not have those fields; this function returns None for now and
# becomes live when quantile forecast support lands (a separate
# CB beyond Phase 5.1).
#
# When the platform grows quantile forecasts, ``ModelScorecard``
# gains ``lower: list[float]`` and ``upper: list[float]`` fields;
# the function reads them and the ``IntervalCoverage`` result
# on the portfolio becomes a number instead of None.

def interval_coverage(
    scorecards: Sequence[ModelScorecard],
) -> float | None:
    """Fraction of actuals within the forecast interval (80% PI by convention).

    Returns None until ``ModelScorecard`` carries ``lower`` and
    ``upper`` arrays (Phase 5.1 wires the seam; the actual
    quantile forecasts are a separate concern).
    """
    # Defensive: the function exists, returns None today. When
    # the model scorecard grows interval fields, replace this
    # body with the actual computation:
    #
    #     total_in = 0
    #     total_n = 0
    #     for card in scorecards:
    #         for a, lo, hi in zip(card.actual, card.lower, card.upper):
    #             total_n += 1
    #             if lo <= a <= hi:
    #                 total_in += 1
    #     return total_in / total_n if total_n else float("nan")
    return None


__all__ = (
    "wape",
    "mean_bias",
    "horizon_level_error",
    "cumulative_mae_at_horizon",
    "expected_stockouts",
    "expected_overstock",
    "interval_coverage",
)