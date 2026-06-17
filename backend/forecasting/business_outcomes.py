"""Phase 7 CB5: business-outcomes rollup between monitoring runs.

The business-outcomes engine consumes the run's scorecards,
approval events, and free-form planner overrides and rolls them
up into a ``BusinessOutcomesReport`` — the typed shape the
cockpit and the four recurring artifacts surface under the
MLOps Monitor tab. The five signal kinds map to the plan's
business-outcomes checklist:

* **Expected stockouts** — per-step ``max(0, actual - forecast)``
  average across all scorecards. Reuses the math in
  ``metrics.expected_stockouts``.
* **Expected overstock** — per-step ``max(0, forecast - actual)``
  average across all scorecards. Reuses the math in
  ``metrics.expected_overstock``.
* **Service level** — fraction of per-step gaps where the
  forecast met actual demand (``forecast >= actual``). Bounded
  to ``[0, 1]`` by the ``BusinessOutcomesReport`` contract.
* **Planner overrides** — free-form descriptions of human
  decisions that diverged from the platform's recommendation.
  Pass-through from the caller; the engine does not interpret.
* **Approval patterns** — count map of ``APPROVE`` / ``REJECT`` /
  ``DEFER`` decisions parsed from the in-process approval audit
  log. The engine reads ``ApprovalEvent.notes`` for the
  ``decision=<VALUE>`` substring (the same format the gateway
  writes). ``raised`` / ``expired`` / ``corrected`` events are
  ignored.

Design:

* **Pure function, no I/O.** ``summarise_business_outcomes`` is
  a pure function of (run_id, scorecards, events,
  planner_overrides). The scheduler reads the previous run's
  audit log and scorecards from disk and calls the function;
  the function does not touch the filesystem.
* **Defaults to 0.0 / empty, never NaN.** Empty inputs produce
  ``(0.0, 0.0)`` for the stockout / overstock pair, ``0.0`` for
  service level, and ``{}`` for the approval map. The cockpit
  never has to special-case the rendering.
* **Reuses ``metrics`` math.** Stockout / overstock rollups
  call into ``metrics.expected_stockouts`` and
  ``metrics.expected_overstock`` so the two modules stay
  consistent — drift, business outcomes, and the cockpit's
  "stockout impact" widget all agree on the same number.
* **No LLM.** Fully deterministic.

The four public functions are:

* ``stockout_overstock_from_scorecards(scorecards)`` — pair
  of mean gaps
* ``approval_pattern_from_events(events)`` — count map
* ``service_level_from_scorecards(scorecards)`` — fraction in
  ``[0, 1]``
* ``summarise_business_outcomes(...)`` — top-level
  orchestrator
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from forecasting.contracts import (
    ApprovalEvent,
    BusinessOutcomesReport,
    ModelScorecard,
)
from forecasting.metrics import (
    expected_overstock,
    expected_stockouts,
)


# The gateway writes ``notes="decision=APPROVE reason=..."``; the
# regex picks the first ``decision=<UPPERCASE>`` token out of
# notes. Anchored to the start of the substring so a free-form
# reason ("decided to APPROVE this") does not collide.
_DECISION_RE = re.compile(r"\bdecision=(APPROVE|REJECT|DEFER)\b")


def _new_id(prefix: str) -> str:
    """Tiny test-friendly id helper — re-imported in the tests.

    The function lives here (rather than in ``approval_gateway``)
    so the test for the approval-pattern engine does not pull
    the gateway in just to build an id. The format matches
    ``approval_gateway._new_id`` (prefix + 8 hex chars).
    """
    import uuid

    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# stockout_overstock_from_scorecards
# ---------------------------------------------------------------------------


def _clamp_zero(value: float) -> float:
    """Convert a NaN / inf to ``0.0``.

    The ``metrics`` layer returns ``float('nan')`` on empty
    input by design (the cockpit renders NaN as "no data" —
    its own contract). The monitoring layer follows the
    opposite convention: empty input is the zero baseline, and
    a NaN / inf in the report would crash the Pydantic
    ``BusinessOutcomesReport`` constructor (a ``float`` field
    is valid for NaN, but the cockpit's downstream rendering
    assumes a number). This helper is the seam: it keeps the
    metrics layer's "no data" semantics where they belong and
    translates them to the monitoring layer's "0.0 baseline."
    """
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def stockout_overstock_from_scorecards(
    scorecards: Sequence[ModelScorecard],
) -> tuple[float, float]:
    """Return ``(expected_stockouts, expected_overstock)``.

    Reuses the math in ``metrics.py`` for the two metrics and
    translates the metrics layer's NaN-on-empty semantics to
    the monitoring layer's ``0.0`` baseline via
    ``_clamp_zero``. The two metrics are computed in a single
    pass over the scorecards, not by re-iterating the input
    twice — the math is cheap either way, but the single-pass
    shape matches the ``metrics.py`` helper layout.
    """
    stockouts = _clamp_zero(expected_stockouts(scorecards))
    overstock = _clamp_zero(expected_overstock(scorecards))
    return stockouts, overstock


# ---------------------------------------------------------------------------
# approval_pattern_from_events
# ---------------------------------------------------------------------------


def approval_pattern_from_events(
    events: Sequence[ApprovalEvent],
) -> dict[str, int]:
    """Count ``APPROVE`` / ``REJECT`` / ``DEFER`` decisions in the audit log.

    The engine reads ``event_type == "decided"`` and parses
    ``notes`` for the ``decision=<VALUE>`` token the gateway
    writes on every decision. Events with no decision token
    (e.g. a malformed audit line) are silently ignored. The
    returned map contains only decisions that were observed
    (no zero-valued keys), so the cockpit can iterate ``keys()``
    without filtering empties.
    """
    counts: dict[str, int] = {}
    for event in events:
        if event.event_type != "decided":
            continue
        match = _DECISION_RE.search(event.notes)
        if match is None:
            continue
        decision = match.group(1)
        counts[decision] = counts.get(decision, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# service_level_from_scorecards
# ---------------------------------------------------------------------------


def service_level_from_scorecards(
    scorecards: Sequence[ModelScorecard],
) -> float:
    """Fraction of per-step gaps where the forecast met actual demand.

    A step is "served" iff ``forecast[h] >= actual[h]`` for
    some index ``h`` in ``(0, len(scorecard.actual))``. The
    service level is the count of served steps divided by the
    total step count. Returns ``0.0`` on empty input (no
    steps, no service).

    The contract layer bounds the result to ``[0, 1]`` via the
    ``ge=0.0, le=1.0`` Pydantic constraint on
    ``BusinessOutcomesReport.service_level``; the engine still
    returns a fraction in that range so a malformed scorecard
    that produces a NaN would still fail Pydantic validation
    at the orchestrator layer.
    """
    if not scorecards:
        return 0.0
    served = 0
    total = 0
    for scorecard in scorecards:
        for forecast, actual in zip(scorecard.forecast, scorecard.actual):
            total += 1
            if forecast >= actual:
                served += 1
    if total == 0:
        return 0.0
    return served / total


# ---------------------------------------------------------------------------
# summarise_business_outcomes — top-level orchestrator
# ---------------------------------------------------------------------------


def summarise_business_outcomes(
    *,
    run_id: str,
    scorecards: Sequence[ModelScorecard],
    events: Sequence[ApprovalEvent],
    planner_overrides: Sequence[str],
) -> BusinessOutcomesReport:
    """Top-level business-outcomes engine.

    Combines the three signal kinds into one typed
    ``BusinessOutcomesReport``. Tolerant of empty inputs on
    all three axes: an empty scorecard list returns zero
    stockout / overstock / service level, an empty event list
    returns an empty approval map, and an empty
    ``planner_overrides`` list returns an empty list.
    """
    stockouts, overstock = stockout_overstock_from_scorecards(scorecards)
    service_level = service_level_from_scorecards(scorecards)
    approval_patterns = approval_pattern_from_events(events)
    return BusinessOutcomesReport(
        run_id=run_id,
        expected_stockouts=stockouts,
        expected_overstock=overstock,
        service_level=service_level,
        planner_overrides=list(planner_overrides),
        approval_patterns=approval_patterns,
    )


__all__ = (
    "approval_pattern_from_events",
    "service_level_from_scorecards",
    "stockout_overstock_from_scorecards",
    "summarise_business_outcomes",
    # _new_id is exposed (with underscore) for tests that need
    # to construct ApprovalEvent ids without pulling in
    # approval_gateway. The pattern matches ``data_drift`` and
    # ``model_drift`` — underscore-prefixed helpers stay out of
    # the public surface but are importable by name.
    "_new_id",
)
