"""Tests for Phase 7 CB5: business_outcomes module.

Covers ``summarise_business_outcomes(...)`` and the three
helper functions the engine uses to roll the inputs up:

* ``stockout_overstock_from_scorecards(scorecards)`` — reuses
  the math in ``metrics.py`` (expected stockouts / overstock per
  step) and returns the pair.
* ``approval_pattern_from_events(events)`` — counts APPROVE /
  REJECT / DEFER decisions from a list of ``ApprovalEvent``s
  (the ``decided`` events; ``raised`` / ``expired`` / ``corrected``
  are ignored).
* ``service_level_from_scorecards(scorecards)`` — fraction of
  per-step gaps where the forecast met actual demand (forecast
  >= actual). Returns 0.0 on empty input; bounded to ``[0, 1]``
  by the contract layer.

The orchestrator ``summarise_business_outcomes(scorecards,
events, planner_overrides)`` combines the three into a
``BusinessOutcomesReport``.
"""

from __future__ import annotations

import pytest

from forecasting.approval_gateway import InProcessApprovalGateway
from forecasting.business_outcomes import (
    _new_id,
    approval_pattern_from_events,
    service_level_from_scorecards,
    stockout_overstock_from_scorecards,
    summarise_business_outcomes,
)
from forecasting.contracts import (
    ApprovalEvent,
    ApprovalRequest,
    BusinessOutcomesReport,
    ModelScorecard,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scorecard(
    *,
    forecast: list[float],
    actual: list[float],
) -> ModelScorecard:
    """Build a tiny ModelScorecard with a given forecast / actual pair.

    MASE / MAE / RMSE are placeholders — the business-outcomes
    engine never reads them, but ``ModelScorecard`` is a
    Pydantic model that requires the fields.
    """
    return ModelScorecard(
        model_family="naive",
        series_key="A",
        fold_cutoff="2026-01-01",
        horizon=len(forecast),
        forecast=forecast,
        actual=actual,
        mae=0.0,
        rmse=0.0,
        mase=0.0,
        bias=0.0,
    )


def _event(
    *,
    event_type: str,
    notes: str = "",
) -> ApprovalEvent:
    """Build a tiny ApprovalEvent for the tests."""
    return ApprovalEvent(
        event_id=_new_id("ev"),
        request_id="req-1",
        run_id="r1",
        event_type=event_type,  # type: ignore[arg-type]
        occurred_at="2026-06-17T00:00:00Z",
        actor="tester",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# stockout_overstock_from_scorecards
# ---------------------------------------------------------------------------


def test_stockout_overstock_handles_under_forecast() -> None:
    """Forecast below actual → positive stockout gap."""
    scorecards = [
        _scorecard(forecast=[5.0, 5.0], actual=[7.0, 9.0]),
    ]
    stockouts, overstock = stockout_overstock_from_scorecards(scorecards)
    assert stockouts == pytest.approx(3.0)  # (2 + 4) / 2
    assert overstock == 0.0


def test_stockout_overstock_handles_over_forecast() -> None:
    """Forecast above actual → positive overstock gap."""
    scorecards = [
        _scorecard(forecast=[7.0, 9.0], actual=[5.0, 5.0]),
    ]
    stockouts, overstock = stockout_overstock_from_scorecards(scorecards)
    assert stockouts == 0.0
    assert overstock == pytest.approx(3.0)


def test_stockout_overstock_handles_mixed() -> None:
    """Mixed forecast / actual pair → both gaps are surfaced."""
    scorecards = [
        _scorecard(forecast=[5.0, 8.0], actual=[7.0, 5.0]),
    ]
    stockouts, overstock = stockout_overstock_from_scorecards(scorecards)
    assert stockouts == pytest.approx(2.0)  # one step: 7 - 5 = 2
    assert overstock == pytest.approx(3.0)  # one step: 8 - 5 = 3


def test_stockout_overstock_handles_exact_match() -> None:
    """Forecast == actual → both gaps are zero (no stockout, no overstock)."""
    scorecards = [
        _scorecard(forecast=[5.0, 5.0], actual=[5.0, 5.0]),
    ]
    stockouts, overstock = stockout_overstock_from_scorecards(scorecards)
    assert stockouts == 0.0
    assert overstock == 0.0


def test_stockout_overstock_handles_empty_input() -> None:
    """Empty input returns (0.0, 0.0) — no NaN, no inf."""
    assert stockout_overstock_from_scorecards([]) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# approval_pattern_from_events
# ---------------------------------------------------------------------------


def test_approval_pattern_counts_only_decided_events() -> None:
    """Only 'decided' events count; raised / expired / corrected are ignored."""
    events = [
        _event(event_type="raised"),
        _event(event_type="decided", notes="decision=APPROVE"),
        _event(event_type="decided", notes="decision=REJECT"),
        _event(event_type="decided", notes="decision=DEFER"),
        _event(event_type="expired"),
        _event(event_type="corrected"),
    ]
    pattern = approval_pattern_from_events(events)
    assert pattern == {"APPROVE": 1, "REJECT": 1, "DEFER": 1}


def test_approval_pattern_empty_input() -> None:
    """Empty input returns an empty dict (no decisions, no map)."""
    assert approval_pattern_from_events([]) == {}


def test_approval_pattern_real_approval_gateway_audit() -> None:
    """The engine reads a real audit log correctly when the gateway writes it."""
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gateway = InProcessApprovalGateway(audit_root=root)
        request = gateway.raise_request(
            run_id="r1",
            kind="data_contract",
            title="t",
            summary="s",
            requested_by="agent",
        )
        gateway.acknowledge(
            request_id=request.request_id,
            decision="APPROVE",
            approver="planner",
            reason="ok",
        )
        events = gateway.read_audit_log("r1")
        pattern = approval_pattern_from_events(events)
        assert pattern == {"APPROVE": 1}


def test_approval_pattern_decision_must_be_known() -> None:
    """A 'decided' event with an unknown decision does not poison the map."""
    events = [
        _event(event_type="decided", notes="decision=APPROVE"),
        _event(event_type="decided", notes="decision=APPROVE"),
        _event(event_type="decided", notes="no decision here"),
    ]
    pattern = approval_pattern_from_events(events)
    assert pattern == {"APPROVE": 2}


# ---------------------------------------------------------------------------
# service_level_from_scorecards
# ---------------------------------------------------------------------------


def test_service_level_meeting_demand_counted_as_served() -> None:
    """A forecast that meets actual demand counts as a served step."""
    scorecards = [
        _scorecard(forecast=[5.0, 8.0], actual=[5.0, 7.0]),
    ]
    # 2 of 2 steps served (forecast >= actual)
    assert service_level_from_scorecards(scorecards) == pytest.approx(1.0)


def test_service_level_undersupply_counted_as_unmet() -> None:
    """A forecast below actual is an unmet step."""
    scorecards = [
        _scorecard(forecast=[3.0, 5.0], actual=[5.0, 7.0]),
    ]
    # 0 of 2 steps served
    assert service_level_from_scorecards(scorecards) == pytest.approx(0.0)


def test_service_level_mixed() -> None:
    """Mixed forecast / actual pairs give the right fraction."""
    scorecards = [
        _scorecard(forecast=[5.0, 5.0], actual=[5.0, 7.0]),
        # step 1 served (5 >= 5), step 2 unmet (5 < 7) → 1 of 2
    ]
    assert service_level_from_scorecards(scorecards) == pytest.approx(0.5)


def test_service_level_empty_input() -> None:
    """Empty input returns 0.0 (no steps, no service)."""
    assert service_level_from_scorecards([]) == 0.0


# ---------------------------------------------------------------------------
# summarise_business_outcomes — top-level orchestrator
# ---------------------------------------------------------------------------


def test_summarise_business_outcomes_composes_all_three() -> None:
    """The orchestrator combines stockouts / overstock / service level,
    approval patterns, and planner overrides into one
    BusinessOutcomesReport.
    """
    scorecards = [
        _scorecard(forecast=[5.0, 8.0], actual=[7.0, 5.0]),
    ]
    events = [
        _event(event_type="decided", notes="decision=APPROVE"),
        _event(event_type="decided", notes="decision=REJECT"),
    ]
    planner_overrides = ["planner reduced SKU_1 order by 50%"]
    report = summarise_business_outcomes(
        run_id="r1",
        scorecards=scorecards,
        events=events,
        planner_overrides=planner_overrides,
    )
    assert isinstance(report, BusinessOutcomesReport)
    assert report.run_id == "r1"
    assert report.expected_stockouts == pytest.approx(2.0)
    assert report.expected_overstock == pytest.approx(3.0)
    assert report.service_level == pytest.approx(0.5)  # 1 of 2
    assert report.planner_overrides == planner_overrides
    assert report.approval_patterns == {"APPROVE": 1, "REJECT": 1}


def test_summarise_business_outcomes_handles_empty_inputs() -> None:
    """All three inputs empty → report is the zero baseline."""
    report = summarise_business_outcomes(
        run_id="r1",
        scorecards=[],
        events=[],
        planner_overrides=[],
    )
    assert report.expected_stockouts == 0.0
    assert report.expected_overstock == 0.0
    assert report.service_level == 0.0
    assert report.planner_overrides == []
    assert report.approval_patterns == {}


def test_summarise_business_outcomes_approval_pattern_includes_defer() -> None:
    """DEFER decisions show up in the approval pattern map."""
    events = [
        _event(event_type="decided", notes="decision=DEFER"),
        _event(event_type="decided", notes="decision=DEFER"),
        _event(event_type="decided", notes="decision=APPROVE"),
    ]
    report = summarise_business_outcomes(
        run_id="r1",
        scorecards=[],
        events=events,
        planner_overrides=[],
    )
    assert report.approval_patterns == {"APPROVE": 1, "DEFER": 2}
