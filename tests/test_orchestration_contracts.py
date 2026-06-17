"""Tests for Phase 6 CB2: Approvals, Scheduling, and ERP Handoff
contracts.

The contracts in ``backend/forecasting/contracts.py`` at the bottom of
the file (ApprovalRequest, ApprovalEvent, ScheduledJobTrigger,
ScheduledJobRun, ErpHandoffPayload, and the associated Literal
aliases) define the typed surface the cockpit UI consumes and (in
the future) any alternative gateway or scheduler implementation
consumes. These tests assert:

* Every approval kind and scheduled job kind the original Phase 6 plan
  calls out is expressible (closed set; the Literal accepts the value).
* Models are constructible with the minimum fields a caller would
  pass in practice.
* Default values match the documented behaviour (pending request,
  no decision recorded, empty payloads, etc.).
* Round-trip through ``model_dump`` / ``model_validate`` preserves
  data — the cockpit consumes JSON, so lossy serialisation is a
  contract violation.
* Invalid values for the closed Literals raise ValidationError — the
  whole point of the closed set is to fail loud at construction.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from forecasting.contracts import (
    ApprovalDecisionValue,
    ApprovalEvent,
    ApprovalKind,
    ApprovalRequest,
    ApprovalStatus,
    ErpHandoffPayload,
    ScheduledJobKind,
    ScheduledJobRun,
    ScheduledJobStatus,
    ScheduledJobTrigger,
)


# ---------------------------------------------------------------------------
# Closed Literal coverage
# ---------------------------------------------------------------------------

ALL_APPROVAL_KINDS = [
    "data_contract",
    "risky_schema_semantics",
    "custom_code_permission",
    "unforecastable_grain_fallback",
    "official_forecast_publication",
    "replenishment_recommendation",
    "erp_procurement_handoff",
]


ALL_SCHEDULED_JOB_KINDS = [
    "data_refresh",
    "validation",
    "forecast_generation",
    "review",
    "monitoring",
    "drift_investigation",
]


@pytest.mark.parametrize("kind", ALL_APPROVAL_KINDS)
def test_approval_request_accepts_every_planned_kind(kind: str) -> None:
    """Every approval kind from the original Phase 6 plan is constructible."""
    req = ApprovalRequest(
        request_id="req-1",
        run_id="run-1",
        kind=kind,  # type: ignore[arg-type]
        title=f"Test {kind}",
        summary="unit test",
        requested_by="tester",
        requested_at="2026-06-17T00:00:00Z",
    )
    assert req.kind == kind
    assert req.status == "pending"
    assert req.decision is None
    assert req.approver is None


def test_approval_kind_literal_is_closed() -> None:
    """A typo or new kind fails loudly instead of silently widening the surface."""
    with pytest.raises(ValidationError):
        ApprovalRequest(
            request_id="req-1",
            run_id="run-1",
            kind="data_contrat",  # typo
            title="t",
            summary="s",
            requested_by="t",
            requested_at="2026-06-17T00:00:00Z",
        )


@pytest.mark.parametrize("kind", ALL_SCHEDULED_JOB_KINDS)
def test_scheduled_job_trigger_accepts_every_planned_kind(kind: str) -> None:
    """Every scheduled job kind from the original Phase 6 plan is constructible."""
    trigger = ScheduledJobTrigger(
        trigger_id="tr-1",
        kind=kind,  # type: ignore[arg-type]
        cron="every 5m",
        created_at="2026-06-17T00:00:00Z",
        created_by="tester",
    )
    assert trigger.kind == kind
    assert trigger.enabled is True
    assert trigger.run_id is None
    assert trigger.params == {}


def test_scheduled_job_kind_literal_is_closed() -> None:
    """A typo fails loudly."""
    with pytest.raises(ValidationError):
        ScheduledJobTrigger(
            trigger_id="tr-1",
            kind="data_refesh",  # typo
            cron="hourly",
            created_at="2026-06-17T00:00:00Z",
            created_by="t",
        )


# ---------------------------------------------------------------------------
# Status / decision Literals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status", ["pending", "approved", "rejected"]
)
def test_approval_status_values(status: str) -> None:
    req = ApprovalRequest(
        request_id="req-1",
        run_id="run-1",
        kind="data_contract",
        title="t",
        summary="s",
        requested_by="t",
        requested_at="2026-06-17T00:00:00Z",
        status=status,  # type: ignore[arg-type]
    )
    assert req.status == status


def test_approval_status_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        ApprovalRequest(
            request_id="req-1",
            run_id="run-1",
            kind="data_contract",
            title="t",
            summary="s",
            requested_by="t",
            requested_at="2026-06-17T00:00:00Z",
            status="PENDING",  # wrong case — Literal is case-sensitive
        )


@pytest.mark.parametrize("decision", ["APPROVE", "REJECT", "DEFER"])
def test_approval_decision_values(decision: str) -> None:
    req = ApprovalRequest(
        request_id="req-1",
        run_id="run-1",
        kind="data_contract",
        title="t",
        summary="s",
        requested_by="t",
        requested_at="2026-06-17T00:00:00Z",
        decision=decision,  # type: ignore[arg-type]
        approver="alice",
        decided_at="2026-06-17T01:00:00Z",
        reason="ok",
    )
    assert req.decision == decision


@pytest.mark.parametrize(
    "status",
    ["queued", "running", "succeeded", "failed", "skipped", "awaiting_approval"],
)
def test_scheduled_job_run_status_values(status: str) -> None:
    run = ScheduledJobRun(
        run_id="run-1",
        trigger_id="tr-1",
        kind="data_refresh",
        started_at="2026-06-17T00:00:00Z",
        status=status,  # type: ignore[arg-type]
    )
    assert run.status == status
    assert run.finished_at is None
    assert run.error is None


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_approval_request_defaults() -> None:
    """Defaults match the documented surface: pending, no decision, empty payload."""
    req = ApprovalRequest(
        request_id="req-1",
        run_id="run-1",
        kind="replenishment_recommendation",
        title="Approve replenishment batch",
        summary="3 series need orders",
        requested_by="replenishment_engine",
        requested_at="2026-06-17T00:00:00Z",
    )
    assert req.status == "pending"
    assert req.decision is None
    assert req.approver is None
    assert req.decided_at is None
    assert req.reason is None
    assert req.payload == {}


def test_scheduled_job_trigger_defaults() -> None:
    trigger = ScheduledJobTrigger(
        trigger_id="tr-1",
        kind="monitoring",
        cron="hourly",
        created_at="2026-06-17T00:00:00Z",
        created_by="ops",
    )
    assert trigger.enabled is True
    assert trigger.run_id is None
    assert trigger.params == {}


def test_scheduled_job_run_defaults() -> None:
    run = ScheduledJobRun(
        run_id="run-1",
        trigger_id="tr-1",
        kind="forecast_generation",
        started_at="2026-06-17T00:00:00Z",
        status="running",
    )
    assert run.finished_at is None
    assert run.result_payload == {}
    assert run.error is None


def test_erp_handoff_payload_defaults() -> None:
    payload = ErpHandoffPayload(
        handoff_id="ho-1",
        run_id="run-1",
        released_at="2026-06-17T02:00:00Z",
        approval_request_id="req-1",
        approver="alice",
        approval_reason="within budget",
        approved_at="2026-06-17T01:30:00Z",
    )
    assert payload.recommendations == []
    assert payload.audit_trail == []


# ---------------------------------------------------------------------------
# ApprovalEvent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event_type", ["raised", "decided", "expired", "corrected"]
)
def test_approval_event_types(event_type: str) -> None:
    event = ApprovalEvent(
        event_id="ev-1",
        request_id="req-1",
        run_id="run-1",
        event_type=event_type,  # type: ignore[arg-type]
        occurred_at="2026-06-17T00:00:00Z",
        actor="gateway",
    )
    assert event.event_type == event_type
    assert event.notes == ""


def test_approval_event_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        ApprovalEvent(
            event_id="ev-1",
            request_id="req-1",
            run_id="run-1",
            event_type="cancelled",  # not in the closed set
            occurred_at="2026-06-17T00:00:00Z",
            actor="gateway",
        )


# ---------------------------------------------------------------------------
# JSON round-trip (the cockpit UI consumes JSON, so this is the wire shape)
# ---------------------------------------------------------------------------


def test_approval_request_json_round_trip() -> None:
    req = ApprovalRequest(
        request_id="req-1",
        run_id="run-1",
        kind="replenishment_recommendation",
        title="Approve batch",
        summary="3 series",
        payload={"series_count": 3, "total_order_qty": 1500.0},
        requested_by="replenishment_engine",
        requested_at="2026-06-17T00:00:00Z",
    )
    dumped = req.model_dump()
    restored = ApprovalRequest.model_validate(dumped)
    assert restored == req


def test_erp_handoff_payload_json_round_trip() -> None:
    payload = ErpHandoffPayload(
        handoff_id="ho-1",
        run_id="run-1",
        released_at="2026-06-17T02:00:00Z",
        approval_request_id="req-1",
        approver="alice",
        approval_reason="within budget",
        approved_at="2026-06-17T01:30:00Z",
        recommendations=[
            {"sku": "SKU_A", "location": "WEST", "quantity": 250.0},
            {"sku": "SKU_B", "location": "EAST", "quantity": 100.0},
        ],
        audit_trail=[
            ApprovalEvent(
                event_id="ev-1",
                request_id="req-1",
                run_id="run-1",
                event_type="raised",
                occurred_at="2026-06-17T00:00:00Z",
                actor="platform",
            ),
            ApprovalEvent(
                event_id="ev-2",
                request_id="req-1",
                run_id="run-1",
                event_type="decided",
                occurred_at="2026-06-17T01:30:00Z",
                actor="alice",
                notes="approved within budget",
            ),
        ],
    )
    dumped = payload.model_dump()
    restored = ErpHandoffPayload.model_validate(dumped)
    assert restored == payload
    assert len(restored.recommendations) == 2
    assert len(restored.audit_trail) == 2


# ---------------------------------------------------------------------------
# Type re-exports (so callers can ``from forecasting.contracts import ApprovalKind``)
# ---------------------------------------------------------------------------


def test_type_aliases_are_exported() -> None:
    """The Literal aliases are importable alongside the model classes."""
    assert ApprovalKind is not None
    assert ApprovalStatus is not None
    assert ApprovalDecisionValue is not None
    assert ScheduledJobKind is not None
    assert ScheduledJobStatus is not None
