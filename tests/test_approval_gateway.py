"""Tests for Phase 6 CB3: in-process approval gateway.

Covers the round-trip the platform uses whenever a human decision is
needed: raise a request, the gateway stores it as pending and writes
a 'raised' event to the audit log; a human acknowledges the request
with APPROVE / REJECT, the gateway transitions status and writes a
'decided' event; subsequent acknowledge calls fail loud; the audit
log is the durable record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forecasting.approval_gateway import (
    AlreadyDecidedError,
    ApprovalGateway,
    InProcessApprovalGateway,
    RequestNotFoundError,
)
from forecasting.contracts import ApprovalRequest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def audit_root(tmp_path: Path) -> Path:
    return tmp_path / "outputs"


@pytest.fixture()
def gateway(audit_root: Path) -> InProcessApprovalGateway:
    return InProcessApprovalGateway(audit_root=audit_root)


# ---------------------------------------------------------------------------
# ABC + interface
# ---------------------------------------------------------------------------


def test_in_process_gateway_is_an_approval_gateway() -> None:
    """The in-process implementation satisfies the abstract interface."""
    assert issubclass(InProcessApprovalGateway, ApprovalGateway)


def test_abstract_gateway_cannot_be_instantiated() -> None:
    """The interface itself is not usable — concrete subclasses only."""
    with pytest.raises(TypeError):
        ApprovalGateway()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# raise_request
# ---------------------------------------------------------------------------


def test_raise_request_returns_pending_approval_request(
    gateway: InProcessApprovalGateway,
) -> None:
    req = gateway.raise_request(
        run_id="run-1",
        kind="replenishment_recommendation",
        title="Approve batch",
        summary="3 series need orders",
        requested_by="replenishment_engine",
        payload={"series_count": 3, "total_order_qty": 1500.0},
    )
    assert isinstance(req, ApprovalRequest)
    assert req.status == "pending"
    assert req.decision is None
    assert req.approver is None
    assert req.decided_at is None
    assert req.reason is None
    assert req.payload == {"series_count": 3, "total_order_qty": 1500.0}
    assert req.request_id.startswith("req-")
    assert req.requested_at.endswith("Z")


def test_raise_request_persists_to_in_memory_dict(
    gateway: InProcessApprovalGateway,
) -> None:
    req = gateway.raise_request(
        run_id="run-1",
        kind="data_contract",
        title="t",
        summary="s",
        requested_by="platform",
    )
    fetched = gateway.get(req.request_id)
    assert fetched is not None
    assert fetched.request_id == req.request_id
    assert fetched.kind == "data_contract"


def test_raise_request_writes_raised_event_to_audit_log(
    gateway: InProcessApprovalGateway,
    audit_root: Path,
) -> None:
    req = gateway.raise_request(
        run_id="run-1",
        kind="custom_code_permission",
        title="Grant custom family",
        summary="agent needs to add XGBoost variant",
        requested_by="foundry_agent",
    )
    log_path = audit_root / "run-1" / "approvals.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_type"] == "raised"
    assert event["request_id"] == req.request_id
    assert event["run_id"] == "run-1"
    assert event["actor"] == "foundry_agent"


def test_raise_request_creates_audit_dir_if_missing(
    gateway: InProcessApprovalGateway,
    audit_root: Path,
) -> None:
    """The platform can raise for a never-before-seen run id."""
    assert not (audit_root / "run-fresh").exists()
    gateway.raise_request(
        run_id="run-fresh",
        kind="erp_procurement_handoff",
        title="t",
        summary="s",
        requested_by="x",
    )
    assert (audit_root / "run-fresh" / "approvals.jsonl").exists()


def test_raise_request_with_empty_payload(
    gateway: InProcessApprovalGateway,
) -> None:
    req = gateway.raise_request(
        run_id="run-1",
        kind="official_forecast_publication",
        title="Publish",
        summary="Foundry report ready",
        requested_by="meridian",
    )
    assert req.payload == {}


# ---------------------------------------------------------------------------
# acknowledge
# ---------------------------------------------------------------------------


def test_acknowledge_approve_transitions_to_approved(
    gateway: InProcessApprovalGateway,
) -> None:
    req = gateway.raise_request(
        run_id="run-1",
        kind="replenishment_recommendation",
        title="t",
        summary="s",
        requested_by="engine",
    )
    decided = gateway.acknowledge(
        request_id=req.request_id,
        decision="APPROVE",
        approver="alice",
        reason="within budget",
    )
    assert decided.status == "approved"
    assert decided.decision == "APPROVE"
    assert decided.approver == "alice"
    assert decided.reason == "within budget"
    assert decided.decided_at is not None
    assert decided.decided_at.endswith("Z")
    # The in-memory state matches.
    assert gateway.get(req.request_id).status == "approved"


def test_acknowledge_reject_transitions_to_rejected(
    gateway: InProcessApprovalGateway,
) -> None:
    req = gateway.raise_request(
        run_id="run-1",
        kind="risky_schema_semantics",
        title="t",
        summary="s",
        requested_by="meridian",
    )
    decided = gateway.acknowledge(
        request_id=req.request_id,
        decision="REJECT",
        approver="bob",
        reason="schema mapping is wrong",
    )
    assert decided.status == "rejected"
    assert decided.decision == "REJECT"


def test_acknowledge_defer_keeps_request_actionable(
    gateway: InProcessApprovalGateway,
) -> None:
    """DEFER is a non-terminal decision: status stays pending so the
    human can be asked again later. A subsequent APPROVE then
    transitions to the terminal approved state."""
    req = gateway.raise_request(
        run_id="run-1",
        kind="data_contract",
        title="t",
        summary="s",
        requested_by="x",
    )
    deferred = gateway.acknowledge(
        request_id=req.request_id,
        decision="DEFER",
        approver="alice",
        reason="come back next week",
    )
    assert deferred.status == "pending"
    assert deferred.decision == "DEFER"
    # Still in the pending list.
    assert req.request_id in [r.request_id for r in gateway.list_pending("run-1")]
    # And re-acknowledge with APPROVE is allowed.
    approved = gateway.acknowledge(
        request_id=req.request_id,
        decision="APPROVE",
        approver="alice",
        reason="ready now",
    )
    assert approved.status == "approved"
    assert approved.decision == "APPROVE"
    # Audit log carries every event in order.
    events = [e for e in gateway.read_audit_log("run-1") if e.request_id == req.request_id]
    assert [e.event_type for e in events] == ["raised", "decided", "decided"]
    assert events[1].notes.startswith("decision=DEFER")
    assert events[2].notes.startswith("decision=APPROVE")


def test_acknowledge_unknown_request_raises(
    gateway: InProcessApprovalGateway,
) -> None:
    with pytest.raises(RequestNotFoundError):
        gateway.acknowledge(
            request_id="req-doesnotexist",
            decision="APPROVE",
            approver="alice",
            reason="x",
        )


def test_acknowledge_twice_raises_already_decided(
    gateway: InProcessApprovalGateway,
) -> None:
    """Idempotency guard: a second decision is a contract violation."""
    req = gateway.raise_request(
        run_id="run-1",
        kind="data_contract",
        title="t",
        summary="s",
        requested_by="x",
    )
    gateway.acknowledge(
        request_id=req.request_id,
        decision="APPROVE",
        approver="alice",
        reason="ok",
    )
    with pytest.raises(AlreadyDecidedError):
        gateway.acknowledge(
            request_id=req.request_id,
            decision="REJECT",
            approver="bob",
            reason="changed mind",
        )


def test_acknowledge_writes_decided_event_after_raised(
    gateway: InProcessApprovalGateway,
) -> None:
    req = gateway.raise_request(
        run_id="run-1",
        kind="erp_procurement_handoff",
        title="t",
        summary="s",
        requested_by="x",
    )
    gateway.acknowledge(
        request_id=req.request_id,
        decision="APPROVE",
        approver="alice",
        reason="release to ERP",
    )
    events = gateway.read_audit_log("run-1")
    assert len(events) == 2
    assert events[0].event_type == "raised"
    assert events[1].event_type == "decided"
    assert events[1].actor == "alice"
    assert events[1].notes.startswith("decision=APPROVE")


# ---------------------------------------------------------------------------
# get / list_pending / list_all
# ---------------------------------------------------------------------------


def test_get_returns_none_for_unknown(
    gateway: InProcessApprovalGateway,
) -> None:
    assert gateway.get("req-missing") is None


def test_list_pending_filters_by_run_id(
    gateway: InProcessApprovalGateway,
) -> None:
    gateway.raise_request(
        run_id="run-1", kind="data_contract", title="t", summary="s", requested_by="x"
    )
    gateway.raise_request(
        run_id="run-1",
        kind="replenishment_recommendation",
        title="t",
        summary="s",
        requested_by="x",
    )
    gateway.raise_request(
        run_id="run-2", kind="data_contract", title="t", summary="s", requested_by="x"
    )
    pending_run1 = gateway.list_pending(run_id="run-1")
    assert len(pending_run1) == 2
    assert all(r.run_id == "run-1" for r in pending_run1)


def test_list_pending_excludes_decided(
    gateway: InProcessApprovalGateway,
) -> None:
    req1 = gateway.raise_request(
        run_id="run-1", kind="data_contract", title="t", summary="s", requested_by="x"
    )
    gateway.raise_request(
        run_id="run-1",
        kind="replenishment_recommendation",
        title="t",
        summary="s",
        requested_by="x",
    )
    gateway.acknowledge(
        request_id=req1.request_id,
        decision="APPROVE",
        approver="a",
        reason="r",
    )
    pending = gateway.list_pending(run_id="run-1")
    assert len(pending) == 1
    assert pending[0].kind == "replenishment_recommendation"


def test_list_pending_all_runs(
    gateway: InProcessApprovalGateway,
) -> None:
    gateway.raise_request(
        run_id="run-1", kind="data_contract", title="t", summary="s", requested_by="x"
    )
    gateway.raise_request(
        run_id="run-2", kind="data_contract", title="t", summary="s", requested_by="x"
    )
    assert len(gateway.list_pending()) == 2


def test_list_all_includes_decided(
    gateway: InProcessApprovalGateway,
) -> None:
    req = gateway.raise_request(
        run_id="run-1", kind="data_contract", title="t", summary="s", requested_by="x"
    )
    gateway.acknowledge(
        request_id=req.request_id, decision="REJECT", approver="a", reason="r"
    )
    all_requests = gateway.list_all(run_id="run-1")
    assert len(all_requests) == 1
    assert all_requests[0].status == "rejected"


# ---------------------------------------------------------------------------
# Persistence: a new gateway instance can read the audit log of the old one
# ---------------------------------------------------------------------------


def test_audit_log_survives_instance_replacement(
    audit_root: Path,
) -> None:
    """The audit file is the durable record; the in-memory dict is rebuilt from it."""
    gw1 = InProcessApprovalGateway(audit_root=audit_root)
    req = gw1.raise_request(
        run_id="run-1",
        kind="data_contract",
        title="t",
        summary="s",
        requested_by="x",
    )
    gw1.acknowledge(
        request_id=req.request_id, decision="APPROVE", approver="a", reason="r"
    )

    gw2 = InProcessApprovalGateway(audit_root=audit_root)
    # gw2 in-memory is empty (the load_from_audit helper does not
    # repopulate decided requests because we cannot reconstruct the
    # full payload from events alone — by design).
    assert gw2.get(req.request_id) is None
    # But the audit log is intact.
    events = gw2.read_audit_log("run-1")
    assert [e.event_type for e in events] == ["raised", "decided"]


# ---------------------------------------------------------------------------
# End-to-end round-trip with all 7 approval kinds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        "data_contract",
        "risky_schema_semantics",
        "custom_code_permission",
        "unforecastable_grain_fallback",
        "official_forecast_publication",
        "replenishment_recommendation",
        "erp_procurement_handoff",
    ],
)
def test_full_round_trip_for_every_approval_kind(
    gateway: InProcessApprovalGateway, kind: str
) -> None:
    """A real human would interact with the gateway through this exact pattern."""
    req = gateway.raise_request(
        run_id="run-1",
        kind=kind,
        title=f"Approve {kind}",
        summary=f"Test for {kind}",
        requested_by="platform",
    )
    assert req.status == "pending"
    decided = gateway.acknowledge(
        request_id=req.request_id,
        decision="APPROVE",
        approver="alice",
        reason="looks good",
    )
    assert decided.status == "approved"
    # Audit log has two events for this request.
    events = [e for e in gateway.read_audit_log("run-1") if e.request_id == req.request_id]
    assert len(events) == 2
    assert events[0].event_type == "raised"
    assert events[1].event_type == "decided"
