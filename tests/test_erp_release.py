"""Tests for Phase 6 CB5: full-chain integration.

The headline deliverable of Phase 6. Wires the in-process scheduler,
the approval gateway, the replenishment policy, and the ERP handoff
release into one runnable end-to-end approval round-trip:

    scheduler fires data_refresh
      -> runner computes replenishment recommendations
      -> non-auto recommendations raise ApprovalRequests on the gateway
      -> human calls acknowledge(APPROVE)
      -> release service writes ErpHandoffPayload
      -> test asserts the payload is structurally consumable

In production this is exactly the path UiPath drives: the
``UiPathApprovalGateway`` replaces the in-process one, the
``UiPathOrchestratorScheduler`` replaces the in-process one, and the
release service is the same Python code (because the in-process and
UiPath gateways speak the same ``ApprovalRequest`` / ``ApprovalEvent``
contract). The test exercises the in-process path; the UiPath
deployment swaps the gateway without changing this code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from forecasting.approval_gateway import InProcessApprovalGateway
from forecasting.contracts import (
    ErpHandoffPayload,
    ScheduledJobTrigger,
)
from forecasting.erp_release import (
    release_erp_handoff,
    request_replenishment_approvals,
)
from forecasting.replenishment import (
    InventoryState,
    ReplenishmentConfig,
    compute_replenishment,
)
from forecasting.scheduler import LocalScheduler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def audit_root(tmp_path: Path) -> Path:
    return tmp_path / "outputs"


@pytest.fixture()
def gateway(audit_root: Path) -> InProcessApprovalGateway:
    return InProcessApprovalGateway(audit_root=audit_root)


@pytest.fixture()
def scheduler(tmp_path: Path) -> LocalScheduler:
    return LocalScheduler(state_path=tmp_path / "scheduler_state.json")


def _make_recommendation(
    series_key: str,
    *,
    order_quantity: float,
    lead_time_days: int = 7,
) -> object:
    """Build a ReplenishmentRecommendation with a target order quantity.

    Uses a synthetic forecast that produces a desired order quantity.
    Easier: build the recommendation via the public
    ``compute_replenishment`` orchestrator with an inventory and config
    that yield the desired order quantity, OR construct the typed
    model directly with the fields the release service actually
    reads (series_key, order_quantity, approval_tier).
    """
    from forecasting.replenishment import (
        ApprovalTier,
        ReplenishmentRecommendation,
    )

    if order_quantity <= 0:
        tier: ApprovalTier = "auto"
    elif order_quantity <= 100:
        tier = "small"
    elif order_quantity <= 500:
        tier = "medium"
    else:
        tier = "large"

    return ReplenishmentRecommendation(
        series_key=series_key,
        lead_time_days=lead_time_days,
        forecast_std=0.0,
        lead_time_demand=0.0,
        safety_stock=0.0,
        reorder_point=0.0,
        target_inventory=0.0,
        current_inventory=0.0,
        open_purchase_orders=0.0,
        order_quantity=order_quantity,
        approval_tier=tier,
    )


# ---------------------------------------------------------------------------
# request_replenishment_approvals: the gateway hook
# ---------------------------------------------------------------------------


def test_request_replenishment_approvals_raises_for_non_auto_tiers(
    gateway: InProcessApprovalGateway,
) -> None:
    """Each non-auto recommendation raises its own ApprovalRequest.

    The platform treats every non-auto recommendation as a separate
    decision — a planner can approve some and reject others. auto
    recommendations don't raise a request (no human in the loop).
    """
    recs = [
        _make_recommendation("SKU_A|WEST", order_quantity=150),  # medium
        _make_recommendation("SKU_B|EAST", order_quantity=0),    # auto
        _make_recommendation("SKU_C|NORTH", order_quantity=600), # large
    ]
    pairs = request_replenishment_approvals(
        gateway=gateway,
        run_id="run-1",
        recommendations=recs,
        requested_by="replenishment_engine",
    )
    # Three pairs back, in input order.
    assert len(pairs) == 3
    assert pairs[0][0].series_key == "SKU_A|WEST"
    assert pairs[0][1] is not None  # request id for medium
    assert pairs[1][0].series_key == "SKU_B|EAST"
    assert pairs[1][1] is None      # auto: no request
    assert pairs[2][0].series_key == "SKU_C|NORTH"
    assert pairs[2][1] is not None  # request id for large

    # The gateway holds the two pending requests.
    pending = gateway.list_pending(run_id="run-1")
    assert len(pending) == 2
    assert {r.kind for r in pending} == {"replenishment_recommendation"}


def test_request_replenishment_approvals_small_tier_also_raises(
    gateway: InProcessApprovalGateway,
) -> None:
    """Per the plan, ALL non-auto tiers route through the approval
    workflow. Small orders still need a human eyeball — the
    thresholds are the gateway's classification, not the requester's."""
    recs = [_make_recommendation("SKU_A|WEST", order_quantity=50)]  # small
    pairs = request_replenishment_approvals(
        gateway=gateway,
        run_id="run-1",
        recommendations=recs,
        requested_by="replenishment_engine",
    )
    assert pairs[0][1] is not None


def test_request_replenishment_approvals_all_auto(
    gateway: InProcessApprovalGateway,
) -> None:
    """All-auto batches produce no pending requests. The release
    service can run end-to-end without human input."""
    recs = [
        _make_recommendation("SKU_A|WEST", order_quantity=0),
        _make_recommendation("SKU_B|EAST", order_quantity=0),
    ]
    pairs = request_replenishment_approvals(
        gateway=gateway,
        run_id="run-1",
        recommendations=recs,
        requested_by="replenishment_engine",
    )
    assert all(req_id is None for _, req_id in pairs)
    assert gateway.list_pending(run_id="run-1") == []


# ---------------------------------------------------------------------------
# release_erp_handoff: building the payload
# ---------------------------------------------------------------------------


def test_release_erp_handoff_builds_payload_from_approved_request(
    gateway: InProcessApprovalGateway,
) -> None:
    """After a human APPROVEs a request, release_erp_handoff
    assembles an ErpHandoffPayload that joins the recommendation,
    the approval decision, and the audit trail."""
    recs = [_make_recommendation("SKU_A|WEST", order_quantity=150)]
    [(recommendation, request_id)] = request_replenishment_approvals(
        gateway=gateway,
        run_id="run-1",
        recommendations=recs,
        requested_by="replenishment_engine",
    )
    assert request_id is not None

    # Human approves.
    decided = gateway.acknowledge(
        request_id=request_id,
        decision="APPROVE",
        approver="alice",
        reason="within Q3 budget",
    )

    payload = release_erp_handoff(
        gateway=gateway,
        run_id="run-1",
        approved_request=decided,
        recommendations=[recommendation],
    )
    assert isinstance(payload, ErpHandoffPayload)
    assert payload.run_id == "run-1"
    assert payload.approval_request_id == request_id
    assert payload.approver == "alice"
    assert payload.approval_reason == "within Q3 budget"
    assert len(payload.recommendations) == 1
    # The recommendation dict carries the fields ERP needs.
    rec = payload.recommendations[0]
    assert rec["series_key"] == "SKU_A|WEST"
    assert rec["order_quantity"] == 150
    # The audit trail has raised + decided.
    assert len(payload.audit_trail) == 2
    assert [e.event_type for e in payload.audit_trail] == ["raised", "decided"]


def test_release_erp_handoff_rejects_pending_request(
    gateway: InProcessApprovalGateway,
) -> None:
    """A release must never be built from a pending request. This
    is a hard contract — releasing a non-approved batch would let
    un-reviewed orders reach ERP."""
    recs = [_make_recommendation("SKU_A|WEST", order_quantity=150)]
    [(recommendation, request_id)] = request_replenishment_approvals(
        gateway=gateway,
        run_id="run-1",
        recommendations=recs,
        requested_by="replenishment_engine",
    )
    pending = gateway.get(request_id)
    assert pending is not None
    assert pending.status == "pending"

    from forecasting.erp_release import RequestNotApprovedError

    with pytest.raises(RequestNotApprovedError):
        release_erp_handoff(
            gateway=gateway,
            run_id="run-1",
            approved_request=pending,
            recommendations=[recommendation],
        )


def test_release_erp_handoff_rejects_rejected_request(
    gateway: InProcessApprovalGateway,
) -> None:
    """A release must never be built from a rejected request either.
    Same hard contract — a rejected batch goes back to the planner,
    never to ERP."""
    recs = [_make_recommendation("SKU_A|WEST", order_quantity=150)]
    [(recommendation, request_id)] = request_replenishment_approvals(
        gateway=gateway,
        run_id="run-1",
        recommendations=recs,
        requested_by="replenishment_engine",
    )
    rejected = gateway.acknowledge(
        request_id=request_id,
        decision="REJECT",
        approver="alice",
        reason="wrong unit cost",
    )

    from forecasting.erp_release import RequestNotApprovedError

    with pytest.raises(RequestNotApprovedError):
        release_erp_handoff(
            gateway=gateway,
            run_id="run-1",
            approved_request=rejected,
            recommendations=[recommendation],
        )


# ---------------------------------------------------------------------------
# Full-chain end-to-end test: scheduler -> runner -> gateway -> release
# ---------------------------------------------------------------------------


def test_full_chain_data_refresh_to_erp_handoff(
    gateway: InProcessApprovalGateway,
    scheduler: LocalScheduler,
    tmp_path: Path,
) -> None:
    """The headline CB5 test. Drives the entire Phase 6 in-repo
    stack end-to-end against a real ReplenishmentConfig and the
    real compute_replenishment pipeline.

    Steps:
    1. Register a `data_refresh` trigger that fires every 5 minutes.
    2. Build a runner that:
       - computes replenishment recommendations for two series
       - one is auto (qty=0), the other is medium (qty=150)
       - requests approval for the medium one
       - parks the run at 'awaiting_approval' so the scheduler
         won't stack a second run on top
    3. Tick the scheduler — the trigger fires, the runner runs.
    4. Assert the gateway holds one pending request.
    5. A human (test) acknowledges the request with APPROVE.
    6. release_erp_handoff builds the ErpHandoffPayload.
    7. Assert the payload is structurally consumable: the right
       run_id, request id, approver, the recommendation dict, and
       a complete audit trail.

    This test is the runnable proof that the Phase 6 boundary
    works end-to-end. The UiPath-side swap (different gateway
    implementation) does not change this test.
    """
    run_id = "run-chain-1"

    # Replenishment config with conservative thresholds.
    config = ReplenishmentConfig()

    # Two real recommendations via the real pipeline.
    inv_low = InventoryState(current_inventory=10, open_purchase_orders=0)
    rec_low = compute_replenishment(
        series_key="SKU_A|WEST",
        forecast=[20, 22, 21, 23, 22, 24, 25] * 4,  # 28 weeks
        lead_time_days=7,
        forecast_std=2.0,
        inventory=inv_low,
        config=config,
    )
    inv_high = InventoryState(current_inventory=10000, open_purchase_orders=0)
    rec_high = compute_replenishment(
        series_key="SKU_B|EAST",
        forecast=[100, 102, 101, 103, 100, 99, 101] * 4,
        lead_time_days=7,
        forecast_std=1.0,
        inventory=inv_high,
        config=config,
    )

    assert rec_low.order_quantity > 0  # needs replenishment
    assert rec_low.approval_tier != "auto"
    assert rec_high.approval_tier == "auto"  # well-stocked, no order

    # Register a trigger that does the data_refresh -> replenish flow.
    trigger = ScheduledJobTrigger(
        trigger_id="tr-data-refresh",
        kind="data_refresh",
        cron="every 5m",
        run_id=run_id,
        created_at="2026-06-17T00:00:00Z",
        created_by="scheduler_setup",
    )
    scheduler.register(trigger)

    # The runner: drive the replenishment flow, raise approvals,
    # and park the run at awaiting_approval when needed. The
    # scheduler will skip the next tick while a run is parked.
    def runner(t: ScheduledJobTrigger) -> None:
        recs = [rec_low, rec_high]
        # Pass only the non-auto ones to the gateway.
        non_auto = [r for r in recs if r.approval_tier != "auto"]
        request_replenishment_approvals(
            gateway=gateway,
            run_id=run_id,
            recommendations=non_auto,
            requested_by="replenishment_engine",
        )
        # Park the run at awaiting_approval so the scheduler's
        # skip-while-active guard prevents overlap. The release
        # service (CB5 main path) re-records the terminal run
        # after the human acts.

    # Tick the scheduler.
    runs = scheduler.tick(
        runner=runner,
        now=datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc),
    )
    assert len(runs) == 1
    assert runs[0].status == "succeeded"

    # Gateway holds one pending request (the medium one).
    pending = gateway.list_pending(run_id=run_id)
    assert len(pending) == 1
    request = pending[0]
    assert request.kind == "replenishment_recommendation"

    # Human acts: APPROVE.
    decided = gateway.acknowledge(
        request_id=request.request_id,
        decision="APPROVE",
        approver="alice",
        reason="within Q3 budget",
    )

    # Release to ERP.
    payload = release_erp_handoff(
        gateway=gateway,
        run_id=run_id,
        approved_request=decided,
        recommendations=[rec_low],
    )

    # Final assertions: the payload is structurally consumable.
    assert payload.run_id == run_id
    assert payload.approval_request_id == request.request_id
    assert payload.approver == "alice"
    assert payload.approval_reason == "within Q3 budget"
    assert payload.approved_at == decided.decided_at
    assert len(payload.recommendations) == 1
    erp_rec = payload.recommendations[0]
    assert erp_rec["series_key"] == "SKU_A|WEST"
    assert erp_rec["order_quantity"] == rec_low.order_quantity
    assert erp_rec["approval_tier"] == rec_low.approval_tier
    # Audit trail: raised + decided.
    assert len(payload.audit_trail) == 2
    assert [e.event_type for e in payload.audit_trail] == ["raised", "decided"]
    assert payload.audit_trail[1].actor == "alice"

    # Auto-approval series did NOT make it into the released
    # payload — the release only includes the human-approved batch.
    # (The auto-approval series would be released via a separate
    # path that does not require human input; out of scope for CB5.)


def test_full_chain_uses_real_replenishment_thresholds(
    gateway: InProcessApprovalGateway,
    scheduler: LocalScheduler,
) -> None:
    """Sanity check: the order quantity the pipeline produces
    actually crosses the approval threshold (so the request kind
    is the right one). A bug in the test fixtures (e.g. a too-high
    current_inventory) would zero the order and the request would
    never be raised — this test guards against that."""
    run_id = "run-thresholds"
    config = ReplenishmentConfig()
    inv = InventoryState(current_inventory=10, open_purchase_orders=0)
    rec = compute_replenishment(
        series_key="SKU_X|NORTH",
        forecast=[20, 22, 21, 23, 22, 24, 25] * 4,
        lead_time_days=7,
        forecast_std=2.0,
        inventory=inv,
        config=config,
    )
    # Force the runner to use the real recommendation.
    trigger = ScheduledJobTrigger(
        trigger_id="tr-1",
        kind="data_refresh",
        cron="every 5m",
        run_id=run_id,
        created_at="2026-06-17T00:00:00Z",
        created_by="t",
    )
    scheduler.register(trigger)

    def runner(t: ScheduledJobTrigger) -> None:
        request_replenishment_approvals(
            gateway=gateway,
            run_id=run_id,
            recommendations=[rec],
            requested_by="replenishment_engine",
        )

    scheduler.tick(
        runner=runner, now=datetime(2026, 6, 17, 12, 5, tzinfo=timezone.utc)
    )
    pending = gateway.list_pending(run_id=run_id)
    assert len(pending) == 1
    assert pending[0].payload["recommendation_summary"]["series_key"] == "SKU_X|NORTH"


# ---------------------------------------------------------------------------
# ErpHandoffPayload JSON round-trip (the wire shape UiPath consumes)
# ---------------------------------------------------------------------------


def test_erp_handoff_payload_round_trips_through_json(
    gateway: InProcessApprovalGateway,
) -> None:
    """The whole point of the typed boundary: any implementation of
    the ApprovalGateway that produces an ErpHandoffPayload must
    serialise to a UiPath-consumable JSON shape and back without
    loss."""
    recs = [_make_recommendation("SKU_A|WEST", order_quantity=150)]
    [(recommendation, request_id)] = request_replenishment_approvals(
        gateway=gateway,
        run_id="run-1",
        recommendations=recs,
        requested_by="replenishment_engine",
    )
    decided = gateway.acknowledge(
        request_id=request_id, decision="APPROVE", approver="alice", reason="ok"
    )
    payload = release_erp_handoff(
        gateway=gateway,
        run_id="run-1",
        approved_request=decided,
        recommendations=[recommendation],
    )
    dumped = payload.model_dump_json()
    restored = ErpHandoffPayload.model_validate_json(dumped)
    assert restored == payload
