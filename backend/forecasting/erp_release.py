"""Phase 6 CB5: ERP release service — assembles the typed handoff
payload from an approved replenishment batch.

Wires the approval gateway and the replenishment pipeline into the
final release to ERP/procurement. Two operations:

* ``request_replenishment_approvals`` — for every recommendation
  whose ``approval_tier`` is not ``"auto"``, raise an
  ``ApprovalRequest`` on the gateway (kind:
  ``replenishment_recommendation``). Return a list of
  ``(recommendation, request_id | None)`` pairs in input order;
  ``request_id`` is ``None`` for the auto-approved rows.
* ``release_erp_handoff`` — after a human APPROVEs a request, build
  an ``ErpHandoffPayload`` that joins the approved recommendations,
  the approval decision, and the audit trail. The payload is the
  contract the cockpit UI displays and (optionally) hands to an
  external ERP connector; the in-process implementation just
  returns it.

Design:

* Pure of I/O: ``release_erp_handoff`` does not write to disk
  itself. The caller (the cockpit UI, a cron job, a webhook) decides
  where the payload lands. This keeps the function deterministic
  and trivial to assert against.
* The release is a one-shot — a request that is still ``pending``
  or has been ``rejected`` cannot produce a payload. This is a hard
  contract: releasing a non-approved batch would let un-reviewed
  orders reach ERP. The error type is specific so the cockpit can
  surface a real reason, not a generic exception.
* The audit trail on the payload is the *full* gateway event log
  for the request, in order, including the original ``raised`` and
  the ``decided`` events. ERP systems can persist the trail
  alongside the receiving transaction.
"""

from __future__ import annotations

from typing import Iterable

from forecasting.approval_gateway import ApprovalGateway
from forecasting.contracts import (
    ApprovalRequest,
    ErpHandoffPayload,
)
from forecasting.replenishment import ReplenishmentRecommendation


class ErpReleaseError(Exception):
    """Base class for ERP-release errors."""


class RequestNotApprovedError(ErpReleaseError):
    """The request is not in a state that permits release.

    The request must be ``status="approved"``. ``pending`` (no human
    decision yet) and ``rejected`` (human said no) are both blocked.
    """


def request_replenishment_approvals(
    gateway: ApprovalGateway,
    run_id: str,
    recommendations: Iterable[ReplenishmentRecommendation],
    requested_by: str,
) -> list[tuple[ReplenishmentRecommendation, str | None]]:
    """Raise an ``ApprovalRequest`` for every non-auto recommendation.

    Returns a list of ``(recommendation, request_id)`` pairs in input
    order. For ``approval_tier == "auto"`` the request_id is
    ``None`` (no human in the loop). The caller is expected to
    iterate the returned pairs and handle each request_id
    individually — a planner can approve some and reject others.
    """
    pairs: list[tuple[ReplenishmentRecommendation, str | None]] = []
    for rec in recommendations:
        if rec.approval_tier == "auto":
            pairs.append((rec, None))
            continue
        # Build a small summary the human can read at a glance. The
        # full recommendation stays in the platform; the gateway
        # payload is a preview.
        summary = {
            "series_key": rec.series_key,
            "order_quantity": rec.order_quantity,
            "approval_tier": rec.approval_tier,
            "reorder_point": rec.reorder_point,
            "current_inventory": rec.current_inventory,
            "lead_time_days": rec.lead_time_days,
        }
        request = gateway.raise_request(
            run_id=run_id,
            kind="replenishment_recommendation",
            title=f"Approve replenishment for {rec.series_key}",
            summary=(
                f"Order {rec.order_quantity:g} units of {rec.series_key} "
                f"(tier: {rec.approval_tier})."
            ),
            requested_by=requested_by,
            payload={"recommendation_summary": summary},
        )
        pairs.append((rec, request.request_id))
    return pairs


def release_erp_handoff(
    gateway: ApprovalGateway,
    run_id: str,
    approved_request: ApprovalRequest,
    recommendations: Iterable[ReplenishmentRecommendation],
) -> ErpHandoffPayload:
    """Build the ERP handoff payload from an approved request.

    The caller is responsible for persisting the payload (writing
    to disk, posting to ERP via a connector, or handing it to the
    cockpit UI). This function builds it and asserts the contract.

    Raises ``RequestNotApprovedError`` if ``approved_request.status``
    is not ``"approved"``.
    """
    if approved_request.status != "approved":
        raise RequestNotApprovedError(
            f"Cannot release request {approved_request.request_id}: "
            f"status is {approved_request.status!r}, must be 'approved'."
        )
    if approved_request.decision is None or approved_request.approver is None:
        # Defensive: an 'approved' status without a decision/approver
        # is an internal inconsistency; the release is blocked.
        raise RequestNotApprovedError(
            f"Request {approved_request.request_id} is 'approved' but "
            f"missing decision/approver fields."
        )
    if approved_request.decided_at is None or approved_request.reason is None:
        raise RequestNotApprovedError(
            f"Request {approved_request.request_id} is 'approved' but "
            f"missing decided_at/reason fields."
        )

    # Build the recommendation dicts the ERP layer needs. Keep the
    # boundary narrow — we ship the fields ERP systems actually use,
    # not the full platform-side recommendation (which carries audit
    # fields, internal model names, etc.).
    erp_recs: list[dict[str, object]] = []
    for rec in recommendations:
        erp_recs.append(
            {
                "series_key": rec.series_key,
                "order_quantity": rec.order_quantity,
                "approval_tier": rec.approval_tier,
                "lead_time_days": rec.lead_time_days,
            }
        )

    # Pull the audit trail from the gateway's audit log. For the
    # in-process implementation that's the JSONL file; for an
    # alternative gateway that doesn't expose the log the trail
    # falls back to empty. The JSONL file is always the durable
    # record; the trail on the payload is best-effort.
    audit_trail = _read_audit_trail(gateway, run_id, approved_request.request_id)

    return ErpHandoffPayload(
        handoff_id=f"ho-{run_id}-{approved_request.request_id}",
        run_id=run_id,
        # Use the decision timestamp as the release timestamp — the
        # release is the human's authorisation, and any time after
        # the authorisation is downstream system noise.
        released_at=approved_request.decided_at,
        approval_request_id=approved_request.request_id,
        approver=approved_request.approver,
        approval_reason=approved_request.reason,
        approved_at=approved_request.decided_at,
        recommendations=erp_recs,
        audit_trail=audit_trail,
    )


def _read_audit_trail(
    gateway: ApprovalGateway,
    run_id: str,
    request_id: str,
) -> list:
    """Read the audit events for one request, in order.

    Uses the in-process gateway's ``read_audit_log`` helper if
    available; falls back to an empty list for any other
    implementation that doesn't expose the log. The audit_trail
    on the payload is best-effort — the JSONL file is always the
    durable record.
    """
    read_log = getattr(gateway, "read_audit_log", None)
    if read_log is None:
        return []
    all_events = read_log(run_id)
    return [e for e in all_events if e.request_id == request_id]
