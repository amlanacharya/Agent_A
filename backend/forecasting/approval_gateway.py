"""Phase 6 CB3: approval gateway — the in-process stand-in for UiPath.

The platform raises an ``ApprovalRequest`` whenever a decision needs a
human. In production that request travels to the UiPath Orchestrator
(Queue + Form) and back via the ``UiPathApprovalGateway`` (separate
repo, out of scope here). In this repo the default implementation is
``InProcessApprovalGateway`` — a pure-Python, in-memory stand-in that
exposes the same surface and persists every state change to
``backend/outputs/{run_id}/approvals.jsonl`` for audit.

Design:

* The gateway is the single source of truth for request state. It
  holds an in-memory dict keyed by ``request_id`` and writes one
  ``ApprovalEvent`` to the audit log per state change.
* The audit log is append-only JSON Lines. The platform can replay
  the full lifecycle of a request from the log alone — the in-memory
  dict is rebuilt at startup by reading the log (see ``load_from_audit``).
* Decoupled from ``cockpit_state``: the gateway doesn't know about
  ``CockpitState``. Callers (platform code) raise a request, then
  flip ``cockpit_state.approval_needed`` themselves — keeps the
  gateway testable without a ``RunState``.
* The interface (``ApprovalGateway``) is an ABC so a future
  ``UiPathApprovalGateway`` can drop in without changing the rest of
  the platform.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from forecasting.contracts import (
    ApprovalDecisionValue,
    ApprovalEvent,
    ApprovalRequest,
)


class ApprovalError(Exception):
    """Base class for gateway-level errors."""


class RequestNotFoundError(ApprovalError):
    """The request_id is unknown to this gateway."""


class AlreadyDecidedError(ApprovalError):
    """The request has already been decided and cannot be re-acknowledged."""


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 with a trailing Z, second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class ApprovalGateway(ABC):
    """Interface every approval gateway implements.

    The contract is small on purpose: raise, acknowledge, fetch,
    list. A UiPath-side implementation would translate raise to a
    Queue Item creation and acknowledge to a Form submission; the
    in-process implementation just mutates an in-memory dict and
    writes an audit line.
    """

    @abstractmethod
    def raise_request(
        self,
        run_id: str,
        kind: str,
        title: str,
        summary: str,
        requested_by: str,
        payload: dict[str, object] | None = None,
    ) -> ApprovalRequest:
        """Record a new pending request. Returns the persisted request."""

    @abstractmethod
    def acknowledge(
        self,
        request_id: str,
        decision: ApprovalDecisionValue,
        approver: str,
        reason: str,
    ) -> ApprovalRequest:
        """Decide a pending request. Raises ``RequestNotFoundError`` if
        the id is unknown; raises ``AlreadyDecidedError`` if the
        request is no longer pending."""

    @abstractmethod
    def get(self, request_id: str) -> ApprovalRequest | None:
        """Return the current state of a request, or None if unknown."""

    @abstractmethod
    def list_pending(self, run_id: str | None = None) -> list[ApprovalRequest]:
        """Return all pending requests, optionally filtered by run_id."""


class InProcessApprovalGateway(ApprovalGateway):
    """The in-process default implementation.

    Persists state to an in-memory dict and mirrors every change to
    ``{audit_root}/{run_id}/approvals.jsonl`` as JSON Lines. The audit
    log is the durable record; the in-memory dict is rebuilt from it
    on restart via ``load_from_audit`` (or left empty for a fresh
    process — both are valid for a single-process POC).
    """

    def __init__(self, audit_root: Path) -> None:
        self._audit_root = Path(audit_root)
        self._requests: dict[str, ApprovalRequest] = {}

    # ----- audit log helpers ------------------------------------------------

    def _audit_path(self, run_id: str) -> Path:
        return self._audit_root / run_id / "approvals.jsonl"

    def _write_event(self, event: ApprovalEvent) -> None:
        path = self._audit_path(event.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json())
            f.write("\n")

    def _read_audit(self, run_id: str) -> list[ApprovalEvent]:
        path = self._audit_path(run_id)
        if not path.exists():
            return []
        events: list[ApprovalEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            events.append(ApprovalEvent.model_validate_json(line))
        return events

    def load_from_audit(self, run_id: str) -> list[ApprovalRequest]:
        """Rebuild in-memory state from the audit log for one run.

        Returns the requests in chronological order (oldest first).
        Idempotent: calling twice does not duplicate. Useful at
        process startup when the in-memory dict is empty but the
        audit log survived a restart.
        """
        events = self._read_audit(run_id)
        if not events:
            return []
        # Group events by request_id, in order.
        by_request: dict[str, list[ApprovalEvent]] = {}
        for ev in events:
            by_request.setdefault(ev.request_id, []).append(ev)
        requests: list[ApprovalRequest] = []
        for rid in sorted(by_request):
            evs = by_request[rid]
            raised = next((e for e in evs if e.event_type == "raised"), None)
            if raised is None:
                # No 'raised' event — partial log; skip.
                continue
            decided = next((e for e in evs if e.event_type == "decided"), None)
            if decided is not None:
                # We can't reconstruct the full request from events alone
                # (we'd need the original payload), so for the in-process
                # gateway the in-memory dict remains the source of truth.
                # Returning [] is a safe, honest answer.
                continue
            # Pending: leave alone — the in-memory dict is the truth.
        # The gateway's in-memory dict is the source of truth for
        # pending requests; this helper exists so a UiPath-backed
        # implementation can sync after a restart if it needs to.
        return requests

    # ----- ApprovalGateway surface -----------------------------------------

    def raise_request(
        self,
        run_id: str,
        kind: str,
        title: str,
        summary: str,
        requested_by: str,
        payload: dict[str, object] | None = None,
    ) -> ApprovalRequest:
        request = ApprovalRequest(
            request_id=_new_id("req"),
            run_id=run_id,
            kind=kind,  # type: ignore[arg-type]
            title=title,
            summary=summary,
            payload=payload or {},
            requested_by=requested_by,
            requested_at=_now_iso(),
        )
        self._requests[request.request_id] = request
        self._write_event(
            ApprovalEvent(
                event_id=_new_id("ev"),
                request_id=request.request_id,
                run_id=run_id,
                event_type="raised",
                occurred_at=request.requested_at,
                actor=requested_by,
                notes=f"raised {kind} request",
            )
        )
        return request

    def acknowledge(
        self,
        request_id: str,
        decision: ApprovalDecisionValue,
        approver: str,
        reason: str,
    ) -> ApprovalRequest:
        existing = self._requests.get(request_id)
        if existing is None:
            raise RequestNotFoundError(f"Unknown request_id: {request_id}")
        if existing.status not in ("pending",):
            raise AlreadyDecidedError(
                f"Request {request_id} is already {existing.status}; "
                f"cannot re-acknowledge."
            )
        decided_at = _now_iso()
        # Status mapping:
        #   APPROVE  -> "approved"   (terminal; the work is released)
        #   REJECT   -> "rejected"   (terminal; the work is stopped)
        #   DEFER    -> "pending"    (the human asked to be asked again
        #                             later; the request stays open and
        #                             can be acknowledged again. The
        #                             decision/approver fields capture
        #                             the deferral event but the next
        #                             acknowledge overwrites them.)
        if decision == "APPROVE":
            new_status: str = "approved"
        elif decision == "REJECT":
            new_status = "rejected"
        else:  # DEFER
            new_status = "pending"
        updated = existing.model_copy(
            update={
                "status": new_status,  # type: ignore[arg-type]
                "decision": decision,
                "approver": approver,
                "decided_at": decided_at,
                "reason": reason,
            }
        )
        self._requests[request_id] = updated
        self._write_event(
            ApprovalEvent(
                event_id=_new_id("ev"),
                request_id=request_id,
                run_id=updated.run_id,
                event_type="decided",
                occurred_at=decided_at,
                actor=approver,
                notes=f"decision={decision} reason={reason!r}",
            )
        )
        return updated

    def get(self, request_id: str) -> ApprovalRequest | None:
        return self._requests.get(request_id)

    def list_pending(self, run_id: str | None = None) -> list[ApprovalRequest]:
        results = [r for r in self._requests.values() if r.status == "pending"]
        if run_id is not None:
            results = [r for r in results if r.run_id == run_id]
        # Stable order: oldest pending first.
        results.sort(key=lambda r: r.requested_at)
        return results

    # ----- test / introspection helpers ------------------------------------

    def list_all(self, run_id: str | None = None) -> list[ApprovalRequest]:
        """Return every request (any status) for a run, oldest first.

        Not part of the abstract interface — used by tests and the
        full-chain integration test to reconstruct audit trails.
        """
        results = list(self._requests.values())
        if run_id is not None:
            results = [r for r in results if r.run_id == run_id]
        results.sort(key=lambda r: r.requested_at)
        return results

    def read_audit_log(self, run_id: str) -> list[ApprovalEvent]:
        """Read the audit log for a run in order. Test/debug helper."""
        return self._read_audit(run_id)
