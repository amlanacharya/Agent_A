from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from forecasting.run_state import (
    HaltedRunError,
    LifecycleError,
    Phase,
    _phase_str,
    advance_phase,
    can_transition,
    load_run_state,
    run_dir,
    save_run_state,
)


# ``ConditionViolationError`` is kept as an alias of ``LifecycleError``
# so existing callers (tests, other tools) that imported the old name
# keep working — both error types describe the same failure mode
# (a phase transition was illegal or its preconditions were unmet).
ConditionViolationError = LifecycleError


def get_run_state(run_id: str) -> dict:
    return load_run_state(run_id).model_dump(mode="json")


def update_run_state(run_id: str, patch: dict) -> dict:
    state = load_run_state(run_id)
    if state.phase == Phase.HALTED:
        raise HaltedRunError(run_id)

    if patch.get("pack_confirmed") is False and state.pack_confirmed:
        raise ValueError("pack_confirmed is a one-way transition (False -> True only)")

    merged = {**state.model_dump(mode="python"), **patch}
    updated = type(state).model_validate(merged)
    save_run_state(updated)
    return updated.model_dump(mode="json")


def advance_to_meridian(run_id: str, user_message: str) -> None:
    del user_message
    state = load_run_state(run_id)
    if state.phase == Phase.PREFLIGHT:
        # Only advance if the transition is legal; HALTED is the only
        # other legal target from PREFLIGHT and we don't auto-halt
        # here. The HALTED guard is the save_run_state one.
        updated = advance_phase(state, Phase.MERIDIAN_SCOPING)
        save_run_state(updated)


def confirm_pack_and_advance(run_id: str) -> None:
    state = load_run_state(run_id)
    # ``advance_phase`` raises LifecycleError on an illegal transition
    # OR a failing precondition — that single check replaces the
    # three inline guards that used to live here.
    updated = advance_phase(
        state,
        Phase.FORGE_EDA,
        is_meridian_scoping=(state.phase == Phase.MERIDIAN_SCOPING),
        pack_not_yet_confirmed=(not state.pack_confirmed),
        open_risks_is_zero=(state.open_risks == 0),
    )
    # Mark the pack confirmed as part of the same atomic update.
    updated = updated.model_copy(update={"pack_confirmed": True})
    save_run_state(updated)


def trigger_foundry(run_id: str) -> None:
    state = load_run_state(run_id)
    updated = advance_phase(
        state,
        Phase.FOUNDRY_MODELLING,
        forge_complete=(state.forge_complete),
    )
    save_run_state(updated)


def create_prism_run(run_id: str, scenario_description: str, entities: dict) -> dict:
    del scenario_description, entities
    state = load_run_state(run_id)
    # A Prism clone is a side-branch from a finished run, NOT a
    # phase transition. The original ``can_transition`` check used
    # here was wrong because REPORT_READY is terminal (its only
    # legal successor is HALTED) — ``can_transition(report_ready,
    # report_ready)`` is False. Prism accepts the run being at
    # REPORT_READY (the finished report) and creates a child
    # whatif directory; the parent run stays at REPORT_READY.
    if _phase_str(state.phase) != Phase.REPORT_READY.value:
        raise LifecycleError(
            f"create_prism_run: phase={_phase_str(state.phase)} cannot host a scenario run; "
            f"need phase=report_ready"
        )
    whatif_id = f"wi-{uuid.uuid4().hex[:8]}"
    (run_dir(run_id) / "whatif" / whatif_id).mkdir(parents=True, exist_ok=True)
    update_run_state(run_id, {"active_whatif_runs": [*state.active_whatif_runs, whatif_id]})
    return {"whatif_id": whatif_id}


def surface_clarification(run_id: str, message: str, sse_emit: Callable) -> None:
    del run_id
    sse_emit("message_done", {"agent": "conductor", "full_text": message})


def log_halt(run_id: str, reason: str, sse_emit: Callable) -> None:
    state = load_run_state(run_id)
    obs = run_dir(run_id) / "obs_log.json"

    if state.phase != Phase.HALTED:
        state.halt_reason = reason
        state.phase = Phase.HALTED
        save_run_state(state)

    _append_obs(obs, {"event": "HALT", "reason": reason})
    sse_emit("error", {"reason": "Run halted - please start a new run.", "halt_reason": reason})


def _append_obs(path: Path, entry: dict) -> None:
    log = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, list):
                log = loaded
        except json.JSONDecodeError:
            log = []
    stamped = {**entry, "ts": datetime.now(timezone.utc).isoformat()}
    log.append(stamped)
    path.write_text(json.dumps(log, indent=2))
