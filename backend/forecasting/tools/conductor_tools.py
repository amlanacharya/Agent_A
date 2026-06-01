from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from forecasting.run_state import (
    HaltedRunError,
    Phase,
    load_run_state,
    run_dir,
    save_run_state,
)


class ConditionViolationError(Exception):
    def __init__(self, tool: str, condition: str):
        super().__init__(f"Conductor tool '{tool}' precondition failed: {condition}")


def get_run_state(run_id: str) -> dict:
    return load_run_state(run_id).model_dump(mode="json")


def update_run_state(run_id: str, patch: dict) -> dict:
    state = load_run_state(run_id)
    if state.phase == Phase.HALTED:
        raise HaltedRunError(run_id)

    if patch.get("pack_confirmed") is False and state.pack_confirmed:
        raise ValueError("pack_confirmed is a one-way transition (False -> True only)")

    updated = state.model_copy(update=patch)
    save_run_state(updated)
    return updated.model_dump(mode="json")


def advance_to_meridian(run_id: str, user_message: str) -> None:
    del user_message
    state = load_run_state(run_id)
    if state.phase == Phase.PREFLIGHT:
        update_run_state(run_id, {"phase": Phase.MERIDIAN_SCOPING.value})


def confirm_pack_and_advance(run_id: str) -> None:
    state = load_run_state(run_id)
    if state.phase != Phase.MERIDIAN_SCOPING:
        raise ConditionViolationError(
            "confirm_pack_and_advance", f"phase={state.phase} not meridian_scoping"
        )
    if state.pack_confirmed:
        raise ConditionViolationError("confirm_pack_and_advance", "pack already confirmed")
    if state.open_risks > 0:
        raise ConditionViolationError(
            "confirm_pack_and_advance", f"open_risks={state.open_risks}"
        )
    update_run_state(run_id, {"pack_confirmed": True, "phase": Phase.FORGE_EDA.value})


def trigger_foundry(run_id: str) -> None:
    state = load_run_state(run_id)
    if not state.forge_complete:
        raise ConditionViolationError("trigger_foundry", "forge_complete is False")
    update_run_state(run_id, {"phase": Phase.FOUNDRY_MODELLING.value})


def create_prism_run(run_id: str, scenario_description: str, entities: dict) -> dict:
    del scenario_description, entities
    state = load_run_state(run_id)
    if state.phase != Phase.REPORT_READY:
        raise ConditionViolationError(
            "create_prism_run", f"phase={state.phase} not report_ready"
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
        log = json.loads(path.read_text())
    stamped = {**entry, "ts": datetime.now(timezone.utc).isoformat()}
    log.append(stamped)
    path.write_text(json.dumps(log, indent=2))
