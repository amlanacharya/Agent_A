from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

OUTPUTS_ROOT = Path("backend/outputs")


class Phase(str, Enum):
    PREFLIGHT = "preflight"
    MERIDIAN_SCOPING = "meridian_scoping"
    FORGE_EDA = "forge_eda"
    FOUNDRY_MODELLING = "foundry_modelling"
    REPORT_READY = "report_ready"
    HALTED = "halted"


class RunState(BaseModel):
    run_id: str
    phase: Phase
    pack_confirmed: bool = False
    meridian_turn_count: int = 0
    forge_complete: bool = False
    foundry_complete: bool = False
    active_whatif_runs: list[str] = Field(default_factory=list)
    open_risks: int = 0
    override_count: int = 0
    halt_reason: str | None = None
    domain: str
    created_at: str

    model_config = ConfigDict(use_enum_values=True)


class RunNotFoundError(Exception):
    def __init__(self, run_id: str):
        super().__init__(f"Run '{run_id}' not found")
        self.run_id = run_id


class HaltedRunError(Exception):
    def __init__(self, run_id: str):
        super().__init__(f"Run '{run_id}' is halted and cannot be mutated")
        self.run_id = run_id


def run_dir(run_id: str) -> Path:
    return OUTPUTS_ROOT / run_id


def state_path(run_id: str) -> Path:
    return run_dir(run_id) / "run_state.json"


def create_run_state(run_id: str, domain: str) -> RunState:
    state = RunState(
        run_id=run_id,
        phase=Phase.PREFLIGHT,
        domain=domain,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    run_dir(run_id).mkdir(parents=True, exist_ok=True)
    save_run_state(state)
    return state


def load_run_state(run_id: str) -> RunState:
    path = state_path(run_id)
    if not path.exists():
        raise RunNotFoundError(run_id)
    return RunState.model_validate_json(path.read_text())


def save_run_state(state: RunState) -> None:
    if state.phase == Phase.HALTED and state.halt_reason is None:
        raise ValueError("halt_reason must be set before saving a HALTED RunState")
    path = state_path(state.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2))
