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

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)


class RunNotFoundError(Exception):
    def __init__(self, run_id: str):
        super().__init__(f"Run '{run_id}' not found")
        self.run_id = run_id


class HaltedRunError(Exception):
    def __init__(self, run_id: str):
        super().__init__(f"Run '{run_id}' is halted and cannot be mutated")
        self.run_id = run_id


class LifecycleError(ValueError):
    """Raised when a phase transition is illegal or a precondition is unmet.

    The conductor / learning workspace / HTTP layer all share this
    error type so the user-facing message is consistent — illegal
    transitions and unmet preconditions are both "you cannot do that
    from here".
    """


# ---------------------------------------------------------------------------
# Phase machine
# ---------------------------------------------------------------------------
#
# The legal Phase transitions for a Run are encoded here as a single
# data structure. Before this lived in three places (HALTED check in
# ``save_run_state`` and ``_assert_mutable_run``, plus the
# phase-specific guards in ``conductor_tools``); a missing check in a
# new tool could silently violate the state machine. The transition
# table IS the source of truth; new tools plug into ``advance_phase``
# and the legal-transition table is the test surface.
#
# Loop-backs are explicit rows:
#
#   forge_eda        -> meridian_scoping  (when Forge finds a gap)
#   foundry_modelling -> meridian_scoping  (when Foundry finds a gap)
#
# Both are bounded — the Run halts when the loop-back cap is reached
# (enforced by the Guard Layer, not here).
LEGAL_TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    Phase.PREFLIGHT: frozenset({Phase.MERIDIAN_SCOPING, Phase.HALTED}),
    Phase.MERIDIAN_SCOPING: frozenset({Phase.FORGE_EDA, Phase.HALTED}),
    Phase.FORGE_EDA: frozenset(
        {Phase.FOUNDRY_MODELLING, Phase.MERIDIAN_SCOPING, Phase.HALTED}
    ),
    Phase.FOUNDRY_MODELLING: frozenset(
        {Phase.REPORT_READY, Phase.MERIDIAN_SCOPING, Phase.HALTED}
    ),
    Phase.REPORT_READY: frozenset({Phase.HALTED}),  # immutable after report
    Phase.HALTED: frozenset(),  # terminal
}


def can_transition(phase: Phase | str, to: Phase | str) -> bool:
    """True iff ``to`` is a legal successor of ``phase``.

    Accepts ``Phase`` enums or the equivalent string values
    (``RunState`` stores the phase as a string because of
    ``use_enum_values=True``). The lookup goes through
    :func:`_phase_key` so the table is keyed on enum values whether
    the caller passes enums or strings.
    """
    return to in LEGAL_TRANSITIONS.get(_phase_key(phase), frozenset())


def _phase_key(phase: Phase | str) -> Phase:
    """Coerce ``phase`` to a :class:`Phase` enum.

    ``RunState.phase`` is a string at runtime because of
    ``use_enum_values=True``; the transition table is keyed on the
    enum form. This helper makes the table usable from both call
    sites.
    """
    if isinstance(phase, Phase):
        return phase
    return Phase(phase)


def legal_next_phases(phase: Phase | str) -> frozenset[Phase]:
    """Return the set of phases ``phase`` may transition to.

    Returns a fresh frozenset each call so callers cannot mutate the
    canonical ``LEGAL_TRANSITIONS`` table. Accepts the same input
    shapes as :func:`can_transition`.
    """
    return LEGAL_TRANSITIONS[_phase_key(phase)]


def advance_phase(state: RunState, to: Phase, **preconditions: bool) -> RunState:
    """Mutate ``state`` if the transition is legal and all preconditions hold.

    ``preconditions`` are named boolean conditions — every entry must
    be True for the transition to proceed. The first failing name is
    reported on the raised :class:`LifecycleError` so the cockpit can
    surface the exact gate that blocked the move.

    Returns a new :class:`RunState` with ``phase`` set to ``to``. The
    caller is responsible for persisting the result via
    :func:`save_run_state` (this function does not write to disk so it
    stays trivially testable).
    """
    if not can_transition(state.phase, to):
        raise LifecycleError(
            f"illegal phase transition {_phase_str(state.phase)} -> {_phase_str(to)}; "
            f"legal next phases are "
            f"{sorted(p.value for p in legal_next_phases(state.phase))}"
        )
    for name, ok in preconditions.items():
        if not ok:
            raise LifecycleError(
                f"precondition {name!r} not met for "
                f"{_phase_str(state.phase)} -> {_phase_str(to)}"
            )
    return state.model_copy(update={"phase": to})


def _phase_str(phase: Phase | str) -> str:
    """Render a Phase (or the string form ``RunState`` uses) for messages."""
    if isinstance(phase, Phase):
        return phase.value
    return str(phase)


def run_dir(run_id: str) -> Path:
    if (
        not run_id
        or Path(run_id).is_absolute()
        or "/" in run_id
        or "\\" in run_id
        or run_id in {".", ".."}
    ):
        raise ValueError("run_id must be a single safe path segment")
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
    if path.exists():
        existing = RunState.model_validate_json(path.read_text())
        if existing.phase == Phase.HALTED:
            raise HaltedRunError(state.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2))
