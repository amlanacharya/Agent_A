"""Tests for the named phase machine (Issue #3).

The legal Phase transitions used to be encoded in three places (HALTED
check in ``save_run_state`` / ``_assert_mutable_run`` and the
phase-specific guards in ``conductor_tools``). These tests pin the
single source of truth — ``LEGAL_TRANSITIONS`` /
:func:`advance_phase` / :func:`can_transition` in
:mod:`forecasting.run_state`.
"""

from __future__ import annotations

import pytest

from forecasting.run_state import (
    LEGAL_TRANSITIONS,
    LifecycleError,
    Phase,
    RunState,
    advance_phase,
    can_transition,
    legal_next_phases,
)


def _state(phase: Phase, **overrides) -> RunState:
    """Build a minimal RunState at the given phase for transition tests."""
    return RunState(
        run_id="r-lifecycle",
        phase=phase,
        domain="fmcg",
        created_at="2026-06-14T00:00:00Z",
        **overrides,
    )


# ---------------------------------------------------------------------------
# legal_next_phases / can_transition
# ---------------------------------------------------------------------------


def test_legal_transitions_table_is_well_formed() -> None:
    """Every Phase must appear as a key; HALTED is terminal (empty set)."""
    assert set(LEGAL_TRANSITIONS) == set(Phase)
    assert LEGAL_TRANSITIONS[Phase.HALTED] == frozenset()
    # HALTED must be reachable from every non-terminal phase
    for phase, targets in LEGAL_TRANSITIONS.items():
        if phase == Phase.HALTED:
            continue
        assert Phase.HALTED in targets, f"{phase.value} cannot halt"


def test_legal_next_phases_returns_fresh_frozenset() -> None:
    """The returned frozenset is a copy; callers cannot mutate the table."""
    snapshot = legal_next_phases(Phase.PREFLIGHT)
    assert snapshot == frozenset({Phase.MERIDIAN_SCOPING, Phase.HALTED})
    # No mutation API on frozenset; the contract is the value, not
    # the identity. Two calls return equal but not necessarily
    # identical values.
    assert legal_next_phases(Phase.PREFLIGHT) == snapshot


def test_can_transition_handles_all_pairs() -> None:
    for from_phase, to_phase in [
        (Phase.PREFLIGHT, Phase.MERIDIAN_SCOPING),
        (Phase.MERIDIAN_SCOPING, Phase.FORGE_EDA),
        (Phase.FORGE_EDA, Phase.FOUNDRY_MODELLING),
        (Phase.FOUNDRY_MODELLING, Phase.REPORT_READY),
        # Loop-backs are explicit (ADR-0005 "lifecycle is not
        # strictly linear").
        (Phase.FORGE_EDA, Phase.MERIDIAN_SCOPING),
        (Phase.FOUNDRY_MODELLING, Phase.MERIDIAN_SCOPING),
        # HALTED is reachable from everywhere.
        (Phase.REPORT_READY, Phase.HALTED),
    ]:
        assert can_transition(from_phase, to_phase), f"{from_phase.value} -> {to_phase.value} should be legal"
    # Illegal: HALTED is terminal.
    assert not can_transition(Phase.HALTED, Phase.PREFLIGHT)
    # Illegal: skipping phases.
    assert not can_transition(Phase.PREFLIGHT, Phase.FORGE_EDA)
    assert not can_transition(Phase.MERIDIAN_SCOPING, Phase.FOUNDRY_MODELLING)


# ---------------------------------------------------------------------------
# advance_phase
# ---------------------------------------------------------------------------


def test_advance_phase_mutates_state_when_legal() -> None:
    state = _state(Phase.PREFLIGHT)
    new_state = advance_phase(state, Phase.MERIDIAN_SCOPING)
    assert new_state.phase == Phase.MERIDIAN_SCOPING
    # Returns a NEW state object (caller is responsible for persistence).
    assert new_state is not state


def test_advance_phase_rejects_illegal_transition() -> None:
    state = _state(Phase.PREFLIGHT)
    with pytest.raises(LifecycleError, match="illegal phase transition"):
        advance_phase(state, Phase.REPORT_READY)


def test_advance_phase_rejects_halted_as_source() -> None:
    state = _state(Phase.HALTED, halt_reason="manual")
    with pytest.raises(LifecycleError, match="halted"):
        advance_phase(state, Phase.PREFLIGHT)


def test_advance_phase_enforces_all_preconditions() -> None:
    state = _state(Phase.MERIDIAN_SCOPING, pack_confirmed=True, open_risks=0)
    # Preconditions are named booleans — all must be True.
    with pytest.raises(LifecycleError, match="pack_not_yet_confirmed"):
        advance_phase(
            state,
            Phase.FORGE_EDA,
            is_meridian_scoping=True,
            pack_not_yet_confirmed=False,  # already confirmed
            open_risks_is_zero=True,
        )


def test_advance_phase_reports_first_failing_precondition() -> None:
    """When multiple preconditions fail, the first one in the call
    order is reported on the error — gives the cockpit a stable
    surface to point at."""
    state = _state(Phase.MERIDIAN_SCOPING, open_risks=2)
    with pytest.raises(LifecycleError, match="open_risks_is_zero"):
        advance_phase(
            state,
            Phase.FORGE_EDA,
            is_meridian_scoping=True,
            pack_not_yet_confirmed=True,
            open_risks_is_zero=False,
        )


def test_advance_phase_succeeds_when_every_precondition_holds() -> None:
    state = _state(Phase.MERIDIAN_SCOPING)
    new_state = advance_phase(
        state,
        Phase.FORGE_EDA,
        is_meridian_scoping=True,
        pack_not_yet_confirmed=True,
        open_risks_is_zero=True,
    )
    assert new_state.phase == Phase.FORGE_EDA


# ---------------------------------------------------------------------------
# Conductor / workspace glue uses the same primitive
# ---------------------------------------------------------------------------


def test_conductor_confirm_pack_uses_state_machine() -> None:
    """The conductor's ``confirm_pack_and_advance`` raises the same
    LifecycleError when the preconditions are not met — the state
    machine IS the source of truth."""
    from forecasting.tools.conductor_tools import confirm_pack_and_advance
    from forecasting.run_state import create_run_state, save_run_state

    state = create_run_state("r-confirm-1", domain="fmcg")
    # Move the run to MERIDIAN_SCOPING so the phase transition is
    # legal at all; then leave open_risks > 0 so the precondition
    # for confirm_pack_and_advance fails.
    state.phase = Phase.MERIDIAN_SCOPING
    state.open_risks = 1
    save_run_state(state)
    with pytest.raises(LifecycleError, match="open_risks_is_zero"):
        confirm_pack_and_advance("r-confirm-1")


def test_conductor_trigger_foundry_uses_state_machine() -> None:
    """``trigger_foundry`` enforces ``forge_complete`` via the state
    machine's precondition, not a separate inline check."""
    from forecasting.tools.conductor_tools import trigger_foundry
    from forecasting.run_state import create_run_state, save_run_state

    state = create_run_state("r-foundry-1", domain="fmcg")
    # Move the Run to FORGE_EDA so the transition is legal at all.
    state.phase = Phase.FORGE_EDA
    save_run_state(state)
    # forge_complete is False by default -> precondition fails.
    with pytest.raises(LifecycleError, match="forge_complete"):
        trigger_foundry("r-foundry-1")
