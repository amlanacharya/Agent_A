from forecasting.cockpit_state import CockpitState
from forecasting.run_state import Phase, RunState
import pytest
from pydantic import ValidationError


def test_cockpit_state_can_be_created_with_required_fields():
    state = CockpitState(
        run_id="run-123",
        current_step="Checking uploaded data",
        active_agent="Meridian",
    )

    assert state.run_id == "run-123"
    assert state.current_step == "Checking uploaded data"
    assert state.active_agent == "Meridian"
    assert state.approval_needed is False
    assert state.blockers == []


def test_to_public_dict_returns_json_safe_status_fields():
    state = CockpitState(
        run_id="run-123",
        current_step="Verifying modelling gate",
        active_agent="Foundry",
        tool_result="EDA passed quality checks",
        code_escalation_status="not_requested",
        code_attempt=1,
        verifier_gate="model_quality",
        approval_needed=True,
        confidence="high",
        blockers=["Awaiting reviewer sign-off"],
    )

    assert state.to_public_dict() == {
        "run_id": "run-123",
        "current_step": "Verifying modelling gate",
        "active_agent": "Foundry",
        "tool_result": "EDA passed quality checks",
        "code_escalation_status": "not_requested",
        "code_attempt": 1,
        "verifier_gate": "model_quality",
        "approval_needed": True,
        "confidence": "high",
        "blockers": ["Awaiting reviewer sign-off"],
    }


def test_with_blocker_appends_blocker_and_lowers_confidence():
    state = CockpitState(
        run_id="run-123",
        current_step="Preparing model run",
        active_agent="Foundry",
        confidence="medium",
        blockers=["Missing holiday calendar"],
    )

    updated = state.with_blocker("Promo mapping is ambiguous")

    assert updated.blockers == [
        "Missing holiday calendar",
        "Promo mapping is ambiguous",
    ]
    assert updated.confidence == "low"
    assert state.blockers == ["Missing holiday calendar"]
    assert state.confidence == "medium"


def test_mark_approval_needed_sets_approval_and_verifier_gate():
    state = CockpitState(
        run_id="run-123",
        current_step="Waiting at checkpoint",
        active_agent="Conductor",
    )

    updated = state.mark_approval_needed("pack_confirmation")

    assert updated.approval_needed is True
    assert updated.verifier_gate == "pack_confirmation"
    assert state.approval_needed is False
    assert state.verifier_gate is None


def test_from_run_state_builds_status_from_run_id():
    run_state = RunState(
        run_id="run-123",
        phase=Phase.MERIDIAN_SCOPING,
        domain="fmcg",
        created_at="2026-06-05T00:00:00+00:00",
    )

    state = CockpitState.from_run_state(
        run_state,
        current_step="Scoping demand drivers",
        active_agent="Meridian",
    )

    assert state.run_id == "run-123"
    assert state.current_step == "Scoping demand drivers"
    assert state.active_agent == "Meridian"
    assert state.blockers == []


def test_from_run_state_maps_halted_phase_to_halt_reason_blocker():
    run_state = RunState(
        run_id="run-123",
        phase=Phase.HALTED,
        domain="fmcg",
        created_at="2026-06-05T00:00:00+00:00",
        halt_reason="guard budget exceeded",
    )

    state = CockpitState.from_run_state(
        run_state,
        current_step="Paused before modelling",
        active_agent="Conductor",
    )

    assert state.blockers == ["Run halted: guard budget exceeded"]
    assert state.confidence == "low"


def test_cockpit_state_rejects_negative_code_attempt():
    with pytest.raises(ValidationError):
        CockpitState(
            run_id="run-123",
            current_step="Preparing code escalation",
            active_agent="Foundry",
            code_attempt=-1,
        )


def test_cockpit_state_rejects_unknown_escalation_status():
    with pytest.raises(ValidationError):
        CockpitState(
            run_id="run-123",
            current_step="Preparing code escalation",
            active_agent="Foundry",
            code_escalation_status="maybe",
        )
