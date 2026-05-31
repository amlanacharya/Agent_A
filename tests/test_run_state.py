import pytest
from forecasting.run_state import (
    HaltedRunError,
    Phase,
    RunNotFoundError,
    RunState,
    create_run_state,
    load_run_state,
    save_run_state,
)


def test_create_run_state(run_id, tmp_outputs):
    state = create_run_state(run_id, domain="fmcg")
    assert state.phase == Phase.PREFLIGHT
    assert state.domain == "fmcg"
    assert state.run_id == run_id
    assert state.pack_confirmed is False


def test_state_persisted_to_disk(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    loaded = load_run_state(run_id)
    assert loaded.run_id == run_id
    assert loaded.domain == "fmcg"


def test_load_missing_run_raises(tmp_outputs):
    with pytest.raises(RunNotFoundError):
        load_run_state("no-such-run")


def test_save_halted_without_reason_raises(run_id, tmp_outputs):
    state = create_run_state(run_id, domain="fmcg")
    state.phase = Phase.HALTED
    with pytest.raises(ValueError, match="halt_reason"):
        save_run_state(state)


def test_save_halted_with_reason_ok(run_id, tmp_outputs):
    state = create_run_state(run_id, domain="fmcg")
    state.halt_reason = "guard budget exceeded"
    state.phase = Phase.HALTED
    save_run_state(state)
    loaded = load_run_state(run_id)
    assert loaded.phase == Phase.HALTED
    assert loaded.halt_reason == "guard budget exceeded"


def test_phase_transitions(run_id, tmp_outputs):
    state = create_run_state(run_id, domain="fmcg")
    state.phase = Phase.MERIDIAN_SCOPING
    save_run_state(state)
    assert load_run_state(run_id).phase == Phase.MERIDIAN_SCOPING
