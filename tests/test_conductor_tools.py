import pytest

from forecasting.run_state import HaltedRunError, Phase, create_run_state, load_run_state
from forecasting.tools.conductor_tools import (
    ConditionViolationError,
    get_run_state,
    log_halt,
    update_run_state,
)


def test_get_run_state_returns_dict(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    state = get_run_state(run_id)
    assert state["run_id"] == run_id
    assert state["phase"] == "preflight"


def test_update_run_state_persists(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    updated = update_run_state(run_id, {"meridian_turn_count": 3})
    assert updated["meridian_turn_count"] == 3
    assert load_run_state(run_id).meridian_turn_count == 3


def test_update_rejects_pack_confirmed_false(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    update_run_state(run_id, {"pack_confirmed": True})
    with pytest.raises(ValueError, match="pack_confirmed"):
        update_run_state(run_id, {"pack_confirmed": False})


def test_update_halted_run_raises(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    log_halt(run_id, "test halt", lambda *a, **k: None)
    with pytest.raises(HaltedRunError):
        update_run_state(run_id, {"meridian_turn_count": 1})


def test_log_halt_sets_phase(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    emitted = []
    log_halt(run_id, "budget exceeded", lambda evt, payload: emitted.append((evt, payload)))
    state = load_run_state(run_id)
    assert state.phase == Phase.HALTED
    assert state.halt_reason == "budget exceeded"
    assert any(event == "error" for event, _ in emitted)
