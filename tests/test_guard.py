import pytest
from forecasting.guard import (
    GuardConfig, AgentGuardState, FoundryRunGuard, GuardHalt
)


def test_conductor_tool_limit_enforced():
    config = GuardConfig(duplicate_hard_stop=10_000)
    guard = AgentGuardState(agent_name="conductor")
    for _ in range(config.max_calls_conductor):
        guard.check_and_record("some_tool", {}, tokens_used=0, config=config)
    with pytest.raises(GuardHalt, match="tool call limit"):
        guard.check_and_record("some_tool", {}, tokens_used=0, config=config)


def test_token_budget_enforced():
    config = GuardConfig()
    guard = AgentGuardState(agent_name="meridian")
    with pytest.raises(GuardHalt, match="token budget"):
        guard.check_and_record("tool", {}, tokens_used=config.token_budget + 1, config=config)


def test_duplicate_call_detected():
    # DUPLICATE_CALL_HARD_STOP=2 (default): the first duplicate is allowed with a
    # warning; the run halts on the second duplicate (the third identical call).
    config = GuardConfig()
    guard = AgentGuardState(agent_name="forge")
    args = {"series_key": "SKU_A|NORTH"}
    guard.check_and_record("classify_demand_profiles", args, tokens_used=0, config=config)
    guard.check_and_record("classify_demand_profiles", args, tokens_used=0, config=config)  # 1st dup — warn
    with pytest.raises(GuardHalt, match="duplicate"):
        guard.check_and_record("classify_demand_profiles", args, tokens_used=0, config=config)  # 2nd dup — halt


def test_foundry_cumulative_limit():
    config = GuardConfig(max_calls_foundry=3)
    g = FoundryRunGuard(run_id="r1")
    for _ in range(3):
        g.check_and_record(config)
    with pytest.raises(GuardHalt, match="cumulative"):
        g.check_and_record(config)


def test_foundry_counter_is_per_run():
    # The counter is per-run (instance), not a process global — run B is unaffected
    # by run A's calls, so concurrent runs don't halt each other. (review §8)
    config = GuardConfig(max_calls_foundry=3)
    g_a, g_b = FoundryRunGuard(run_id="A"), FoundryRunGuard(run_id="B")
    for _ in range(3):
        g_a.check_and_record(config)
    g_b.check_and_record(config)
    assert g_b.count == 1


def test_different_args_not_duplicate():
    config = GuardConfig()
    guard = AgentGuardState(agent_name="forge")
    guard.check_and_record("classify_demand_profiles", {"x": 1}, tokens_used=0, config=config)
    guard.check_and_record("classify_demand_profiles", {"x": 2}, tokens_used=0, config=config)
