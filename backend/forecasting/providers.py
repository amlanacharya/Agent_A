from __future__ import annotations

from typing import Any, Callable

from forecasting.guard import AgentGuardState, FoundryRunGuard, GuardConfig

_TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}

_default_config = GuardConfig()


def dispatch_tool(
    tool_name: str,
    args: dict[str, Any],
    guard: AgentGuardState,
    tokens_used: int,
    foundry_guard: FoundryRunGuard | None = None,
    config: GuardConfig | None = None,
) -> Any:
    cfg = config or _default_config
    guard.check_and_record(tool_name, args, tokens_used, cfg)
    if foundry_guard is not None:
        foundry_guard.check_and_record(cfg)
    fn = _TOOL_REGISTRY.get(tool_name)
    if fn is None:
        raise KeyError(f"Unknown tool: '{tool_name}'. Is it registered in _TOOL_REGISTRY?")
    return fn(**args)
