from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


class GuardHalt(Exception):
    pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class GuardConfig:
    """All limits .env-configurable; values here are defaults only."""

    token_budget: int = field(default_factory=lambda: _env_int("GUARD_TOKEN_BUDGET", 80_000))
    max_calls_conductor: int = field(default_factory=lambda: _env_int("GUARD_MAX_CALLS_CONDUCTOR", 20))
    max_calls_meridian: int = field(default_factory=lambda: _env_int("GUARD_MAX_CALLS_MERIDIAN", 20))
    max_calls_forge: int = field(default_factory=lambda: _env_int("GUARD_MAX_CALLS_FORGE", 20))
    max_calls_prism: int = field(default_factory=lambda: _env_int("GUARD_MAX_CALLS_PRISM", 20))
    max_calls_foundry: int = field(default_factory=lambda: _env_int("GUARD_MAX_CALLS_FOUNDRY", 500))
    duplicate_hard_stop: int = field(default_factory=lambda: _env_int("GUARD_DUPLICATE_HARD_STOP", 2))


_AGENT_LIMITS = {
    "conductor": "max_calls_conductor",
    "meridian": "max_calls_meridian",
    "forge": "max_calls_forge",
    "prism": "max_calls_prism",
    "foundry": "max_calls_foundry",
}


def _call_hash(tool_name: str, args: dict[str, Any]) -> str:
    payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


@dataclass
class AgentGuardState:
    agent_name: str
    _call_count: int = field(default=0, init=False)
    _hash_counts: dict[str, int] = field(default_factory=dict, init=False)

    def check_and_record(
        self,
        tool_name: str,
        args: dict[str, Any],
        tokens_used: int,
        config: GuardConfig,
    ) -> None:
        if tokens_used >= config.token_budget:
            raise GuardHalt(f"token budget exceeded: {tokens_used} >= {config.token_budget}")

        limit_attr = _AGENT_LIMITS.get(self.agent_name)
        if limit_attr:
            limit = getattr(config, limit_attr)
            if self._call_count >= limit:
                raise GuardHalt(
                    f"{self.agent_name} tool call limit reached: {self._call_count} >= {limit}"
                )

        h = _call_hash(tool_name, args)
        prior = self._hash_counts.get(h, 0)
        if prior >= config.duplicate_hard_stop:
            raise GuardHalt(f"duplicate tool call hard stop ({prior + 1}x): {tool_name}({args})")
        if prior >= 1:
            log.warning("duplicate tool call (%dx): %s(%s)", prior + 1, tool_name, args)
        self._hash_counts[h] = prior + 1

        self._call_count += 1


@dataclass
class FoundryRunGuard:
    """Per-run cumulative Foundry call counter (instance state, not process global)."""

    run_id: str
    count: int = field(default=0, init=False)

    def check_and_record(self, config: GuardConfig) -> None:
        if self.count >= config.max_calls_foundry:
            raise GuardHalt(
                f"Foundry cumulative limit reached for run {self.run_id}: "
                f"{self.count} >= {config.max_calls_foundry}"
            )
        self.count += 1
