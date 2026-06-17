"""Marginal-gain stop condition for the two-path escalation loop.

CB4 of Phase 4.1: the "are we fitting noise?" guardrail. A
principal DS stops iterating when the next change does not pay
for itself; this module makes that rule explicit, deterministic,
and `.env`-configurable. The same stop condition is consumed by
the config-escalation loop (CB5) and the code-escalation path
(`model_escalation`), so a single set of thresholds governs the
whole escalation ladder.

Design rules:

* **Pure stop function, separate loader.** ``should_stop`` is a
  pure function: it takes a history and a config, returns a
  bool. No I/O, no globals. ``load_marginal_gain_config`` is the
  one function that reads ``.env``; tests pass the dataclass
  directly and never touch the environment.
* **Symmetric on improvement and regression.** A tiny regression
  is also "no improvement" in the absolute sense — the loop
  must not bounce between near-equal MASE values forever. The
  function compares the *absolute* delta to ``min_mase_delta``.
* **Patience is consecutive.** Two consecutive non-improvements
  stop the loop; one bad attempt in the middle of a good
  streak does not. Cumulative non-improvement is a different
  problem and the harness's per-knob cap handles it.
* **Target hits stop immediately.** ``current_mase <= target_mase``
  is checked first; the patience logic never runs once the
  target is met. The DS doesn't keep tinkering once the model
  is good enough.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    """Read a float from .env, falling back to ``default`` on any error.

    Mirror of the ``_env_int`` helper in ``forecasting.guard`` —
    same shape, same return-on-error contract. Reads ``os.environ``
    directly rather than going through pydantic-settings because
    the rest of the platform does it this way (see ``guard.py``).
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    """Read an int from .env, falling back to ``default`` on any error."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MarginalGainConfig:
    """Thresholds the marginal-gain stop condition uses.

    All fields are `.env`-configurable. The dataclass freezes
    after construction so the config cannot be mutated mid-run
    (a mid-run change to the threshold would make the stop
    condition's history inconsistent).

    Defaults match what a principal DS would use on weekly retail
    SKU data: stop when 2 consecutive attempts improve MASE by
    less than 0.02, or when absolute MASE reaches 1.0 (the
    naive-baseline threshold from the ``MASE Target`` glossary).
    """

    min_mase_delta: float = field(default_factory=lambda: _env_float("FOUNDRY_MIN_MASE_DELTA", 0.02))
    patience: int = field(default_factory=lambda: _env_int("FOUNDRY_MARGINAL_GAIN_PATIENCE", 2))
    target_mase: float = field(default_factory=lambda: _env_float("FOUNDRY_TARGET_MASE", 1.0))


def load_marginal_gain_config() -> MarginalGainConfig:
    """Build a :class:`MarginalGainConfig` from the current environment.

    A separate loader keeps the pure stop function testable
    without touching ``os.environ``. Returns a fresh dataclass
    on each call so the caller can mutate / cache as needed
    (the dataclass itself is frozen, so the returned object is
    immutable).
    """
    return MarginalGainConfig()


def should_stop(
    history: list[float],
    config: MarginalGainConfig,
    *,
    current_mase: float,
) -> bool:
    """Decide whether the loop should stop after the latest attempt.

    Parameters
    ----------
    history
        MASE values per attempt, most recent last. The just-finished
        attempt is the last element (which is also ``current_mase``;
        the caller passes both for clarity at the call site).
        May be empty (no attempts yet).
    config
        The thresholds. Frozen dataclass.
    current_mase
        The MASE of the just-finished attempt. Passed explicitly
        so the caller doesn't have to remember the convention
        that ``history[-1] == current_mase``.

    Returns
    -------
    bool
        True if the loop should stop, False if it should continue.
        The function does not raise; callers compose it with
        their own attempt cap.

    Stop conditions (checked in order):

    1. ``current_mase <= config.target_mase`` — target met.
    2. ``len(history) >= config.patience`` and the last
       ``patience`` MASE deltas are all below
       ``config.min_mase_delta`` in absolute value — we are
       fitting noise.
    """
    if current_mase <= config.target_mase:
        return True
    if len(history) < config.patience:
        return False
    # Compute the last `patience` deltas. history is MASE per
    # attempt; a delta is history[i] - history[i-1] (negative when
    # MASE improved, positive when it regressed). The patience
    # check needs the deltas of consecutive *pairs*; for patience=2
    # that's one delta (between the last two attempts); for
    # patience=3 that's two deltas (last three attempts).
    if config.patience < 2:
        # Edge case: patience < 2 means "stop on the first
        # non-improvement". With patience=0, every call returns
        # True (the deltas check is vacuous); with patience=1,
        # we need at least one delta, which means len(history) >= 2.
        if config.patience <= 0:
            return True
        # patience == 1: stop on the very latest non-improvement
        if len(history) < 2:
            return False
        last_delta = history[-1] - history[-2]
        return abs(last_delta) < config.min_mase_delta
    recent = history[-config.patience:]
    deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
    return all(abs(d) < config.min_mase_delta for d in deltas)


__all__ = (
    "MarginalGainConfig",
    "load_marginal_gain_config",
    "should_stop",
)
