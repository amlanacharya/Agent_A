"""Tests for ``forecasting.marginal_gain`` (Phase 4.1 CB4).

The marginal-gain stop condition is the "are we fitting noise?"
guardrail. It is a small pure function with several subtle
branches (target-met first, patience-only-when-enough-history,
symmetric on improvement/regression, edge-case patience values).
These tests pin each branch.
"""

from __future__ import annotations

import pytest

from forecasting.marginal_gain import (
    MarginalGainConfig,
    load_marginal_gain_config,
    should_stop,
)


# ---------------------------------------------------------------------------
# Target met
# ---------------------------------------------------------------------------


def test_should_stop_when_target_met_exactly() -> None:
    """current_mase == target_mase -> stop (target met)."""
    cfg = MarginalGainConfig(min_mase_delta=0.02, patience=2, target_mase=1.0)
    assert should_stop([1.5, 1.2, 1.0], cfg, current_mase=1.0) is True


def test_should_stop_when_target_met_with_margin() -> None:
    """current_mase strictly below target -> stop."""
    cfg = MarginalGainConfig(min_mase_delta=0.02, patience=2, target_mase=1.0)
    assert should_stop([1.5, 1.2, 0.95], cfg, current_mase=0.95) is True


# ---------------------------------------------------------------------------
# Patience insufficient
# ---------------------------------------------------------------------------


def test_should_not_stop_with_empty_history() -> None:
    """No attempts yet -> never stop on the patience rule."""
    cfg = MarginalGainConfig(min_mase_delta=0.02, patience=2, target_mase=1.0)
    assert should_stop([], cfg, current_mase=1.5) is False


def test_should_not_stop_with_insufficient_history() -> None:
    """patience=2 needs at least 2 attempts; 1 attempt -> no stop."""
    cfg = MarginalGainConfig(min_mase_delta=0.02, patience=2, target_mase=1.0)
    assert should_stop([1.5], cfg, current_mase=1.5) is False


# ---------------------------------------------------------------------------
# Patience met — improvement and regression both count
# ---------------------------------------------------------------------------


def test_should_stop_after_two_consecutive_non_improvements() -> None:
    """Last 2 attempts each improved by less than 0.02 -> stop."""
    cfg = MarginalGainConfig(min_mase_delta=0.02, patience=2, target_mase=1.0)
    # Improvements of 0.005 and 0.001 — both below 0.02 threshold.
    history = [1.500, 1.495, 1.494]
    assert should_stop(history, cfg, current_mase=1.494) is True


def test_should_not_stop_on_one_non_improvement() -> None:
    """patience=2 means 2 consecutive; 1 below-threshold is not enough."""
    cfg = MarginalGainConfig(min_mase_delta=0.02, patience=2, target_mase=1.0)
    # Attempt 1: 1.500 -> 1.495, delta 0.005 (below threshold).
    # Attempt 2: 1.495 -> 1.430, delta 0.065 (above threshold).
    # The LAST delta is large, so the loop continues.
    history = [1.500, 1.495, 1.430]
    assert should_stop(history, cfg, current_mase=1.430) is False


def test_should_not_stop_on_mixed_deltas() -> None:
    """Alternating small improvement and large improvement -> continue."""
    cfg = MarginalGainConfig(min_mase_delta=0.02, patience=2, target_mase=1.0)
    # 1.500 -> 1.495 (small), 1.495 -> 1.400 (large). The patience
    # window is the last 2 attempts (1.495 and 1.400) — deltas
    # 0.005 (small) and 0.095 (large). The large delta rescues
    # the loop, so no stop.
    history = [1.500, 1.495, 1.400]
    assert should_stop(history, cfg, current_mase=1.400) is False


def test_should_stop_on_consecutive_regressions_too() -> None:
    """Two consecutive regressions (MASE got worse) also count as no-improvement."""
    cfg = MarginalGainConfig(min_mase_delta=0.02, patience=2, target_mase=1.0)
    # 1.500 -> 1.505 -> 1.510 — both deltas are +0.005, well
    # below the 0.02 absolute threshold. The absolute-value rule
    # in the implementation catches this; a small regression is
    # the same signal as a small improvement (we are not making
    # meaningful progress).
    history = [1.500, 1.505, 1.510]
    assert should_stop(history, cfg, current_mase=1.510) is True


# ---------------------------------------------------------------------------
# Symmetric on absolute value
# ---------------------------------------------------------------------------


def test_min_mase_delta_is_absolute_value() -> None:
    """A small improvement and a small regression are both 'no improvement'."""
    cfg = MarginalGainConfig(min_mase_delta=0.02, patience=2, target_mase=1.0)
    # Both deltas have abs < 0.02 -> stop, regardless of sign.
    history_small_improvement = [1.500, 1.495, 1.494]
    history_small_regression = [1.500, 1.505, 1.510]
    assert should_stop(history_small_improvement, cfg, current_mase=1.494) is True
    assert should_stop(history_small_regression, cfg, current_mase=1.510) is True


# ---------------------------------------------------------------------------
# Patience edge cases
# ---------------------------------------------------------------------------


def test_patience_one_stops_on_first_non_improvement() -> None:
    """patience=1 means 'stop on the most recent non-improvement'."""
    cfg = MarginalGainConfig(min_mase_delta=0.02, patience=1, target_mase=1.0)
    # Last delta is 0.005, below threshold -> stop.
    assert should_stop([1.500, 1.495], cfg, current_mase=1.495) is True
    # Last delta is 0.05, above threshold -> continue.
    assert should_stop([1.500, 1.450], cfg, current_mase=1.450) is False


def test_patience_zero_always_stops() -> None:
    """patience=0 is a degenerate 'always stop on the patience rule'.

    The function is total — callers should never pass patience=0
    in production. The behaviour here is documented so a future
    misconfiguration surfaces as a test failure, not a silent
    loop.
    """
    cfg = MarginalGainConfig(min_mase_delta=0.02, patience=0, target_mase=1.0)
    assert should_stop([1.500, 1.400], cfg, current_mase=1.400) is True
    assert should_stop([], cfg, current_mase=1.500) is True


def test_patience_three_needs_three_attempts() -> None:
    """patience=3 considers the last 3 attempts (2 deltas)."""
    cfg = MarginalGainConfig(min_mase_delta=0.02, patience=3, target_mase=1.0)
    # 1.500 -> 1.495 (small), 1.495 -> 1.494 (small) — both
    # deltas below 0.02 -> stop.
    history = [1.500, 1.495, 1.494]
    assert should_stop(history, cfg, current_mase=1.494) is True
    # With only 2 attempts, patience=3's deltas window is empty
    # (we need at least patience entries), so the function falls
    # through and returns False.
    assert should_stop([1.500, 1.495], cfg, current_mase=1.495) is False


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


def test_load_marginal_gain_config_uses_defaults_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No .env overrides -> dataclass defaults."""
    for name in ("FOUNDRY_MIN_MASE_DELTA", "FOUNDRY_MARGINAL_GAIN_PATIENCE", "FOUNDRY_TARGET_MASE"):
        monkeypatch.delenv(name, raising=False)
    cfg = load_marginal_gain_config()
    assert cfg.min_mase_delta == 0.02
    assert cfg.patience == 2
    assert cfg.target_mase == 1.0


def test_load_marginal_gain_config_honours_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three .env values are honoured when set."""
    monkeypatch.setenv("FOUNDRY_MIN_MASE_DELTA", "0.05")
    monkeypatch.setenv("FOUNDRY_MARGINAL_GAIN_PATIENCE", "3")
    monkeypatch.setenv("FOUNDRY_TARGET_MASE", "0.8")
    cfg = load_marginal_gain_config()
    assert cfg.min_mase_delta == 0.05
    assert cfg.patience == 3
    assert cfg.target_mase == 0.8


def test_load_marginal_gain_config_partial_env_overrides_only_set_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the .env values present are honoured; missing values use defaults."""
    monkeypatch.setenv("FOUNDRY_MIN_MASE_DELTA", "0.05")
    # patience and target_mase are unset
    monkeypatch.delenv("FOUNDRY_MARGINAL_GAIN_PATIENCE", raising=False)
    monkeypatch.delenv("FOUNDRY_TARGET_MASE", raising=False)
    cfg = load_marginal_gain_config()
    assert cfg.min_mase_delta == 0.05
    assert cfg.patience == 2  # default
    assert cfg.target_mase == 1.0  # default


def test_load_marginal_gain_config_invalid_env_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unparseable .env value falls back to the default."""
    monkeypatch.setenv("FOUNDRY_MIN_MASE_DELTA", "not-a-float")
    cfg = load_marginal_gain_config()
    assert cfg.min_mase_delta == 0.02


# ---------------------------------------------------------------------------
# Config is frozen
# ---------------------------------------------------------------------------


def test_marginal_gain_config_is_frozen() -> None:
    """A mid-run threshold change would invalidate the stop history.

    The dataclass is frozen so the only way to "change" the
    threshold mid-run is to load a new config. This is a
    defensive property — the test pins it.
    """
    cfg = MarginalGainConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.patience = 99  # type: ignore[misc]
