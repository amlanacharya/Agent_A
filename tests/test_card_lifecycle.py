"""Tests for the card lifecycle in ``forecasting.learning_workspace``
(Phase 4.1 CB6).

The lifecycle is a small pure-function state machine. The tests
pin every transition:

* Pending -> Active: runs_validated >= config.min_runs_to_activate.
* Active -> Retired: consecutive_regressions >= threshold.
* Active -> Expired: now - last_validated_at >= max_age_days.
* Active stays Active: improving run resets consecutive_regressions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from forecasting.contracts import (
    Claim,
    Proposal,
    ProposalTarget,
)
from forecasting.learning_workspace import (
    Card,
    CardLifecycleConfig,
    CardStatus,
    is_card_active,
    load_card_lifecycle_config,
    should_retire_on_regression,
    update_card_after_run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _proposal() -> Proposal:
    """A minimal config Proposal used as the card's content."""
    return Proposal(
        kind="config",
        config_action="enable_promo_indicator",
        target=ProposalTarget(scope="series", series_key="A", segment_id=None),
        rationale="test",
        evidence=Claim(
            claim_id="c-1",
            claim="synthetic",
            verification_status="SUPPORTED",
            evidence_type="pattern",
            evidence_ref="x",
            applies_to="A",
            downstream_impact="test",
            created_at="2026-06-17T00:00:00+00:00",
        ),
    )


def _card(
    *,
    runs_validated: int = 2,
    consecutive_regressions: int = 0,
    status: CardStatus = "active",
    last_validated_at: datetime | None = None,
) -> Card:
    """Build a Card with sensible test defaults."""
    last = last_validated_at or datetime(2026, 6, 17, tzinfo=timezone.utc)
    return Card(
        card_id="c-test",
        proposal=_proposal(),
        created_at=last - timedelta(days=30),
        last_validated_at=last,
        runs_validated=runs_validated,
        consecutive_regressions=consecutive_regressions,
        status=status,
    )


def _now() -> datetime:
    return datetime(2026, 6, 17, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# is_card_active
# ---------------------------------------------------------------------------


def test_card_pending_until_runs_validated_threshold() -> None:
    """runs_validated=1 with default config (threshold=2) -> not active.

    The status field is "active" but the runs_validated gate
    keeps it inactive. The is_card_active check is the strict
    one — a card is active iff all three conditions hold.
    """
    card = _card(runs_validated=1, status="active")
    cfg = CardLifecycleConfig()
    assert is_card_active(card, now=_now(), config=cfg) is False


def test_card_active_at_runs_validated_threshold() -> None:
    """runs_validated=2 with default config -> active."""
    card = _card(runs_validated=2, status="active")
    cfg = CardLifecycleConfig()
    assert is_card_active(card, now=_now(), config=cfg) is True


def test_card_not_active_when_status_pending() -> None:
    """A pending card is not active even with enough runs_validated.

    The status field is the primary gate. A card with
    status=pending has not been promoted yet.
    """
    card = _card(runs_validated=5, status="pending")
    cfg = CardLifecycleConfig()
    assert is_card_active(card, now=_now(), config=cfg) is False


def test_card_not_active_when_retired() -> None:
    card = _card(runs_validated=5, status="retired")
    cfg = CardLifecycleConfig()
    assert is_card_active(card, now=_now(), config=cfg) is False


def test_card_not_active_when_expired() -> None:
    card = _card(runs_validated=5, status="expired")
    cfg = CardLifecycleConfig()
    assert is_card_active(card, now=_now(), config=cfg) is False


def test_card_not_active_when_expired_by_age() -> None:
    """A card with status=active but aged past max_age_days is not active."""
    old = datetime(2026, 1, 1, tzinfo=timezone.utc)  # well past 90 days
    card = _card(runs_validated=5, status="active", last_validated_at=old)
    cfg = CardLifecycleConfig(max_age_days=90)
    now = datetime(2026, 6, 17, tzinfo=timezone.utc)
    assert is_card_active(card, now=now, config=cfg) is False


# ---------------------------------------------------------------------------
# should_retire_on_regression
# ---------------------------------------------------------------------------


def test_should_retire_when_consecutive_regressions_reach_threshold() -> None:
    """consecutive_regressions == threshold -> retire."""
    card = _card(consecutive_regressions=2)
    cfg = CardLifecycleConfig(consecutive_regressions_to_retire=2)
    assert should_retire_on_regression(card, config=cfg) is True


def test_should_not_retire_on_single_regression() -> None:
    """consecutive_regressions < threshold -> not retire."""
    card = _card(consecutive_regressions=1)
    cfg = CardLifecycleConfig(consecutive_regressions_to_retire=2)
    assert should_retire_on_regression(card, config=cfg) is False


# ---------------------------------------------------------------------------
# update_card_after_run — improving run
# ---------------------------------------------------------------------------


def test_update_card_after_improving_run_resets_regressions() -> None:
    """A single improving run resets consecutive_regressions to 0."""
    card = _card(runs_validated=2, consecutive_regressions=1, status="active")
    cfg = CardLifecycleConfig()
    updated = update_card_after_run(
        card, mase_improved=True, now=_now(), config=cfg
    )
    assert updated.consecutive_regressions == 0
    assert updated.runs_validated == 3
    assert updated.last_validated_at == _now()
    assert updated.status == "active"


def test_update_card_pending_becomes_active_at_threshold() -> None:
    """A pending card with runs_validated=1 -> active after the 2nd run."""
    card = _card(runs_validated=1, status="pending")
    cfg = CardLifecycleConfig()
    updated = update_card_after_run(
        card, mase_improved=True, now=_now(), config=cfg
    )
    # The 2nd run pushes runs_validated to 2, status flips to active.
    assert updated.runs_validated == 2
    assert updated.status == "active"


# ---------------------------------------------------------------------------
# update_card_after_run — regression
# ---------------------------------------------------------------------------


def test_update_card_after_regression_increments_counter() -> None:
    """A single regression bumps consecutive_regressions by 1, status unchanged."""
    card = _card(runs_validated=2, consecutive_regressions=0, status="active")
    cfg = CardLifecycleConfig(consecutive_regressions_to_retire=2)
    updated = update_card_after_run(
        card, mase_improved=False, now=_now(), config=cfg
    )
    assert updated.consecutive_regressions == 1
    assert updated.runs_validated == 2  # unchanged
    assert updated.status == "active"  # not yet retired


def test_update_card_retires_on_second_consecutive_regression() -> None:
    """Two consecutive regressions trigger retirement."""
    card = _card(runs_validated=5, consecutive_regressions=0, status="active")
    cfg = CardLifecycleConfig(consecutive_regressions_to_retire=2)
    # First regression: counter -> 1, status still active.
    after_first = update_card_after_run(
        card, mase_improved=False, now=_now(), config=cfg
    )
    assert after_first.consecutive_regressions == 1
    assert after_first.status == "active"
    # Second regression: counter -> 2, status flips to retired.
    after_second = update_card_after_run(
        after_first, mase_improved=False, now=_now(), config=cfg
    )
    assert after_second.consecutive_regressions == 2
    assert after_second.status == "retired"


# ---------------------------------------------------------------------------
# update_card_after_run — interleaved improvements and regressions
# ---------------------------------------------------------------------------


def test_improvement_after_regression_resets_counter() -> None:
    """An improving run after a regression resets the counter to 0."""
    card = _card(runs_validated=2, consecutive_regressions=1, status="active")
    cfg = CardLifecycleConfig(consecutive_regressions_to_retire=2)
    # An improvement resets the counter, preventing retirement
    # on the next regression.
    updated = update_card_after_run(
        card, mase_improved=True, now=_now(), config=cfg
    )
    assert updated.consecutive_regressions == 0
    assert updated.status == "active"


# ---------------------------------------------------------------------------
# update_card_after_run — age check
# ---------------------------------------------------------------------------


def test_card_expires_when_old_at_regression_run() -> None:
    """A card older than max_age_days is expired at the next non-improvement."""
    old = datetime(2026, 1, 1, tzinfo=timezone.utc)
    card = _card(
        runs_validated=2,
        consecutive_regressions=0,
        status="active",
        last_validated_at=old,
    )
    cfg = CardLifecycleConfig(max_age_days=90)
    now = datetime(2026, 6, 17, tzinfo=timezone.utc)
    updated = update_card_after_run(
        card, mase_improved=False, now=now, config=cfg
    )
    # The card is past max_age_days AND a regression happened
    # in the same run -> expired.
    assert updated.status == "expired"


def test_card_does_not_expire_on_improving_run() -> None:
    """An improving run refreshes last_validated_at, so the card stays active."""
    old = datetime(2026, 1, 1, tzinfo=timezone.utc)
    card = _card(
        runs_validated=2,
        consecutive_regressions=0,
        status="active",
        last_validated_at=old,
    )
    cfg = CardLifecycleConfig(max_age_days=90)
    now = datetime(2026, 6, 17, tzinfo=timezone.utc)
    updated = update_card_after_run(
        card, mase_improved=True, now=now, config=cfg
    )
    # last_validated_at is bumped to `now`, so the age check
    # is a no-op. Status stays active.
    assert updated.last_validated_at == now
    assert updated.status == "active"


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


def test_load_card_lifecycle_config_defaults_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No .env overrides -> dataclass defaults."""
    for name in (
        "LEARNING_MIN_RUNS_TO_ACTIVATE",
        "LEARNING_CARD_MAX_AGE_DAYS",
        "LEARNING_REGRESSIONS_TO_RETIRE",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = load_card_lifecycle_config()
    assert cfg.min_runs_to_activate == 2
    assert cfg.max_age_days == 90
    assert cfg.consecutive_regressions_to_retire == 2


def test_load_card_lifecycle_config_honours_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three .env values are honoured when set."""
    monkeypatch.setenv("LEARNING_MIN_RUNS_TO_ACTIVATE", "3")
    monkeypatch.setenv("LEARNING_CARD_MAX_AGE_DAYS", "30")
    monkeypatch.setenv("LEARNING_REGRESSIONS_TO_RETIRE", "5")
    cfg = load_card_lifecycle_config()
    assert cfg.min_runs_to_activate == 3
    assert cfg.max_age_days == 30
    assert cfg.consecutive_regressions_to_retire == 5


def test_load_card_lifecycle_config_invalid_env_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unparseable .env value falls back to the default."""
    monkeypatch.setenv("LEARNING_MIN_RUNS_TO_ACTIVATE", "not-an-int")
    cfg = load_card_lifecycle_config()
    assert cfg.min_runs_to_activate == 2


# ---------------------------------------------------------------------------
# Config is frozen
# ---------------------------------------------------------------------------


def test_card_lifecycle_config_is_frozen() -> None:
    """Defensive: a mid-run threshold change would invalidate card history."""
    cfg = CardLifecycleConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.min_runs_to_activate = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Card is frozen
# ---------------------------------------------------------------------------


def test_card_is_frozen() -> None:
    """A card's history must be immutable after creation."""
    card = _card()
    with pytest.raises((AttributeError, TypeError)):
        card.runs_validated = 99  # type: ignore[misc]