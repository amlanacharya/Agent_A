from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, Field

from forecasting.contracts import Proposal
from forecasting.run_state import HaltedRunError, Phase, legal_next_phases, load_run_state, run_dir

MemoryLayer = Literal["global", "customer", "project"]
LearningTier = Literal["auto", "verifier", "human"]
LearningStatus = Literal["proposed", "approved", "rejected"]

REQUIRED_ARTIFACTS: dict[str, str] = {
    "CONTEXT.md": "# Run Context\n",
    "DATA_CONTRACT.md": "# Data Contract\n",
    "LEARNINGS.md": "# Learnings\n",
    "ASSUMPTIONS.md": "# Assumptions\n",
    "DECISIONS.md": "# Decisions\n",
    "RUNBOOK.md": "# Runbook\n",
    "MODEL_REGISTRY.md": "# Model Registry\n",
    "PROMOTION_DECISIONS.md": "# Promotion Decisions\n",
}

TIER_HEADINGS: dict[LearningTier, str] = {
    "auto": "Auto-Promoted Learnings",
    "verifier": "Verifier-Promoted Learnings",
    "human": "Human-Approved Learnings",
}


class MemoryLayerError(ValueError):
    pass


class LearningPromotionError(ValueError):
    pass


class LearningEntry(BaseModel):
    tier: LearningTier
    category: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    status: LearningStatus


@dataclass(frozen=True)
class RunWorkspace:
    run_id: str
    path: Path
    artifacts: dict[str, Path]


def validate_memory_layer(layer: str) -> MemoryLayer:
    if layer not in set(get_args(MemoryLayer)):
        raise MemoryLayerError(f"Unknown memory layer: {layer}")
    return layer  # type: ignore[return-value]


def create_run_workspace(run_id: str) -> RunWorkspace:
    _assert_mutable_run(run_id)
    workspace_path = run_dir(run_id) / "workspace"
    workspace_path.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}
    for name, initial_content in REQUIRED_ARTIFACTS.items():
        path = workspace_path / name
        if not path.exists():
            path.write_text(initial_content)
        artifacts[name] = path

    return RunWorkspace(run_id=run_id, path=workspace_path, artifacts=artifacts)


def promote_learning(workspace: RunWorkspace, entry: LearningEntry) -> None:
    _assert_mutable_run(workspace.run_id)
    if entry.status != "approved":
        raise LearningPromotionError("Only approved learning entries can be promoted")

    learnings_path = workspace.artifacts["LEARNINGS.md"]
    heading = TIER_HEADINGS[entry.tier]
    text = learnings_path.read_text()
    learning_block = (
        f"- **{entry.category}**: {entry.statement}\n"
        f"  Evidence: {entry.evidence}\n"
    )
    learnings_path.write_text(_append_under_heading(text, heading, learning_block))


def _assert_mutable_run(run_id: str) -> None:
    """Reject writes to a HALTED Run.

    The "is this Run mutable?" question is now a one-liner over the
    state machine: a Run is mutable iff it has at least one legal
    successor phase. HALTED has none, so this is the same check the
    HALTED guard in ``save_run_state`` performs, just expressed
    against the same data structure the conductor uses.
    """
    state = load_run_state(run_id)
    if not legal_next_phases(state.phase):
        raise HaltedRunError(run_id)


def _append_under_heading(text: str, heading: str, block: str) -> str:
    marker = f"## {heading}"
    if marker not in text:
        return text.rstrip() + f"\n\n{marker}\n\n{block}"

    start = text.index(marker)
    next_heading = text.find("\n## ", start + len(marker))
    insertion_point = len(text) if next_heading == -1 else next_heading
    before = text[:insertion_point].rstrip()
    after = text[insertion_point:].lstrip("\n")

    updated = before + "\n\n" + block
    if after:
        updated += "\n" + after
    return updated


# ---------------------------------------------------------------------------
# Card lifecycle (Phase 4.1 CB6)
# ---------------------------------------------------------------------------
# A "card" is the durable artifact the proposal tool (CB3) emits
# when a Proposal is verified. The card is the bridge between
# the in-run judgement (CB3 + CB5) and the cross-run learning
# (next Run reads the cards and applies them). The lifecycle
# state machine is the governance: cards are pending until they
# have been validated enough times to trust, become active when
# they reach the threshold, retire when they stop helping, and
# expire when the world has moved on.
#
# Design rules:
#
# * **Pure functions, separate loader.** ``is_card_active``,
#   ``should_retire_on_regression``, and ``update_card_after_run``
#   are all pure functions of (card, now, config). ``load_card_lifecycle_config``
#   is the one .env reader. Tests pass the dataclass directly.
# * **Retired cards stay in LEARNINGS.md.** The status flag
#   changes; the file is not deleted. The plan's audit trail
#   requires the card's history to be visible even after
#   retirement.
# * **Activation is gated on runs_validated >= 2.** A card with
#   one successful run is still pending. The second run that
#   successfully validates the card promotes it to active.
# * **Regressions are consecutive, not cumulative.** A single
#   bad run in the middle of a good streak does not retire
#   the card. Same vocabulary as the marginal-gain patience
#   rule (CB4).
# * **Age-out is from last_validated_at, not created_at.**
#   A card that was last validated yesterday is one day old,
#   regardless of when it was created. The "the card is alive
#   if it keeps working" rule.

CardStatus = Literal["pending", "active", "retired", "expired"]


def _env_int(name: str, default: int) -> int:
    """Read an int from .env, falling back to ``default`` on any error.

    Mirror of the helpers in ``forecasting.guard`` and
    ``forecasting.marginal_gain`` — same return-on-error contract.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class CardLifecycleConfig:
    """Card-lifecycle thresholds. All fields are .env-configurable.

    The dataclass is frozen so a mid-run threshold change cannot
    silently re-evaluate a card (a card's history is timestamped
    against a single config snapshot).
    """

    min_runs_to_activate: int = field(
        default_factory=lambda: _env_int("LEARNING_MIN_RUNS_TO_ACTIVATE", 2)
    )
    max_age_days: int = field(
        default_factory=lambda: _env_int("LEARNING_CARD_MAX_AGE_DAYS", 90)
    )
    consecutive_regressions_to_retire: int = field(
        default_factory=lambda: _env_int("LEARNING_REGRESSIONS_TO_RETIRE", 2)
    )


def load_card_lifecycle_config() -> CardLifecycleConfig:
    """Build a :class:`CardLifecycleConfig` from the current environment.

    Returns a fresh dataclass on each call so the caller can
    cache or mutate as needed (the dataclass itself is frozen).
    """
    return CardLifecycleConfig()


@dataclass(frozen=True)
class Card:
    """A learning card — a verified Proposal + its application history.

    The card is the durable cross-run artifact. Its markdown
    form lives in ``LEARNINGS.md`` (existing
    :func:`promote_learning` handles the append). The dataclass
    is the in-memory surface the lifecycle functions consume.
    """

    card_id: str
    proposal: Proposal
    created_at: datetime
    last_validated_at: datetime
    runs_validated: int
    consecutive_regressions: int
    status: CardStatus


def is_card_active(
    card: Card,
    *,
    now: datetime,
    config: CardLifecycleConfig,
) -> bool:
    """Decide whether a card is currently active.

    A card is active iff all of:

    1. ``status == "active"`` (not "retired" or "expired")
    2. ``runs_validated >= config.min_runs_to_activate`` (has been
       validated enough times to trust)
    3. The card is not older than ``config.max_age_days`` from
       ``last_validated_at`` (the world hasn't moved on).
    """
    if card.status != "active":
        return False
    if card.runs_validated < config.min_runs_to_activate:
        return False
    age = now - card.last_validated_at
    if age >= timedelta(days=config.max_age_days):
        return False
    return True


def should_retire_on_regression(
    card: Card,
    *,
    config: CardLifecycleConfig,
) -> bool:
    """A card should be retired when consecutive_regressions reaches
    ``config.consecutive_regressions_to_retire``.

    A single regression in the middle of a good streak does
    *not* retire the card — the rule is consecutive, mirroring
    the marginal-gain patience rule.
    """
    return card.consecutive_regressions >= config.consecutive_regressions_to_retire


def update_card_after_run(
    card: Card,
    *,
    mase_improved: bool,
    now: datetime,
    config: CardLifecycleConfig,
) -> Card:
    """Apply one run's outcome to the card. Returns the updated card.

    Behaviour:

    * ``mase_improved`` True: reset ``consecutive_regressions`` to
      0, bump ``runs_validated``, update ``last_validated_at``,
      and set ``status`` to ``"active"`` (the card has been
      validated at least once after this run). A pending card
      becomes active when ``runs_validated`` reaches the
      activation threshold.
    * ``mase_improved`` False: bump ``consecutive_regressions``;
      if the threshold is hit, set ``status`` to ``"retired"``.
    * If ``now - last_validated_at > max_age_days``: set
      ``status`` to ``"expired"`` (the world has moved on since
      the last successful run).

    The age check fires *after* the mase_improved update so a
    card that was last validated long ago but is now improving
    is still considered fresh (last_validated_at is bumped).
    """
    if mase_improved:
        new_consecutive = 0
        new_runs = card.runs_validated + 1
        new_last_validated = now
        new_status: CardStatus = "active"
    else:
        new_consecutive = card.consecutive_regressions + 1
        new_runs = card.runs_validated
        new_last_validated = card.last_validated_at
        # Construct the post-regression card to test retirement
        # against the threshold. The status of this synthetic
        # card doesn't matter for the threshold check — only
        # consecutive_regressions does.
        candidate = Card(
            card_id=card.card_id,
            proposal=card.proposal,
            created_at=card.created_at,
            last_validated_at=card.last_validated_at,
            runs_validated=card.runs_validated,
            consecutive_regressions=new_consecutive,
            status=card.status,
        )
        if should_retire_on_regression(candidate, config=config):
            new_status = "retired"
        else:
            new_status = card.status

    # Age check: if last_validated_at is older than max_age_days,
    # expire the card. Only meaningful when the card was not
    # just updated (a fresh card's last_validated_at == now,
    # so the check is a no-op for improving runs).
    if not mase_improved:
        age = now - new_last_validated
        if age >= timedelta(days=config.max_age_days):
            new_status = "expired"

    return Card(
        card_id=card.card_id,
        proposal=card.proposal,
        created_at=card.created_at,
        last_validated_at=new_last_validated,
        runs_validated=new_runs,
        consecutive_regressions=new_consecutive,
        status=new_status,
    )


__all__ = (
    "MemoryLayer",
    "LearningTier",
    "LearningStatus",
    "LearningEntry",
    "RunWorkspace",
    "LearningPromotionError",
    "REQUIRED_ARTIFACTS",
    "create_run_workspace",
    "promote_learning",
    "validate_memory_layer",
    # Card lifecycle (CB6)
    "Card",
    "CardStatus",
    "CardLifecycleConfig",
    "is_card_active",
    "load_card_lifecycle_config",
    "should_retire_on_regression",
    "update_card_after_run",
)