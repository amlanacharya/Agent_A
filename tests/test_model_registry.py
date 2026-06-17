"""Tests for the MODEL_REGISTRY provenance in
``forecasting.learning_workspace`` (Phase 4.1 CB7).

The provenance is a small append-only audit log for model
selection decisions: which model family was registered, the
MASE that justified the registration, and the Proposal[]
trail that produced the change. The tests pin the format
(pure ``format_provenance_entry``) and the I/O contract
(``record_proposal_provenance`` appends under the heading
without overwriting existing entries).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from forecasting.contracts import (
    Claim,
    Proposal,
    ProposalTarget,
)
from forecasting.learning_workspace import (
    MODEL_REGISTRY_HEADING,
    RunWorkspace,
    create_run_workspace,
    format_provenance_entry,
    record_proposal_provenance,
)
from forecasting.run_state import create_run_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _proposal(
    *,
    config_action: str | None = "enable_promo_indicator",
    code_action: str | None = None,
    series_key: str = "A",
    rationale: str = "promo residuals spiked",
) -> Proposal:
    """Build a minimal Proposal for the registry entry.

    Defaults to a config-kind proposal; tests that exercise
    code-kind set ``code_action`` and clear ``config_action``.
    """
    return Proposal(
        kind="config" if config_action else "code",
        config_action=config_action,
        code_action=code_action,
        target=ProposalTarget(scope="series", series_key=series_key, segment_id=None),
        rationale=rationale,
        evidence=Claim(
            claim_id="c-1",
            claim="synthetic",
            verification_status="SUPPORTED",
            evidence_type="pattern",
            evidence_ref="x",
            applies_to=series_key,
            downstream_impact="test",
            created_at="2026-06-17T00:00:00+00:00",
        ),
    )


def _ts() -> datetime:
    return datetime(2026, 6, 17, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# format_provenance_entry
# ---------------------------------------------------------------------------


def test_format_includes_model_family() -> None:
    """The entry's `### heading` carries the model family name."""
    entry = format_provenance_entry(
        model_family="xgboost_global",
        proposals=[],
        mase=0.85,
        created_at=_ts(),
    )
    assert "### xgboost_global" in entry


def test_format_includes_mase_with_three_decimals() -> None:
    """MASE is formatted with 3-decimal precision so entries are comparable."""
    entry = format_provenance_entry(
        model_family="naive",
        proposals=[],
        mase=1.23456,
        created_at=_ts(),
    )
    assert "MASE=1.235" in entry


def test_format_includes_timestamp() -> None:
    """The entry carries an ISO-8601 timestamp so the audit can sort by time."""
    entry = format_provenance_entry(
        model_family="naive",
        proposals=[],
        mase=1.0,
        created_at=_ts(),
    )
    assert "2026-06-17T12:00:00" in entry


def test_format_handles_empty_proposals_with_placeholder() -> None:
    """An empty proposals list is recorded as 'no proposals; baseline model'."""
    entry = format_provenance_entry(
        model_family="naive",
        proposals=[],
        mase=1.5,
        created_at=_ts(),
    )
    assert "(no proposals; baseline model)" in entry


def test_format_renders_each_proposal_as_one_line() -> None:
    """Each Proposal becomes a bullet line with action + target + rationale."""
    proposals = [
        _proposal(config_action="enable_promo_indicator", rationale="promo spike"),
        _proposal(config_action="enable_stockout_features", series_key="B", rationale="stockout heavy"),
    ]
    entry = format_provenance_entry(
        model_family="xgboost_global",
        proposals=proposals,
        mase=0.85,
        created_at=_ts(),
    )
    # The action is wrapped in backticks for markdown readability.
    assert "`enable_promo_indicator` -> A: promo spike" in entry
    assert "`enable_stockout_features` -> B: stockout heavy" in entry


def test_format_handles_code_kind_proposal() -> None:
    """A code-kind proposal's code_action shows up in the bullet."""
    proposals = [
        _proposal(config_action=None, code_action="new_model_family", rationale="lgbm would help"),
    ]
    entry = format_provenance_entry(
        model_family="new_family",
        proposals=proposals,
        mase=0.7,
        created_at=_ts(),
    )
    assert "`new_model_family` -> A: lgbm would help" in entry


def test_format_handles_segment_target() -> None:
    """A segment-target proposal renders segment_id, not series_key."""
    p = Proposal(
        kind="config",
        config_action="enable_hierarchy_features",
        target=ProposalTarget(scope="segment", series_key=None, segment_id="G1"),
        rationale="segment-wide",
        evidence=Claim(
            claim_id="c-1",
            claim="synthetic",
            verification_status="SUPPORTED",
            evidence_type="pattern",
            evidence_ref="x",
            applies_to="G1",
            downstream_impact="test",
            created_at="2026-06-17T00:00:00+00:00",
        ),
    )
    entry = format_provenance_entry(
        model_family="xgboost_global",
        proposals=[p],
        mase=0.85,
        created_at=_ts(),
    )
    assert "`enable_hierarchy_features` -> G1" in entry


def test_format_is_deterministic() -> None:
    """Same inputs -> same markdown block. The audit must be reproducible."""
    proposals = [_proposal()]
    e1 = format_provenance_entry(
        model_family="naive", proposals=proposals, mase=1.0, created_at=_ts()
    )
    e2 = format_provenance_entry(
        model_family="naive", proposals=proposals, mase=1.0, created_at=_ts()
    )
    assert e1 == e2


# ---------------------------------------------------------------------------
# record_proposal_provenance — file I/O
# ---------------------------------------------------------------------------


def test_record_creates_heading_on_first_call(run_id, tmp_outputs) -> None:
    """The first call to record creates the '## Registered Models' heading."""
    create_run_state(run_id, domain="fmcg")
    workspace = create_run_workspace(run_id)
    record_proposal_provenance(
        workspace,
        model_family="naive",
        proposals=[],
        mase=1.5,
        created_at=_ts(),
    )
    text = workspace.artifacts["MODEL_REGISTRY.md"].read_text()
    assert MODEL_REGISTRY_HEADING in text
    assert "### naive" in text


def test_record_appends_a_second_entry_without_overwriting_first(
    run_id, tmp_outputs
) -> None:
    """A second call appends; both entries are present."""
    create_run_state(run_id, domain="fmcg")
    workspace = create_run_workspace(run_id)
    record_proposal_provenance(
        workspace,
        model_family="naive",
        proposals=[],
        mase=1.5,
        created_at=_ts(),
    )
    later = datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)
    record_proposal_provenance(
        workspace,
        model_family="xgboost_global",
        proposals=[_proposal(rationale="promo residuals")],
        mase=0.85,
        created_at=later,
    )
    text = workspace.artifacts["MODEL_REGISTRY.md"].read_text()
    assert "### naive" in text
    assert "### xgboost_global" in text
    # Both timestamps appear.
    assert "2026-06-17T12:00:00" in text
    assert "2026-06-18T12:00:00" in text


def test_record_rejects_halted_run(run_id, tmp_outputs) -> None:
    """A halted run cannot have its registry written (consistent with promote_learning)."""
    from forecasting.learning_workspace import _assert_mutable_run  # noqa: F401
    from forecasting.run_state import HaltedRunError, save_run_state, Phase

    create_run_state(run_id, domain="fmcg")
    workspace = create_run_workspace(run_id)
    # Halt the run. The state machine requires halt_reason to
    # be set when transitioning to HALTED.
    state = create_run_state(run_id, domain="fmcg")
    state.phase = Phase.HALTED
    state.halt_reason = "test halt"
    save_run_state(state)
    with pytest.raises(HaltedRunError):
        record_proposal_provenance(
            workspace,
            model_family="naive",
            proposals=[],
            mase=1.5,
            created_at=_ts(),
        )