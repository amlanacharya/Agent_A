"""Tests for the PROMOTION_DECISIONS.md generator (Phase 5.2 CB4).

CB4 is the last sub-checkbox of Phase 5.2. The format is a
pure function; the I/O is a thin wrapper that reuses the
workspace ``_append_under_heading`` helper from Phase 4.1.

Tests use the ``run_id`` / ``tmp_outputs`` fixtures from
``tests/conftest.py`` (the same pattern the Phase 4.1
``test_model_registry.py`` uses).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from forecasting.contracts import ModelScorecard
from forecasting.learning_workspace import (
    RunWorkspace,
    create_run_workspace,
)
from forecasting.promotion import (
    Champion,
    PromotionCandidate,
    PromotionComparison,
    ShadowModeResult,
    format_promotion_decision,
    write_promotion_decision,
)
from forecasting.run_state import create_run_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _card(
    series_key: str,
    fold_cutoff: str,
    forecast: list[float],
    actual: list[float] | None = None,
) -> ModelScorecard:
    """Build a ModelScorecard for the comparison fixture."""
    if actual is None:
        actual = list(forecast)
    f = np.asarray(forecast, dtype=float)
    a = np.asarray(actual, dtype=float)
    residuals = a - f
    mae = float(np.mean(np.abs(residuals))) if len(f) else 0.0
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if len(f) else 0.0
    return ModelScorecard(
        model_family="xgboost_global",
        series_key=series_key,
        fold_cutoff=fold_cutoff,
        horizon=len(forecast),
        forecast=forecast,
        actual=actual,
        mae=mae,
        rmse=rmse,
        mase=mae,
        bias=float(np.mean(residuals)),
    )


def _ts() -> datetime:
    return datetime(2026, 6, 17, 12, 0, 0, tzinfo=timezone.utc)


def _candidate(scorecards: list[ModelScorecard], reason: str = "from CB3 propose_feature_changes") -> PromotionCandidate:
    return PromotionCandidate(
        run_id="r-candidate",
        model_family="xgboost_global",
        scorecards=scorecards,
        reason=reason,
    )


def _champion(scorecards: list[ModelScorecard]) -> Champion:
    return Champion(
        model_family="naive",
        scorecards=scorecards,
        promoted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _comparison(**overrides) -> PromotionComparison:
    """Build a PromotionComparison with sensible defaults."""
    defaults = dict(
        candidate_wape=0.05,
        champion_wape=0.08,
        wape_delta=-0.03,
        segments_compared=["G1", "G2"],
        segments_improved=["G1"],
        segments_regressed=[],
        promotion_outcome="promote",
    )
    defaults.update(overrides)
    return PromotionComparison(**defaults)


def _shadow(agreement_rate: float = 0.95, tolerance: float = 0.05) -> ShadowModeResult:
    return ShadowModeResult(
        candidate_family="xgboost_global",
        champion_family="naive",
        tolerance=tolerance,
        per_series_pairs={"A": [(10.0, 10.1), (20.0, 20.2)]},
        agreement_rate=agreement_rate,
    )


# ---------------------------------------------------------------------------
# format_promotion_decision
# ---------------------------------------------------------------------------


def test_format_includes_candidate_and_champion_family() -> None:
    candidate = _candidate([_card("A", "2026-03-01T00:00:00", [10.0, 20.0])])
    champion = _champion([_card("A", "2026-03-01T00:00:00", [11.0, 19.0])])
    block = format_promotion_decision(
        candidate=candidate,
        champion=champion,
        comparison=_comparison(),
        shadow=_shadow(),
        decided_at=_ts(),
    )
    assert "xgboost_global vs naive" in block


def test_format_includes_wape_delta_with_sign() -> None:
    """WAPE delta is formatted with a leading + or - for the sign."""
    candidate = _candidate([_card("A", "2026-03-01T00:00:00", [10.0])])
    champion = _champion([_card("A", "2026-03-01T00:00:00", [10.0])])
    block = format_promotion_decision(
        candidate=candidate,
        champion=champion,
        comparison=_comparison(wape_delta=-0.03),
        shadow=_shadow(),
        decided_at=_ts(),
    )
    assert "delta=-0.030" in block


def test_format_includes_per_segment_lists() -> None:
    candidate = _candidate([_card("A", "2026-03-01T00:00:00", [10.0])])
    champion = _champion([_card("A", "2026-03-01T00:00:00", [10.0])])
    block = format_promotion_decision(
        candidate=candidate,
        champion=champion,
        comparison=_comparison(
            segments_compared=["G1", "G2"],
            segments_improved=["G1", "G2"],
            segments_regressed=["G3"],
        ),
        shadow=_shadow(),
        decided_at=_ts(),
    )
    assert "G1, G2" in block  # compared + improved
    assert "G3" in block  # regressed


def test_format_includes_shadow_agreement_rate() -> None:
    """Shadow-mode agreement rate is formatted as a percentage."""
    candidate = _candidate([_card("A", "2026-03-01T00:00:00", [10.0])])
    champion = _champion([_card("A", "2026-03-01T00:00:00", [10.0])])
    block = format_promotion_decision(
        candidate=candidate,
        champion=champion,
        comparison=_comparison(),
        shadow=_shadow(agreement_rate=0.95, tolerance=0.05),
        decided_at=_ts(),
    )
    assert "Shadow-mode agreement: 95.0%" in block
    assert "tolerance=5.00%" in block


def test_format_includes_promotion_outcome() -> None:
    candidate = _candidate([_card("A", "2026-03-01T00:00:00", [10.0])])
    champion = _champion([_card("A", "2026-03-01T00:00:00", [10.0])])
    block = format_promotion_decision(
        candidate=candidate,
        champion=champion,
        comparison=_comparison(promotion_outcome="leakage_failed"),
        shadow=_shadow(),
        decided_at=_ts(),
    )
    assert "promotion_outcome=leakage_failed" in block


def test_format_includes_timestamp() -> None:
    candidate = _candidate([_card("A", "2026-03-01T00:00:00", [10.0])])
    champion = _champion([_card("A", "2026-03-01T00:00:00", [10.0])])
    block = format_promotion_decision(
        candidate=candidate,
        champion=champion,
        comparison=_comparison(),
        shadow=_shadow(),
        decided_at=_ts(),
    )
    assert "2026-06-17T12:00:00" in block


def test_format_includes_candidate_reason() -> None:
    """The candidate's reason shows up in the entry for the audit log."""
    candidate = _candidate(
        [_card("A", "2026-03-01T00:00:00", [10.0])],
        reason="from CB3 propose_feature_changes (enable_promo_indicator)",
    )
    champion = _champion([_card("A", "2026-03-01T00:00:00", [10.0])])
    block = format_promotion_decision(
        candidate=candidate,
        champion=champion,
        comparison=_comparison(),
        shadow=_shadow(),
        decided_at=_ts(),
    )
    assert "from CB3 propose_feature_changes" in block


def test_format_is_deterministic() -> None:
    """Same inputs -> same block. Audit must be reproducible."""
    candidate = _candidate([_card("A", "2026-03-01T00:00:00", [10.0])])
    champion = _champion([_card("A", "2026-03-01T00:00:00", [10.0])])
    args = dict(
        candidate=candidate,
        champion=champion,
        comparison=_comparison(),
        shadow=_shadow(),
        decided_at=_ts(),
    )
    assert format_promotion_decision(**args) == format_promotion_decision(**args)


def test_format_handles_empty_segments_with_placeholder() -> None:
    """An empty segments_compared list renders as the literal string '(none)'."""
    candidate = _candidate([_card("A", "2026-03-01T00:00:00", [10.0])])
    champion = _champion([_card("A", "2026-03-01T00:00:00", [10.0])])
    block = format_promotion_decision(
        candidate=candidate,
        champion=champion,
        comparison=_comparison(segments_compared=[], segments_improved=[], segments_regressed=[]),
        shadow=_shadow(),
        decided_at=_ts(),
    )
    # The "(none)" placeholder appears three times (compared, improved, regressed).
    assert block.count("(none)") == 3


# ---------------------------------------------------------------------------
# write_promotion_decision — file I/O
# ---------------------------------------------------------------------------


def test_write_creates_heading_on_first_call(run_id, tmp_outputs) -> None:
    """The first call to write creates the '## Promotion Decisions' heading."""
    create_run_state(run_id, domain="fmcg")
    workspace = create_run_workspace(run_id)
    candidate = _candidate([_card("A", "2026-03-01T00:00:00", [10.0])])
    champion = _champion([_card("A", "2026-03-01T00:00:00", [10.0])])
    write_promotion_decision(
        workspace,
        candidate=candidate,
        champion=champion,
        comparison=_comparison(),
        shadow=_shadow(),
        decided_at=_ts(),
    )
    text = workspace.artifacts["PROMOTION_DECISIONS.md"].read_text()
    assert "Promotion Decisions" in text
    assert "xgboost_global vs naive" in text


def test_write_appends_second_entry_without_overwriting_first(run_id, tmp_outputs) -> None:
    """A second call appends; both entries are present in the file."""
    create_run_state(run_id, domain="fmcg")
    workspace = create_run_workspace(run_id)
    cutoff_a = "2026-03-01T00:00:00"
    cutoff_b = "2025-12-01T00:00:00"
    # First entry
    write_promotion_decision(
        workspace,
        candidate=_candidate([_card("A", cutoff_a, [10.0, 20.0])]),
        champion=_champion([_card("A", cutoff_a, [11.0, 19.0])]),
        comparison=_comparison(promotion_outcome="promote"),
        shadow=_shadow(agreement_rate=0.95),
        decided_at=_ts(),
    )
    # Second entry with a different timestamp.
    later = datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)
    write_promotion_decision(
        workspace,
        candidate=_candidate(
            [_card("B", cutoff_b, [50.0, 60.0])],
            reason="second decision",
        ),
        champion=_champion([_card("B", cutoff_b, [55.0, 65.0])]),
        comparison=_comparison(promotion_outcome="reject"),
        shadow=_shadow(agreement_rate=0.80),
        decided_at=later,
    )
    text = workspace.artifacts["PROMOTION_DECISIONS.md"].read_text()
    # Both timestamps appear.
    assert "2026-06-17T12:00:00" in text
    assert "2026-06-18T12:00:00" in text
    # Both outcomes appear.
    assert "promotion_outcome=promote" in text
    assert "promotion_outcome=reject" in text


def test_write_rejects_halted_run(run_id, tmp_outputs) -> None:
    """A halted run cannot have its decision log written (consistent with other paths)."""
    from forecasting.run_state import HaltedRunError, save_run_state, Phase

    create_run_state(run_id, domain="fmcg")
    workspace = create_run_workspace(run_id)
    # Halt the run.
    state = create_run_state(run_id, domain="fmcg")
    state.phase = Phase.HALTED
    state.halt_reason = "test halt"
    save_run_state(state)
    candidate = _candidate([_card("A", "2026-03-01T00:00:00", [10.0])])
    champion = _champion([_card("A", "2026-03-01T00:00:00", [10.0])])
    with pytest.raises(HaltedRunError):
        write_promotion_decision(
            workspace,
            candidate=candidate,
            champion=champion,
            comparison=_comparison(),
            shadow=_shadow(),
            decided_at=_ts(),
        )