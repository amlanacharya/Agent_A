"""Tests for the Phase 5.2 CB3 shadow-mode runner.

The shadow-mode runner compares (candidate, champion) forecasts
step-by-step and reports the agreement rate. Tests use
synthetic scorecards with known forecasts; the function is
pure and deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from forecasting.contracts import ModelScorecard
from forecasting.promotion import (
    Champion,
    PromotionCandidate,
    ShadowModeResult,
    run_shadow_mode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _card(
    series_key: str,
    fold_cutoff: str,
    forecast: list[float],
    actual: list[float] | None = None,
    *,
    model_family: str = "naive",
) -> ModelScorecard:
    """Build a ModelScorecard. ``actual`` defaults to ``forecast``
    (perfect forecast) so the card passes Pydantic validation
    without the test having to spell out 4+ identical numbers.
    """
    if actual is None:
        actual = list(forecast)
    f = np.asarray(forecast, dtype=float)
    a = np.asarray(actual, dtype=float)
    residuals = a - f
    mae = float(np.mean(np.abs(residuals))) if len(f) else 0.0
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if len(f) else 0.0
    return ModelScorecard(
        model_family=model_family,
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


def _candidate(scorecards: list[ModelScorecard]) -> PromotionCandidate:
    return PromotionCandidate(
        run_id="r-candidate",
        model_family="xgboost_global",
        scorecards=scorecards,
        reason="test",
    )


def _champion(scorecards: list[ModelScorecard]) -> Champion:
    return Champion(
        model_family="naive",
        scorecards=scorecards,
        promoted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Perfect agreement
# ---------------------------------------------------------------------------


def test_shadow_perfect_agreement_when_scorecards_identical() -> None:
    """Identical forecasts on both sides -> agreement_rate=1.0."""
    cutoff = "2026-03-01T00:00:00"
    fc = [10.0, 20.0, 30.0, 40.0]
    candidate = _candidate([_card("A", cutoff, fc, model_family="xgboost_global")])
    champion = _champion([_card("A", cutoff, fc, model_family="naive")])
    result = run_shadow_mode(candidate, champion)
    assert isinstance(result, ShadowModeResult)
    assert result.agreement_rate == 1.0
    assert result.candidate_family == "xgboost_global"
    assert result.champion_family == "naive"


def test_shadow_per_series_pairs_are_step_by_step() -> None:
    """per_series_pairs[series_key] is the list of (cand, champ) per step."""
    cutoff = "2026-03-01T00:00:00"
    cand_fc = [10.0, 20.0, 30.0, 40.0]
    champ_fc = [11.0, 19.0, 31.0, 39.0]
    candidate = _candidate([_card("A", cutoff, cand_fc)])
    champion = _champion([_card("A", cutoff, champ_fc)])
    result = run_shadow_mode(candidate, champion)
    assert result.per_series_pairs["A"] == [
        (10.0, 11.0),
        (20.0, 19.0),
        (30.0, 31.0),
        (40.0, 39.0),
    ]


# ---------------------------------------------------------------------------
# Partial / zero agreement
# ---------------------------------------------------------------------------


def test_shadow_agreement_rate_one_when_all_pairs_close() -> None:
    """All pairs within the (5%) tolerance -> agreement_rate=1.0."""
    cutoff = "2026-03-01T00:00:00"
    cand_fc = [10.0, 20.0, 30.0, 40.0]
    # 4% relative difference: within 5% tolerance.
    champ_fc = [10.4, 20.4, 30.4, 40.4]
    candidate = _candidate([_card("A", cutoff, cand_fc)])
    champion = _champion([_card("A", cutoff, champ_fc)])
    result = run_shadow_mode(candidate, champion)
    assert result.agreement_rate == 1.0


def test_shadow_agreement_rate_zero_when_all_pairs_far() -> None:
    """All pairs well outside the tolerance -> agreement_rate=0.0."""
    cutoff = "2026-03-01T00:00:00"
    cand_fc = [10.0, 20.0, 30.0, 40.0]
    # 50% relative difference: well outside the 5% tolerance.
    champ_fc = [5.0, 10.0, 15.0, 20.0]
    candidate = _candidate([_card("A", cutoff, cand_fc)])
    champion = _champion([_card("A", cutoff, champ_fc)])
    result = run_shadow_mode(candidate, champion)
    assert result.agreement_rate == 0.0


def test_shadow_agreement_handles_zero_champion_forecast() -> None:
    """A champion forecast of 0 with a non-zero candidate is a disagreement.

    The ``_ZERO_EPS`` branch in the relative-error computation
    treats 0 as a small-but-nonzero denominator, so a candidate
    of 5 vs champion of 0 produces a relative error of 5/eps =
    ~5e9 -> not within tolerance. Agreement rate is 0.
    """
    cutoff = "2026-03-01T00:00:00"
    cand_fc = [5.0, 5.0]
    champ_fc = [0.0, 0.0]
    candidate = _candidate([_card("A", cutoff, cand_fc)])
    champion = _champion([_card("A", cutoff, champ_fc)])
    result = run_shadow_mode(candidate, champion)
    assert result.agreement_rate == 0.0


def test_shadow_zero_zero_pair_agrees() -> None:
    """Champion 0 / candidate 0: |delta| = 0 -> within any tolerance."""
    cutoff = "2026-03-01T00:00:00"
    fc = [0.0, 0.0]
    candidate = _candidate([_card("A", cutoff, fc)])
    champion = _champion([_card("A", cutoff, fc)])
    result = run_shadow_mode(candidate, champion)
    assert result.agreement_rate == 1.0


# ---------------------------------------------------------------------------
# Tolerance parameter
# ---------------------------------------------------------------------------


def test_shadow_tolerance_parameter_respected() -> None:
    """A 10% difference is in agreement at tolerance=0.20 but out at tolerance=0.05."""
    cutoff = "2026-03-01T00:00:00"
    cand_fc = [10.0, 20.0, 30.0]
    champ_fc = [11.0, 22.0, 33.0]  # 10% relative diff
    candidate = _candidate([_card("A", cutoff, cand_fc)])
    champion = _champion([_card("A", cutoff, champ_fc)])
    # Loose tolerance: in agreement.
    loose = run_shadow_mode(candidate, champion, tolerance=0.20)
    assert loose.agreement_rate == 1.0
    assert loose.tolerance == 0.20
    # Tight tolerance: not in agreement.
    tight = run_shadow_mode(candidate, champion, tolerance=0.05)
    assert tight.agreement_rate == 0.0
    assert tight.tolerance == 0.05


# ---------------------------------------------------------------------------
# Multi-series / multi-fold
# ---------------------------------------------------------------------------


def test_shadow_runs_across_multiple_series() -> None:
    """The agreement rate is aggregated across all (series, step) pairs."""
    cutoff = "2026-03-01T00:00:00"
    cand_fc_a = [10.0, 20.0]
    champ_fc_a = [10.0, 20.0]  # agree on A
    cand_fc_b = [50.0, 60.0]
    champ_fc_b = [100.0, 120.0]  # disagree on B
    candidate = _candidate(
        [_card("A", cutoff, cand_fc_a), _card("B", cutoff, cand_fc_b)]
    )
    champion = _champion(
        [_card("A", cutoff, champ_fc_a), _card("B", cutoff, champ_fc_b)]
    )
    result = run_shadow_mode(candidate, champion)
    # 2 pairs on A (both agree), 2 pairs on B (both disagree).
    # Agreement rate = 2/4 = 0.5.
    assert result.agreement_rate == 0.5
    assert set(result.per_series_pairs) == {"A", "B"}


def test_shadow_pairs_by_fold_cutoff_first() -> None:
    """Two scorecards on each side, same fold_cutoffs -> pairs by cutoff."""
    cutoff_a = "2026-03-01T00:00:00"
    cutoff_b = "2025-12-01T00:00:00"
    cand_cards = [
        _card("A", cutoff_a, [10.0, 20.0]),
        _card("A", cutoff_b, [30.0, 40.0]),
    ]
    champ_cards = [
        _card("A", cutoff_a, [10.0, 20.0]),
        _card("A", cutoff_b, [30.0, 40.0]),
    ]
    candidate = _candidate(cand_cards)
    champion = _champion(champ_cards)
    result = run_shadow_mode(candidate, champion)
    # 4 pairs (2 folds * 2 steps), all identical -> 100% agreement.
    assert len(result.per_series_pairs["A"]) == 4
    assert result.agreement_rate == 1.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_shadow_is_deterministic() -> None:
    """Same inputs -> same result. The audit must be reproducible."""
    cutoff = "2026-03-01T00:00:00"
    cand_fc = [10.0, 20.0, 30.0]
    champ_fc = [11.0, 19.0, 31.0]
    candidate = _candidate([_card("A", cutoff, cand_fc)])
    champion = _champion([_card("A", cutoff, champ_fc)])
    r1 = run_shadow_mode(candidate, champion)
    r2 = run_shadow_mode(candidate, champion)
    assert r1.agreement_rate == r2.agreement_rate
    assert r1.per_series_pairs == r2.per_series_pairs
    assert r1.tolerance == r2.tolerance


def test_shadow_empty_scorecards_yields_zero_agreement_rate() -> None:
    """Both sides empty -> total_pairs=0, agreement_rate=0 (no disagreement, no agreement)."""
    candidate = _candidate([])
    champion = _champion([])
    result = run_shadow_mode(candidate, champion)
    assert result.agreement_rate == 0.0
    assert result.per_series_pairs == {}