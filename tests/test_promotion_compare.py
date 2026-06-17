"""Tests for the Phase 5.2 CB2 candidate vs champion comparison.

The comparison function consumes ``PromotionCandidate`` /
``Champion`` records and a fixed ``BacktestWindow``. Tests
use synthetic scorecards with known WAPE values; the
function is pure and deterministic.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from forecasting.contracts import ModelScorecard
from forecasting.promotion import (
    BacktestWindow,
    Champion,
    PromotionCandidate,
    PromotionComparison,
    build_default_backtest_window,
    compare_candidate_to_champion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _card(
    series_key: str,
    fold_cutoff: str,
    forecast: list[float],
    actual: list[float],
    *,
    model_family: str = "naive",
) -> ModelScorecard:
    """Build a ModelScorecard with sensible test defaults."""
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


def _window_clean() -> BacktestWindow:
    """A clean backtest window with 2 cutoffs, horizon=4.

    Cutoffs are newest-first (matches the builder's convention).
    """
    return BacktestWindow(
        start="2025-01-01T00:00:00",
        end="2026-12-31T00:00:00",
        cutoffs=[
            "2026-03-01T00:00:00",  # most recent
            "2025-12-01T00:00:00",
        ],
        horizon=4,
    )


def _candidate_better_than_champion() -> tuple[PromotionCandidate, Champion, dict]:
    """Candidate has lower WAPE than champion.

    Series A: actual [10, 20, 30, 40]
    - Champion forecast [12, 18, 30, 38] -> abs err [2, 2, 0, 2]
    - Candidate forecast [11, 19, 30, 39] -> abs err [1, 1, 0, 1]

    Champion WAPE on A: 6 / 100 = 0.06
    Candidate WAPE on A: 3 / 100 = 0.03
    """
    cutoff = "2026-03-01T00:00:00"
    candidate = PromotionCandidate(
        run_id="r-candidate",
        model_family="xgboost_global",
        scorecards=[
            _card("A", cutoff, [11.0, 19.0, 30.0, 39.0], [10.0, 20.0, 30.0, 40.0]),
        ],
        reason="from CB3 propose_feature_changes",
    )
    champion = Champion(
        model_family="naive",
        scorecards=[
            _card("A", cutoff, [12.0, 18.0, 30.0, 38.0], [10.0, 20.0, 30.0, 40.0]),
        ],
        promoted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    segment_map = {"A": "G1"}
    return candidate, champion, segment_map


def _candidate_worse_than_champion() -> tuple[PromotionCandidate, Champion, dict]:
    """Candidate has higher WAPE than champion."""
    cutoff = "2026-03-01T00:00:00"
    candidate = PromotionCandidate(
        run_id="r-candidate",
        model_family="xgboost_global",
        scorecards=[
            _card("A", cutoff, [15.0, 25.0, 35.0, 45.0], [10.0, 20.0, 30.0, 40.0]),
        ],
        reason="test",
    )
    champion = Champion(
        model_family="naive",
        scorecards=[
            _card("A", cutoff, [10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0]),
        ],
        promoted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return candidate, champion, {"A": "G1"}


# ---------------------------------------------------------------------------
# PromotionComparison basic shape
# ---------------------------------------------------------------------------


def test_compare_candidate_better_returns_promote() -> None:
    """A candidate with lower WAPE on the window is promoted."""
    candidate, champion, segment_map = _candidate_better_than_champion()
    result = compare_candidate_to_champion(
        candidate,
        champion,
        window=_window_clean(),
        canonical_table_end="2026-06-17T00:00:00",
        segment_map=segment_map,
    )
    assert isinstance(result, PromotionComparison)
    assert result.promotion_outcome == "promote"
    assert result.wape_delta < 0  # negative = candidate better
    assert result.candidate_wape < result.champion_wape


def test_compare_candidate_worse_returns_reject() -> None:
    """A candidate with higher WAPE is rejected."""
    candidate, champion, segment_map = _candidate_worse_than_champion()
    result = compare_candidate_to_champion(
        candidate,
        champion,
        window=_window_clean(),
        canonical_table_end="2026-06-17T00:00:00",
        segment_map=segment_map,
    )
    assert result.promotion_outcome == "reject"
    assert result.wape_delta > 0  # positive = candidate worse


def test_compare_tie_returns_reject_by_default() -> None:
    """A perfect tie (no improvement) goes to the champion -> reject.

    ``min_improvement=0`` means "any improvement counts". A
    tie has wape_delta=0, which is not <= -min_improvement (=
    0). The function rejects ties by default — promotion
    requires strict improvement.
    """
    cutoff = "2026-03-01T00:00:00"
    scorecards = [
        _card("A", cutoff, [11.0, 21.0, 31.0, 41.0], [10.0, 20.0, 30.0, 40.0]),
    ]
    candidate = PromotionCandidate(
        run_id="r1", model_family="xgboost_global", scorecards=scorecards, reason="test"
    )
    champion = Champion(
        model_family="naive",
        scorecards=scorecards,  # same scorecards -> identical WAPE
        promoted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    result = compare_candidate_to_champion(
        candidate,
        champion,
        window=_window_clean(),
        canonical_table_end="2026-06-17T00:00:00",
        segment_map={"A": "G1"},
    )
    assert result.wape_delta == 0.0
    assert result.promotion_outcome == "reject"


# ---------------------------------------------------------------------------
# Leakage short-circuit
# ---------------------------------------------------------------------------


def test_compare_short_circuits_on_leakage_failure() -> None:
    """A non-clean leakage check short-circuits to leakage_failed.

    No numeric comparison is performed; the comparison
    surfaces the failure as the outcome.
    """
    candidate, champion, segment_map = _candidate_better_than_champion()
    # Cutoffs AFTER canonical_table_end -> future_cutoff
    leaky_window = BacktestWindow(
        start="2025-01-01T00:00:00",
        end="2027-12-31T00:00:00",
        cutoffs=[
            "2027-06-01T00:00:00",  # future
            "2026-03-01T00:00:00",
        ],
        horizon=4,
    )
    result = compare_candidate_to_champion(
        candidate,
        champion,
        window=leaky_window,
        canonical_table_end="2026-06-17T00:00:00",
        segment_map=segment_map,
    )
    assert result.promotion_outcome == "leakage_failed"
    assert math.isnan(result.candidate_wape)
    assert math.isnan(result.champion_wape)
    assert math.isnan(result.wape_delta)


# ---------------------------------------------------------------------------
# Window filter
# ---------------------------------------------------------------------------


def test_compare_uses_only_scorecards_in_window() -> None:
    """Scorecards with fold_cutoff outside the window are ignored."""
    cutoff_in = "2026-03-01T00:00:00"
    cutoff_out = "2025-01-01T00:00:00"  # outside the window's cutoffs list
    candidate = PromotionCandidate(
        run_id="r1",
        model_family="xgboost_global",
        scorecards=[
            _card("A", cutoff_in, [11.0, 19.0, 30.0, 39.0], [10.0, 20.0, 30.0, 40.0]),
            # This scorecard is outside the window -> must be ignored.
            _card("A", cutoff_out, [100.0, 100.0, 100.0, 100.0], [10.0, 20.0, 30.0, 40.0]),
        ],
        reason="test",
    )
    champion = Champion(
        model_family="naive",
        scorecards=[
            _card("A", cutoff_in, [12.0, 18.0, 30.0, 38.0], [10.0, 20.0, 30.0, 40.0]),
            _card("A", cutoff_out, [12.0, 18.0, 30.0, 38.0], [10.0, 20.0, 30.0, 40.0]),
        ],
        promoted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    result = compare_candidate_to_champion(
        candidate,
        champion,
        window=_window_clean(),
        canonical_table_end="2026-06-17T00:00:00",
        segment_map={"A": "G1"},
    )
    # Only the in-window scorecard is compared; candidate still
    # wins on the single in-window cutoff.
    assert result.candidate_wape == pytest.approx(0.03)
    assert result.champion_wape == pytest.approx(0.06)
    assert result.promotion_outcome == "promote"


# ---------------------------------------------------------------------------
# Per-segment rollup
# ---------------------------------------------------------------------------


def test_compare_segments_improved_listed() -> None:
    """A segment where the candidate's WAPE dropped is in segments_improved."""
    cutoff = "2026-03-01T00:00:00"
    candidate = PromotionCandidate(
        run_id="r1",
        model_family="xgboost_global",
        scorecards=[
            _card("A", cutoff, [11.0, 19.0, 30.0, 39.0], [10.0, 20.0, 30.0, 40.0]),
            _card("B", cutoff, [11.0, 19.0, 30.0, 39.0], [10.0, 20.0, 30.0, 40.0]),
        ],
        reason="test",
    )
    champion = Champion(
        model_family="naive",
        scorecards=[
            _card("A", cutoff, [12.0, 18.0, 30.0, 38.0], [10.0, 20.0, 30.0, 40.0]),
            _card("B", cutoff, [12.0, 18.0, 30.0, 38.0], [10.0, 20.0, 30.0, 40.0]),
        ],
        promoted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    result = compare_candidate_to_champion(
        candidate,
        champion,
        window=_window_clean(),
        canonical_table_end="2026-06-17T00:00:00",
        segment_map={"A": "G1", "B": "G2"},
    )
    assert "G1" in result.segments_improved
    assert "G2" in result.segments_improved


def test_compare_segments_regressed_blocks_promotion() -> None:
    """A segment regression is a hard fail — even if overall WAPE improves."""
    cutoff = "2026-03-01T00:00:00"
    candidate = PromotionCandidate(
        run_id="r1",
        model_family="xgboost_global",
        # Candidate is better on A but worse on B.
        scorecards=[
            _card("A", cutoff, [11.0, 19.0, 30.0, 39.0], [10.0, 20.0, 30.0, 40.0]),
            _card("B", cutoff, [50.0, 50.0, 50.0, 50.0], [10.0, 20.0, 30.0, 40.0]),
        ],
        reason="test",
    )
    champion = Champion(
        model_family="naive",
        scorecards=[
            _card("A", cutoff, [12.0, 18.0, 30.0, 38.0], [10.0, 20.0, 30.0, 40.0]),
            _card("B", cutoff, [12.0, 18.0, 30.0, 38.0], [10.0, 20.0, 30.0, 40.0]),
        ],
        promoted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    result = compare_candidate_to_champion(
        candidate,
        champion,
        window=_window_clean(),
        canonical_table_end="2026-06-17T00:00:00",
        segment_map={"A": "G1", "B": "G2"},
    )
    # G1 improved (candidate better) but G2 regressed (candidate
    # worse on B). The platform rejects: a segment regression
    # is a hard fail.
    assert "G1" in result.segments_improved
    assert "G2" in result.segments_regressed
    assert result.promotion_outcome == "reject"


def test_compare_min_improvement_threshold_respected() -> None:
    """min_improvement=0.05 forces the candidate to improve WAPE by 5% absolute."""
    cutoff = "2026-03-01T00:00:00"
    # Candidate improves by 0.03 (from 0.06 to 0.03) -- below the 0.05 threshold.
    candidate = PromotionCandidate(
        run_id="r1",
        model_family="xgboost_global",
        scorecards=[
            _card("A", cutoff, [11.0, 19.0, 30.0, 39.0], [10.0, 20.0, 30.0, 40.0]),
        ],
        reason="test",
    )
    champion = Champion(
        model_family="naive",
        scorecards=[
            _card("A", cutoff, [12.0, 18.0, 30.0, 38.0], [10.0, 20.0, 30.0, 40.0]),
        ],
        promoted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    # With min_improvement=0.05, the 0.03 improvement is not enough.
    result = compare_candidate_to_champion(
        candidate,
        champion,
        window=_window_clean(),
        canonical_table_end="2026-06-17T00:00:00",
        segment_map={"A": "G1"},
        min_improvement=0.05,
    )
    assert result.wape_delta < 0  # candidate still better
    assert result.promotion_outcome == "reject"  # but below the threshold


def test_compare_series_without_segment_assignment_silently_skipped() -> None:
    """A series_key not in segment_map does not break the comparison.

    The series is excluded from the per-segment rollup but
    still contributes to the overall WAPE.
    """
    cutoff = "2026-03-01T00:00:00"
    candidate = PromotionCandidate(
        run_id="r1",
        model_family="xgboost_global",
        scorecards=[
            _card("A", cutoff, [11.0, 19.0, 30.0, 39.0], [10.0, 20.0, 30.0, 40.0]),
            _card("unsegmented", cutoff, [11.0, 19.0, 30.0, 39.0], [10.0, 20.0, 30.0, 40.0]),
        ],
        reason="test",
    )
    champion = Champion(
        model_family="naive",
        scorecards=[
            _card("A", cutoff, [12.0, 18.0, 30.0, 38.0], [10.0, 20.0, 30.0, 40.0]),
            _card("unsegmented", cutoff, [12.0, 18.0, 30.0, 38.0], [10.0, 20.0, 30.0, 40.0]),
        ],
        promoted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    result = compare_candidate_to_champion(
        candidate,
        champion,
        window=_window_clean(),
        canonical_table_end="2026-06-17T00:00:00",
        segment_map={"A": "G1"},  # 'unsegmented' is NOT in the map
    )
    # Both segments_compared only contains G1; unsegmented is
    # silently dropped from the per-segment rollup.
    assert result.segments_compared == ["G1"]
    assert result.promotion_outcome == "promote"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_compare_is_deterministic() -> None:
    """Same inputs -> same comparison record."""
    candidate, champion, segment_map = _candidate_better_than_champion()
    r1 = compare_candidate_to_champion(
        candidate,
        champion,
        window=_window_clean(),
        canonical_table_end="2026-06-17T00:00:00",
        segment_map=segment_map,
    )
    r2 = compare_candidate_to_champion(
        candidate,
        champion,
        window=_window_clean(),
        canonical_table_end="2026-06-17T00:00:00",
        segment_map=segment_map,
    )
    assert r1.candidate_wape == r2.candidate_wape
    assert r1.champion_wape == r2.champion_wape
    assert r1.wape_delta == r2.wape_delta
    assert r1.promotion_outcome == r2.promotion_outcome
    assert r1.segments_improved == r2.segments_improved
    assert r1.segments_regressed == r2.segments_regressed