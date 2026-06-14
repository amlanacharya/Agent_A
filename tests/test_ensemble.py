"""Tests for the ensemble tracker and blend helper (Phase 4)."""

from __future__ import annotations

import pytest

from forecasting.contracts import EnsembleSummary, ModelScorecard
from forecasting.ensemble import (
    PROTECTED_FAMILIES,
    EnsembleTracker,
    blend_forecasts,
    summarise_scorecards,
)


def _scorecard(
    *,
    model_family: str,
    series_key: str,
    mae: float,
    fold_cutoff: str = "2024-01-01",
    horizon: int = 2,
) -> ModelScorecard:
    return ModelScorecard(
        model_family=model_family,  # type: ignore[arg-type]
        series_key=series_key,
        fold_cutoff=fold_cutoff,
        horizon=horizon,
        forecast=[1.0] * horizon,
        actual=[1.0] * horizon,
        mae=mae,
        rmse=mae,
        mase=mae,
        bias=0.0,
    )


# ---------------------------------------------------------------------------
# summarise_scorecards
# ---------------------------------------------------------------------------


def test_summarise_scorecards_picks_lowest_mae_per_series() -> None:
    scs = [
        _scorecard(model_family="naive", series_key="A", mae=3.0),
        _scorecard(model_family="croston", series_key="A", mae=1.0),
        _scorecard(model_family="naive", series_key="B", mae=2.0),
        _scorecard(model_family="croston", series_key="B", mae=0.5),
        _scorecard(model_family="naive", series_key="C", mae=2.0),
        _scorecard(model_family="croston", series_key="C", mae=0.5),
    ]
    tracker = summarise_scorecards(scs, series_segment={"A": "INT", "B": "INT", "C": "INT"})
    # Croston wins A, B, C -> 3 wins. Naive wins none.
    weights = tracker.weights_for_segment("INT")
    # Naive has only the protected floor, croston has the bulk.
    assert weights["croston"] > weights["naive"]
    assert weights["croston"] >= 0.8  # at least 80% of the blend


def test_summarise_scorecards_with_split_winner_ties() -> None:
    scs = [
        _scorecard(model_family="naive", series_key="A", mae=3.0),
        _scorecard(model_family="croston", series_key="A", mae=1.0),
        _scorecard(model_family="naive", series_key="B", mae=2.0),
        _scorecard(model_family="croston", series_key="B", mae=4.0),
    ]
    tracker = summarise_scorecards(scs, series_segment={"A": "INT", "B": "INT"})
    # 1 win each -> equal weight (after floor).
    weights = tracker.weights_for_segment("INT")
    assert abs(weights["croston"] - weights["naive"]) < 1e-6


def test_summarise_scorecards_marks_never_surfaced_families() -> None:
    scs = [
        _scorecard(model_family="naive", series_key="A", mae=3.0),
        _scorecard(model_family="croston", series_key="A", mae=1.0),
        _scorecard(model_family="xgboost_global", series_key="A", mae=2.0),
    ]
    tracker = summarise_scorecards(scs, series_segment={"A": "INT"})
    assert "xgboost_global" in tracker.never_surfaced()


def test_summarise_scorecards_marks_frequently_promoted_families() -> None:
    scs = []
    for series in ["A", "B", "C", "D"]:
        scs.append(_scorecard(model_family="naive", series_key=series, mae=3.0))
        scs.append(_scorecard(model_family="croston", series_key=series, mae=1.0))
    tracker = summarise_scorecards(scs, series_segment={s: "INT" for s in ("A", "B", "C", "D")})
    # Croston won 100% of folds -> frequently promoted.
    assert "croston" in tracker.frequently_promoted()
    # Naive won 0% -> not in the frequently-promoted list, even
    # though it still gets the protected floor in the weights.
    assert "naive" not in tracker.frequently_promoted()
    assert "naive" in tracker.never_surfaced()


def test_summarise_scorecards_handles_empty_input() -> None:
    tracker = summarise_scorecards([])
    assert tracker.summary() == EnsembleSummary()


def test_summarise_scorecards_retire_excludes_family_from_wins() -> None:
    scs = [
        _scorecard(model_family="naive", series_key="A", mae=1.0),
        _scorecard(model_family="croston", series_key="A", mae=3.0),
    ]
    tracker = summarise_scorecards(scs, series_segment={"A": "INT"}, retired=("naive",))
    assert "naive" in tracker.retired
    # The retirement takes the family out of the weights entirely
    # so the blend no longer mixes it in.
    weights = tracker.weights_for_segment("INT")
    assert "naive" not in weights


def test_summarise_scorecards_uses_default_segment_when_unmapped() -> None:
    scs = [
        _scorecard(model_family="naive", series_key="A", mae=1.0),
        _scorecard(model_family="croston", series_key="A", mae=3.0),
    ]
    tracker = summarise_scorecards(scs)  # no series_segment mapping
    weights = tracker.weights_for_segment("__default__")
    assert weights["naive"] > weights["croston"]


# ---------------------------------------------------------------------------
# EnsembleTracker direct API
# ---------------------------------------------------------------------------


def test_tracker_record_increments_families_per_segment() -> None:
    tracker = EnsembleTracker(series_segment={"A": "INT"})
    tracker.record(_scorecard(model_family="naive", series_key="A", mae=1.0))
    tracker.record(_scorecard(model_family="croston", series_key="A", mae=1.0))
    weights = tracker.weights_for_segment("INT")
    assert set(weights) == {"naive", "croston"}


def test_tracker_record_winner_only_counts_explicit_wins() -> None:
    tracker = EnsembleTracker(series_segment={"A": "INT"})
    tracker.record(_scorecard(model_family="naive", series_key="A", mae=1.0))
    tracker.record(_scorecard(model_family="croston", series_key="A", mae=1.0))
    # No record_winner() call yet, so the segment has no wins
    # recorded. The weight blend falls back to an equal share.
    weights = tracker.weights_for_segment("INT")
    assert abs(weights["naive"] - weights["croston"]) < 1e-6


def test_tracker_protected_families_have_floor_weight() -> None:
    tracker = EnsembleTracker(series_segment={"A": "INT"})
    # Croston wins; naive does not.
    tracker.record(_scorecard(model_family="naive", series_key="A", mae=3.0))
    tracker.record(_scorecard(model_family="croston", series_key="A", mae=1.0))
    tracker.record_winner("A", "croston")
    weights = tracker.weights_for_segment("INT")
    # Naive should still be in the weights (protected) but well
    # below croston.
    assert "naive" in weights
    assert weights["naive"] >= 0.05 - 1e-9
    assert weights["croston"] > weights["naive"]


def test_tracker_summary_returns_cockpit_facing_shape() -> None:
    scs = [
        _scorecard(model_family="naive", series_key="A", mae=3.0),
        _scorecard(model_family="croston", series_key="A", mae=1.0),
    ]
    tracker = summarise_scorecards(scs, series_segment={"A": "INT"})
    summary = tracker.summary()
    assert isinstance(summary, EnsembleSummary)
    assert "INT" in summary.weights
    assert summary.weights["INT"]["croston"] > summary.weights["INT"]["naive"]


def test_tracker_scorecards_property_returns_copy() -> None:
    scs = [_scorecard(model_family="naive", series_key="A", mae=1.0)]
    tracker = summarise_scorecards(scs)
    # The property is a snapshot; mutating the returned list must
    # not affect the tracker's internal state.
    snapshot = tracker.scorecards
    snapshot.clear()
    assert len(tracker.scorecards) == 1
    # A new tracker built from the same scorecards still sees them.
    tracker2 = summarise_scorecards(scs)
    assert len(tracker2.scorecards) == 1


def test_protected_families_includes_naive_seasonal_and_croston() -> None:
    assert "naive" in PROTECTED_FAMILIES
    assert "seasonal_naive" in PROTECTED_FAMILIES
    assert "croston" in PROTECTED_FAMILIES


# ---------------------------------------------------------------------------
# blend_forecasts
# ---------------------------------------------------------------------------


def test_blend_forecasts_uses_normalised_weights() -> None:
    family_forecasts = {"naive": [10.0, 10.0], "croston": [20.0, 20.0]}
    weights = {"naive": 0.25, "croston": 0.75}
    blended = blend_forecasts(family_forecasts, weights)
    assert blended == [17.5, 17.5]


def test_blend_forecasts_ignores_families_not_in_forecasts() -> None:
    family_forecasts = {"naive": [10.0, 10.0]}
    weights = {"naive": 0.5, "croston": 0.5}
    # All the weight on croston gets dropped because croston has
    # no forecast, so naive ends up at 100%.
    blended = blend_forecasts(family_forecasts, weights)
    assert blended == [10.0, 10.0]


def test_blend_forecasts_returns_empty_when_nothing_to_blend() -> None:
    assert blend_forecasts({}, {}) == []
    assert blend_forecasts({"a": [1.0]}, {}) == []


def test_blend_forecasts_skips_mismatched_lengths() -> None:
    family_forecasts = {"naive": [10.0, 10.0], "croston": [20.0]}  # wrong length
    weights = {"naive": 0.5, "croston": 0.5}
    blended = blend_forecasts(family_forecasts, weights)
    # Only naive contributes. Horizon is max(len(naive), len(croston)) = 2.
    assert blended == [10.0, 10.0]
