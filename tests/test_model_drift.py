"""Tests for Phase 7 CB4: model_drift module.

Covers ``detect_model_drift(previous_scorecards, current_scorecards,
segment_map)`` and the helper functions the engine uses to roll
up the per-series ModelScorecards into a per-run ModelDriftReport.

The engine must:

* Return a valid ``ModelDriftReport`` for any input, including the
  empty case (no previous, no current)
* Be pure — no I/O, no globals, no LLM
* Default the MASE / bias to ``0.0`` (no regression) on either side
  missing, so the report never carries NaN / inf into the cockpit
* Report per-segment degradation by walking the ``segment_map``
  provided by the caller
* Leave ``interval_calibration`` as ``None`` (the platform does
  not yet emit quantile forecasts — same seam pattern as
  ``interval_coverage`` in ``metrics.py``)
"""

from __future__ import annotations

import pytest

from forecasting.contracts import (
    ModelDriftReport,
    ModelScorecard,
    SegmentDegradation,
    SegmentDef,
    SegmentMap,
)
from forecasting.model_drift import (
    _scorecards_to_mase_bias,
    detect_model_drift,
    per_segment_mase,
)


def _scorecard(
    *,
    series_key: str,
    mase: float,
    bias: float = 0.0,
) -> ModelScorecard:
    """Build a tiny ModelScorecard for the tests."""
    return ModelScorecard(
        model_family="naive",
        series_key=series_key,
        fold_cutoff="2026-01-01",
        horizon=1,
        forecast=[10.0],
        actual=[12.0],
        mae=2.0,
        rmse=2.0,
        mase=mase,
        bias=bias,
    )


def _segment_map(items: dict[str, str]) -> SegmentMap:
    """Build a SegmentMap from a ``{series_key: segment_id}`` dict.

    The engine walks ``SegmentMap.segments[i].series_keys`` (not
    a top-level ``map`` field), so the test helper groups series
    by segment id and builds the segment list the engine expects.
    """
    by_segment: dict[str, list[str]] = {}
    for series_key, segment_id in items.items():
        by_segment.setdefault(segment_id, []).append(series_key)
    segments = [
        SegmentDef(
            segment_id=segment_id,
            label=f"segment {segment_id}",
            series_keys=sorted(keys),
            provisional=True,
        )
        for segment_id, keys in sorted(by_segment.items())
    ]
    return SegmentMap(
        run_id="r1",
        segments=segments,
        provisional=True,
        derived_by="test:helper",
    )


# ---------------------------------------------------------------------------
# _scorecards_to_mase_bias
# ---------------------------------------------------------------------------


def test_scorecards_to_mase_bias_averages_correctly() -> None:
    """The helper averages MASE and bias across the input scorecards."""
    scorecards = [
        _scorecard(series_key="A", mase=0.80, bias=0.10),
        _scorecard(series_key="B", mase=1.20, bias=-0.10),
    ]
    mase, bias = _scorecards_to_mase_bias(scorecards)
    assert mase == pytest.approx(1.0)
    assert bias == pytest.approx(0.0)


def test_scorecards_to_mase_bias_empty_input() -> None:
    """Empty input returns (0.0, 0.0) — no NaN, no inf."""
    assert _scorecards_to_mase_bias([]) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# per_segment_mase
# ---------------------------------------------------------------------------


def test_per_segment_mase_groups_scorecards_correctly() -> None:
    """Scorecards in segment G1 are averaged together, separately from G2."""
    scorecards = [
        _scorecard(series_key="A", mase=0.80),  # G1
        _scorecard(series_key="B", mase=0.90),  # G1
        _scorecard(series_key="C", mase=1.20),  # G2
    ]
    segment_map = _segment_map({"A": "G1", "B": "G1", "C": "G2"})
    per_seg = per_segment_mase(scorecards, segment_map)
    assert per_seg == {"G1": pytest.approx(0.85), "G2": pytest.approx(1.20)}


def test_per_segment_mase_skips_unmapped_keys() -> None:
    """Scorecards whose series_key is not in the segment_map are skipped."""
    scorecards = [
        _scorecard(series_key="A", mase=0.80),  # G1
        _scorecard(series_key="UNKNOWN", mase=9.99),  # not mapped
    ]
    segment_map = _segment_map({"A": "G1"})
    per_seg = per_segment_mase(scorecards, segment_map)
    assert per_seg == {"G1": pytest.approx(0.80)}


def test_per_segment_mase_empty_input() -> None:
    """Empty input returns an empty dict."""
    assert per_segment_mase([], _segment_map({})) == {}


# ---------------------------------------------------------------------------
# detect_model_drift — top-level engine
# ---------------------------------------------------------------------------


def test_detect_model_drift_empty_previous_returns_zero_deltas() -> None:
    """A first run has no previous to compare against — no deltas.

    The population-level MASE / bias defaults to ``0.0`` on the
    previous side (no NaN, no inf). The per-segment view still
    surfaces the current segment as a "no baseline" observation:
    the previous MASE is 0.0 and the current is the actual
    value, so the delta is positive (the engine surfaces the
    full sign of the change). This is the platform's "we have
    no baseline" signal — the planner reads the per-segment
    delta as "this is fresh, we have no prior to compare
    against."
    """
    report = detect_model_drift(
        run_id="r1",
        previous_run_id="",
        previous_scorecards=[],
        current_scorecards=[_scorecard(series_key="A", mase=0.80)],
        segment_map=_segment_map({"A": "G1"}),
    )
    assert isinstance(report, ModelDriftReport)
    assert report.run_id == "r1"
    assert report.previous_run_id == ""
    assert report.mase_previous == 0.0
    assert report.mase_current == pytest.approx(0.80)
    assert report.mase_delta == pytest.approx(0.80)
    assert report.interval_calibration is None
    # The per-segment view: G1 has no previous (0.0) and a
    # current MASE of 0.80. Delta is the full value — the
    # planner reads it as "this is a new segment, we have no
    # baseline."
    assert len(report.segment_degradation) == 1
    assert report.segment_degradation[0].segment_id == "G1"
    assert report.segment_degradation[0].mase_previous == 0.0
    assert report.segment_degradation[0].mase_current == pytest.approx(0.80)
    assert report.segment_degradation[0].mase_delta == pytest.approx(0.80)


def test_detect_model_drift_reports_mase_and_bias_deltas() -> None:
    """Population-level MASE / bias deltas surface on the report."""
    previous = [
        _scorecard(series_key="A", mase=0.80, bias=0.0),
        _scorecard(series_key="B", mase=0.80, bias=0.0),
    ]
    current = [
        _scorecard(series_key="A", mase=0.90, bias=0.10),
        _scorecard(series_key="B", mase=0.90, bias=0.10),
    ]
    report = detect_model_drift(
        run_id="r2",
        previous_run_id="r1",
        previous_scorecards=previous,
        current_scorecards=current,
        segment_map=_segment_map({"A": "G1", "B": "G1"}),
    )
    assert report.mase_previous == pytest.approx(0.80)
    assert report.mase_current == pytest.approx(0.90)
    assert report.mase_delta == pytest.approx(0.10)
    assert report.bias_previous == pytest.approx(0.0)
    assert report.bias_current == pytest.approx(0.10)
    assert report.bias_delta == pytest.approx(0.10)


def test_detect_model_drift_reports_per_segment_degradation() -> None:
    """Per-segment deltas come from the per-segment mase helper."""
    previous = [
        _scorecard(series_key="A", mase=0.80),  # G1
        _scorecard(series_key="C", mase=1.00),  # G2
    ]
    current = [
        _scorecard(series_key="A", mase=0.85),  # G1: small regression
        _scorecard(series_key="C", mase=1.50),  # G2: big regression
    ]
    report = detect_model_drift(
        run_id="r2",
        previous_run_id="r1",
        previous_scorecards=previous,
        current_scorecards=current,
        segment_map=_segment_map({"A": "G1", "C": "G2"}),
    )
    by_segment = {s.segment_id: s for s in report.segment_degradation}
    assert by_segment["G1"].mase_previous == pytest.approx(0.80)
    assert by_segment["G1"].mase_current == pytest.approx(0.85)
    assert by_segment["G1"].mase_delta == pytest.approx(0.05)
    assert by_segment["G2"].mase_previous == pytest.approx(1.00)
    assert by_segment["G2"].mase_current == pytest.approx(1.50)
    assert by_segment["G2"].mase_delta == pytest.approx(0.50)


def test_detect_model_drift_improvement_is_negative_delta() -> None:
    """A positive MASE improvement surfaces as a negative delta."""
    previous = [_scorecard(series_key="A", mase=1.20)]
    current = [_scorecard(series_key="A", mase=0.80)]
    report = detect_model_drift(
        run_id="r2",
        previous_run_id="r1",
        previous_scorecards=previous,
        current_scorecards=current,
        segment_map=_segment_map({"A": "G1"}),
    )
    assert report.mase_delta == pytest.approx(-0.40)
    assert report.segment_degradation[0].mase_delta == pytest.approx(-0.40)


def test_detect_model_drift_default_segment_map_is_empty() -> None:
    """An empty SegmentMap produces an empty segment_degradation list
    even when scorecards are present (the engine never invents
    segment ids)."""
    report = detect_model_drift(
        run_id="r2",
        previous_run_id="r1",
        previous_scorecards=[_scorecard(series_key="A", mase=0.80)],
        current_scorecards=[_scorecard(series_key="A", mase=0.90)],
        # Empty segment map → no segments to roll up. The engine
    # silently returns an empty list rather than raising or
    # inventing segment ids.
    segment_map=_segment_map({}),
    )
    assert report.segment_degradation == []


def test_detect_model_drift_handles_empty_current() -> None:
    """A run with no current scorecards (e.g. the harness didn't
    finish) returns a zero current mase / bias and a per-segment
    degradation that still includes the previous side as a
    reference. The engine never invents a MASE of NaN."""
    previous = [_scorecard(series_key="A", mase=0.80)]
    report = detect_model_drift(
        run_id="r2",
        previous_run_id="r1",
        previous_scorecards=previous,
        current_scorecards=[],
        segment_map=_segment_map({"A": "G1"}),
    )
    assert report.mase_current == 0.0
    assert report.mase_delta == pytest.approx(-0.80)
    # Per-segment view: previous was 0.80, current is 0.0, so the
    # delta is the negative of the previous MASE.
    by_segment = {s.segment_id: s for s in report.segment_degradation}
    assert by_segment["G1"].mase_previous == pytest.approx(0.80)
    assert by_segment["G1"].mase_current == 0.0
    assert by_segment["G1"].mase_delta == pytest.approx(-0.80)


def test_detect_model_drift_ignores_interval_calibration() -> None:
    """The interval_calibration field stays None — the platform
    does not yet emit quantile forecasts."""
    report = detect_model_drift(
        run_id="r1",
        previous_run_id="r0",
        previous_scorecards=[],
        current_scorecards=[_scorecard(series_key="A", mase=0.80)],
        segment_map=_segment_map({"A": "G1"}),
    )
    assert report.interval_calibration is None


def test_detect_model_drift_segment_degradation_typed() -> None:
    """The per-segment list is a typed ``list[SegmentDegradation]``."""
    report = detect_model_drift(
        run_id="r2",
        previous_run_id="r1",
        previous_scorecards=[_scorecard(series_key="A", mase=0.80)],
        current_scorecards=[_scorecard(series_key="A", mase=0.90)],
        segment_map=_segment_map({"A": "G1"}),
    )
    for seg in report.segment_degradation:
        assert isinstance(seg, SegmentDegradation)
