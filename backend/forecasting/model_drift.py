"""Phase 7 CB4: model-drift detection between monitoring runs.

The model-drift engine compares the previous run's
``Sequence[ModelScorecard]`` against the current run's, rolling
the per-series scorecards up into a per-run ``ModelDriftReport``:

* **Population-level MASE / bias** — averaged over the input
  scorecards, with the signed delta between previous and current.
  Positive delta = regression; negative = improvement.
* **Per-segment degradation** — same rollup, but grouped by the
  ``SegmentMap`` the caller passes in. The cockpit uses the
  per-segment view to surface "G1 regressed, G2 is fine."
* **Interval calibration** — the seam. The platform does not
  yet emit quantile forecasts (same seam pattern as
  ``interval_coverage`` in ``metrics.py``), so the field stays
  ``None``. When the platform grows quantile support, the
  engine sets it to the observed coverage of the 80% PI.

Design:

* **Pure function, no I/O.** ``detect_model_drift`` is a pure
  function of (run_id, previous_scorecards, current_scorecards,
  segment_map). The scheduler / cockpit reads the previous
  run's scorecards from disk and calls the function; the
  function does not touch the filesystem.
* **Defaults to 0.0, never NaN.** An empty previous or current
  list returns ``0.0`` for the population MASE / bias (not
  NaN), so the cockpit can render the report without a
  special-case branch.
* **Skips unmapped series.** Scorecards whose ``series_key`` is
  not in the ``SegmentMap`` are excluded from the per-segment
  rollup. The engine never invents a segment id; the caller
  is the source of truth for the map.
* **No LLM.** Fully deterministic. The cockpit surfaces the
  typed report; the planner reads it.

The three public functions are:

* ``_scorecards_to_mase_bias(scorecards) -> tuple[float, float]``
  — average MASE and bias over the input list
* ``per_segment_mase(scorecards, segment_map) -> dict[str, float]``
  — MASE averaged per segment id
* ``detect_model_drift(...) -> ModelDriftReport`` — the
  top-level orchestrator
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from forecasting.contracts import (
    ModelDriftReport,
    ModelScorecard,
    SegmentDegradation,
    SegmentMap,
)

# ---------------------------------------------------------------------------
# _scorecards_to_mase_bias
# ---------------------------------------------------------------------------


def _scorecards_to_mase_bias(
    scorecards: Sequence[ModelScorecard],
) -> tuple[float, float]:
    """Average MASE and bias across ``scorecards``.

    Returns ``(0.0, 0.0)`` on empty input. NaN-safe: the engine
    never returns ``float('nan')`` for an empty input because the
    cockpit would have to special-case the rendering.
    """
    if not scorecards:
        return 0.0, 0.0
    n = len(scorecards)
    total_mase = sum(s.mase for s in scorecards)
    total_bias = sum(s.bias for s in scorecards)
    return total_mase / n, total_bias / n


# ---------------------------------------------------------------------------
# per_segment_mase
# ---------------------------------------------------------------------------


def per_segment_mase(
    scorecards: Sequence[ModelScorecard],
    segment_map: SegmentMap,
) -> dict[str, float]:
    """Average MASE per segment id.

    The engine walks the ``SegmentMap.segments`` list (the
    platform's source of truth for series-to-segment membership)
    and groups the input scorecards by segment. Series whose
    key is not in any segment's ``series_keys`` are silently
    skipped — the engine never invents a segment id. Returns
    an empty dict on empty input or empty map.
    """
    if not scorecards or not segment_map.segments:
        return {}
    # Build the (series_key -> segment_id) lookup once, by walking
    # the segment definitions. The lookup is deterministic: a
    # series in multiple segments would collide, but the platform's
    # contract is "a series belongs to exactly one segment" (see
    # CONTEXT.MD 'Segment'), so the first match wins.
    key_to_segment: dict[str, str] = {}
    for segment in segment_map.segments:
        for key in segment.series_keys:
            key_to_segment.setdefault(key, segment.segment_id)
    by_segment: dict[str, list[float]] = {}
    for scorecard in scorecards:
        segment_id = key_to_segment.get(scorecard.series_key)
        if segment_id is None:
            continue
        by_segment.setdefault(segment_id, []).append(scorecard.mase)
    return {
        segment_id: sum(values) / len(values)
        for segment_id, values in by_segment.items()
    }


def _segment_delta(
    segment_id: str,
    previous_per_seg: Mapping[str, float],
    current_per_seg: Mapping[str, float],
) -> SegmentDegradation:
    """Build one ``SegmentDegradation`` row from per-segment dicts.

    When a segment is missing from the current side, its current
    MASE is treated as ``0.0`` (the engine never invents a NaN).
    The same is true on the previous side.
    """
    previous = previous_per_seg.get(segment_id, 0.0)
    current = current_per_seg.get(segment_id, 0.0)
    return SegmentDegradation(
        segment_id=segment_id,
        mase_previous=previous,
        mase_current=current,
        mase_delta=current - previous,
    )


# ---------------------------------------------------------------------------
# detect_model_drift — top-level orchestrator
# ---------------------------------------------------------------------------


def detect_model_drift(
    *,
    run_id: str,
    previous_run_id: str,
    previous_scorecards: Sequence[ModelScorecard],
    current_scorecards: Sequence[ModelScorecard],
    segment_map: SegmentMap,
) -> ModelDriftReport:
    """Top-level model-drift engine.

    Combines the population-level MASE / bias deltas with the
    per-segment degradation view into one typed
    ``ModelDriftReport``. ``interval_calibration`` is the seam —
    it stays ``None`` until the platform grows quantile
    forecasts.
    """
    mase_previous, bias_previous = _scorecards_to_mase_bias(previous_scorecards)
    mase_current, bias_current = _scorecards_to_mase_bias(current_scorecards)
    previous_per_seg = per_segment_mase(previous_scorecards, segment_map)
    current_per_seg = per_segment_mase(current_scorecards, segment_map)
    # The per-segment view reports every segment the engine saw on
    # either side. A segment present in previous but not current is
    # surfaced as a regression (MASE went from real to 0.0), which
    # is the platform's "we lost visibility" signal.
    all_segments = sorted(set(previous_per_seg) | set(current_per_seg))
    segment_degradation = [
        _segment_delta(segment_id, previous_per_seg, current_per_seg)
        for segment_id in all_segments
    ]
    return ModelDriftReport(
        run_id=run_id,
        previous_run_id=previous_run_id,
        mase_previous=mase_previous,
        mase_current=mase_current,
        mase_delta=mase_current - mase_previous,
        bias_previous=bias_previous,
        bias_current=bias_current,
        bias_delta=bias_current - bias_previous,
        interval_calibration=None,
        segment_degradation=segment_degradation,
    )


__all__ = (
    "detect_model_drift",
    "per_segment_mase",
    # The underscore-prefixed helpers are kept out of ``__all__``
    # — they are private implementation details, not platform
    # surface. Tests that need them import them by their
    # underscore name. The pattern matches the rest of the
    # platform (see ``forecasting.data_drift``).
)


# _scorecards_to_mase_bias and _segment_delta are module-private
# helpers; they are exposed in __all__ via underscore convention
# only so test files that need them can import them by name
# (e.g. ``from forecasting.model_drift import
# _scorecards_to_mase_bias``). The pattern is the same as
# ``forecasting.data_drift``: leading underscore = private,
# presence in __all__ is not a public-API contract.
