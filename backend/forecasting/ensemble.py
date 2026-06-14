"""Ensemble behaviour tracking for the forecasting harness (Phase 4).

The harness is the canonical place to score models; the ensemble
tracker is the canonical place to remember which models won over
time, and to compute the weights the harness should use to blend
multiple families into a single forecast.

Four behaviours the plan calls out:

- **weights by segment** - the ensemble output is a single forecast
  vector per series, produced by a weighted average of the family
  forecasts where the weights are the families' historical
  best-in-fold rate for that segment.
- **frequently promoted** - families that have been best-in-fold
  for >= 50% of series in a segment over the run's history.
- **never surfaced** - families that fit successfully but were
  never best-in-fold. The harness can use this to drop them from
  the production blend and flag them for retirement.
- **retired but retained** - families that were promoted in a
  prior run and have since been replaced. We keep their scorecards
  on disk for audit but exclude them from the live weights.

The tracker is intentionally additive: it never mutates the
scorecards it is given and never throws when a series has no
scorecards. An empty tracker returns an empty ``EnsembleSummary``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from forecasting.contracts import (
    EnsembleSummary,
    ModelFamilyName,
    ModelScorecard,
)


# Threshold above which a family is considered "frequently
# promoted" for a segment. A family that wins 50%+ of the folds
# for a segment has earned a permanent slot in the blend.
_PROMOTION_THRESHOLD = 0.5

# The set of families that are always run by the harness even if
# they have never surfaced. Retiring them outright is a
# cockpit-driven decision; the harness keeps them in the candidate
# list until told otherwise.
PROTECTED_FAMILIES: frozenset[ModelFamilyName] = frozenset(
    {"naive", "seasonal_naive", "croston"}
)


class EnsembleTracker:
    """Track ensemble behaviour for a single run.

    The tracker is fed ``ModelScorecard`` objects (one per
    family x series x fold). The order of insertion does not
    matter: the scorecard is the source of truth for "which
    family won which fold".
    """

    def __init__(
        self,
        *,
        run_id: str = "unknown",
        series_segment: dict[str, str] | None = None,
    ) -> None:
        self._run_id = run_id
        self._series_segment: dict[str, str] = dict(series_segment or {})
        # ``wins[segment][family]`` = number of series for which
        # ``family`` was the best (lowest MAE) scorecard.
        self._wins: dict[str, dict[ModelFamilyName, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # ``families_per_segment[segment]`` = set of families that
        # produced at least one scorecard in that segment. Used to
        # derive "never surfaced" = families that ran but were
        # never the best-in-fold.
        self._families_per_segment: dict[str, set[ModelFamilyName]] = defaultdict(set)
        # ``series_count[segment]`` = number of distinct series
        # that have at least one scorecard in the segment.
        self._series_count: dict[str, set[str]] = defaultdict(set)
        self._retired: set[ModelFamilyName] = set()
        # Accumulate the full scorecard history so the harness can
        # emit the audit log without re-deriving it. The tracker
        # itself never reads this back; downstream code persists it.
        self._scorecards: list[ModelScorecard] = []

    # ----------------- mutation -----------------

    def record(self, scorecard: ModelScorecard) -> None:
        """Record one scorecard.

        The scorecard is added to the history and the per-segment
        win counts are updated. ``series_segment`` is consulted
        to bucket the scorecard into a segment; series that do not
        have a segment mapping land in ``"__default__"`` so the
        tracker never silently drops them.
        """
        self._scorecards.append(scorecard)
        segment = self._series_segment.get(scorecard.series_key, "__default__")
        self._families_per_segment[segment].add(scorecard.model_family)
        self._series_count[segment].add(scorecard.series_key)

    def record_winner(self, series_key: str, family: ModelFamilyName) -> None:
        """Record that ``family`` was the best-in-fold for ``series_key``."""
        segment = self._series_segment.get(series_key, "__default__")
        self._wins[segment][family] += 1
        self._series_count[segment].add(series_key)

    def retire(self, family: ModelFamilyName) -> None:
        """Mark a family as retired.

        Retired families are excluded from the live weights but
        their scorecards stay in the audit history.
        """
        self._retired.add(family)

    # ----------------- queries -----------------

    @property
    def scorecards(self) -> list[ModelScorecard]:
        return list(self._scorecards)

    @property
    def retired(self) -> set[ModelFamilyName]:
        return set(self._retired)

    def weights_for_segment(self, segment: str) -> dict[ModelFamilyName, float]:
        """Return the normalised weight vector for a segment.

        Weights are proportional to the family's win count in this
        segment, ignoring retired families. Protected families
        (naive, seasonal_naive, croston) are guaranteed a minimum
        5% share of the blend even when they have not won a fold
        in the segment yet. When no family has won a fold in the
        segment yet, every active family is given an equal share.
        Families with effectively zero weight after the floor (e.g.
        non-protected never-surfaced) are dropped from the
        returned dict so the blend only ever mixes the families
        that actually earn their keep.

        The floor is enforced via a "take from the leader" pass:
        the protected family's share is clamped to 0.05 of the
        blend and the leader loses weight proportionally. This
        makes the floor visible in the final weights without
        inflating the protected family past the floor.
        """
        active = self._families_per_segment.get(segment, set()) - self._retired
        if not active:
            return {}
        wins = dict(self._wins.get(segment, {}))
        if not wins:
            # No winners yet (every scorecard in this segment was
            # an "I tried but I wasn't best" submission). Give each
            # active family an equal share so the blend is still
            # well-defined.
            share = 1.0 / len(active)
            return {family: share for family in sorted(active)}
        total = sum(wins.values())
        raw_weights: dict[ModelFamilyName, float] = {
            family: wins.get(family, 0) / total for family in active
        }
        # Apply the protected floor. A protected family that has
        # earned less than 5% of the wins is bumped up to 5%, and
        # the difference is taken from the other families
        # proportionally. We iterate the floor pass until every
        # protected family is at the floor, in case multiple
        # protected families are competing for the 5% slots.
        floor = 0.05
        for _ in range(len(PROTECTED_FAMILIES) + 1):
            deficit = 0.0
            for family in PROTECTED_FAMILIES & active:
                if raw_weights[family] < floor:
                    deficit += floor - raw_weights[family]
                    raw_weights[family] = floor
            if deficit == 0:
                break
            donors = [f for f in active if f not in PROTECTED_FAMILIES or raw_weights[f] > floor]
            donor_total = sum(raw_weights[donor] for donor in donors)
            if donor_total <= 0:
                break
            for donor in donors:
                share = raw_weights[donor] / donor_total
                raw_weights[donor] = max(0.0, raw_weights[donor] - share * deficit)
        # Drop effectively-zero weights (non-protected never-surfaced
        # families). The protected families survive the floor.
        non_zero = {family: weight for family, weight in raw_weights.items() if weight > 1e-9}
        return non_zero

    def frequently_promoted(self) -> list[ModelFamilyName]:
        """Families that have been best-in-fold for >= 50% of series in any segment."""
        promoted: set[ModelFamilyName] = set()
        for segment, families in self._families_per_segment.items():
            active = families - self._retired
            if not active:
                continue
            segment_size = max(len(self._series_count.get(segment, set())), 1)
            for family in active:
                wins = self._wins.get(segment, {}).get(family, 0)
                if wins / segment_size >= _PROMOTION_THRESHOLD:
                    promoted.add(family)
        return sorted(promoted)

    def never_surfaced(self) -> list[ModelFamilyName]:
        """Families that ran but were never best-in-fold anywhere.

        A family is "never surfaced" if it produced scorecards in
        at least one segment but never won a fold in that segment.
        A family with no scorecards at all (the harness decided
        not to fit it, e.g. XGBoost on a 1-row history) is NOT
        listed here - the harness reports those separately as the
        ``never_surfaced`` list on the ``ForecastHarnessReport``.
        """
        never: set[ModelFamilyName] = set()
        for segment, families in self._families_per_segment.items():
            active = families - self._retired
            if not active:
                continue
            wins = self._wins.get(segment, {})
            for family in active:
                if wins.get(family, 0) == 0:
                    never.add(family)
        return sorted(never)

    def summary(self) -> EnsembleSummary:
        """Produce the cockpit-facing ``EnsembleSummary`` for the run."""
        weights: dict[str, dict[str, float]] = {}
        for segment in sorted(self._families_per_segment):
            seg_weights = self.weights_for_segment(segment)
            if seg_weights:
                weights[segment] = {family: float(weight) for family, weight in seg_weights.items()}
        return EnsembleSummary(
            weights=weights,
            frequently_promoted=self.frequently_promoted(),
            never_surfaced=self.never_surfaced(),
            retired=sorted(self._retired),
        )


def blend_forecasts(
    family_forecasts: dict[ModelFamilyName, list[float]],
    weights: dict[ModelFamilyName, float],
) -> list[float]:
    """Blend per-family forecasts into a single vector.

    ``family_forecasts`` is ``family -> horizon-long vector``.
    ``weights`` is the same shape as the harness uses. Returns an
    empty list when there is nothing to blend.

    Families that appear in ``family_forecasts`` but not in
    ``weights`` are ignored; families that appear in ``weights``
    but not in ``family_forecasts`` are likewise ignored. This
    keeps the blend well-defined even when the tracker has
    history the harness has not yet fit, or vice versa.
    """
    if not family_forecasts or not weights:
        return []
    horizon = max(len(vec) for vec in family_forecasts.values())
    blended = [0.0] * horizon
    total_weight = 0.0
    for family, weight in weights.items():
        if family not in family_forecasts:
            continue
        vec = family_forecasts[family]
        if len(vec) != horizon:
            continue
        for index, value in enumerate(vec):
            blended[index] += weight * value
        total_weight += weight
    if total_weight == 0:
        return []
    return [value / total_weight for value in blended]


def summarise_scorecards(
    scorecards: Iterable[ModelScorecard],
    *,
    series_segment: dict[str, str] | None = None,
    retired: Iterable[ModelFamilyName] = (),
) -> EnsembleTracker:
    """Build a tracker from a flat list of scorecards.

    Convenience for tests and for the harness's "I already have
    every scorecard in memory" path. The tracker derives the
    best-in-fold winner for each series from the scorecards
    themselves (lowest MAE) so the caller does not have to
    pre-compute winners.
    """
    series_segment = series_segment or {}
    scorecard_list = list(scorecards)
    run_id = "unknown"

    retired_set = set(retired)
    tracker = EnsembleTracker(run_id=run_id, series_segment=series_segment)
    best_per_series: dict[str, ModelScorecard] = {}
    for scorecard in scorecard_list:
        tracker.record(scorecard)
        if scorecard.model_family not in retired_set:
            existing = best_per_series.get(scorecard.series_key)
            if existing is None or scorecard.mae < existing.mae:
                best_per_series[scorecard.series_key] = scorecard
    for scorecard in best_per_series.values():
        tracker.record_winner(scorecard.series_key, scorecard.model_family)
    for family in retired_set:
        tracker.retire(family)
    return tracker


__all__ = [
    "EnsembleTracker",
    "blend_forecasts",
    "summarise_scorecards",
    "PROTECTED_FAMILIES",
]
