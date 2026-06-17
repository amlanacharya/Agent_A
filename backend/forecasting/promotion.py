"""Champion/challenger promotion layer for Phase 5.2.

The promotion layer takes a candidate model and the current
champion, runs both on the same fixed backtest window, and
returns a typed comparison. The actual decision (promote or
reject) is a separate concern. The function shape:

* ``build_default_backtest_window(forecast_horizon, ...)`` —
  the platform default window: N cutoffs each ``horizon`` apart,
  starting from the most-recent cutoff working backwards.
* ``check_window_leakage(window, canonical_table_end)`` —
  detect when a candidate's scorecard cutoffs include dates
  after the canonical table's last-known date (which would
  mean the candidate was scored on data the platform doesn't
  have).

CB1 ships the window + leakage check. CB2 adds the
candidate/champion contracts + comparison function. CB3 adds
the shadow-mode runner. CB4 adds the PROMOTION_DECISIONS.md
generator.

Design rules:

* **Pure functions, no I/O.** The window is built from
  parameters; the leakage check is a function of (window,
  end_date). Tests pass synthetic inputs and assert outputs.
* **Closed Literal surface.** The leakage check returns one of
  a small set of ``LeakageCheck`` outcomes, never a free-text
  string.
* **Reuse the metric portfolio.** Phase 5.1's ``wape`` and
  per-step helpers are the inputs to the comparison function
  (CB2); this module only defines the window spec and the
  leakage check in CB1.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel

from forecasting.contracts import (
    BacktestWindow,
    ModelFamilyName,
    ModelScorecard,
)
from forecasting.metrics import wape


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_default_backtest_window(
    *,
    forecast_horizon: int,
    num_cutoffs: int = 4,
    end: datetime | None = None,
    spacing_units: int = 1,
) -> BacktestWindow:
    """Build the platform default ``BacktestWindow``.

    The default is N cutoffs each ``forecast_horizon`` apart,
    ending at ``end`` (default: now). ``start`` is
    ``end - (num_cutoffs * forecast_horizon * spacing_units)``.
    Each cutoff is on the boundary between folds so a fold has
    a full ``forecast_horizon`` of held-out data on each side.

    Parameters
    ----------
    forecast_horizon
        The forecast horizon in the same time unit as the
        cutoff timestamps. A weekly forecast on a 4-week
        horizon passes ``forecast_horizon=4``.
    num_cutoffs
        Number of cutoffs to include. The platform default is
        4 (Phase 2 says "minimum 2 folds required for valid
        walk-forward validation", so 4 is comfortably above the
        floor).
    end
        The most-recent cutoff. ``None`` means "now" (UTC).
        The window's ``end`` field is set to ``end`` (the
        window's outer boundary, not the cutoff list's end).
    spacing_units
        Number of time-units between adjacent cutoffs.
        Defaults to 1 (consecutive cutoffs are 1 horizon apart).
        A value of 2 spreads the cutoffs further apart (every
        other horizon), useful when the data is too dense for
        a 4-cutoff default.

    Notes
    -----
    The timestamps are stored as ISO-8601 strings (matching
    ``ModelScorecard.fold_cutoff``). The harness interprets
    them with ``pd.Timestamp``; the comparison layer does not
    need to do timestamp arithmetic itself.
    """
    if num_cutoffs < 2:
        raise ValueError(
            f"num_cutoffs must be >= 2 (the Phase 2 walk-forward "
            f"validation floor); got {num_cutoffs}"
        )
    if forecast_horizon < 1:
        raise ValueError(
            f"forecast_horizon must be >= 1; got {forecast_horizon}"
        )
    if end is None:
        end = datetime.now()
    # ``end`` is the most-recent cutoff. The window's outer end
    # is one horizon past the most-recent cutoff (the fold
    # after the last cutoff has a full horizon to score on).
    # The window's outer start is num_cutoffs * horizon back
    # from the most-recent cutoff.
    window_end = end + timedelta(days=forecast_horizon * spacing_units)
    window_start = end - timedelta(days=forecast_horizon * spacing_units * num_cutoffs)
    cutoffs = [
        (end - timedelta(days=forecast_horizon * spacing_units * i)).isoformat()
        for i in range(num_cutoffs)
    ]
    return BacktestWindow(
        start=window_start.isoformat(),
        end=window_end.isoformat(),
        cutoffs=cutoffs,
        horizon=forecast_horizon,
    )


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------


LeakageCheck = Literal["clean", "future_cutoff", "out_of_order", "empty_cutoffs"]


def check_window_leakage(
    window: BacktestWindow,
    *,
    canonical_table_end: str,
) -> LeakageCheck:
    """Detect cutoffs that fall after the canonical table's last-known date.

    A candidate whose scorecards were generated from cutoffs
    after the canonical table's last-known date is scored on
    data the platform does not actually have — a leakage red
    flag. The promotion layer must reject such candidates
    before any comparison.

    Outcomes:

    * ``"clean"`` — every cutoff is at or before
      ``canonical_table_end``.
    * ``"future_cutoff"`` — at least one cutoff is strictly
      after ``canonical_table_end``.
    * ``"out_of_order"`` — the cutoffs are not in non-increasing
      chronological order. (The builder produces them in
      chronological order, but a manually-constructed window
      might not.)
    * ``"empty_cutoffs"`` — the window has no cutoffs at all.
      The platform default builder prevents this, but a
      manually-constructed window may not.

    The function is pure and side-effect-free. The caller
    decides what to do with a non-clean outcome (typically:
    refuse to compare, surface a ``leakage_check_failed`` reason
    on the ``PromotionComparison``).
    """
    if not window.cutoffs:
        return "empty_cutoffs"
    parsed: list[datetime] = []
    for cutoff in window.cutoffs:
        try:
            parsed.append(datetime.fromisoformat(cutoff))
        except (TypeError, ValueError):
            # Unparseable cutoff is treated as "out of order" —
            # it cannot be a real time, so it cannot be on the
            # right side of any ordering check.
            return "out_of_order"
    end = datetime.fromisoformat(canonical_table_end)
    if any(c > end for c in parsed):
        return "future_cutoff"
    # Non-increasing order: each cutoff <= the previous one.
    for previous, current in zip(parsed, parsed[1:]):
        if current > previous:
            return "out_of_order"
    return "clean"


# ---------------------------------------------------------------------------
# Promotion candidate / champion / comparison (Phase 5.2 CB2)
# ---------------------------------------------------------------------------
# CB2 ships the actual side-by-side scoring. CB1 set the window
# spec; CB2 takes two sets of scorecards (candidate's and
# champion's) and returns a typed ``PromotionComparison`` record.
#
# Why the segment_map is a kwarg and not on the scorecard:
# ``ModelScorecard`` is a per-(model, series, fold) record. The
# segment_id is a (run-level) lookup, not a per-scorecard field,
# because the same scorecard can belong to different segments
# depending on which run produced it. Passing the map at the
# call site keeps the scorecard type stable and the comparison
# function pure.


PromotionOutcome = Literal["promote", "reject", "leakage_failed", "human_required"]


class PromotionCandidate(BaseModel):
    """A model proposed for promotion to champion.

    Carries the scorecards the candidate produced on the
    ``BacktestWindow``, plus metadata for the audit log.
    ``reason`` is a free-text explanation of why this
    candidate was generated (e.g. "from CB3 propose_feature_changes").
    """

    run_id: str
    model_family: ModelFamilyName
    scorecards: list[ModelScorecard]
    reason: str


class Champion(BaseModel):
    """The current production model. Carries its scorecards on the same window."""

    model_family: ModelFamilyName
    scorecards: list[ModelScorecard]
    promoted_at: datetime


class PromotionComparison(BaseModel):
    """The outcome of one candidate vs champion comparison.

    ``promotion_outcome`` is the closed Literal decision:
    ``promote`` when the candidate improves WAPE past
    ``min_improvement`` and does not regress any segment,
    ``reject`` otherwise, ``leakage_failed`` when the
    ``BacktestWindow`` itself fails the leakage check,
    ``human_required`` when the comparison cannot be decided
    by metrics alone (reserved for future use).
    """

    candidate_wape: float
    champion_wape: float
    wape_delta: float  # negative = candidate better
    segments_compared: list[str]
    segments_improved: list[str]
    segments_regressed: list[str]
    promotion_outcome: PromotionOutcome


def _scorecards_in_window(
    scorecards: list[ModelScorecard],
    window: BacktestWindow,
) -> list[ModelScorecard]:
    """Filter a scorecard list to those whose fold_cutoff is in the window.

    The comparison must use exactly the same cutoffs for
    candidate and champion, so anything outside the window is
    silently dropped. Returns an empty list when no scorecard
    matches (a degenerate case the caller must surface).
    """
    in_window = set(window.cutoffs)
    return [s for s in scorecards if s.fold_cutoff in in_window]


def _per_segment_wape(
    scorecards: list[ModelScorecard],
    segment_map: dict[str, str],
) -> dict[str, float]:
    """WAPE per segment, keyed by segment_id.

    Returns ``{}`` for a scorecard whose series_key is not in
    the segment map. The caller decides whether that is a
    silent skip or a hard error (today: silent skip, with
    the unsegmented series excluded from the per-segment
    rollup).
    """
    grouped: dict[str, list[ModelScorecard]] = {}
    for card in scorecards:
        segment_id = segment_map.get(card.series_key)
        if segment_id is None:
            continue
        grouped.setdefault(segment_id, []).append(card)
    return {seg: wape(cards) for seg, cards in grouped.items()}


def compare_candidate_to_champion(
    candidate: PromotionCandidate,
    champion: Champion,
    *,
    window: BacktestWindow,
    canonical_table_end: str,
    segment_map: dict[str, str],
    min_improvement: float = 0.0,
) -> PromotionComparison:
    """Compare a candidate to the champion on the fixed ``BacktestWindow``.

    Steps:

    1. Run the leakage check. A non-clean outcome short-circuits
       to a ``leakage_failed`` comparison (no WAPE delta).
    2. Filter both scorecard lists to the window's cutoffs.
    3. Compute WAPE for the candidate and the champion
       (Phase 5.1's ``wape``).
    4. Compute per-segment WAPE for both, then per-segment
       delta. ``segments_improved`` are the segments where the
       candidate's WAPE dropped; ``segments_regressed`` are
       the segments where it rose.
    5. Decide the outcome: ``promote`` when ``wape_delta <=
       -min_improvement`` AND no segment regressed. Otherwise
       ``reject``. ``human_required`` is reserved for cases
       where the platform defers the call to a human reviewer
       (today: not emitted; future Phase 6 hook).

    The ``min_improvement`` default is 0 (a tie goes to the
    champion — no change). A positive value (e.g. 0.01) means
    "the candidate must improve WAPE by at least 1% absolute
    to be promoted", a safety floor the platform can tune
    via ``PROMOTION_MIN_IMPROVEMENT`` in ``.env``.
    """
    leakage = check_window_leakage(
        window, canonical_table_end=canonical_table_end
    )
    if leakage != "clean":
        # The leakage check failed. Surface the outcome with no
        # numeric comparison — the candidate is rejected before
        # any WAPE delta is computed.
        return PromotionComparison(
            candidate_wape=float("nan"),
            champion_wape=float("nan"),
            wape_delta=float("nan"),
            segments_compared=[],
            segments_improved=[],
            segments_regressed=[],
            promotion_outcome="leakage_failed",
        )

    candidate_cards = _scorecards_in_window(candidate.scorecards, window)
    champion_cards = _scorecards_in_window(champion.scorecards, window)
    candidate_wape = wape(candidate_cards)
    champion_wape = wape(champion_cards)
    # wape_delta is positive when the candidate is worse (its
    # WAPE is higher than the champion's). A negative delta
    # means the candidate is better.
    wape_delta = candidate_wape - champion_wape

    candidate_segments = _per_segment_wape(candidate_cards, segment_map)
    champion_segments = _per_segment_wape(champion_cards, segment_map)
    segments_compared = sorted(set(candidate_segments) | set(champion_segments))
    segments_improved: list[str] = []
    segments_regressed: list[str] = []
    for seg in segments_compared:
        cand_seg = candidate_segments.get(seg, float("nan"))
        champ_seg = champion_segments.get(seg, float("nan"))
        # NaN-safe comparison: a NaN segment is treated as
        # neither improved nor regressed (the segment is
        # missing data on one side).
        if cand_seg != cand_seg or champ_seg != champ_seg:
            continue
        if cand_seg < champ_seg:
            segments_improved.append(seg)
        elif cand_seg > champ_seg:
            segments_regressed.append(seg)

    # Promote when the overall WAPE improved past the
    # threshold AND no segment regressed. A regressed segment
    # is a hard fail — the candidate might be better on
    # average but worse on the segments that matter.
    #
    # Strict ``<`` (not ``<=``): a tie (wape_delta == 0) does not
    # promote. With min_improvement=0, the rule reduces to
    # "promote only on strict WAPE improvement"; a candidate
    # that matches the champion's WAPE exactly does not displace
    # the champion. This is the docstring's "tie goes to the
    # champion -- no change" semantic.
    promoted = wape_delta < -min_improvement and not segments_regressed
    outcome: PromotionOutcome = "promote" if promoted else "reject"
    return PromotionComparison(
        candidate_wape=candidate_wape,
        champion_wape=champion_wape,
        wape_delta=wape_delta,
        segments_compared=segments_compared,
        segments_improved=segments_improved,
        segments_regressed=segments_regressed,
        promotion_outcome=outcome,
    )


# ---------------------------------------------------------------------------
# Shadow-mode runner (Phase 5.2 CB3)
# ---------------------------------------------------------------------------
# Shadow mode runs both models on the same canonical table and
# compares their forecasts step-by-step. It is *not* a metric
# comparison (the WAPE / per-segment WAPE from CB2 cover that).
# It is a check on whether the candidate's forecasts are
# numerically close to the champion's — a large disagreement
# at the same (series, fold) is a red flag that the
# candidate is producing qualitatively different output, even
# if its average WAPE is comparable.
#
# Design rules:
#
# * **Pure function.** No I/O, no LLM, no harness re-run.
#   The scorecards are inputs; the function returns a typed
#   ``ShadowModeResult``. The scorecards are produced by the
#   same canonical table (the candidate and champion were
#   both scored on the same folds), so the comparison is
#   well-defined.
# * **Per-series pairs.** The output is a dict[series_key,
#   list[(candidate_forecast, champion_forecast)]] — one
#   tuple per forecast step. The list length matches the
#   scorecard's horizon. The audit can grep the result for
#   specific (series, step) disagreements.
# * **Relative agreement.** A pair is "in agreement" when
#   ``|cand - champ| / max(|champ|, eps) <= tolerance``. The
#   ``max(..., eps)`` branch handles the champion-forecast-
#   equals-zero case where relative error is undefined.
# * **No window filter.** Shadow mode is independent of
#   ``BacktestWindow`` — every scorecard on each side is
#   compared. The window is a comparison constraint; shadow
#   mode is an inspection tool.


# Threshold below which |champion| is treated as "zero" for the
# relative-error denominator. Without this, a champion
# forecast of exactly 0 would make relative error explode
# (division by zero) or every disagreement look like 100%.
_ZERO_EPS = 1e-9


class ShadowModeResult(BaseModel):
    """The outcome of one shadow-mode run.

    ``per_series_pairs`` maps ``series_key`` to a list of
    ``(candidate_forecast, champion_forecast)`` tuples, one
    per forecast step. The audit uses this to inspect any
    specific disagreement.

    ``agreement_rate`` is the fraction of (series, step)
    pairs where the candidate and champion forecasts are
    within ``tolerance`` of each other (relative tolerance,
    ``|delta| / max(|champion|, eps) <= tolerance``).
    1.0 = perfect agreement; 0.0 = total disagreement.
    """

    candidate_family: ModelFamilyName
    champion_family: ModelFamilyName
    tolerance: float
    per_series_pairs: dict[str, list[tuple[float, float]]]
    agreement_rate: float


def _index_scorecards_by_series(
    scorecards: list[ModelScorecard],
) -> dict[str, list[ModelScorecard]]:
    """Group scorecards by ``series_key``.

    The function emits one pair-list per series; multiple
    folds per series produce multiple scorecards that get
    paired fold-by-fold (best-effort: same fold_cutoff first,
    then any leftover).
    """
    out: dict[str, list[ModelScorecard]] = {}
    for card in scorecards:
        out.setdefault(card.series_key, []).append(card)
    return out


def _pair_forecasts(
    candidate_cards: list[ModelScorecard],
    champion_cards: list[ModelScorecard],
) -> list[tuple[float, float]]:
    """Pair (candidate, champion) forecasts step-by-step.

    Pairs by ``fold_cutoff`` first (the natural fold alignment),
    falling back to positional pairing for scorecards whose
    fold_cutoff doesn't match. The result length is the
    candidate's horizon (the runner treats the candidate's
    list as the canonical ordering — if the lists differ in
    length, the longer is truncated).

    Returns a flat list of (candidate_step, champion_step)
    tuples across all folds. The caller splits by series.
    """
    by_cutoff: dict[str, ModelScorecard] = {c.fold_cutoff: c for c in champion_cards}
    pairs: list[tuple[float, float]] = []
    for cand_card in candidate_cards:
        champ_card = by_cutoff.get(cand_card.fold_cutoff)
        if champ_card is None:
            # No matching fold on the champion side. Pair
            # positionally with the next unused champion card.
            # In practice this is rare because both sides were
            # scored on the same canonical table.
            for i in range(min(len(cand_card.forecast), len(champion_cards[i].forecast) if i < len(champion_cards) else 0)):
                pairs.append((cand_card.forecast[i], champion_cards[i].forecast[i]))
            continue
        n = min(len(cand_card.forecast), len(champ_card.forecast))
        for i in range(n):
            pairs.append((cand_card.forecast[i], champ_card.forecast[i]))
    return pairs


def run_shadow_mode(
    candidate: PromotionCandidate,
    champion: Champion,
    *,
    tolerance: float = 0.05,
) -> ShadowModeResult:
    """Run shadow mode: pair the candidate's and champion's forecasts step-by-step.

    Parameters
    ----------
    candidate
        The candidate ``PromotionCandidate``. Its scorecards
        are the "shadow" output (the candidate's forecasts
        produced on the same canonical table as the champion's).
    champion
        The current production ``Champion``. Its scorecards
        are the baseline.
    tolerance
        Relative agreement threshold. A pair is "in agreement"
        when ``|cand - champ| / max(|champion|, eps) <= tolerance``.
        Default 0.05 (5% relative disagreement is the cutoff).

    Returns
    -------
    ShadowModeResult
        The per-series pair list and the overall agreement
        rate. Pure function: same inputs -> same result.
    """
    cand_by_series = _index_scorecards_by_series(candidate.scorecards)
    champ_by_series = _index_scorecards_by_series(champion.scorecards)

    per_series_pairs: dict[str, list[tuple[float, float]]] = {}
    total_pairs = 0
    agreeing_pairs = 0
    # Iterate the union of series keys so a series present
    # on only one side still appears in the output (with
    # whatever pairs we can compute — none if the other side
    # is missing, so it contributes zero to the agreement
    # rate).
    for series_key in sorted(set(cand_by_series) | set(champ_by_series)):
        cand_cards = cand_by_series.get(series_key, [])
        champ_cards = champ_by_series.get(series_key, [])
        pairs = _pair_forecasts(cand_cards, champ_cards)
        per_series_pairs[series_key] = pairs
        total_pairs += len(pairs)
        for cand_step, champ_step in pairs:
            # Relative disagreement, with the eps branch to
            # handle champion-forecast-equals-zero.
            denom = max(abs(champ_step), _ZERO_EPS)
            if abs(cand_step - champ_step) / denom <= tolerance:
                agreeing_pairs += 1
    agreement_rate = agreeing_pairs / total_pairs if total_pairs else 0.0
    return ShadowModeResult(
        candidate_family=candidate.model_family,
        champion_family=champion.model_family,
        tolerance=tolerance,
        per_series_pairs=per_series_pairs,
        agreement_rate=agreement_rate,
    )


__all__ = (
    "LeakageCheck",
    "PromotionCandidate",
    "Champion",
    "PromotionComparison",
    "PromotionOutcome",
    "ShadowModeResult",
    "build_default_backtest_window",
    "check_window_leakage",
    "compare_candidate_to_champion",
    "run_shadow_mode",
)