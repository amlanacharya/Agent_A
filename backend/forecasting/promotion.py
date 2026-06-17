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

from forecasting.contracts import BacktestWindow


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


__all__ = (
    "LeakageCheck",
    "build_default_backtest_window",
    "check_window_leakage",
)