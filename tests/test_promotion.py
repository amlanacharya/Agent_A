"""Tests for ``forecasting.promotion`` Phase 5.2 CB1.

The window spec and leakage check are pure functions on a
typed contract. Tests pin the builder's defaults, the
builder's edge cases, and every branch of the leakage check.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from forecasting.contracts import BacktestWindow
from forecasting.promotion import (
    build_default_backtest_window,
    check_window_leakage,
)


# ---------------------------------------------------------------------------
# build_default_backtest_window
# ---------------------------------------------------------------------------


def test_build_default_window_has_num_cutoffs_cutoffs() -> None:
    """The default builder produces exactly ``num_cutoffs`` cutoffs."""
    end = datetime(2026, 6, 17, tzinfo=timezone.utc)
    window = build_default_backtest_window(forecast_horizon=4, num_cutoffs=4, end=end)
    assert len(window.cutoffs) == 4


def test_build_default_window_cutoffs_are_in_chrono_order() -> None:
    """Cutoffs are emitted most-recent-first; parsed they are non-increasing."""
    end = datetime(2026, 6, 17, tzinfo=timezone.utc)
    window = build_default_backtest_window(forecast_horizon=4, num_cutoffs=4, end=end)
    parsed = [datetime.fromisoformat(c) for c in window.cutoffs]
    for previous, current in zip(parsed, parsed[1:]):
        assert current <= previous


def test_build_default_window_cutoffs_are_horizon_apart() -> None:
    """Adjacent cutoffs differ by exactly ``forecast_horizon * spacing_units`` days."""
    end = datetime(2026, 6, 17, tzinfo=timezone.utc)
    window = build_default_backtest_window(forecast_horizon=4, num_cutoffs=4, end=end)
    parsed = [datetime.fromisoformat(c) for c in window.cutoffs]
    deltas = [
        (parsed[i] - parsed[i + 1]).days for i in range(len(parsed) - 1)
    ]
    for d in deltas:
        assert d == 4  # forecast_horizon=4, spacing_units=1


def test_build_default_window_horizon_field_matches_argument() -> None:
    window = build_default_backtest_window(forecast_horizon=7, num_cutoffs=3)
    assert window.horizon == 7


def test_build_default_window_spacing_units_respected() -> None:
    """spacing_units=2 doubles the gap between cutoffs."""
    end = datetime(2026, 6, 17, tzinfo=timezone.utc)
    window = build_default_backtest_window(
        forecast_horizon=4, num_cutoffs=4, end=end, spacing_units=2
    )
    parsed = [datetime.fromisoformat(c) for c in window.cutoffs]
    deltas = [
        (parsed[i] - parsed[i + 1]).days for i in range(len(parsed) - 1)
    ]
    for d in deltas:
        assert d == 8  # forecast_horizon=4, spacing_units=2


def test_build_default_window_rejects_num_cutoffs_below_2() -> None:
    """The Phase 2 walk-forward validation floor is 2 folds."""
    with pytest.raises(ValueError, match="num_cutoffs must be >= 2"):
        build_default_backtest_window(forecast_horizon=4, num_cutoffs=1)


def test_build_default_window_rejects_zero_horizon() -> None:
    with pytest.raises(ValueError, match="forecast_horizon must be >= 1"):
        build_default_backtest_window(forecast_horizon=0, num_cutoffs=4)


def test_build_default_window_end_defaults_to_now_when_omitted() -> None:
    """When end is None, the most-recent cutoff is approximately now."""
    before = datetime.now()
    window = build_default_backtest_window(forecast_horizon=4, num_cutoffs=2)
    after = datetime.now()
    most_recent = datetime.fromisoformat(window.cutoffs[0])
    # The most-recent cutoff should be within the before/after window.
    assert before <= most_recent <= after + timedelta(seconds=1)


def test_build_default_window_start_and_end_are_set() -> None:
    """The window's outer start and end are populated (not just the cutoffs)."""
    end = datetime(2026, 6, 17, tzinfo=timezone.utc)
    window = build_default_backtest_window(forecast_horizon=4, num_cutoffs=4, end=end)
    # start < earliest cutoff < most-recent cutoff < end.
    parsed_start = datetime.fromisoformat(window.start)
    parsed_end = datetime.fromisoformat(window.end)
    earliest_cutoff = datetime.fromisoformat(window.cutoffs[-1])
    most_recent_cutoff = datetime.fromisoformat(window.cutoffs[0])
    assert parsed_start < earliest_cutoff
    assert most_recent_cutoff < parsed_end


# ---------------------------------------------------------------------------
# check_window_leakage
# ---------------------------------------------------------------------------


def _window(*cutoffs: str, horizon: int = 4) -> BacktestWindow:
    """Build a BacktestWindow from raw cutoff strings."""
    return BacktestWindow(
        start="2025-01-01T00:00:00",
        end="2026-12-31T00:00:00",
        cutoffs=list(cutoffs),
        horizon=horizon,
    )


def test_leakage_check_clean_when_all_cutoffs_are_past() -> None:
    """Every cutoff <= canonical_table_end -> clean.

    Note: cutoffs are emitted newest-first by
    ``build_default_backtest_window``; the leakage check
    accepts that order.
    """
    window = _window(
        "2026-03-01T00:00:00",
        "2026-02-01T00:00:00",
        "2026-01-01T00:00:00",
    )
    assert check_window_leakage(window, canonical_table_end="2026-06-17T00:00:00") == "clean"


def test_leakage_check_future_cutoff_detected() -> None:
    """A cutoff after the canonical table's end is a leakage red flag."""
    window = _window(
        "2026-08-01T00:00:00",  # future
        "2026-01-01T00:00:00",
    )
    assert check_window_leakage(window, canonical_table_end="2026-06-17T00:00:00") == "future_cutoff"


def test_leakage_check_out_of_order_detected() -> None:
    """Cutoffs not in non-increasing chronological order -> out_of_order.

    Cutoffs must be newest-first (matches the builder's output).
    Ascending order is flagged as out_of_order so a future
    reader sees the convention violation immediately.
    """
    window = _window(
        "2026-01-01T00:00:00",
        "2026-03-01T00:00:00",  # later than the previous one
    )
    assert check_window_leakage(window, canonical_table_end="2026-06-17T00:00:00") == "out_of_order"


def test_leakage_check_empty_cutoffs_reported() -> None:
    """An empty cutoff list is its own outcome, not a leakage-free clean."""
    window = _window()
    assert check_window_leakage(window, canonical_table_end="2026-06-17T00:00:00") == "empty_cutoffs"


def test_leakage_check_single_cutoff_is_in_order() -> None:
    """A single cutoff trivially satisfies the ordering check."""
    window = _window("2026-01-01T00:00:00")
    assert check_window_leakage(window, canonical_table_end="2026-06-17T00:00:00") == "clean"


def test_leakage_check_cutoff_at_boundary_is_clean() -> None:
    """A cutoff exactly at the canonical table's end is the boundary case.

    ``<=`` not ``<`` — the cutoff AT the table's end is the
    last fully-observed fold. Cutoffs are newest-first.
    """
    window = _window(
        "2026-06-17T00:00:00",  # exactly at the boundary (most recent)
        "2026-01-01T00:00:00",
    )
    assert check_window_leakage(window, canonical_table_end="2026-06-17T00:00:00") == "clean"


def test_leakage_check_unparseable_cutoff_reports_out_of_order() -> None:
    """An unparseable cutoff string is treated as out_of_order (cannot be on any ordering)."""
    window = _window(
        "not-a-timestamp",
        "2026-01-01T00:00:00",
    )
    assert check_window_leakage(window, canonical_table_end="2026-06-17T00:00:00") == "out_of_order"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_build_default_window_is_deterministic_given_end() -> None:
    """Same end -> same window string. The audit must be reproducible."""
    end = datetime(2026, 6, 17, tzinfo=timezone.utc)
    w1 = build_default_backtest_window(forecast_horizon=4, num_cutoffs=4, end=end)
    w2 = build_default_backtest_window(forecast_horizon=4, num_cutoffs=4, end=end)
    assert w1.start == w2.start
    assert w1.end == w2.end
    assert w1.cutoffs == w2.cutoffs
    assert w1.horizon == w2.horizon