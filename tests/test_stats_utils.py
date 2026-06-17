"""Tests for the shared per-series stats utilities (Issue #5).

The Phase 1 ``preflight_stats`` and Phase 2 ``eda_probes`` modules used
to each carry their own copy of the same helpers
(``_autocorr`` and the boolean-string set). Both now import from
:mod:`forecasting.stats_utils`. These tests pin the shared contract.
"""

from __future__ import annotations

import numpy as np

from forecasting.canonical_data import CANONICAL_REQUIRED_NON_NULL_COLUMNS
from forecasting.stats_utils import BOOLEAN_FLAG_TEXT, autocorr


def test_autocorr_returns_zero_for_constant_series() -> None:
    """A constant series has std=0 -> the helper returns 0.0 (not NaN).

    Returning 0.0 keeps the EDA report JSON-serialisable; returning
    NaN would break Pydantic validation downstream.
    """
    constant = np.full(20, 5.0)
    assert autocorr(constant, lag=1) == 0.0
    assert autocorr(constant, lag=4) == 0.0


def test_autocorr_returns_zero_for_degenerate_lag() -> None:
    """Lag <= 0 or lag >= length returns 0.0 (not a meaningless number)."""
    x = np.arange(10, dtype=float)
    assert autocorr(x, lag=0) == 0.0
    assert autocorr(x, lag=10) == 0.0
    assert autocorr(x, lag=-1) == 0.0


def test_autocorr_matches_naive_implementation() -> None:
    """The shared helper produces the same value as a hand-rolled
    Pearson correlation on x[:-lag] vs x[lag:]."""
    rng = np.random.default_rng(seed=42)
    x = rng.normal(size=50)
    a = x[:-4]
    b = x[4:]
    expected = float(np.corrcoef(a, b)[0, 1])
    assert autocorr(x, lag=4) == expected


def test_boolean_flag_text_covers_the_canonical_vocabulary() -> None:
    """The shared vocabulary includes the values the canonical layer
    accepts and the probes label as boolean. Adding a new accepted
    flag text is one edit here."""
    expected = {"true", "false", "yes", "no", "y", "n", "1", "0"}
    assert BOOLEAN_FLAG_TEXT == frozenset(expected)


def test_canonical_required_non_null_columns_is_exported() -> None:
    """The canonical layer now exports its required-column tuple so the
    EDA probes no longer inline their own copy.

    This is the seam that lets ``eda_probes._REQUIRED_NON_NULL_COLUMNS``
    derive from the canonical contract: adding a new required column
    to the contract is one edit in ``canonical_data``, and the
    probes' ``rows_with_missing`` metric picks it up automatically.
    """
    assert "sku_id" in CANONICAL_REQUIRED_NON_NULL_COLUMNS
    assert "location_id" in CANONICAL_REQUIRED_NON_NULL_COLUMNS
    assert "week_start" in CANONICAL_REQUIRED_NON_NULL_COLUMNS
    assert "demand_qty" in CANONICAL_REQUIRED_NON_NULL_COLUMNS
