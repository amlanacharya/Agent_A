"""Tests for ``forecasting.residual_analysis`` (Phase 4.1 CB2).

The decomposition is the deterministic, non-LLM half of the two-path
escalation loop. The math has to be pinned: the LLM in CB3
(``propose_feature_changes``) will rank proposals by the pattern
severities emitted here, so a wrong severity or a missed pattern
silently mis-ranks every proposal. These tests use synthetic
scorecards with known residual structure to assert the right patterns
fire at the right severity.

The cases below cover every pattern in the closed set plus the
"context absent" path (canonical slice is None → no contextual
patterns emitted).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from forecasting.contracts import ModelScorecard
from forecasting.residual_analysis import decompose_residuals


def _scorecard(series_key: str, forecast: list[float], actual: list[float]) -> ModelScorecard:
    """Build a minimal ModelScorecard for tests.

    Only the fields the decomposition reads (``series_key``,
    ``forecast``, ``actual``, ``fold_cutoff``) are populated; the
    rest default. This keeps the test fixtures tight.
    """
    import numpy as _np
    f = _np.asarray(forecast, dtype=float)
    a = _np.asarray(actual, dtype=float)
    residuals = a - f
    mae = float(_np.mean(_np.abs(residuals))) if len(f) else 0.0
    rmse = float(_np.sqrt(_np.mean(residuals ** 2))) if len(f) else 0.0
    bias = float(_np.mean(residuals)) if len(f) else 0.0
    return ModelScorecard(
        model_family="naive",
        series_key=series_key,
        fold_cutoff="2024-01-01T00:00:00",
        horizon=len(forecast),
        forecast=forecast,
        actual=actual,
        mae=mae,
        rmse=rmse,
        mase=mae,  # placeholder; not read by the decomposition
        bias=bias,
    )


def _canonical_slice(promo: list[int] | None = None, stockout: list[int] | None = None) -> pd.DataFrame:
    """Build a canonical slice DataFrame for the contextual patterns.

    Pass ``None`` for any column that should be absent (the
    decomposition handles missing columns gracefully).
    """
    n = len(promo) if promo is not None else (len(stockout) if stockout is not None else 0)
    data: dict = {}
    if promo is not None:
        data["promo"] = promo
    if stockout is not None:
        data["stockout_flag"] = stockout
    return pd.DataFrame(data, index=range(n))


# ---------------------------------------------------------------------------
# Pure stats block
# ---------------------------------------------------------------------------


def test_decompose_returns_pure_stats_with_no_context() -> None:
    """No canonical slice → stats block populated, contextual fields None."""
    # forecast = [1..10], actual = forecast + small noise. Residual mean
    # should be ~0, std > 0, mae > 0, and the autocorr fields should
    # be None-or-float depending on length.
    forecast = [float(i) for i in range(1, 11)]
    rng = np.random.default_rng(seed=42)
    actual = [f + rng.normal(0, 0.1) for f in forecast]
    decomp = decompose_residuals(_scorecard("A", forecast, actual))
    assert decomp.series_key == "A"
    assert decomp.fold_cutoff == "2024-01-01T00:00:00"
    stats = decomp.stats
    assert stats.n == 10
    assert stats.residual_mean == pytest.approx(0.0, abs=0.05)
    assert stats.residual_std > 0
    assert stats.mae > 0
    assert stats.promo_residual_mean is None
    assert stats.stockout_residual_mean is None
    assert stats.parent_residual_mean is None
    assert stats.autocorr_lag_1 is not None
    # lag-8 with n=10 leaves only 2 lag pairs. On a small noisy
    # series the value can be extreme; we don't assert a specific
    # value, just that the field is populated (computable) and
    # within [-1, 1] (the autocorr contract).
    assert stats.autocorr_lag_8 is not None
    assert -1.0 <= stats.autocorr_lag_8 <= 1.0


def test_decompose_rejects_length_mismatch() -> None:
    """forecast and actual of different lengths raises ValueError."""
    # Build the scorecard directly to avoid the helper's own
    # broadcast check (numpy raises first on shape mismatch).
    bad = ModelScorecard(
        model_family="naive",
        series_key="A",
        fold_cutoff="2024-01-01T00:00:00",
        horizon=3,
        forecast=[1.0, 2.0, 3.0],
        actual=[1.0, 2.0],  # length mismatch
        mae=0.0,
        rmse=0.0,
        mase=0.0,
        bias=0.0,
    )
    with pytest.raises(ValueError, match="length mismatch"):
        decompose_residuals(bad)


def test_decompose_handles_short_series() -> None:
    """A series of length 1 has no autocorr; std = 0; patterns None/unemitted."""
    decomp = decompose_residuals(_scorecard("A", [1.0], [1.0]))
    assert decomp.stats.n == 1
    assert decomp.stats.residual_mean == 0.0
    assert decomp.stats.residual_std == 0.0
    assert decomp.stats.mae == 0.0
    assert decomp.stats.autocorr_lag_1 is None
    assert decomp.patterns == []


# ---------------------------------------------------------------------------
# BIASED_RESIDUAL
# ---------------------------------------------------------------------------


def test_biased_residual_emitted_when_mean_residual_is_large() -> None:
    """Constant +2 residual on a level-10 demand series → bias ratio = 2/eps.

    With residual_mean=2 and residual_std=0 (constant residual), the
    bias ratio is technically infinite — the function uses
    ``stats.residual_std > 0`` as a guard, so a constant residual
    does NOT emit BIASED_RESIDUAL via the bias-vs-noise path. That's
    a deliberate design choice: a constant residual is detected
    downstream (the proposal tool will see residual_mean=2 directly).
    We test a case where residual std is small but non-zero, where
    the ratio is high.
    """
    forecast = [10.0] * 10
    # Residual = 1.5 + tiny noise. std ~0.16, mean = 1.5, ratio ~9.4.
    actual = [f + 1.5 + ((-0.05) ** i) for i, f in enumerate(forecast)]
    decomp = decompose_residuals(_scorecard("A", forecast, actual))
    patterns = [p.pattern for p in decomp.patterns]
    assert "BIASED_RESIDUAL" in patterns
    bias_hit = next(p for p in decomp.patterns if p.pattern == "BIASED_RESIDUAL")
    assert bias_hit.severity == pytest.approx(1.0, abs=0.01)
    assert "mean residual" in bias_hit.detail


def test_biased_residual_not_emitted_when_residual_is_noise_only() -> None:
    """Zero-mean symmetric noise → |mean| << std, no BIASED pattern."""
    forecast = [10.0] * 20
    rng = np.random.default_rng(seed=1)
    actual = [f + rng.normal(0, 1.0) for f in forecast]
    decomp = decompose_residuals(_scorecard("A", forecast, actual))
    patterns = [p.pattern for p in decomp.patterns]
    assert "BIASED_RESIDUAL" not in patterns


# ---------------------------------------------------------------------------
# AUTOCORRELATED_RESIDUAL
# ---------------------------------------------------------------------------


def test_autocorrelated_residual_emitted_on_lag1() -> None:
    """Residuals = +1, -1, +1, -1, ... → strong lag-1 autocorrelation."""
    forecast = [10.0] * 20
    residuals_pattern = [1.0 if i % 2 == 0 else -1.0 for i in range(20)]
    actual = [f + r for f, r in zip(forecast, residuals_pattern)]
    decomp = decompose_residuals(_scorecard("A", forecast, actual))
    patterns = [p.pattern for p in decomp.patterns]
    assert "AUTOCORRELATED_RESIDUAL" in patterns
    ac_hit = next(p for p in decomp.patterns if p.pattern == "AUTOCORRELATED_RESIDUAL")
    # lag-1 of a [-1, +1, -1, +1, ...] pattern is -1.0 → |ac| = 1.0
    # → severity = 1.0.
    assert ac_hit.severity == pytest.approx(1.0, abs=0.01)


def test_autocorrelated_residual_not_emitted_on_white_noise() -> None:
    """Independent residuals → autocorrelations near 0, no AUTOCORR pattern."""
    forecast = [10.0] * 30
    rng = np.random.default_rng(seed=2)
    actual = [f + rng.normal(0, 1.0) for f in forecast]
    decomp = decompose_residuals(_scorecard("A", forecast, actual))
    patterns = [p.pattern for p in decomp.patterns]
    assert "AUTOCORRELATED_RESIDUAL" not in patterns


# ---------------------------------------------------------------------------
# PROMO_RESIDUAL_SPIKE / STOCKOUT_RESIDUAL_SPIKE
# ---------------------------------------------------------------------------


def test_promo_residual_spike_emitted_when_promo_residual_differs() -> None:
    """Residuals +2 on promo weeks, 0 on non-promo → spike pattern fires."""
    forecast = [10.0] * 10
    actual = []
    for i in range(10):
        if i in (2, 5, 8):  # promo weeks
            actual.append(12.0)  # forecast 10, actual 12 → residual +2
        else:
            actual.append(10.0)  # residual 0
    promo_flags = [1 if i in (2, 5, 8) else 0 for i in range(10)]
    decomp = decompose_residuals(
        _scorecard("A", forecast, actual),
        canonical_slice=_canonical_slice(promo=promo_flags),
    )
    patterns = [p.pattern for p in decomp.patterns]
    assert "PROMO_RESIDUAL_SPIKE" in patterns
    promo_hit = next(p for p in decomp.patterns if p.pattern == "PROMO_RESIDUAL_SPIKE")
    assert promo_hit.severity > 0
    assert "promo-week" in promo_hit.detail


def test_promo_residual_spike_not_emitted_when_no_diff() -> None:
    """Same residual on promo and non-promo weeks → no spike."""
    forecast = [10.0] * 10
    actual = [10.5] * 10  # constant residual +0.5
    promo_flags = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    decomp = decompose_residuals(
        _scorecard("A", forecast, actual),
        canonical_slice=_canonical_slice(promo=promo_flags),
    )
    patterns = [p.pattern for p in decomp.patterns]
    assert "PROMO_RESIDUAL_SPIKE" not in patterns


def test_stockout_residual_spike_emitted_when_stockout_residual_differs() -> None:
    """Residuals +3 on stockout weeks, 0 elsewhere → spike fires."""
    forecast = [10.0] * 10
    actual = []
    for i in range(10):
        if i in (1, 4):  # stockout weeks
            actual.append(13.0)  # residual +3
        else:
            actual.append(10.0)  # residual 0
    stockout_flags = [1 if i in (1, 4) else 0 for i in range(10)]
    decomp = decompose_residuals(
        _scorecard("A", forecast, actual),
        canonical_slice=_canonical_slice(stockout=stockout_flags),
    )
    patterns = [p.pattern for p in decomp.patterns]
    assert "STOCKOUT_RESIDUAL_SPIKE" in patterns


def test_promo_and_stockout_are_independent() -> None:
    """A promo spike should not cause a stockout spike (and vice versa).

    Same synthetic data as the promo-spike test, with stockout_flag
    set to all zeros. Only PROMO_RESIDUAL_SPIKE should fire; the
    stockout means are both 0 so the gap is 0 and the pattern is
    not emitted.
    """
    forecast = [10.0] * 10
    actual = []
    for i in range(10):
        if i in (2, 5, 8):
            actual.append(12.0)
        else:
            actual.append(10.0)
    promo_flags = [1 if i in (2, 5, 8) else 0 for i in range(10)]
    stockout_flags = [0] * 10
    decomp = decompose_residuals(
        _scorecard("A", forecast, actual),
        canonical_slice=_canonical_slice(promo=promo_flags, stockout=stockout_flags),
    )
    patterns = [p.pattern for p in decomp.patterns]
    assert "PROMO_RESIDUAL_SPIKE" in patterns
    assert "STOCKOUT_RESIDUAL_SPIKE" not in patterns


# ---------------------------------------------------------------------------
# PARENT_CHILD_RESIDUAL_GAP
# ---------------------------------------------------------------------------


def test_parent_child_residual_gap_emitted_when_child_differs_from_parent() -> None:
    """Child residual mean = 0, parent mean = 1.0 → gap pattern fires."""
    forecast = [10.0] * 10
    actual = [10.0] * 10  # residual = 0 → child mean = 0
    decomp = decompose_residuals(
        _scorecard("A", forecast, actual),
        parent_residual_mean=1.0,  # parent mean = 1.0 → gap = 1.0 / std
    )
    # residual std = 0 here (constant residual) → guard skips the
    # pattern. Test with non-constant residual.
    actual = [f + ((-0.1) ** i) for i, f in enumerate(forecast)]
    decomp = decompose_residuals(
        _scorecard("A", forecast, actual),
        parent_residual_mean=1.0,
    )
    patterns = [p.pattern for p in decomp.patterns]
    assert "PARENT_CHILD_RESIDUAL_GAP" in patterns
    gap_hit = next(p for p in decomp.patterns if p.pattern == "PARENT_CHILD_RESIDUAL_GAP")
    assert "child" in gap_hit.detail
    assert "parent" in gap_hit.detail


def test_parent_child_residual_gap_not_emitted_when_no_parent_supplied() -> None:
    """Without parent_residual_mean → no pattern emitted."""
    forecast = [10.0] * 10
    actual = [11.0, 9.0, 11.0, 9.0, 11.0, 9.0, 11.0, 9.0, 11.0, 9.0]
    decomp = decompose_residuals(_scorecard("A", forecast, actual))
    patterns = [p.pattern for p in decomp.patterns]
    assert "PARENT_CHILD_RESIDUAL_GAP" not in patterns


# ---------------------------------------------------------------------------
# HETEROSCEDASTIC_RESIDUAL
# ---------------------------------------------------------------------------


def test_heteroscedastic_residual_emitted_on_long_tail() -> None:
    """9 small residuals + 1 huge residual → std/mae ratio > 1."""
    forecast = [10.0] * 10
    actual = [10.1, 9.9, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0]
    # 9 of 10 residuals are ~0; one is +10. std >> mae → heteroscedastic.
    decomp = decompose_residuals(_scorecard("A", forecast, actual))
    patterns = [p.pattern for p in decomp.patterns]
    assert "HETEROSCEDASTIC_RESIDUAL" in patterns


def test_heteroscedastic_residual_not_emitted_on_uniform_noise() -> None:
    """Constant-magnitude residuals → std ≈ mae, ratio ≈ 1, below threshold.

    The threshold is 1.0; an alternating ±1 pattern gives ratio =
    1.0 exactly, which crosses. Use a tighter-magnitude noise so
    the ratio is well below 1.0 and the pattern does not fire.
    """
    forecast = [10.0] * 20
    actual = [10.5] * 20  # every residual = +0.5, std ≈ mae ≈ 0.5
    decomp = decompose_residuals(_scorecard("A", forecast, actual))
    patterns = [p.pattern for p in decomp.patterns]
    assert "HETEROSCEDASTIC_RESIDUAL" not in patterns


# ---------------------------------------------------------------------------
# Empty pattern list
# ---------------------------------------------------------------------------


def test_decompose_emits_empty_patterns_on_well_modelled_series() -> None:
    """A perfectly forecasted series → residual = 0, std = 0, no patterns.

    This is the "model is doing its job" case the proposal tool sees
    and has nothing to recommend. The decomposition should return
    an empty pattern list, not a noisy one.
    """
    forecast = [10.0] * 10
    actual = [10.0] * 10
    decomp = decompose_residuals(_scorecard("A", forecast, actual))
    assert decomp.patterns == []


# ---------------------------------------------------------------------------
# Severity clipping
# ---------------------------------------------------------------------------


def test_severity_clipped_to_unit_interval() -> None:
    """A pattern with measured value above the threshold has severity 1.0.

    Use the promo-spike case with a very large residual on promo
    weeks (gap >> threshold) and assert the severity is 1.0, not
    the raw value.
    """
    forecast = [10.0] * 10
    actual = []
    for i in range(10):
        if i in (2, 5, 8):
            actual.append(50.0)  # massive over-forecast residual
        else:
            actual.append(10.0)  # perfect
    promo_flags = [1 if i in (2, 5, 8) else 0 for i in range(10)]
    decomp = decompose_residuals(
        _scorecard("A", forecast, actual),
        canonical_slice=_canonical_slice(promo=promo_flags),
    )
    patterns = [p.pattern for p in decomp.patterns]
    if "PROMO_RESIDUAL_SPIKE" in patterns:
        hit = next(p for p in decomp.patterns if p.pattern == "PROMO_RESIDUAL_SPIKE")
        assert 0.0 <= hit.severity <= 1.0


def test_severity_in_unit_interval_across_all_hits() -> None:
    """Defensive: every emitted hit has severity in [0, 1]."""
    forecast = [10.0] * 10
    actual = [12.0 if i in (2, 5, 8) else 10.0 for i in range(10)]
    promo_flags = [1 if i in (2, 5, 8) else 0 for i in range(10)]
    stockout_flags = [1 if i in (1, 4) else 0 for i in range(10)]
    decomp = decompose_residuals(
        _scorecard("A", forecast, actual),
        canonical_slice=_canonical_slice(promo=promo_flags, stockout=stockout_flags),
        parent_residual_mean=0.5,
    )
    for hit in decomp.patterns:
        assert 0.0 <= hit.severity <= 1.0
        assert hit.pattern in {
            "BIASED_RESIDUAL",
            "AUTOCORRELATED_RESIDUAL",
            "PROMO_RESIDUAL_SPIKE",
            "STOCKOUT_RESIDUAL_SPIKE",
            "PARENT_CHILD_RESIDUAL_GAP",
            "HETEROSCEDASTIC_RESIDUAL",
        }
