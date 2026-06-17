"""Tests for the deterministic stubs in :mod:`tests.stubs`."""

from __future__ import annotations

import pytest

from forecasting.config_escalation import MeasureMASE as MeasureMASE_from_config
from forecasting.contracts import FeatureFlags, MeasureMASE
from tests.stubs import StubMeasureMASE


def _flags(**overrides) -> FeatureFlags:
    base = {
        "use_fourier": False,
        "use_lag_features": True,
        "use_promo_indicator": False,
        "fourier_terms": 3,
        "use_stockout_features": False,
        "use_hierarchy_features": False,
        "use_lifecycle_features": False,
        "use_intermittency_features": False,
    }
    base.update(overrides)
    return FeatureFlags(**base)


def test_stub_measure_mase_constant_value_returns_value_every_call() -> None:
    """``StubMeasureMASE(value=...)`` returns the same value on every call."""
    stub = StubMeasureMASE(value=0.42)
    flags = _flags()
    assert stub(flags, "naive") == 0.42
    assert stub(flags, "seasonal_naive") == 0.42
    assert stub(flags, "moving_average") == 0.42
    assert stub.calls == 3


def test_stub_measure_mase_iterates_values_in_order() -> None:
    """``StubMeasureMASE(values=[...])`` returns one value per call, in order."""
    stub = StubMeasureMASE(values=[1.5, 1.3, 1.1])
    flags = _flags()
    assert stub(flags, "naive") == 1.5
    assert stub(flags, "naive") == 1.3
    assert stub(flags, "naive") == 1.1
    # Exhausted: holds the last value, like the old _stub_measure helper.
    assert stub(flags, "naive") == 1.1
    assert stub(flags, "naive") == 1.1
    assert stub.calls == 5


def test_stub_measure_mase_ignores_flags_and_family() -> None:
    """The stub returns its configured value regardless of the call args."""
    stub = StubMeasureMASE(value=0.7)
    flags_a = _flags(use_promo_indicator=True)
    flags_b = _flags(use_lag_features=False)
    assert stub(flags_a, "naive") == 0.7
    assert stub(flags_b, "exponential_smoothing") == 0.7


def test_stub_measure_mase_satisfies_protocol() -> None:
    """``StubMeasureMASE`` structurally satisfies :class:`MeasureMASE`.

    With ``@runtime_checkable`` on the Protocol, ``isinstance``
    gives us a real structural check at test time. Production
    code still treats ``MeasureMASE`` as a duck-typed Protocol;
    the runtime check is a test-time convenience.
    """
    stub = StubMeasureMASE(value=0.5)
    assert isinstance(stub, MeasureMASE)
    # And the call signature is what the loop expects.
    flags = _flags()
    assert stub(flags, "naive") == 0.5


def test_measure_mase_is_importable_from_contracts() -> None:
    """``MeasureMASE`` is now importable from :mod:`forecasting.contracts`."""
    # The import at the top of this file already proves it; the
    # assertion here documents the AC explicitly.
    assert MeasureMASE is not None
    # And the re-export from config_escalation is preserved.
    assert MeasureMASE is MeasureMASE_from_config


def test_stub_measure_mase_constructor_rejects_both_and_neither() -> None:
    """The constructor enforces exactly-one of ``value`` / ``values``."""
    with pytest.raises(ValueError, match="exactly one"):
        StubMeasureMASE(value=0.1, values=[0.2, 0.3])
    with pytest.raises(ValueError, match="one of"):
        StubMeasureMASE()
