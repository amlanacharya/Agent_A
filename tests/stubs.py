"""Deterministic test stubs for the forecasting pipeline.

The forecasting loop code has several injectable seams (Protocol
types in :mod:`forecasting.contracts`) that production satisfies
with real harness-backed adapters. Tests need a deterministic
stand-in so the harness isn't actually called inside unit
tests - the loop's behaviour is pinned against a known
per-call return sequence instead.

Each stub in this module is:

* Deterministic: returns whatever the constructor was given,
  nothing more.
* Self-contained: no I/O, no harness calls, no randomness.
* Protocol-conforming: structurally satisfies the corresponding
  :class:`forecasting.contracts` Protocol, asserted explicitly
  in :mod:`tests.test_stubs`.
* Documented: each constructor's intent is on the class docstring
  so the test author's choice between ``value`` and ``values``
  is obvious at the call site.

Adding a new stub? Pattern after ``StubMeasureMASE`` below.
"""

from __future__ import annotations

from typing import Iterable, Union

from forecasting.contracts import FeatureFlags, MeasureMASE, ModelFamilyName


class StubMeasureMASE:
    """A deterministic ``MeasureMASE`` for unit-testing the config loop.

    Two constructor forms:

    * ``StubMeasureMASE(value: float)`` - return that single
      value on every call, regardless of flags or family.
    * ``StubMeasureMASE(values: list[float])`` - return one
      value per call, in order. The last value is held once
      the sequence is exhausted (matches the old test helper's
      "pop and hold" semantics so existing test scripts keep
      their meaning).

    No ad-hoc branching inside ``__call__`` - the constructor
    normalises both forms to a single ``_values`` list.
    """

    def __init__(self, value: float | None = None, values: list[float] | None = None) -> None:
        if value is not None and values is not None:
            raise ValueError("pass exactly one of `value` or `values`, not both")
        if value is None and values is None:
            raise ValueError("pass one of `value` (a float) or `values` (a list[float])")
        if value is not None:
            self._values: list[float] = [float(value)]
        else:
            assert values is not None  # for type-checkers
            self._values = [float(v) for v in values]
        self._calls = 0

    def __call__(self, flags: FeatureFlags, model_family: ModelFamilyName) -> float:
        index = min(self._calls, len(self._values) - 1)
        self._calls += 1
        return self._values[index]

    @property
    def calls(self) -> int:
        """How many times the stub has been invoked (for assertions)."""
        return self._calls


# Accept either the value= or values= kwarg shape - the AC's
# documented ``StubMeasureMASE(value=...)`` and
# ``StubMeasureMASE(values=...)`` forms both go through the same
# constructor above. ``__all__`` is the public surface for
# ``from tests.stubs import ...`` callers.
__all__ = ["StubMeasureMASE"]
