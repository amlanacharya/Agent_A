"""Tests for the extracted backtest module (Issue #2).

The harness's old ``_backtest_one_fold`` returned ``ModelScorecard | None``,
swallowing every failure. The new module returns a typed
:class:`BacktestResult` whose ``failure`` field names the reason —
that's what these tests pin down.
"""

from __future__ import annotations

import pandas as pd
import pytest

from forecasting.backtest import (
    BacktestFailure,
    BacktestFailureKind,
    BacktestResult,
    backtest_one_fold,
)
from forecasting.contracts import ModelScorecard
from forecasting.forecasting_models import (
    ForecastingModel,
    ForecastingModelError,
    NaiveModel,
)


def _weekly(values, name: str = "demand"):
    return pd.Series(
        values,
        index=pd.date_range("2024-01-01", periods=len(values), freq="W-MON"),
        name=name,
        dtype=float,
    )


def test_backtest_one_fold_returns_a_scorecard_on_success() -> None:
    history = _weekly([1.0, 2.0, 3.0, 4.0, 5.0])
    features = pd.DataFrame(
        {
            "date": list(history.index),
            "series_key": ["A"] * len(history),
            "demand": list(history.to_numpy()),
        }
    )
    cutoff = history.index[2]  # fit on 2 rows, predict 1
    result = backtest_one_fold(
        model=NaiveModel(),
        family="naive",
        series_key="A",
        history=history,
        features=features,
        cutoff=cutoff,
        horizon=1,
    )
    assert isinstance(result, BacktestResult)
    assert result.succeeded
    assert result.failure is None
    assert isinstance(result.scorecard, ModelScorecard)
    assert result.scorecard.model_family == "naive"
    # Naive forecast = last in-sample value (3.0); actuals is [4.0].
    assert result.scorecard.forecast == [3.0]


def test_backtest_one_fold_reports_insufficient_history() -> None:
    """An in-sample slice that is too short to fit produces a typed
    recoverable failure, not a None."""
    history = _weekly([1.0])
    features = pd.DataFrame({"date": list(history.index), "series_key": ["A"], "demand": [1.0]})
    result = backtest_one_fold(
        model=NaiveModel(),
        family="naive",
        series_key="A",
        history=history,
        features=features,
        cutoff=history.index[0],
        horizon=1,
    )
    assert result.failure is not None
    assert result.scorecard is None
    assert result.failure.kind == BacktestFailureKind.INSUFFICIENT_HISTORY
    assert result.failure.recoverable is True


def test_backtest_one_fold_reports_fit_error() -> None:
    """A model that raises during fit is captured as a non-recoverable
    BacktestFailure with kind=FIT_ERROR, not swallowed as None."""

    class _CrashingModel(ForecastingModel):
        family = "naive"  # type: ignore[assignment]

        def _fit_series(self, history, *, features):  # type: ignore[override]
            raise ForecastingModelError("simulated fit failure")

        def _predict_series(self, payload, *, horizon):  # type: ignore[override]
            return []

    history = _weekly([1.0, 2.0, 3.0, 4.0, 5.0])
    features = pd.DataFrame(
        {"date": list(history.index), "series_key": ["A"] * len(history), "demand": list(history.to_numpy())}
    )
    result = backtest_one_fold(
        model=_CrashingModel(),
        family="naive",
        series_key="A",
        history=history,
        features=features,
        cutoff=history.index[2],
        horizon=1,
    )
    assert result.failure is not None
    assert result.failure.kind == BacktestFailureKind.FIT_ERROR
    assert "simulated fit failure" in result.failure.reason
    assert result.failure.recoverable is False


def test_backtest_one_fold_reports_predict_error() -> None:
    class _PredictCrashingModel(ForecastingModel):
        family = "naive"  # type: ignore[assignment]

        def _fit_series(self, history, *, features):  # type: ignore[override]
            return {"value": float(history.iloc[-1])}

        def _predict_series(self, payload, *, horizon):  # type: ignore[override]
            raise ForecastingModelError("simulated predict failure")

    history = _weekly([1.0, 2.0, 3.0, 4.0, 5.0])
    features = pd.DataFrame(
        {"date": list(history.index), "series_key": ["A"] * len(history), "demand": list(history.to_numpy())}
    )
    result = backtest_one_fold(
        model=_PredictCrashingModel(),
        family="naive",
        series_key="A",
        history=history,
        features=features,
        cutoff=history.index[2],
        horizon=1,
    )
    assert result.failure is not None
    assert result.failure.kind == BacktestFailureKind.PREDICT_ERROR
    assert result.failure.recoverable is False


def test_backtest_one_fold_reports_data_contract_failure() -> None:
    """A model that returns a horizon-long forecast with a non-finite
    value triggers a recoverable DATA_CONTRACT_FAILURE (the data
    contract gate catches NaN / inf in the forecast vector)."""

    class _InfForecastModel(ForecastingModel):
        family = "naive"  # type: ignore[assignment]

        def _fit_series(self, history, *, features):  # type: ignore[override]
            return {"value": float(history.iloc[-1])}

        def _predict_series(self, payload, *, horizon):  # type: ignore[override]
            # Correct length, but one value is non-finite — the
            # ``check_data_contract`` gate (inside the harness's data
            # contract check) is the one that catches this.
            return [float(payload["value"])] * (horizon - 1) + [float("inf")]

    history = _weekly([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    features = pd.DataFrame(
        {"date": list(history.index), "series_key": ["A"] * len(history), "demand": list(history.to_numpy())}
    )
    result = backtest_one_fold(
        model=_InfForecastModel(),
        family="naive",
        series_key="A",
        history=history,
        features=features,
        cutoff=history.index[2],
        horizon=3,  # model returns [v, v, inf] -> contract fail on finiteness
    )
    assert result.failure is not None
    assert result.failure.kind == BacktestFailureKind.DATA_CONTRACT_FAILURE
    assert result.failure.recoverable is True


def test_backtest_failure_is_json_serialisable() -> None:
    """The failure kind uses str-Enum values so it round-trips through
    Pydantic / JSON when the harness persists the report."""
    f = BacktestFailure(kind=BacktestFailureKind.FIT_ERROR, reason="x")
    # The str-Enum base gives us serialisation for free.
    assert f.kind.value == "fit_error"
    assert isinstance(f.kind, str)
