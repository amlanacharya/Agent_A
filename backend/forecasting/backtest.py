"""Backtest one model on one fold — extracted from the harness.

The forecast harness used to inline this operation inside its main
loop, with ``_safe_fit`` / ``_safe_predict`` swallowing
``ForecastingModelError`` and returning ``None``. That made the
failure mode invisible to the caller — a fold that failed because the
data was too short, the model crashed, or the forecast was NaN all
looked the same.

This module is the single, named operation: fit one model on the
in-sample slice of one fold, predict the next ``horizon`` rows, score
the result, and return a typed :class:`BacktestResult`. The harness
consumes the typed failure; the model_escalation layer can read it
when deciding which gate the candidate failed; the model tests can
import the same function the harness does instead of reinventing the
fold-cut / actuals-head / metric computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forecasting.contracts import ModelFamilyName, ModelScorecard, RobustnessCheck
from forecasting.forecasting_models import (
    ForecastingModel,
    ForecastingModelError,
)


if TYPE_CHECKING:
    from forecasting.forecasting_models import ForecastingModel as _FM  # noqa: F401


# ---------------------------------------------------------------------------
# Typed failure
# ---------------------------------------------------------------------------


class BacktestFailureKind(str, Enum):
    """The reason a backtest fold could not produce a scorecard.

    Stored on the string value so :class:`BacktestResult` round-trips
    through JSON / Pydantic serialisation when the harness persists
    the report.
    """

    INSUFFICIENT_HISTORY = "insufficient_history"
    FIT_ERROR = "fit_error"
    PREDICT_ERROR = "predict_error"
    DATA_CONTRACT_FAILURE = "data_contract_failure"


@dataclass(frozen=True)
class BacktestFailure:
    """Typed record of why a backtest fold did not produce a scorecard.

    ``recoverable`` is True for failures the harness can drop the
    family for (``InsufficientHistory``, ``DataContractFailure``) and
    False for ones that suggest a code bug (``FitError``,
    ``PredictError`` — a model raised when it should not have).
    """

    kind: BacktestFailureKind
    reason: str

    @property
    def recoverable(self) -> bool:
        return self.kind in {
            BacktestFailureKind.INSUFFICIENT_HISTORY,
            BacktestFailureKind.DATA_CONTRACT_FAILURE,
        }


@dataclass(frozen=True)
class BacktestResult:
    """The outcome of a single backtest fold.

    Exactly one of ``scorecard`` / ``failure`` is set. Callers branch
    on ``result.failure is None``; the harness propagates the typed
    failure to its per-segment failure set so the model_escalation
    layer can read the kind when it gates promotion.
    """

    scorecard: ModelScorecard | None
    failure: BacktestFailure | None

    @property
    def succeeded(self) -> bool:
        return self.scorecard is not None and self.failure is None


# ---------------------------------------------------------------------------
# Naive MAE (used for MASE)
# ---------------------------------------------------------------------------


def _naive_mae(history: pd.Series) -> float:
    if len(history) < 2:
        return float("nan")
    naive_errors = (history.iloc[1:] - history.shift(1).iloc[1:]).abs()
    return float(naive_errors.mean())


# ---------------------------------------------------------------------------
# Data-contract gate
# ---------------------------------------------------------------------------


def _check_data_contract(
    *,
    forecast: list[float],
    actual: list[float] | None,
    horizon: int,
) -> RobustnessCheck:
    """Validate the data-contract gate (private to this module).

    The forecast must be a horizon-long numeric vector with no NaN
    or infinity. The actuals are checked for length parity but are
    not required to be all-present (an empty actuals list means we
    are in inference mode, not a backtest).

    Lives here, not in :mod:`forecasting.model_escalation`, because
    it is a pure evaluation primitive on a single fold's output, not
    a gate over the whole report. Inlining keeps the dependency
    direction honest: ``backtest`` depends on ``contracts`` and
    ``forecasting_models`` only.
    """
    issues: list[str] = []
    if not isinstance(forecast, list):
        issues.append("forecast must be a list")
    if len(forecast) != horizon:
        issues.append(f"forecast length {len(forecast)} != horizon {horizon}")
    for index, value in enumerate(forecast):
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            issues.append(f"forecast[{index}] is not finite")
            break
    if actual is not None and len(actual) != len(forecast):
        issues.append(f"actual length {len(actual)} != forecast length {len(forecast)}")
    return RobustnessCheck(
        check="data_contract",
        passed=not issues,
        detail="; ".join(issues) if issues else "forecast shape and finiteness ok",
    )


# ---------------------------------------------------------------------------
# Public operation
# ---------------------------------------------------------------------------


def backtest_one_fold(
    *,
    model: ForecastingModel,
    family: ModelFamilyName,
    series_key: str,
    history: pd.Series,
    features: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
) -> BacktestResult:
    """Backtest one model on one (series, fold) pair.

    The fold's training history is ``history[history.index <= cutoff]``;
    the actuals are the next ``horizon`` observations strictly after
    the cutoff. Returns a :class:`BacktestResult` whose
    ``failure`` field carries the typed reason when no scorecard is
    produced — never raises ``ForecastingModelError`` out of this
    function; failures are part of the result.
    """
    in_sample_mask = history.index <= cutoff
    in_sample = history[in_sample_mask].dropna()
    actuals = history[(history.index > cutoff)].head(horizon).dropna()
    if in_sample.empty or len(actuals) < horizon:
        return BacktestResult(
            scorecard=None,
            failure=BacktestFailure(
                kind=BacktestFailureKind.INSUFFICIENT_HISTORY,
                reason=(
                    f"in_sample rows={len(in_sample)}, actuals rows={len(actuals)}"
                    f" (horizon={horizon})"
                ),
            ),
        )
    in_sample_features = (
        features[features["date"] <= cutoff].reset_index(drop=True)
        if not features.empty
        else features
    )
    try:
        state = model.fit(in_sample, series_key=series_key, features=in_sample_features)
    except ForecastingModelError as exc:
        return BacktestResult(
            scorecard=None,
            failure=BacktestFailure(
                kind=BacktestFailureKind.FIT_ERROR,
                reason=str(exc),
            ),
        )
    try:
        forecast = model.predict(state, horizon=horizon)
    except ForecastingModelError as exc:
        return BacktestResult(
            scorecard=None,
            failure=BacktestFailure(
                kind=BacktestFailureKind.PREDICT_ERROR,
                reason=str(exc),
            ),
        )

    # The data-contract check runs first - it is the cheapest gate
    # and the only one that catches degenerate forecasts (NaN,
    # wrong length, infinity). The other gates live in
    # ``model_escalation`` and are run on the final report.
    data_check = _check_data_contract(forecast=forecast, actual=actuals.tolist(), horizon=horizon)
    if not data_check.passed:
        return BacktestResult(
            scorecard=None,
            failure=BacktestFailure(
                kind=BacktestFailureKind.DATA_CONTRACT_FAILURE,
                reason=data_check.detail,
            ),
        )

    forecast_arr = np.asarray(forecast, dtype=float)
    actual_arr = np.asarray(actuals, dtype=float)
    errors = actual_arr - forecast_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))
    naive_mae = _naive_mae(in_sample)
    mase = float("nan") if naive_mae == 0 or math.isnan(naive_mae) else mae / naive_mae

    return BacktestResult(
        scorecard=ModelScorecard(
            model_family=family,
            series_key=series_key,
            fold_cutoff=cutoff.isoformat(),
            horizon=horizon,
            forecast=[float(v) for v in forecast_arr],
            actual=[float(v) for v in actual_arr],
            mae=mae,
            rmse=rmse,
            mase=mase,
            bias=bias,
        ),
        failure=None,
    )


__all__ = (
    "BacktestFailureKind",
    "BacktestFailure",
    "BacktestResult",
    "backtest_one_fold",
)
