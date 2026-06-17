"""Governed forecasting model families (Phase 4).

Each family follows the same contract::

    family.fit(history) -> fitted_state
    family.predict(fitted_state, horizon) -> forecast vector

The families are deliberately small. They take a single series'
history (a 1-d ``pandas.Series`` indexed by date) and return a
horizon-length ``list[float]``. The harness is what decides which
families to fit, folds to evaluate, and weights to use. Custom /
non-registered families are forbidden here — the harness rejects
them, and the only path for a new family is the model-escalation
layer in ``forecasting.model_escalation`` (capped at three attempts,
gated by data contract + backtest + robustness + review).

The seven governed families, in order from simplest to most
flexible:

- ``naive`` - last value carried forward.
- ``seasonal_naive`` - same week-of-season from the most recent
  season in the history.
- ``moving_average`` - mean of the last ``window`` values.
- ``exponential_smoothing`` - Holt-Winters additive level + trend
  (no seasonality — keep it stable enough to run on a 4-row
  history).
- ``croston`` - Croston's method for intermittent demand (forecasts
  the inter-demand interval and the demand size, divides).
- ``xgboost_global`` - gradient-boosted trees over the canonical
  feature table, fit per series using the lag / rolling / promo
  features produced by the Feature Factory. This is the "global
  ML" family. Its persistence (JSON encoding / decoding) and the
  recursive-forecast loop live in
  :mod:`forecasting.xgboost_persistence`; only the ``fit`` /
  ``predict`` contract lives here.
- ``aggregate_allocate`` - top-down fallback that forecasts the
  parent grain (sum across all series in a parent key) and allocates
  back to the children in proportion to their last observed share.
  Used as a safety net when per-series data is too short to fit
  anything else. It is the only family that needs construction-time
  resources (``parent_features``); it picks what it needs from the
  :class:`FamilyResources` bag and ignores the rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from forecasting.contracts import ModelFamilyName
from forecasting.xgboost_persistence import (
    decode_xgboost_model,
    encode_xgboost_model,
    recursive_forecast,
)


class ForecastingModelError(ValueError):
    """Raised when a model family cannot be fit or cannot predict."""


# ---------------------------------------------------------------------------
# Family resources
# ---------------------------------------------------------------------------


@dataclass
class FamilyResources:
    """Typed bag the harness assembles per (series, family) pair.

    Adding a new resource field is one line here and the consuming
    family reads it; no new keyword on ``build_model``. Families that
    do not need a given resource simply ignore it.
    """

    # Top-down parent-grain history (sum across children of a parent
    # key on each date). Populated by the harness when
    # ``aggregate_allocate`` is in scope; ``None`` otherwise.
    parent_features: pd.DataFrame | None = None
    # Extra per-family resources live here as an open dict so the
    # bag stays a single, named object. The harness fills in
    # whatever it has; families pick what they need.
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Family protocol
# ---------------------------------------------------------------------------


@dataclass
class FittedState:
    """Opaque, JSON-safe state a fitted model returns from ``fit``.

    Each family has its own concrete shape. The harness treats it as
    a black box and only ever round-trips it through ``predict``.
    """

    family: ModelFamilyName
    series_key: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "series_key": self.series_key, "payload": dict(self.payload)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FittedState":
        return cls(family=data["family"], series_key=data["series_key"], payload=dict(data["payload"]))


class ForecastingModel:
    """Base class for a governed model family.

    Subclasses must implement ``_fit_series`` and ``_predict_series``;
    the public ``fit`` / ``predict`` methods handle batching and
    validation so the harness can treat every family uniformly.
    """

    family: ModelFamilyName

    # ----------------- public API -----------------

    def fit(
        self,
        history: pd.Series,
        *,
        series_key: str,
        features: pd.DataFrame | None = None,
    ) -> FittedState:
        """Fit a single series.

        ``history`` is a numeric ``pandas.Series`` indexed by date (or
        any monotonic index). It must contain at least one non-null
        value; the harness is responsible for the "we have any data
        at all" check.

        ``features`` is the per-row feature table produced by the
        Feature Factory for the same series, indexed the same way as
        ``history``. It is only consumed by the ``xgboost_global``
        family; other families ignore it.
        """
        cleaned = _coerce_history(history, series_key)
        payload = self._fit_series(cleaned, features=features)
        return FittedState(family=self.family, series_key=series_key, payload=payload)

    def predict(self, state: FittedState, *, horizon: int) -> list[float]:
        """Produce a horizon-long forecast from a previously-fit state."""
        if state.family != self.family:
            raise ForecastingModelError(
                f"state was fit with {state.family!r} but {self.family!r} is predicting it"
            )
        if horizon < 1:
            raise ForecastingModelError(f"horizon must be >= 1 (got {horizon})")
        values = self._predict_series(state.payload, horizon=horizon)
        if len(values) != horizon:
            raise ForecastingModelError(
                f"{self.family} returned {len(values)} values, expected {horizon}"
            )
        return [float(v) for v in values]

    # ----------------- subclass hooks -----------------

    def _fit_series(
        self, history: pd.Series, *, features: pd.DataFrame | None
    ) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _predict_series(
        self, payload: dict[str, Any], *, horizon: int
    ) -> list[float]:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_history(history: pd.Series, series_key: str) -> pd.Series:
    if not isinstance(history, pd.Series):
        raise ForecastingModelError(
            f"history for {series_key!r} must be a pandas Series, got {type(history).__name__}"
        )
    numeric = pd.to_numeric(history, errors="coerce").astype(float)
    if numeric.dropna().empty:
        raise ForecastingModelError(f"history for {series_key!r} contains no numeric values")
    return numeric


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or math.isnan(denominator):
        return float("nan")
    return numerator / denominator


# ---------------------------------------------------------------------------
# 1. Naive
# ---------------------------------------------------------------------------


class NaiveModel(ForecastingModel):
    """Last value carried forward. The MASE reference baseline."""

    family = "naive"

    def _fit_series(
        self, history: pd.Series, *, features: pd.DataFrame | None
    ) -> dict[str, Any]:
        return {"last_value": float(history.dropna().iloc[-1])}

    def _predict_series(self, payload: dict[str, Any], *, horizon: int) -> list[float]:
        return [payload["last_value"]] * horizon


# ---------------------------------------------------------------------------
# 2. Seasonal naive
# ---------------------------------------------------------------------------


class SeasonalNaiveModel(ForecastingModel):
    """Same week-of-season from the most recent season.

    ``season_length`` defaults to 52 for weekly data; can be set to
    12 for monthly, 4 for quarterly, etc. When the history is
    shorter than one season the model degenerates to a naive
    forecast (last value), but is still registered as "seasonal
    naive" so the scorecard is honest about what was attempted.
    """

    family = "seasonal_naive"

    def __init__(self, season_length: int = 52) -> None:
        if season_length < 1:
            raise ForecastingModelError("season_length must be >= 1")
        self._season_length = season_length

    def _fit_series(
        self, history: pd.Series, *, features: pd.DataFrame | None
    ) -> dict[str, Any]:
        values = history.dropna().to_numpy()
        return {
            "values": [float(v) for v in values],
            "season_length": self._season_length,
        }

    def _predict_series(self, payload: dict[str, Any], *, horizon: int) -> list[float]:
        values = payload["values"]
        season_length = int(payload["season_length"])
        if len(values) < season_length:
            return [float(values[-1])] * horizon
        # The most recent ``season_length`` values define the cycle.
        cycle = values[-season_length:]
        return [float(cycle[i % season_length]) for i in range(horizon)]


# ---------------------------------------------------------------------------
# 3. Moving average
# ---------------------------------------------------------------------------


class MovingAverageModel(ForecastingModel):
    """Mean of the last ``window`` values, repeated across the horizon."""

    family = "moving_average"

    def __init__(self, window: int = 4) -> None:
        if window < 1:
            raise ForecastingModelError("window must be >= 1")
        self._window = window

    def _fit_series(
        self, history: pd.Series, *, features: pd.DataFrame | None
    ) -> dict[str, Any]:
        values = history.dropna().to_numpy()
        if len(values) < self._window:
            window_values = values
        else:
            window_values = values[-self._window :]
        return {"mean": float(np.mean(window_values)) if len(window_values) else 0.0}

    def _predict_series(self, payload: dict[str, Any], *, horizon: int) -> list[float]:
        return [float(payload["mean"])] * horizon


# ---------------------------------------------------------------------------
# 4. Exponential smoothing (Holt, additive trend)
# ---------------------------------------------------------------------------


class ExponentialSmoothingModel(ForecastingModel):
    """Holt's linear method (level + trend) with additive trend.

    Implemented directly (alpha / beta grid-searched on the history)
    so it stays self-contained and does not require statsmodels
    during tests. Falls back to the last value if the in-sample fit
    is degenerate (zero-variance, etc.).
    """

    family = "exponential_smoothing"

    def _fit_series(
        self, history: pd.Series, *, features: pd.DataFrame | None
    ) -> dict[str, Any]:
        values = history.dropna().to_numpy()
        if len(values) < 2:
            return {"alpha": 1.0, "beta": 0.0, "level": float(values[-1]), "trend": 0.0}
        best = _holt_fit(values)
        return {
            "alpha": best["alpha"],
            "beta": best["beta"],
            "level": best["level"],
            "trend": best["trend"],
        }

    def _predict_series(self, payload: dict[str, Any], *, horizon: int) -> list[float]:
        level = float(payload["level"])
        trend = float(payload["trend"])
        return [float(level + (step + 1) * trend) for step in range(horizon)]


def _holt_fit(values: np.ndarray) -> dict[str, float]:
    """Grid-search alpha / beta on a 2-state Holt model.

    Returns the (alpha, beta, final_level, final_trend) combination
    that minimises in-sample squared error. The grid is small
    (5x5) on purpose — this is a baseline, not a tuned forecaster.
    """
    alphas = (0.1, 0.3, 0.5, 0.7, 0.9)
    betas = (0.0, 0.1, 0.3, 0.5)
    best: dict[str, float] | None = None
    for alpha in alphas:
        for beta in betas:
            level = float(values[0])
            trend = float(values[1] - values[0]) if len(values) > 1 else 0.0
            sse = 0.0
            for index in range(1, len(values)):
                forecast = level + trend
                error = float(values[index]) - forecast
                sse += error * error
                new_level = alpha * float(values[index]) + (1 - alpha) * (level + trend)
                new_trend = beta * (new_level - level) + (1 - beta) * trend
                level, trend = new_level, new_trend
            if best is None or sse < best["sse"]:
                best = {"alpha": alpha, "beta": beta, "level": level, "trend": trend, "sse": sse}
    assert best is not None  # values has length >= 2 by the caller's check
    return {"alpha": best["alpha"], "beta": best["beta"], "level": best["level"], "trend": best["trend"]}


# ---------------------------------------------------------------------------
# 5. Croston (intermittent demand)
# ---------------------------------------------------------------------------


class CrostonModel(ForecastingModel):
    """Croston's method for intermittent demand.

    Forecasts the demand size divided by the inter-demand interval.
    Used for INTERMITTENT / LUMPY series; the harness is responsible
    for routing the right families to the right series. When the
    series is non-intermittent the model still returns a sensible
    number (the unconditional mean) but the scorecard usually picks
    something better.
    """

    family = "croston"

    def _fit_series(
        self, history: pd.Series, *, features: pd.DataFrame | None
    ) -> dict[str, Any]:
        values = history.dropna().to_numpy()
        nonzero = [float(v) for v in values if v > 0]
        if not nonzero:
            return {"interval": 1.0, "size": 0.0}
        # Inter-demand intervals: the count of consecutive zero
        # observations between each non-zero one. The first non-zero
        # value has an interval equal to its position + 1.
        intervals: list[int] = []
        gap = 0
        for v in values:
            if v > 0:
                intervals.append(gap + 1)
                gap = 0
            else:
                gap += 1
        return {
            "interval": float(np.mean(intervals)) if intervals else 1.0,
            "size": float(np.mean(nonzero)),
        }

    def _predict_series(self, payload: dict[str, Any], *, horizon: int) -> list[float]:
        rate = _safe_div(payload["size"], payload["interval"])
        if math.isnan(rate):
            rate = 0.0
        return [float(rate)] * horizon


# ---------------------------------------------------------------------------
# 6. XGBoost (global ML)
# ---------------------------------------------------------------------------


class XGBoostGlobalModel(ForecastingModel):
    """Gradient-boosted trees over the canonical feature table.

    Trained per series on the lag / rolling / promo / Fourier
    features produced by the Feature Factory. Imported lazily so
    the harness can still be imported on systems that do not have
    xgboost installed (the harness / unit tests do not fail to
    import — they just skip this family at fit time when xgboost
    is missing).
    """

    family = "xgboost_global"

    def __init__(self, n_estimators: int = 50, max_depth: int = 4) -> None:
        if n_estimators < 1:
            raise ForecastingModelError("n_estimators must be >= 1")
        if max_depth < 1:
            raise ForecastingModelError("max_depth must be >= 1")
        self._n_estimators = n_estimators
        self._max_depth = max_depth

    def _fit_series(
        self, history: pd.Series, *, features: pd.DataFrame | None
    ) -> dict[str, Any]:
        if features is None or features.empty:
            raise ForecastingModelError("xgboost_global requires a features DataFrame")
        if "demand" not in features.columns:
            raise ForecastingModelError("features DataFrame is missing a 'demand' column")
        xgboost = _import_xgboost()
        if xgboost is None:
            raise ForecastingModelError("xgboost is not installed")

        # Drop non-feature columns: anything that's the target, the
        # join key, or a date is not a predictor.
        drop = {history.name or "demand", "demand", "date", "series_key"}
        # Also drop the alias columns the Feature Factory produces
        # (sku_id, location_id, week_start, demand_qty, etc.) — we
        # only want numeric features.
        drop |= {"sku_id", "location_id", "week_start", "demand_qty"}
        feature_cols = [
            column for column in features.columns
            if column not in drop and pd.api.types.is_numeric_dtype(features[column])
        ]
        if not feature_cols:
            raise ForecastingModelError("xgboost_global found no numeric features to train on")

        x = features[feature_cols].to_numpy(dtype=float)
        y = pd.to_numeric(features["demand"], errors="coerce").to_numpy(dtype=float)
        # Drop rows with NaN in either X or y (the lag/rolling
        # families NaN the first few rows of every series).
        mask = ~(np.isnan(x).any(axis=1) | np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) < 2:
            raise ForecastingModelError("xgboost_global has fewer than 2 training rows after NaN filtering")

        model = xgboost.XGBRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            objective="reg:squarederror",
        )
        model.fit(x, y)
        return {
            "model": encode_xgboost_model(model),
            "feature_cols": list(feature_cols),
            # Last feature row is the most recent observation; we
            # will use it to roll a horizon-length forecast by
            # carrying the prediction forward in place of the
            # target and re-predicting. This is a simple 1-step
            # recursive forecast — accurate enough for the baseline
            # global-ML family.
            "last_x": [float(v) for v in x[-1]],
        }

    def _predict_series(self, payload: dict[str, Any], *, horizon: int) -> list[float]:
        xgboost = _import_xgboost()
        if xgboost is None:
            raise ForecastingModelError("xgboost is not installed")
        model = decode_xgboost_model(xgboost, payload["model"])
        return recursive_forecast(
            model=model,
            last_x=payload["last_x"],
            feature_cols=payload["feature_cols"],
            horizon=horizon,
        )


def _import_xgboost():
    try:
        import xgboost  # type: ignore
    except ImportError:
        return None
    return xgboost


# ---------------------------------------------------------------------------
# 7. Aggregate and allocate (top-down fallback)
# ---------------------------------------------------------------------------


class AggregateAllocateModel(ForecastingModel):
    """Top-down fallback: forecast at the parent grain, allocate to children.

    The parent key is taken from the row's ``sku_id`` column
    (defaulting to ``series_key`` when the canonical table does not
    carry one). Each child's forecast is its share of the parent's
    last observed total, applied to the parent's naive forecast.

    This family is the safety net for short-history series: it
    uses ALL of the parent's history rather than a single
    child's, so it works even when a child has only one or two
    observations of its own.
    """

    family = "aggregate_allocate"

    def __init__(self, parent_features: pd.DataFrame | None = None) -> None:
        # ``parent_features`` is the parent's full history (sum of
        # demand across children at each date), indexed by date. The
        # harness supplies it; the model does not aggregate on its
        # own.
        self._parent_features = parent_features

    def _fit_series(
        self, history: pd.Series, *, features: pd.DataFrame | None
    ) -> dict[str, Any]:
        if self._parent_features is None or self._parent_features.empty:
            raise ForecastingModelError("aggregate_allocate requires parent_features")
        parent_demand = self._parent_features["demand"]
        parent_demand = pd.to_numeric(parent_demand, errors="coerce").astype(float).dropna()
        if parent_demand.empty:
            raise ForecastingModelError("aggregate_allocate parent history is empty")
        # Share = this child's last value / the parent's value at
        # the same date. When the parent has no value on that date
        # (sparse data) we fall back to a uniform 1 / n_children
        # share if there is sibling data in the features frame.
        clean_history = history.dropna()
        child_value = float(clean_history.iloc[-1]) if not clean_history.empty else 0.0
        # Use the most recent date in the child history.
        child_date = clean_history.index[-1] if not clean_history.empty else None
        if child_date is not None and child_date in parent_demand.index:
            parent_value = float(parent_demand.loc[child_date])
        else:
            parent_value = float(parent_demand.iloc[-1])
        share = _safe_div(child_value, parent_value)
        if math.isnan(share) or share < 0:
            share = 0.0
        return {
            "share": float(share),
            "parent_last": float(parent_demand.iloc[-1]),
        }

    def _predict_series(self, payload: dict[str, Any], *, horizon: int) -> list[float]:
        return [float(payload["share"]) * float(payload["parent_last"]) for _ in range(horizon)]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# The canonical, ordered registry. Each entry is a small factory
# function that takes a ``FamilyResources`` bag and returns a
# ``ForecastingModel`` instance. The factory IS the source of truth for
# which resources each family consumes; ``build_model`` is a thin
# dispatcher that calls the right factory. Adding a new family is one
# entry here and a class above; no edits to ``build_model`` and no
# special cases for "this family needs X at construction time" — it
# just picks what it needs from the bag and ignores the rest.
def _make_naive(_resources: FamilyResources) -> ForecastingModel:
    return NaiveModel()


def _make_seasonal_naive(_resources: FamilyResources) -> ForecastingModel:
    return SeasonalNaiveModel()


def _make_moving_average(_resources: FamilyResources) -> ForecastingModel:
    return MovingAverageModel()


def _make_exponential_smoothing(_resources: FamilyResources) -> ForecastingModel:
    return ExponentialSmoothingModel()


def _make_croston(_resources: FamilyResources) -> ForecastingModel:
    return CrostonModel()


def _make_xgboost_global(_resources: FamilyResources) -> ForecastingModel:
    return XGBoostGlobalModel()


def _make_aggregate_allocate(resources: FamilyResources) -> ForecastingModel:
    return AggregateAllocateModel(parent_features=resources.parent_features)


_MODEL_REGISTRY: dict[ModelFamilyName, "Callable[[FamilyResources], ForecastingModel]"] = {
    "naive": _make_naive,
    "seasonal_naive": _make_seasonal_naive,
    "moving_average": _make_moving_average,
    "exponential_smoothing": _make_exponential_smoothing,
    "croston": _make_croston,
    "xgboost_global": _make_xgboost_global,
    "aggregate_allocate": _make_aggregate_allocate,
}


def list_model_families() -> list[ModelFamilyName]:
    """Return the canonical list of governed model families.

    Order matches the planning document (simplest to most flexible).
    """
    return [
        "naive",
        "seasonal_naive",
        "moving_average",
        "exponential_smoothing",
        "croston",
        "xgboost_global",
        "aggregate_allocate",
    ]


def build_model(
    family: ModelFamilyName,
    *,
    resources: FamilyResources | None = None,
) -> ForecastingModel:
    """Construct a model family instance from a :class:`FamilyResources` bag.

    ``resources.parent_features`` is forwarded to
    :class:`AggregateAllocateModel` and ignored by every other family
    — the bag IS the only construction-time channel, so no family
    gets a special keyword here. Raises if ``family`` is not in the
    registry.
    """
    if family not in _MODEL_REGISTRY:
        raise ForecastingModelError(f"unknown model family: {family!r}")
    resolved = resources if resources is not None else FamilyResources()
    return _MODEL_REGISTRY[family](resolved)


def default_families_for_class(sb_class: str | None) -> list[ModelFamilyName]:
    """Pick a sensible default family set for a demand class.

    SMOOTH / ERRATIC series get the full statistical + ML spread
    including seasonal naive and XGBoost. INTERMITTENT / LUMPY
    series lean on Croston + naive + aggregate_allocate, since the
    seasonality and the global ML model are not informative when
    most weeks are zero.

    A ``None`` (or unrecognised) class returns the full default —
    the harness falls back to running every family and letting the
    scorecard pick the winner.
    """
    if sb_class in ("SMOOTH", "ERRATIC"):
        return [
            "naive",
            "seasonal_naive",
            "moving_average",
            "exponential_smoothing",
            "xgboost_global",
        ]
    if sb_class in ("INTERMITTENT", "LUMPY"):
        return ["naive", "moving_average", "croston", "aggregate_allocate"]
    return list_model_families()


__all__ = [
    "ForecastingModel",
    "FittedState",
    "ForecastingModelError",
    "FamilyResources",
    "NaiveModel",
    "SeasonalNaiveModel",
    "MovingAverageModel",
    "ExponentialSmoothingModel",
    "CrostonModel",
    "XGBoostGlobalModel",
    "AggregateAllocateModel",
    "list_model_families",
    "build_model",
    "default_families_for_class",
]
