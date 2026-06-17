"""XGBoost persistence + recursive-forecast helper.

The XGBoost model family is the only one that needs a JSON-safe
serialization story (so ``FittedState.payload`` round-trips through the
Pydantic model and the on-disk run state) and the only one whose
``predict`` is a multi-step recursive forecast (it rolls the prediction
back into ``lag_1`` and re-derives ``lag_2`` / ``rolling_mean_4``).

Both concerns are persistence / inference mechanics, not part of the
``fit`` / ``predict`` contract. Keeping them out of ``forecasting_models``
shrinks the model file and lets the XGBoost family read like the other
six (which are 5–10 lines each).
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def encode_xgboost_model(model: Any) -> str:
    """JSON-encode an XGBoost model so it round-trips through a Pydantic payload."""
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    return booster.save_raw().decode("latin-1")


def decode_xgboost_model(xgboost: Any, raw: str) -> Any:
    """Inverse of :func:`encode_xgboost_model` — reconstructs a fitted XGBoost model."""
    booster = xgboost.Booster()
    booster.load_model(bytearray(raw.encode("latin-1")))
    wrapper = xgboost.XGBRegressor()
    wrapper._Booster = booster
    return wrapper


# ---------------------------------------------------------------------------
# Recursive forecast
# ---------------------------------------------------------------------------


def recursive_forecast(
    *,
    model: Any,
    last_x: list[float],
    feature_cols: list[str],
    horizon: int,
) -> list[float]:
    """Roll a horizon-long forecast out of a single fitted XGBoost model.

    At every step the model's prediction is clamped at zero (demand is
    non-negative) and rolled into ``lag_1``; ``lag_2`` and
    ``rolling_mean_4`` are re-derived as a simple mean of the last
    few predictions. This is deliberately naive — Phase 5 will refine
    with proper backtest gates.
    """
    idx_lag_1 = feature_cols.index("lag_1") if "lag_1" in feature_cols else None
    idx_lag_2 = feature_cols.index("lag_2") if "lag_2" in feature_cols else None
    idx_roll = feature_cols.index("rolling_mean_4") if "rolling_mean_4" in feature_cols else None
    current = list(last_x)
    forecasts: list[float] = []
    for _ in range(horizon):
        prediction = float(model.predict(np.array([current], dtype=float))[0])
        forecasts.append(max(0.0, prediction))
        if idx_lag_1 is not None:
            current[idx_lag_1] = forecasts[-1]
        if idx_lag_2 is not None:
            current[idx_lag_2] = forecasts[-2] if len(forecasts) >= 2 else forecasts[-1]
        if idx_roll is not None:
            recent = forecasts[-min(4, len(forecasts)) :]
            current[idx_roll] = float(np.mean(recent)) if recent else 0.0
    return forecasts


__all__ = (
    "encode_xgboost_model",
    "decode_xgboost_model",
    "recursive_forecast",
)
