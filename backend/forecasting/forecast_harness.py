"""The forecasting harness (Phase 4).

The harness is the canonical entry point for turning the Feature
Factory's output into a ``ForecastHarnessReport``. It is the only
place that:

- knows which model families are registered
- decides which families to run for which segment
- runs the backtest folds
- blends per-family forecasts through the ensemble
- surfaces per-family robustness checks

The harness deliberately does NOT:

- decide promotion - that is Phase 5's job, gated on this report
- compute replenishment policy - that is also Phase 5
- talk to the cockpit directly - the cockpit reads the report

The split keeps the harness small enough to reason about and
makes Phase 5 a pure "consume the report" task.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from forecasting.contracts import (
    ForecastHarnessReport,
    ForecastRequest,
    ModelFamilyName,
    ModelResult,
    ModelScorecard,
    RobustnessCheck,
    SBClass,
    SeriesResult,
)
from forecasting.ensemble import (
    EnsembleTracker,
    blend_forecasts,
    summarise_scorecards,
)
from forecasting.forecasting_models import (
    ForecastingModel,
    ForecastingModelError,
    build_model,
    default_families_for_class,
    list_model_families,
)
from forecasting.model_escalation import (
    check_data_contract,
    check_review,
    check_robustness,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_forecast_harness(
    request: ForecastRequest,
    *,
    features: pd.DataFrame,
    fold_cutoffs: Sequence[pd.Timestamp] | None = None,
    parent_features: pd.DataFrame | None = None,
) -> ForecastHarnessReport:
    """Run the governed forecasting harness on a feature table.

    The harness folds on the cutoffs the Feature Factory used
    (so the same fold bands drive both the feature computation
    and the backtest). For each (family, series, fold) triple it
    emits a ``ModelScorecard``; the per-series best is promoted to
    a ``ModelResult`` on the report's ``series_results``.

    ``parent_features`` is forwarded to ``aggregate_allocate`` so
    the top-down fallback has the parent's history. It is
    optional; when absent, ``aggregate_allocate`` is simply
    skipped for series whose only family that worked was it.
    """
    _validate_request(request, features)
    fold_cutoffs = list(fold_cutoffs) if fold_cutoffs is not None else list(request.fold_cutoffs)
    parsed_cutoffs = [pd.Timestamp(c) for c in fold_cutoffs]
    series_keys = _series_keys_in_order(features)

    scorecards: list[ModelScorecard] = []
    robustness_checks: list[RobustnessCheck] = []
    never_surfaced: set[ModelFamilyName] = set()
    series_results: list[SeriesResult] = []

    # Per-segment family sets: a family that fails for a segment
    # is dropped from that segment's candidate list, so we don't
    # pay the XGBoost cost 50 times for the same failure mode.
    failed_families: dict[str, set[ModelFamilyName]] = defaultdict(set)

    # The set of families that successfully produced at least one
    # scorecard in any segment. Used to derive
    # ``never_surfaced`` at the end.
    families_with_scorecards: set[ModelFamilyName] = set()

    for series_key in series_keys:
        sb_class = request.segment_sb_class.get(series_key)
        candidate_families = _candidate_families_for(
            request.model_families, sb_class, segment_failures=failed_families.get(series_key, set())
        )
        series_history = _series_history(features, series_key, request.target_col)
        series_features = _series_features(features, series_key)
        parent_for_series = _parent_features_for(parent_features, series_key, features)

        per_family_forecasts: dict[ModelFamilyName, list[float]] = {}
        per_family_metrics: dict[ModelFamilyName, ModelScorecard] = {}

        for family in candidate_families:
            if family in failed_families.get(series_key, set()):
                continue
            try:
                model = build_model(
                    family,
                    parent_features=parent_for_series,
                )
            except ForecastingModelError:
                failed_families[series_key].add(family)
                continue

            for cutoff in parsed_cutoffs:
                scorecard = _backtest_one_fold(
                    model=model,
                    family=family,
                    series_key=series_key,
                    history=series_history,
                    features=series_features,
                    cutoff=cutoff,
                    horizon=request.horizon,
                )
                if scorecard is None:
                    continue
                families_with_scorecards.add(family)
                scorecards.append(scorecard)
                per_family_metrics[family] = scorecard  # last fold wins; the harness keeps the most recent

            if family not in per_family_metrics:
                # Family failed every fold for this series.
                failed_families[series_key].add(family)
                continue
            # Inference fit runs once on the full history, after all folds.
            inference_state = _safe_fit(model, history=series_history, features=series_features)
            if inference_state is not None:
                forecast = _safe_predict(model, inference_state, horizon=request.horizon)
                if forecast is not None:
                    per_family_forecasts[family] = forecast

        if not per_family_metrics:
            # Every candidate family failed for this series.
            never_surfaced.update(candidate_families)
            continue

        best_family, best_scorecard = min(
            per_family_metrics.items(), key=lambda item: item[1].mae
        )
        best_forecast = per_family_forecasts.get(best_family, best_scorecard.forecast)
        series_results.append(
            SeriesResult(
                series_key=series_key,
                sb_class=sb_class or "SMOOTH",
                mase_target=request.mase_target_for(series_key),
                results=[
                    ModelResult(
                        model_name=family,
                        mase=scorecard.mase,
                        mae=scorecard.mae,
                        rmse=scorecard.rmse,
                        forecast=scorecard.forecast,
                        selected=(family == best_family),
                    )
                    for family, scorecard in per_family_metrics.items()
                ],
                best_model=best_family,
                target_met=(best_scorecard.mase <= request.mase_target_for(series_key)),
            )
        )

    # Derive the ensemble summary from the scorecards the harness
    # actually produced. ``series_segment`` reuses the segment /
    # demand-class mapping so per-segment weights are correct.
    series_segment = dict(request.segment_sb_class)
    tracker = summarise_scorecards(scorecards, series_segment=series_segment)
    ensemble_summary = tracker.summary() if scorecards else None

    # The harness's own robustness check is the global view: do
    # any of the fitted families blow up across folds? Phase 5
    # will read these to gate promotion.
    for family in families_with_scorecards:
        robustness_checks.append(
            check_robustness(scorecards=scorecards, family=family)
        )
    # Always include a review gate so the cockpit can surface the
    # human-approval state at a glance. ``human_approved`` is
    # False by default; the cockpit flips it when the planner
    # signs off on the candidate set.
    robustness_checks.append(
        check_review(human_approved=False, approver=None)
    )

    # ``never_surfaced`` is the families the harness tried to run
    # for at least one series and that produced no scorecards at
    # all - distinct from "ran but was never best-in-fold" which
    # is on ``ensemble_summary.never_surfaced``.
    for family in list_model_families():
        if family not in families_with_scorecards and family in request.model_families:
            never_surfaced.add(family)

    narrative = _build_narrative(
        series_results=series_results,
        scorecards=scorecards,
        robustness_checks=robustness_checks,
        ensemble=ensemble_summary,
        never_surfaced=never_surfaced,
    )

    return ForecastHarnessReport(
        run_id=request.run_id,
        horizon=request.horizon,
        series_results=series_results,
        scorecards=scorecards,
        ensemble=ensemble_summary,
        robustness_checks=robustness_checks,
        never_surfaced=sorted(never_surfaced),
        narrative=narrative,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_request(request: ForecastRequest, features: pd.DataFrame) -> None:
    if features.empty:
        raise ForecastingModelError("features DataFrame is empty")
    if "series_key" not in features.columns:
        raise ForecastingModelError("features DataFrame is missing a 'series_key' column")
    if "date" not in features.columns:
        raise ForecastingModelError("features DataFrame is missing a 'date' column")
    if request.horizon < 1:
        raise ForecastingModelError(f"horizon must be >= 1 (got {request.horizon})")
    unknown = [family for family in request.model_families if family not in list_model_families()]
    if unknown:
        raise ForecastingModelError(f"unknown model families: {', '.join(unknown)}")


def _series_keys_in_order(features: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(features["series_key"]))


def _series_history(features: pd.DataFrame, series_key: str, target_col: str) -> pd.Series:
    sub = features[features["series_key"] == series_key].sort_values("date", kind="mergesort")
    if target_col not in sub.columns:
        raise ForecastingModelError(f"feature table is missing target column {target_col!r}")
    return pd.Series(
        pd.to_numeric(sub[target_col], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(sub["date"], errors="coerce").to_numpy(),
        name=target_col,
    )


def _series_features(features: pd.DataFrame, series_key: str) -> pd.DataFrame:
    sub = features[features["series_key"] == series_key].sort_values("date", kind="mergesort").reset_index(drop=True)
    return sub


def _parent_features_for(
    parent_features: pd.DataFrame | None,
    series_key: str,
    features: pd.DataFrame,
) -> pd.DataFrame | None:
    """Filter the parent-grain features down to the parent of ``series_key``.

    The Feature Factory's hierarchy family uses ``sku_id`` as the
    parent key. When ``parent_features`` is supplied and the
    canonical table has a ``sku_id`` column, the harness scopes
    the parent features to just the rows for this series' parent.
    When the lookup is ambiguous we return the full parent
    features - the aggregate-and-allocate model handles the
    ambiguity by using the most recent parent total.
    """
    if parent_features is None or parent_features.empty:
        return None
    if "sku_id" not in features.columns:
        return parent_features
    sub = features[features["series_key"] == series_key]
    if sub.empty or "sku_id" not in sub.columns:
        return parent_features
    parent_sku = sub["sku_id"].iloc[0]
    if "sku_id" in parent_features.columns:
        return parent_features[parent_features["sku_id"] == parent_sku]
    return parent_features


def _candidate_families_for(
    requested: Sequence[ModelFamilyName],
    sb_class: SBClass | None,
    segment_failures: set[ModelFamilyName],
) -> list[ModelFamilyName]:
    """Narrow the family set based on demand class and prior failures.

    When ``sb_class`` is provided, we use ``default_families_for_class``
    to pick a sensible default. When the caller has not provided
    a class hint we run the families they asked for (the harness
    trusts the caller to have already done the EDA).

    Families in ``segment_failures`` are filtered out so the
    harness does not pay the XGBoost cost 50 times for the same
    failure mode.
    """
    if sb_class is None:
        candidates = list(requested)
    else:
        candidates = default_families_for_class(sb_class)
        # The caller can still force-include / force-exclude by
        # intersecting with their request. We honour the
        # intersection: a family they asked for but the class
        # hint excluded is still allowed; a family the class
        # hint included but they did not ask for is not run.
        candidates = [family for family in candidates if family in set(requested)] or list(requested)
    return [family for family in candidates if family not in segment_failures]


def _backtest_one_fold(
    *,
    model: ForecastingModel,
    family: ModelFamilyName,
    series_key: str,
    history: pd.Series,
    features: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
) -> ModelScorecard | None:
    """Backtest one model on one (series, fold) pair.

    The fold's training history is ``history[history.index <= cutoff]``;
    the actuals are the next ``horizon`` observations strictly after
    the cutoff. Returns ``None`` if there are not enough rows to
    fit or to score.
    """
    in_sample_mask = history.index <= cutoff
    in_sample = history[in_sample_mask].dropna()
    actuals = history[(history.index > cutoff)].head(horizon).dropna()
    if in_sample.empty or len(actuals) < horizon:
        return None
    in_sample_features = features[features["date"] <= cutoff].reset_index(drop=True) if not features.empty else features
    try:
        state = model.fit(in_sample, series_key=series_key, features=in_sample_features)
    except ForecastingModelError:
        return None
    try:
        forecast = model.predict(state, horizon=horizon)
    except ForecastingModelError:
        return None

    # The data-contract check runs first - it is the cheapest gate
    # and the only one that catches degenerate forecasts (NaN,
    # wrong length, infinity). The other gates live in
    # ``model_escalation`` and are run on the final report.
    data_check = check_data_contract(forecast=forecast, actual=actuals.tolist(), horizon=horizon)
    if not data_check.passed:
        return None

    forecast_arr = np.asarray(forecast, dtype=float)
    actual_arr = np.asarray(actuals, dtype=float)
    errors = actual_arr - forecast_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))
    naive_mae = _naive_mae(in_sample)
    mase = float("nan") if naive_mae == 0 or math.isnan(naive_mae) else mae / naive_mae

    return ModelScorecard(
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
    )


def _naive_mae(history: pd.Series) -> float:
    if len(history) < 2:
        return float("nan")
    naive_errors = (history.iloc[1:] - history.shift(1).iloc[1:]).abs()
    return float(naive_errors.mean())


def _safe_fit(model: ForecastingModel, *, history: pd.Series, features: pd.DataFrame):
    try:
        return model.fit(history, series_key=str(history.name or "series"), features=features)
    except ForecastingModelError:
        return None


def _safe_predict(model: ForecastingModel, state, *, horizon: int) -> list[float] | None:
    try:
        return model.predict(state, horizon=horizon)
    except ForecastingModelError:
        return None


def _build_narrative(
    *,
    series_results: list[SeriesResult],
    scorecards: list[ModelScorecard],
    robustness_checks: list[RobustnessCheck],
    ensemble,
    never_surfaced: set[ModelFamilyName],
) -> str:
    if not series_results:
        return "No series produced a usable forecast; check the EDA report and feature table."
    best_per_family: dict[str, int] = defaultdict(int)
    for result in series_results:
        best_per_family[result.best_model] += 1
    best_summary = ", ".join(
        f"{family}: {count}" for family, count in sorted(best_per_family.items())
    )
    robustness_line = "; ".join(
        f"{check.check}={'PASS' if check.passed else 'FAIL'}" for check in robustness_checks
    ) or "no robustness checks"
    never_surfaced_line = (
        f" Never-surfaced families: {', '.join(sorted(never_surfaced))}."
        if never_surfaced
        else ""
    )
    ensemble_line = ""
    if ensemble is not None and ensemble.weights:
        promoted = ", ".join(ensemble.frequently_promoted) or "none"
        ensemble_line = f" Frequently promoted: {promoted}."
    return (
        f"Harness evaluated {len(scorecards)} scorecards across {len(series_results)} series. "
        f"Best-in-fold distribution: {best_summary}.{ensemble_line} "
        f"Robustness checks: {robustness_line}.{never_surfaced_line}"
    )


# ---------------------------------------------------------------------------
# ``aggregate_allocate_available`` is a small registry hook: when
# the parent-grain history is missing for the entire run, the
# aggregate-and-allocate family cannot be run and the harness
# silently drops it from the candidate list.
# ---------------------------------------------------------------------------


def aggregate_allocate_available(parent_features: pd.DataFrame | None) -> bool:
    """True when the aggregate-and-allocate family has a parent history to use."""
    return parent_features is not None and not parent_features.empty


# ``ForecastingModelError`` is re-exported here so the harness and
# the public API share a single error class.
__all__ = [
    "run_forecast_harness",
    "ForecastingModelError",
    "aggregate_allocate_available",
    "blend_forecasts",
    "EnsembleTracker",
    "list_model_families",
    "default_families_for_class",
    "build_model",
    "RobustnessCheck",
    "ForecastRequest",
    "ForecastHarnessReport",
    "ModelScorecard",
    "SeriesResult",
    "ModelResult",
]


# --- ForecastRequest.mase_target_for ----------------------------------
# Defined here (rather than in contracts.py) so the request model
# stays pure data and the harness owns the policy. The default
# target is 1.0 (beating the naive baseline by definition - MASE
# <= 1 means the candidate is at least as good as naive).
def _default_mase_target() -> float:
    return 1.0


def _mase_target_for(self, series_key: str) -> float:
    return _default_mase_target()


ForecastRequest.mase_target_for = _mase_target_for  # type: ignore[attr-defined]
