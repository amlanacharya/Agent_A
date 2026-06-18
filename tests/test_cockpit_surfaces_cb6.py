"""Tests for Phase 8 CB6: EDA Explorer + Feature Factory + Model Arena + Forecast Review surfaces.

Four more cockpit surfaces, all thin aggregators over
existing platform state — no new business logic (same
pattern as the Phase 6 CB4 scheduler).

* ``EdaExplorerSurface`` — surfaces per-series EDA detail
  (the ``series_profiles`` from the EDA report, plus
  per-series ADI / CV² / trend / seasonality, plus the
  recommended-models list). The cockpit's EDA Explorer
  lets the planner drill into one series at a time.

* ``FeatureFactorySurface`` — surfaces the per-series
  ``FeatureFlags`` (which feature families are enabled)
  + the canonical features (lag_1, lag_2, rolling_mean_4,
  Fourier terms, etc.) for a selected series. The cockpit
  shows the feature config + a feature-importance chart
  link.

* ``ModelArenaSurface`` — surfaces the
  ``ForecastHarnessReport`` (the per-model scorecards, the
  per-segment MASE, the frequently-promoted + never-surfaced
  models). The cockpit's Model Arena is the per-run
  leaderboard.

* ``ForecastReviewSurface`` — surfaces the latest
  ``FoundryReport`` (per-series forecast, MASE target,
  demand class, best model, target met). The cockpit's
  Forecast Review is the per-series drill-down.

All four use the provider-injection pattern (CB4/CB5):
the production wiring (CB8) reads from disk; tests
pass in-memory providers.
"""

from __future__ import annotations

import pytest

from api.models import SurfaceSnapshot
from api.surfaces import (
    EdaExplorerSurface,
    FeatureFactorySurface,
    ForecastReviewSurface,
    ModelArenaSurface,
)


def _eda_report() -> "object":
    from forecasting.contracts import (
        EDAReport,
        MissingnessReport,
        SegmentProfile,
        SeriesDemandProfile,
    )
    return EDAReport(
        run_id="r1",
        segment_profiles=[
            SegmentProfile(
                segment_id="G1",
                series_count=2,
                demand_class_distribution={"SMOOTH": 1, "ERRATIC": 1},
                median_adi=1.2,
                median_cv2=0.5,
                forecastability_breakdown={"forecastable": 2},
            )
        ],
        series_profiles=[
            SeriesDemandProfile(
                series_key="SKU_1",
                sb_class="SMOOTH",
                adi=1.0,
                cv2=0.4,
                trend_strength=0.1,
                seasonal_strength=0.2,
                recommended_models=["croston", "sba"],
            ),
            SeriesDemandProfile(
                series_key="SKU_2",
                sb_class="ERRATIC",
                adi=1.4,
                cv2=0.8,
                trend_strength=0.2,
                seasonal_strength=0.1,
                recommended_models=["croston", "sba", "ets"],
            ),
        ],
        feature_config={
            "SKU_1": {"use_lag_features": True},
            "SKU_2": {"use_lag_features": True, "use_promo_indicator": True},
        },
        narrative="Two-series run.",
        missingness=MissingnessReport(
            per_column=[], per_row_count=0, rows_with_missing=0, rows_total=0
        ),
    )


def _harness_report() -> "object":
    from forecasting.contracts import (
        EnsembleSummary,
        ForecastHarnessReport,
        ModelScorecard,
        SeriesResult,
    )
    return ForecastHarnessReport(
        run_id="r1",
        horizon=1,
        series_results=[
            SeriesResult(
                series_key="SKU_1",
                sb_class="SMOOTH",
                mase_target=0.80,
                results=[],
                best_model="naive",
                target_met=True,
            ),
        ],
        scorecards=[
            ModelScorecard(
                model_family="naive",
                series_key="SKU_1",
                fold_cutoff="2026-01-01",
                horizon=1,
                forecast=[10.0],
                actual=[11.0],
                mae=1.0,
                rmse=1.0,
                mase=0.85,
                bias=0.05,
            ),
            ModelScorecard(
                model_family="seasonal_naive",
                series_key="SKU_1",
                fold_cutoff="2026-01-01",
                horizon=1,
                forecast=[10.5],
                actual=[11.0],
                mae=0.5,
                rmse=0.5,
                mase=0.75,
                bias=0.02,
            ),
        ],
        ensemble=EnsembleSummary(
            weights={"G1": {"naive": 0.45, "seasonal_naive": 0.55}},
            frequently_promoted=["naive"],
            never_surfaced=[],
            retired=[],
        ),
    )


def _foundry_report() -> "object":
    from forecasting.contracts import FoundryReport, ModelResult, SeriesResult

    return FoundryReport(
        run_id="r1",
        series_results=[
            SeriesResult(
                series_key="SKU_1",
                sb_class="SMOOTH",
                mase_target=0.80,
                results=[
                    ModelResult(
                        model_name="naive", mase=0.85, mae=1.0, rmse=1.0,
                        forecast=[10.0], selected=True,
                    ),
                ],
                best_model="naive",
                target_met=False,
            ),
            SeriesResult(
                series_key="SKU_2",
                sb_class="ERRATIC",
                mase_target=0.80,
                results=[
                    ModelResult(
                        model_name="croston", mase=0.75, mae=0.8, rmse=0.8,
                        forecast=[5.0], selected=True,
                    ),
                ],
                best_model="croston",
                target_met=True,
            ),
        ],
        overall_mase=0.80,
        target_met_fraction=0.5,
        narrative="Mixed outcome.",
    )


# ---------------------------------------------------------------------------
# EdaExplorerSurface
# ---------------------------------------------------------------------------


def test_eda_explorer_surfaces_per_series_profiles() -> None:
    """The EDA Explorer surfaces the per-series profiles for the cockpit drill-down."""
    surface = EdaExplorerSurface(eda_report_provider=lambda rid: _eda_report())
    snapshot = surface.render("r1")
    assert snapshot.surface == "eda_explorer"
    assert snapshot.state["series_count"] == 2
    by_key = {p["series_key"]: p for p in snapshot.state["series_profiles"]}
    assert by_key["SKU_1"]["sb_class"] == "SMOOTH"
    assert by_key["SKU_2"]["sb_class"] == "ERRATIC"


def test_eda_explorer_surfaces_demand_class_distribution() -> None:
    """The EDA Explorer surfaces the demand-class breakdown for the cockpit."""
    surface = EdaExplorerSurface(eda_report_provider=lambda rid: _eda_report())
    snapshot = surface.render("r1")
    assert snapshot.state["demand_class_distribution"] == {"SMOOTH": 1, "ERRATIC": 1}


def test_eda_explorer_handles_missing_eda_report() -> None:
    """A run with no EDA report surfaces an empty drill-down."""
    surface = EdaExplorerSurface(eda_report_provider=lambda rid: None)
    snapshot = surface.render("r-empty")
    assert snapshot.state["series_count"] == 0
    assert snapshot.state["series_profiles"] == []


# ---------------------------------------------------------------------------
# FeatureFactorySurface
# ---------------------------------------------------------------------------


def test_feature_factory_surfaces_per_series_feature_flags() -> None:
    """The Feature Factory surface surfaces the per-series FeatureFlags."""
    surface = FeatureFactorySurface(feature_flags_provider=lambda rid: _eda_report().feature_config)
    snapshot = surface.render("r1")
    assert snapshot.surface == "feature_factory"
    assert snapshot.state["SKU_1"]["use_lag_features"] is True
    assert snapshot.state["SKU_2"]["use_lag_features"] is True
    assert snapshot.state["SKU_2"]["use_promo_indicator"] is True


def test_feature_factory_handles_missing_flags() -> None:
    """A run with no feature flags yet surfaces an empty dict."""
    surface = FeatureFactorySurface(feature_flags_provider=lambda rid: None)
    snapshot = surface.render("r-empty")
    assert snapshot.state == {}


def test_feature_factory_includes_recommended_models_per_series() -> None:
    """The Feature Factory surfaces the per-series recommended-models list."""
    surface = FeatureFactorySurface(
        feature_flags_provider=lambda rid: _eda_report().feature_config,
        series_profiles_provider=lambda rid: _eda_report().series_profiles,
    )
    snapshot = surface.render("r1")
    recommended = snapshot.state["recommended_models_per_series"]
    assert recommended["SKU_1"] == ["croston", "sba"]
    assert recommended["SKU_2"] == ["croston", "sba", "ets"]


# ---------------------------------------------------------------------------
# ModelArenaSurface
# ---------------------------------------------------------------------------


def test_model_arena_surfaces_scorecards_and_ensemble() -> None:
    """The Model Arena surfaces the scorecards + the ensemble summary."""
    surface = ModelArenaSurface(harness_report_provider=lambda rid: _harness_report())
    snapshot = surface.render("r1")
    assert snapshot.surface == "model_arena"
    assert snapshot.state["scorecard_count"] == 2
    by_family = {s["model_family"]: s for s in snapshot.state["scorecards"]}
    assert by_family["naive"]["mase"] == pytest.approx(0.85)
    assert by_family["seasonal_naive"]["mase"] == pytest.approx(0.75)
    assert snapshot.state["ensemble_weights"] == {"G1": {"naive": 0.45, "seasonal_naive": 0.55}}
    assert snapshot.state["frequently_promoted"] == ["naive"]


def test_model_arena_surfaces_never_surfaced() -> None:
    """The Model Arena surfaces the never-surfaced list (fit-failed models)."""
    from forecasting.contracts import EnsembleSummary, ForecastHarnessReport, ModelScorecard
    report = ForecastHarnessReport(
        run_id="r1",
        horizon=1,
        series_results=[],
        scorecards=[
            ModelScorecard(
                model_family="naive", series_key="A", fold_cutoff="2026-01-01",
                horizon=1, forecast=[1.0], actual=[1.0],
                mae=0.0, rmse=0.0, mase=0.0, bias=0.0,
            ),
        ],
        ensemble=EnsembleSummary(
            weights={"G1": {"naive": 1.0}},
            frequently_promoted=[],
            never_surfaced=["xgboost_global"],  # fit failed
            retired=[],
        ),
    )
    surface = ModelArenaSurface(harness_report_provider=lambda rid: report)
    snapshot = surface.render("r1")
    assert "xgboost_global" in snapshot.state["never_surfaced"]


def test_model_arena_handles_missing_harness_report() -> None:
    """A run with no harness report yet surfaces an empty leaderboard."""
    surface = ModelArenaSurface(harness_report_provider=lambda rid: None)
    snapshot = surface.render("r-empty")
    assert snapshot.state["scorecard_count"] == 0
    assert snapshot.state["ensemble_weights"] == {}


# ---------------------------------------------------------------------------
# ForecastReviewSurface
# ---------------------------------------------------------------------------


def test_forecast_review_surfaces_per_series_results() -> None:
    """The Forecast Review surfaces per-series MASE / target / best model."""
    surface = ForecastReviewSurface(foundry_report_provider=lambda rid: _foundry_report())
    snapshot = surface.render("r1")
    assert snapshot.surface == "forecast_review"
    assert snapshot.state["overall_mase"] == pytest.approx(0.80)
    assert snapshot.state["target_met_fraction"] == pytest.approx(0.5)
    by_key = {r["series_key"]: r for r in snapshot.state["series_results"]}
    assert by_key["SKU_1"]["best_model"] == "naive"
    assert by_key["SKU_1"]["target_met"] is False
    assert by_key["SKU_2"]["best_model"] == "croston"
    assert by_key["SKU_2"]["target_met"] is True


def test_forecast_review_handles_missing_foundry_report() -> None:
    """A run with no Foundry report yet surfaces a 'no data' placeholder."""
    surface = ForecastReviewSurface(foundry_report_provider=lambda rid: None)
    snapshot = surface.render("r-empty")
    assert snapshot.state["overall_mase"] == 0.0
    assert snapshot.state["series_results"] == []


def test_forecast_review_includes_narrative() -> None:
    """The Forecast Review surfaces the Foundry narrative for the cockpit."""
    surface = ForecastReviewSurface(foundry_report_provider=lambda rid: _foundry_report())
    snapshot = surface.render("r1")
    assert "Mixed outcome" in snapshot.state["narrative"]
