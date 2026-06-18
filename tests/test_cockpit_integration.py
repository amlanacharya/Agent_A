"""Tests for Phase 8 CB8: full-chain cockpit FastAPI integration.

End-to-end test of the FastAPI surface the cockpit UI
consumes. The chain mirrors the production path:

1. Construct a ``SurfaceRegistry`` with all 9 surfaces
   wired to in-memory providers.
2. Build a FastAPI app that exposes:
   * ``GET /surfaces`` — list of registered surface names
   * ``GET /surfaces/{surface_name}/{run_id}`` — render a
     specific surface for a run
   * ``POST /plots`` — render a plot via the PlotEngine
3. Hit the endpoints via ``fastapi.testclient.TestClient``
   and assert the JSON responses are typed, valid, and
   match the surface / plot contracts.

The tests assert:

* All 9 surfaces are registered and render successfully
* Each surface's response is a valid ``SurfaceSnapshot``
* Each plot kind renders to a valid ``PlotResponse``
* Unknown surface name returns 404
* Unknown plot kind returns a typed error
* The 9 surface names match the plan's checklist
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.app import build_cockpit_app
from api.models import PlotKind, SurfaceName
from api.plot_engine import InProcessPlotEngine
from api.surfaces import (
    CanonicalTableBuilderSurface,
    DataHealthSurface,
    EdaExplorerSurface,
    FeatureFactorySurface,
    ForecastReviewSurface,
    LearningJournalSurface,
    MissionControlSurface,
    MlopsMonitorSurface,
    ModelArenaSurface,
    ReplenishmentBoardSurface,
    SurfaceRegistry,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def run_id() -> str:
    return "r-test-001"


@pytest.fixture()
def registry(run_id: str, tmp_path: Path) -> SurfaceRegistry:
    """Build a registry with all 9 surfaces wired to in-memory providers."""
    from forecasting.cockpit_state import CockpitState
    from forecasting.contracts import (
        EDAReport,
        EnsembleSummary,
        FeatureFlags,
        ForecastHarnessReport,
        FoundryReport,
        MissingnessReport,
        ModelResult,
        ModelScorecard,
        SegmentProfile,
        SeriesDemandProfile,
        SeriesResult,
    )
    from forecasting.replenishment import ReplenishmentRecommendation

    # Mission Control: a fixed live state.
    cockpit_state = CockpitState(
        run_id=run_id,
        current_step="foundry_modelling",
        active_agent="foundry",
        confidence="high",
    )
    # Data Health + EDA Explorer + Feature Factory: a small EDA report.
    eda_report = EDAReport(
        run_id=run_id,
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
                series_key="SKU_1", sb_class="SMOOTH", adi=1.0, cv2=0.4,
                trend_strength=0.1, seasonal_strength=0.2,
                recommended_models=["croston"],
            ),
            SeriesDemandProfile(
                series_key="SKU_2", sb_class="ERRATIC", adi=1.4, cv2=0.8,
                trend_strength=0.2, seasonal_strength=0.1,
                recommended_models=["croston", "ets"],
            ),
        ],
        feature_config={
            "SKU_1": FeatureFlags(),
            "SKU_2": FeatureFlags(),
        },
        narrative="Two-series run.",
        missingness=MissingnessReport(
            per_column=[], per_row_count=0, rows_with_missing=0, rows_total=0
        ),
    )
    # Canonical Table: a small DataFrame.
    df = pd.DataFrame({
        "week_start": ["2024-W01", "2024-W02", "2024-W03"],
        "sku_id": ["SKU_1", "SKU_1", "SKU_2"],
        "location_id": ["WEST", "WEST", "EAST"],
        "demand_qty": [10.0, 12.0, 8.0],
    })
    # Model Arena: a small harness report.
    harness_report = ForecastHarnessReport(
        run_id=run_id,
        horizon=1,
        series_results=[],
        scorecards=[
            ModelScorecard(
                model_family="naive", series_key="SKU_1",
                fold_cutoff="2026-01-01", horizon=1,
                forecast=[10.0], actual=[11.0],
                mae=1.0, rmse=1.0, mase=0.85, bias=0.05,
            ),
        ],
        ensemble=EnsembleSummary(
            weights={"G1": {"naive": 1.0}},
            frequently_promoted=[],
            never_surfaced=[],
            retired=[],
        ),
    )
    # Forecast Review: a small Foundry report.
    foundry_report = FoundryReport(
        run_id=run_id,
        series_results=[
            SeriesResult(
                series_key="SKU_1", sb_class="SMOOTH", mase_target=0.80,
                results=[
                    ModelResult(
                        model_name="naive", mase=0.85, mae=1.0, rmse=1.0,
                        forecast=[10.0], selected=True,
                    ),
                ],
                best_model="naive", target_met=True,
            ),
        ],
        overall_mase=0.85,
        target_met_fraction=1.0,
        narrative="Single-series success.",
    )
    # Replenishment Board: a single recommendation.
    recommendations = [
        ReplenishmentRecommendation(
            series_key="SKU_1", lead_time_days=7, forecast_std=2.0,
            lead_time_demand=70.0, safety_stock=15.0, reorder_point=85.0,
            target_inventory=100.0, current_inventory=20.0,
            open_purchase_orders=0.0, order_quantity=80.0,
            approval_tier="medium",
        ),
    ]
    # MLOps Monitor: write the four Phase 7 markdown artifacts.
    artifacts_root = tmp_path / "outputs"
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "MONITORING_REPORT.md").write_text("# Monitoring\n")
    (run_dir / "DRIFT_REPORT.md").write_text("# Drift\n")
    (run_dir / "OVERRIDE_ANALYSIS.md").write_text("# Overrides\n")
    (run_dir / "MODEL_HEALTH.md").write_text("# Health\n")
    # Learning Journal: a workspace with the six artifacts.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "LEARNINGS.md").write_text(
        "# LEARNINGS\n\n## Active\n\n- card-1\n- card-2\n\n## Retired\n\n- card-3\n"
    )
    (workspace / "DECISIONS.md").write_text("# Decisions\n")
    (workspace / "ASSUMPTIONS.md").write_text("# Assumptions\n")
    (workspace / "RUNBOOK.md").write_text("# Runbook\n")
    (workspace / "MODEL_REGISTRY.md").write_text("# Registry\n")
    (workspace / "PROMOTION_DECISIONS.md").write_text("# Promotion\n")

    registry = SurfaceRegistry()
    registry.register(MissionControlSurface(
        cockpit_state_provider=lambda rid: cockpit_state,
    ))
    registry.register(DataHealthSurface(
        eda_report_provider=lambda rid: eda_report,
    ))
    registry.register(CanonicalTableBuilderSurface(
        canonical_table_provider=lambda rid: df,
    ))
    registry.register(EdaExplorerSurface(
        eda_report_provider=lambda rid: eda_report,
    ))
    registry.register(FeatureFactorySurface(
        feature_flags_provider=lambda rid: eda_report.feature_config,
        series_profiles_provider=lambda rid: eda_report.series_profiles,
    ))
    registry.register(ModelArenaSurface(
        harness_report_provider=lambda rid: harness_report,
    ))
    registry.register(ForecastReviewSurface(
        foundry_report_provider=lambda rid: foundry_report,
    ))
    registry.register(ReplenishmentBoardSurface(
        recommendations_provider=lambda rid: recommendations,
    ))
    registry.register(MlopsMonitorSurface(
        artifacts_root=artifacts_root,
    ))
    registry.register(LearningJournalSurface(
        workspace_root=workspace,
    ))
    return registry


@pytest.fixture()
def app(registry: SurfaceRegistry) -> FastAPI:
    """Build the FastAPI app wired to the in-memory registry + engine."""
    return build_cockpit_app(registry=registry, engine=InProcessPlotEngine())


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Surface list endpoint
# ---------------------------------------------------------------------------


def test_surfaces_endpoint_lists_all_nine_surfaces(client: TestClient) -> None:
    """The /surfaces endpoint exposes all 9 surfaces from the plan checklist."""
    response = client.get("/surfaces")
    assert response.status_code == 200
    surfaces = response.json()["surfaces"]
    expected = {
        "mission_control",
        "data_health",
        "canonical_table_builder",
        "eda_explorer",
        "feature_factory",
        "model_arena",
        "forecast_review",
        "replenishment_board",
        "mlops_monitor",
        "learning_journal",
    }
    assert set(surfaces) == expected


# ---------------------------------------------------------------------------
# Per-surface render endpoint
# ---------------------------------------------------------------------------


def test_mission_control_endpoint_renders(client: TestClient) -> None:
    response = client.get("/surfaces/mission_control/r-test-001")
    assert response.status_code == 200
    data = response.json()
    assert data["surface"] == "mission_control"
    assert data["run_id"] == "r-test-001"
    assert data["state"]["current_step"] == "foundry_modelling"
    assert data["state"]["confidence"] == "high"


def test_data_health_endpoint_renders(client: TestClient) -> None:
    response = client.get("/surfaces/data_health/r-test-001")
    assert response.status_code == 200
    data = response.json()
    assert data["surface"] == "data_health"
    assert data["state"]["series_count"] == 2


def test_canonical_table_builder_endpoint_renders(client: TestClient) -> None:
    response = client.get("/surfaces/canonical_table_builder/r-test-001")
    assert response.status_code == 200
    data = response.json()
    assert data["state"]["row_count"] == 3
    assert data["state"]["columns"] == ["week_start", "sku_id", "location_id", "demand_qty"]


def test_eda_explorer_endpoint_renders(client: TestClient) -> None:
    response = client.get("/surfaces/eda_explorer/r-test-001")
    assert response.status_code == 200
    data = response.json()
    assert data["state"]["series_count"] == 2


def test_feature_factory_endpoint_renders(client: TestClient) -> None:
    response = client.get("/surfaces/feature_factory/r-test-001")
    assert response.status_code == 200
    data = response.json()
    assert "SKU_1" in data["state"]


def test_model_arena_endpoint_renders(client: TestClient) -> None:
    response = client.get("/surfaces/model_arena/r-test-001")
    assert response.status_code == 200
    data = response.json()
    assert data["state"]["scorecard_count"] == 1


def test_forecast_review_endpoint_renders(client: TestClient) -> None:
    response = client.get("/surfaces/forecast_review/r-test-001")
    assert response.status_code == 200
    data = response.json()
    assert data["state"]["overall_mase"] == 0.85


def test_replenishment_board_endpoint_renders(client: TestClient) -> None:
    response = client.get("/surfaces/replenishment_board/r-test-001")
    assert response.status_code == 200
    data = response.json()
    assert data["state"]["recommendation_count"] == 1
    assert data["state"]["total_order_quantity"] == 80.0


def test_mlops_monitor_endpoint_renders(client: TestClient) -> None:
    response = client.get("/surfaces/mlops_monitor/r-test-001")
    assert response.status_code == 200
    data = response.json()
    assert data["state"]["MONITORING_REPORT.md"] == "# Monitoring\n"


def test_learning_journal_endpoint_renders(client: TestClient) -> None:
    response = client.get("/surfaces/learning_journal/r-test-001")
    assert response.status_code == 200
    data = response.json()
    assert data["state"]["active_cards"] == 2
    assert data["state"]["retired_cards"] == 1


# ---------------------------------------------------------------------------
# Error surface
# ---------------------------------------------------------------------------


def test_unknown_surface_returns_404(client: TestClient) -> None:
    """An unknown surface name is a 404 — not a 500."""
    response = client.get("/surfaces/nope/r-test-001")
    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Plot endpoint
# ---------------------------------------------------------------------------


def test_plot_endpoint_renders_demand_curve(client: TestClient) -> None:
    """A plot request returns a valid PlotResponse with PNG bytes."""
    response = client.post(
        "/plots",
        json={
            "run_id": "r1",
            "kind": "demand_curve",
            "params": {
                "weeks": ["W1", "W2", "W3"],
                "actual": [1.0, 2.0, 3.0],
                "forecast": [1.1, 1.9, 3.1],
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["kind"] == "demand_curve"
    assert data["content_type"] == "image/png"
    raw = base64.b64decode(data["bytes_b64"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("kind", list(PlotKind.__args__))
def test_plot_endpoint_handles_all_seven_kinds(client: TestClient, kind: str) -> None:
    """The plot endpoint dispatches to all 7 kinds."""
    params = _params_for(kind)
    response = client.post(
        "/plots",
        json={"run_id": "r1", "kind": kind, "params": params},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["kind"] == kind
    assert data["content_type"] == "image/png"
    assert base64.b64decode(data["bytes_b64"])[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_endpoint_rejects_unknown_kind(client: TestClient) -> None:
    """An unknown plot kind is a 422 (Pydantic validation) before the engine sees it."""
    response = client.post(
        "/plots",
        json={"run_id": "r1", "kind": "bogus", "params": {}},
    )
    assert response.status_code == 422


def test_plot_endpoint_returns_typed_error_for_missing_params(client: TestClient) -> None:
    """A plot request with missing required params is a 422 (engine-side)."""
    response = client.post(
        "/plots",
        json={"run_id": "r1", "kind": "demand_curve", "params": {}},
    )
    assert response.status_code == 400
    assert "Missing required params" in response.json()["detail"]


def _params_for(kind: str) -> dict:
    if kind == "demand_curve":
        return {
            "weeks": ["W1", "W2", "W3"],
            "actual": [1.0, 2.0, 3.0],
            "forecast": [1.1, 1.9, 3.1],
        }
    if kind == "sparsity":
        return {"series": [{"series_key": "A", "adi": 1.0, "cv2": 0.5}]}
    if kind == "anomalies":
        return {
            "weeks": ["W1", "W2", "W3"],
            "values": [1.0, 50.0, 2.0],
            "flags": [False, True, False],
        }
    if kind == "forecast_band":
        return {
            "weeks": ["W1", "W2"],
            "forecast": [1.0, 2.0],
            "lower": [0.8, 1.8],
            "upper": [1.2, 2.2],
        }
    if kind == "backtest":
        return {
            "folds": ["f1", "f2"],
            "actual": [1.0, 2.0],
            "forecast": [1.1, 1.9],
        }
    if kind == "feature_importance":
        return {
            "features": [
                {"name": "a", "importance": 0.5},
                {"name": "b", "importance": 0.3},
            ],
        }
    if kind == "drift_chart":
        return {
            "runs": ["r1", "r2"],
            "segments": {"G1": [0.8, 0.9]},
        }
    return {}


# ---------------------------------------------------------------------------
# Plan-checklist coverage
# ---------------------------------------------------------------------------


def test_all_nine_plan_surfaces_are_registered(registry: SurfaceRegistry) -> None:
    """The registry covers every surface the plan calls for."""
    expected = {
        "mission_control",
        "data_health",
        "canonical_table_builder",
        "eda_explorer",
        "feature_factory",
        "model_arena",
        "forecast_review",
        "replenishment_board",
        "mlops_monitor",
        "learning_journal",
    }
    actual = set(registry.list_surfaces())
    assert actual == expected
