"""Tests for Phase 8 CB8: full-chain cockpit FastAPI integration.

End-to-end test of the FastAPI surface the cockpit UI
consumes. The chain mirrors the production path:

1. Construct a ``SurfaceRegistry`` with all 10 surfaces
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

* All 10 surfaces are registered and render successfully
* Each surface's response is a valid ``SurfaceSnapshot``
* Each plot kind renders to a valid ``PlotResponse``
* Unknown surface name returns 404
* Unknown plot kind returns a typed error
* The 10 surface names match the plan's checklist
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
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


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def run_id() -> str:
    return "r-test-001"


@pytest.fixture()
def registry(run_id: str, tmp_path: Path) -> SurfaceRegistry:
    """Build a registry with all 10 surfaces wired to in-memory providers."""
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
    # MLOps Monitor + Learning Journal: write the markdown artifacts.
    artifacts_root = tmp_path / "outputs"
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True)
    _write_many(run_dir, {
        "MONITORING_REPORT.md": "# Monitoring\n",
        "DRIFT_REPORT.md": "# Drift\n",
        "OVERRIDE_ANALYSIS.md": "# Overrides\n",
        "MODEL_HEALTH.md": "# Health\n",
    })
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_many(workspace, {
        "LEARNINGS.md": "# LEARNINGS\n\n## Active\n\n- card-1\n- card-2\n\n## Retired\n\n- card-3\n",
        "DECISIONS.md": "# Decisions\n",
        "ASSUMPTIONS.md": "# Assumptions\n",
        "RUNBOOK.md": "# Runbook\n",
        "MODEL_REGISTRY.md": "# Registry\n",
        "PROMOTION_DECISIONS.md": "# Promotion\n",
    })

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


def _write_many(root: Path, mapping: dict[str, str]) -> None:
    """Write each (filename, content) pair in ``mapping`` under ``root``."""
    for name, content in mapping.items():
        (root / name).write_text(content)


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


def test_surfaces_endpoint_lists_all_registered_surfaces(client: TestClient) -> None:
    """The /surfaces endpoint exposes every surface the plan calls for."""
    response = client.get("/surfaces")
    assert response.status_code == 200
    surfaces = response.json()["surfaces"]
    assert set(surfaces) == set(SurfaceName.__args__)


# ---------------------------------------------------------------------------
# Per-surface render endpoint
# ---------------------------------------------------------------------------


# Per-surface expected state. The (name, state_key, state_value) tuples let
# one parametrized test cover all 10 surfaces without 10 near-identical
# functions. ``Ellipsis`` (``...``) is a sentinel meaning "presence check
# only" (used for surfaces whose state value is itself a dict, e.g.
# Feature Factory's ``{"SKU_1": FeatureFlags(...)}``). The list keeps the
# per-surface contract visible at a glance.
_SURFACE_EXPECTATIONS: list[tuple[str, str, object]] = [
    ("mission_control", "current_step", "foundry_modelling"),
    ("data_health", "series_count", 2),
    ("canonical_table_builder", "row_count", 3),
    ("eda_explorer", "series_count", 2),
    ("feature_factory", "SKU_1", ...),  # presence check (state is a dict)
    ("model_arena", "scorecard_count", 1),
    ("forecast_review", "overall_mase", 0.85),
    ("replenishment_board", "recommendation_count", 1),
    ("mlops_monitor", "MONITORING_REPORT.md", "# Monitoring\n"),
    ("learning_journal", "active_cards", 2),
]


@pytest.mark.parametrize("name,state_key,state_value", _SURFACE_EXPECTATIONS)
def test_surface_endpoint_renders(
    client: TestClient, name: str, state_key: str, state_value: object
) -> None:
    """Each registered surface returns 200 + a SurfaceSnapshot with the expected state."""
    response = client.get(f"/surfaces/{name}/r-test-001")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["surface"] == name
    assert data["run_id"] == "r-test-001"
    if state_value is ...:
        assert state_key in data["state"]
    else:
        assert data["state"][state_key] == state_value


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
    from tests.test_plot_engine import _params_for
    response = client.post(
        "/plots",
        json={"run_id": "r1", "kind": kind, "params": _params_for(kind)},
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
    """A plot request with missing required params is a 400 (engine-side)."""
    response = client.post(
        "/plots",
        json={"run_id": "r1", "kind": "demand_curve", "params": {}},
    )
    assert response.status_code == 400
    assert "Missing required params" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Plan-checklist coverage
# ---------------------------------------------------------------------------


def test_all_plan_surfaces_are_registered(registry: SurfaceRegistry) -> None:
    """The registry covers every surface the plan calls for."""
    assert set(registry.list_surfaces()) == set(SurfaceName.__args__)
