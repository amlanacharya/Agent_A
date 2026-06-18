"""Run the cockpit FastAPI app via ``uv run python -m api``.

Used for local dev + the Phase 9 contract test. Production deploys
should pass a real ``SurfaceRegistry`` + ``PlotEngine``; this module
wires the in-process defaults (all 10 surfaces with stubbed providers
that return empty / canned data — good enough for end-to-end UI
development, NOT for production correctness).

The integration test in ``tests/test_cockpit_integration.py`` uses a
separate ``build_cockpit_app(registry, engine)`` factory with a
fixture registry; this module is the dev-time entry point only.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# pyproject.toml's [tool.pytest.ini_options] pythonpath adds backend/ for
# tests, but ``python -m api`` runs outside pytest. Add it manually so the
# ``forecasting`` package resolves.
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pandas as pd
import uvicorn

from api.app import build_cockpit_app
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
    ModelResult,
    ModelScorecard,
    SegmentProfile,
    SeriesDemandProfile,
    SeriesResult,
)
from forecasting.replenishment import ReplenishmentRecommendation


def _build_dev_registry(artifacts_root: Path, workspace_root: Path) -> SurfaceRegistry:
    """Build a registry with all 10 surfaces wired to canned fixtures.

    Mirrors ``tests/test_cockpit_integration.py::registry`` but uses
    ``tempfile.TemporaryDirectory`` instead of pytest's ``tmp_path``
    so this works as a standalone dev launcher.

    The data is intentionally minimal — the cockpit UI is wired to
    these providers via the FastAPI surface, so the planner sees the
    shape of every surface without a full backend run.
    """
    run_id = "dev-run"
    cockpit_state = CockpitState(
        run_id=run_id,
        current_step="foundry_modelling",
        active_agent="foundry",
        confidence="high",
    )
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
        feature_config={"SKU_1": FeatureFlags(), "SKU_2": FeatureFlags()},
        narrative="Dev-mode two-series run.",
    )
    df = pd.DataFrame({
        "week_start": ["2024-W01", "2024-W02", "2024-W03"],
        "sku_id": ["SKU_1", "SKU_1", "SKU_2"],
        "location_id": ["WEST", "WEST", "EAST"],
        "demand_qty": [10.0, 12.0, 8.0],
    })
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
            frequently_promoted=[], never_surfaced=[], retired=[],
        ),
    )
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
        narrative="Dev-mode single-series success.",
    )
    recommendations = [
        ReplenishmentRecommendation(
            series_key="SKU_1", lead_time_days=7, forecast_std=2.0,
            lead_time_demand=70.0, safety_stock=15.0, reorder_point=85.0,
            target_inventory=100.0, current_inventory=20.0,
            open_purchase_orders=0.0, order_quantity=80.0,
            approval_tier="medium",
        ),
    ]
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("MONITORING_REPORT.md", "DRIFT_REPORT.md",
                 "OVERRIDE_ANALYSIS.md", "MODEL_HEALTH.md"):
        (run_dir / name).write_text(f"# {name}\n")
    workspace_root.mkdir(parents=True, exist_ok=True)
    for name in ("LEARNINGS.md", "DECISIONS.md", "ASSUMPTIONS.md",
                 "RUNBOOK.md", "MODEL_REGISTRY.md", "PROMOTION_DECISIONS.md"):
        (workspace_root / name).write_text(
            "# LEARNINGS\n\n## Active\n\n- dev-card\n\n## Retired\n\n"
            if name == "LEARNINGS.md" else f"# {name}\n"
        )

    registry = SurfaceRegistry()
    registry.register(MissionControlSurface(cockpit_state_provider=lambda rid: cockpit_state))
    registry.register(DataHealthSurface(eda_report_provider=lambda rid: eda_report))
    registry.register(CanonicalTableBuilderSurface(canonical_table_provider=lambda rid: df))
    registry.register(EdaExplorerSurface(eda_report_provider=lambda rid: eda_report))
    registry.register(FeatureFactorySurface(
        feature_flags_provider=lambda rid: eda_report.feature_config,
        series_profiles_provider=lambda rid: eda_report.series_profiles,
    ))
    registry.register(ModelArenaSurface(harness_report_provider=lambda rid: harness_report))
    registry.register(ForecastReviewSurface(foundry_report_provider=lambda rid: foundry_report))
    registry.register(ReplenishmentBoardSurface(recommendations_provider=lambda rid: recommendations))
    registry.register(MlopsMonitorSurface(artifacts_root=artifacts_root))
    registry.register(LearningJournalSurface(workspace_root=workspace_root))
    return registry


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifacts_root = tmp_path / "outputs"
        workspace_root = tmp_path / "workspace"
        registry = _build_dev_registry(artifacts_root, workspace_root)
        engine = InProcessPlotEngine()
        app = build_cockpit_app(registry=registry, engine=engine)
        uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
