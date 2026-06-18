"""Tests for Phase 8 CB4: Mission Control + MLOps Monitor surfaces.

Covers the two surfaces the plan groups under 'platform state'
+ 'monitoring':

* ``MissionControlSurface`` — reads ``CockpitState`` and
  returns a ``SurfaceSnapshot`` with the 7 live-state fields
  the plan calls for (current_step, tool_result, code escalation
  status, attempt count, verifier gate, approval needed,
  confidence / blockers).
* ``MlopsMonitorSurface`` — reads the four Phase 7 markdown
  artifacts from ``outputs/{run_id}/`` and returns a
  ``SurfaceSnapshot`` whose state carries each artifact's
  content (or None if the artifact does not exist yet).

The surfaces are pure functions of (run_id, optional inputs)
plus a typed ``SurfaceSnapshot`` return. The FastAPI router
(CB8) calls them. The engine is a thin aggregator over the
existing platform's data — no new business logic, same
pattern as Phase 6 CB4's scheduler (glue, not engine).

The surface registry (``SurfaceRegistry``) is the typed
dispatch seam: the router calls ``registry.get(surface_name)``
and the registry routes to the right surface. A future
external surface can register itself behind the same
``CockpitSurface`` interface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.models import SurfaceName, SurfaceSnapshot
from api.surfaces import (
    CockpitSurface,
    MissionControlSurface,
    MlopsMonitorSurface,
    SurfaceRegistry,
)


# ---------------------------------------------------------------------------
# MissionControlSurface
# ---------------------------------------------------------------------------


def test_mission_control_surface_returns_seven_live_state_fields() -> None:
    """The plan calls for 7 live-state fields. The surface surfaces them all."""
    from forecasting.cockpit_state import CockpitState

    cockpit_state = CockpitState(
        run_id="r1",
        current_step="foundry_modelling",
        active_agent="foundry",
        tool_result="model_arena",
        code_escalation_status="not_requested",
        code_attempt=1,
        verifier_gate="backtest",
        approval_needed=False,
        confidence="high",
        blockers=[],
    )
    surface = MissionControlSurface(cockpit_state_provider=lambda run_id: cockpit_state)
    snapshot = surface.render("r1")
    assert isinstance(snapshot, SurfaceSnapshot)
    assert snapshot.run_id == "r1"
    assert snapshot.surface == "mission_control"
    # The 7 live-state fields are all present in the state dict.
    state = snapshot.state
    assert state["current_step"] == "foundry_modelling"
    assert state["active_agent"] == "foundry"
    assert state["tool_result"] == "model_arena"
    assert state["code_escalation_status"] == "not_requested"
    assert state["code_attempt"] == 1
    assert state["verifier_gate"] == "backtest"
    assert state["approval_needed"] is False
    assert state["confidence"] == "high"
    assert state["blockers"] == []


def test_mission_control_surface_handles_halted_run() -> None:
    """A halted Run surfaces a blocker + low confidence."""
    from forecasting.cockpit_state import CockpitState
    from forecasting.run_state import Phase, RunState

    run_state = RunState(
        run_id="r1",
        phase=Phase.HALTED,
        domain="fmcg",
        created_at="2026-06-17T00:00:00Z",
        halt_reason="open risks unacknowledged",
    )
    surface = MissionControlSurface(cockpit_state_provider=lambda rid: CockpitState.from_run_state(run_state, "halted", "conductor"))
    snapshot = surface.render("r1")
    assert snapshot.state["confidence"] == "low"
    assert any("halted" in b.lower() for b in snapshot.state["blockers"])


# ---------------------------------------------------------------------------
# MlopsMonitorSurface
# ---------------------------------------------------------------------------


def test_mlops_monitor_surface_reads_phase_7_artifacts(tmp_path: Path) -> None:
    """The surface reads the four Phase 7 markdown artifacts from outputs/{run_id}/."""
    output_dir = tmp_path / "outputs" / "r1"
    output_dir.mkdir(parents=True)
    (output_dir / "MONITORING_REPORT.md").write_text("# Monitoring\n")
    (output_dir / "DRIFT_REPORT.md").write_text("# Drift\n")
    (output_dir / "OVERRIDE_ANALYSIS.md").write_text("# Overrides\n")
    (output_dir / "MODEL_HEALTH.md").write_text("# Health\n")
    surface = MlopsMonitorSurface(artifacts_root=tmp_path / "outputs")
    snapshot = surface.render("r1")
    assert snapshot.surface == "mlops_monitor"
    assert snapshot.run_id == "r1"
    assert snapshot.state["MONITORING_REPORT.md"] == "# Monitoring\n"
    assert snapshot.state["DRIFT_REPORT.md"] == "# Drift\n"
    assert snapshot.state["OVERRIDE_ANALYSIS.md"] == "# Overrides\n"
    assert snapshot.state["MODEL_HEALTH.md"] == "# Health\n"


def test_mlops_monitor_surface_handles_missing_artifacts(tmp_path: Path) -> None:
    """A run with no monitoring artifacts yet surfaces all four as None."""
    output_dir = tmp_path / "outputs" / "r-empty"
    output_dir.mkdir(parents=True)
    surface = MlopsMonitorSurface(artifacts_root=tmp_path / "outputs")
    snapshot = surface.render("r-empty")
    assert snapshot.state == {
        "MONITORING_REPORT.md": None,
        "DRIFT_REPORT.md": None,
        "OVERRIDE_ANALYSIS.md": None,
        "MODEL_HEALTH.md": None,
    }


# ---------------------------------------------------------------------------
# SurfaceRegistry
# ---------------------------------------------------------------------------


def test_surface_registry_routes_by_name() -> None:
    """The registry dispatches by SurfaceName to the right surface."""
    from forecasting.cockpit_state import CockpitState

    cockpit = CockpitState(
        run_id="r1",
        current_step="preflight",
        active_agent="preflight",
    )
    registry = SurfaceRegistry()
    registry.register(MissionControlSurface(cockpit_state_provider=lambda rid: cockpit))
    snapshot = registry.render("mission_control", "r1")
    assert snapshot.surface == "mission_control"
    assert snapshot.state["current_step"] == "preflight"


def test_surface_registry_unknown_surface_raises() -> None:
    """An unknown surface name is a 404 (registry raises a typed error)."""
    from api.surfaces import UnknownSurfaceError

    registry = SurfaceRegistry()
    with pytest.raises(UnknownSurfaceError):
        registry.render("nope", "r1")  # type: ignore[arg-type]


def test_surface_registry_duplicate_registration_raises() -> None:
    """Registering the same surface twice is a programming error, not a silent override."""
    from forecasting.cockpit_state import CockpitState
    from api.surfaces import DuplicateSurfaceError

    cockpit = CockpitState(run_id="r1", current_step="x", active_agent="y")
    registry = SurfaceRegistry()
    registry.register(MissionControlSurface(cockpit_state_provider=lambda rid: cockpit))
    with pytest.raises(DuplicateSurfaceError):
        registry.register(MissionControlSurface(cockpit_state_provider=lambda rid: cockpit))


def test_surface_registry_lists_registered_surfaces() -> None:
    """The registry exposes the set of registered surface names (for the UI menu)."""
    from forecasting.cockpit_state import CockpitState

    cockpit = CockpitState(run_id="r1", current_step="x", active_agent="y")
    registry = SurfaceRegistry()
    registry.register(MissionControlSurface(cockpit_state_provider=lambda rid: cockpit))
    registry.register(MlopsMonitorSurface(artifacts_root=Path("/tmp")))
    names = registry.list_surfaces()
    assert "mission_control" in names
    assert "mlops_monitor" in names
