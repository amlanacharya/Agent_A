"""Tests for GET /cockpit-state/{run_id} — Phase 10 CB6.

The RunConsole page polls this every 5s for the live
CockpitState (current_step, active_agent, blockers) without
re-fetching the full surface. The endpoint also returns a
top-level ``phase`` field the UI's dispatch logic reads.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _seed_run_state(tmp_outputs: Path, run_id: str, phase: str) -> Path:
    """Write ``outputs/{run_id}/run_state.json`` so the route can load it."""
    run_dir = tmp_outputs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "phase": phase,
                "domain": "fmcg",
                "pack_confirmed": False,
                "open_risks": 0,
                "override_count": 0,
                "forge_complete": False,
                "foundry_complete": False,
                "halt_reason": None,
                "active_whatif_runs": [],
                "created_at": "2026-06-22T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return state_path


@pytest.fixture()
def app(tmp_outputs: Path):
    """FastAPI app wired with all Phase 10 routes."""
    from api.app import build_cockpit_app
    from api.playbooks import playbook_for
    from api.plot_engine import InProcessPlotEngine
    from api.surfaces import SurfaceRegistry

    registry = SurfaceRegistry()
    engine = InProcessPlotEngine()
    return build_cockpit_app(
        registry=registry,
        engine=engine,
        outputs_root=tmp_outputs,
        playbook_loader=playbook_for,
    )


@pytest.fixture()
def client(app) -> TestClient:
    return TestClient(app)


def test_cockpit_state_for_preflight_phase(client, tmp_outputs):
    run_id = "r-cs-preflight"
    _seed_run_state(tmp_outputs, run_id, phase="preflight")

    response = client.get(f"/cockpit-state/{run_id}")

    assert response.status_code == 200, response.text
    body = response.json()
    # The spec-required top-level ``phase`` field for UI dispatch.
    assert body["phase"] == "preflight"
    assert body["current_step"] == "preflight"
    # preflight is not the chat-loop phase — conductor owns it.
    assert body["active_agent"] == "conductor"


def test_cockpit_state_for_meridian_scoping_uses_meridian_agent(client, tmp_outputs):
    run_id = "r-cs-meridian"
    _seed_run_state(tmp_outputs, run_id, phase="meridian_scoping")

    response = client.get(f"/cockpit-state/{run_id}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["phase"] == "meridian_scoping"
    # meridian_scoping is the chat loop — Meridian is the active agent.
    assert body["active_agent"] == "meridian"


def test_cockpit_state_for_report_ready(client, tmp_outputs):
    run_id = "r-cs-ready"
    _seed_run_state(tmp_outputs, run_id, phase="report_ready")

    response = client.get(f"/cockpit-state/{run_id}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["phase"] == "report_ready"
    assert body["active_agent"] == "conductor"


def test_cockpit_state_unknown_run_returns_404(client):
    response = client.get("/cockpit-state/r-does-not-exist")

    assert response.status_code == 404, response.text


def test_cockpit_state_for_halted_run_includes_halt_blocker(client, tmp_outputs):
    run_id = "r-cs-halted"
    run_dir = tmp_outputs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "phase": "halted",
                "domain": "fmcg",
                "pack_confirmed": False,
                "open_risks": 0,
                "override_count": 0,
                "forge_complete": False,
                "foundry_complete": False,
                # halt_reason drives the blocker text the
                # left-rail surfaces to the user.
                "halt_reason": "missing required columns: series_key",
                "active_whatif_runs": [],
                "created_at": "2026-06-22T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    response = client.get(f"/cockpit-state/{run_id}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["phase"] == "halted"
    # The blocker surfaces the halt reason so the operator
    # can see why the run stopped without scrolling.
    assert any("halted" in b.lower() for b in body.get("blockers", []))