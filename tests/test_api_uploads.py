"""Tests for POST /uploads — Phase 10 CB1.

The cockpit uploads a CSV here; the server creates a RunState, runs
Preflight, and returns the bundle + the run_id + the initial CockpitState.
"""
from __future__ import annotations

import io
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


FMCG_PLAYBOOK = {
    "common_grains": ["sku", "region"],
    "time_col": "week",
    "demand_col": "demand",
    "min_series": 1,
    "min_history_periods": 4,
}


def _csv_bytes(n_weeks: int = 12) -> bytes:
    rows = [
        f"2024-W{w + 1:02d},{sku},NORTH,{float(w + 1)}"
        for sku in ["SKU_A", "SKU_B"]
        for w in range(n_weeks)
    ]
    return ("week,sku,region,demand\n" + "\n".join(rows)).encode()


@pytest.fixture()
def app(tmp_outputs: Path):
    """Build a FastAPI app with the upload route wired.

    The test's ``tmp_outputs`` fixture (tests/conftest.py) monkeypatches
    ``forecasting.run_state.OUTPUTS_ROOT`` so the run dirs land under
    pytest's ``tmp_path``.
    """
    from api.app import build_cockpit_app
    from api.playbooks import playbook_for
    from api.surfaces import SurfaceRegistry
    from api.plot_engine import InProcessPlotEngine

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


def test_post_uploads_returns_run_id_and_preflight_bundle(client, tmp_outputs):
    files = {"file": ("input.csv", _csv_bytes(), "text/csv")}
    data = {"domain": "fmcg"}
    response = client.post("/uploads", files=files, data=data)

    assert response.status_code == 200, response.text
    body = response.json()
    assert "run_id" in body
    assert "preflight" in body
    assert "state" in body
    # PreflightBundle is a Pydantic dict; spot-check the canonical fields.
    assert body["preflight"]["run_id"] == body["run_id"]
    assert body["preflight"]["data_quality_report"]["series_count"] == 2
    assert body["state"]["current_step"] == "preflight"
    assert body["state"]["active_agent"] == "conductor"

    # Run dir was created under tmp_outputs/{run_id}/.
    run_id = body["run_id"]
    run_dir = tmp_outputs / run_id
    assert run_dir.is_dir()
    assert (run_dir / "input.csv").exists()
    assert (run_dir / "run_state.json").exists()


def test_post_uploads_with_malformed_csv_returns_422(client):
    files = {"file": ("broken.csv", b"\x00\x01\x02corrupted", "text/csv")}
    data = {"domain": "fmcg"}
    response = client.post("/uploads", files=files, data=data)

    assert response.status_code == 422, response.text
    body = response.json()
    assert "detail" in body
    # The blocking issue code surfaces so the cockpit can render it.
    detail = body["detail"]
    assert "UNPARSEABLE_FILE" in str(detail) or any(
        "UNPARSEABLE_FILE" in str(item) for item in (detail if isinstance(detail, list) else [detail])
    )


def test_post_uploads_with_unknown_domain_returns_400(client):
    files = {"file": ("input.csv", _csv_bytes(), "text/csv")}
    data = {"domain": "imaginary"}
    response = client.post("/uploads", files=files, data=data)

    assert response.status_code == 400
    assert "unknown domain" in response.json()["detail"].lower()
