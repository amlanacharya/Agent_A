"""Tests for POST /runs/{run_id}/advance — Phase 10 CB4.

The cockpit's "Advance to next phase" button posts here. The
route delegates to ``Conductor.drive_run_to_next`` which dispatches
based on the current phase. Each test seeds a minimal
``run_state.json`` (and the prior-phase artifacts the
Conductor's idempotency branch reads) so the test doesn't need
real data.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _seed_run_state(tmp_outputs: Path, run_id: str, phase: str) -> str:
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
    return str(state_path)


def _seed_phase_artifacts(tmp_outputs: Path, run_id: str, current_phase: str) -> None:
    """Write the artifacts the Conductor's drive methods need.

    The Conductor's ``drive_run_to_<phase>`` methods are idempotent —
    if the artifact the method would write already exists, the
    method just transitions and returns. Seeding the artifacts for
    every prior phase lets ``drive_run_to_next`` walk the chain
    without re-running the real EDA / Foundry / Report pipelines.

    Also seeds ``preflight.json`` because ``_load_canonical_and_segments``
    (which the foundry step calls before the idempotency check
    fires) requires it. The test doesn't exercise the real
    canonical-data pipeline; the seed is enough to satisfy the
    conductor's hard contract.
    """
    from forecasting.contracts import SegmentDef, SegmentMap

    run_dir = tmp_outputs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    segment_map = SegmentMap(
        run_id=run_id,
        segments=[
            SegmentDef(
                segment_id="G1",
                label="region=NORTH",
                series_keys=["sku_a|north"],
            )
        ],
        provisional=True,
        derived_by="playbook:segment_by=region",
    )
    (run_dir / "preflight.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "segment_map": segment_map.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    if current_phase in ("forge_eda", "foundry_modelling", "report_ready"):
        (run_dir / "eda_report.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "series_profiles": [],
                    "segment_profiles": [],
                    "feature_config": {},
                    "narrative": "test seed",
                }
            ),
            encoding="utf-8",
        )
    # ``forecast_harness_report.json`` is the gate for
    # ``drive_run_to_foundry`` — seeding it for ``forge_eda`` makes
    # the chained ``drive_run_to_foundry`` call (from
    # ``drive_run_to_next`` when the run is at forge_eda) skip the
    # harness call and just transition.
    if current_phase in ("forge_eda", "foundry_modelling", "report_ready"):
        (run_dir / "forecast_harness_report.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "horizon": 1,
                    "series_results": [],
                    "scorecards": [],
                    "ensemble": {
                        "weights": {},
                        "frequently_promoted": [],
                        "never_surfaced": [],
                        "retired": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "ensemble_summary.json").write_text(
            json.dumps(
                {
                    "weights": {},
                    "frequently_promoted": [],
                    "never_surfaced": [],
                    "retired": [],
                }
            ),
            encoding="utf-8",
        )
    # ``foundry_report.json`` is the gate for the report step —
    # ``drive_run_to_report`` loads it via ``_load_foundry_report``
    # before checking idempotency. Seeding it for ``forge_eda``
    # too lets the chained ``drive_run_to_report`` (from
    # ``drive_run_to_next`` when the run is at forge_eda) find the
    # artifact and proceed to write ``report.json``.
    if current_phase in ("forge_eda", "foundry_modelling", "report_ready"):
        (run_dir / "foundry_report.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "series_results": [],
                    "overall_mase": 0.0,
                    "target_met_fraction": 0.0,
                    "narrative": "test seed foundry",
                    "ensemble": {
                        "weights": {},
                        "frequently_promoted": [],
                        "never_surfaced": [],
                        "retired": [],
                    },
                }
            ),
            encoding="utf-8",
        )


@pytest.fixture()
def app(tmp_outputs: Path):
    """FastAPI app with the read + upload + advance routes wired."""
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


# ---------------------------------------------------------------------------
# Six phase-dispatch tests + one not-found test.
# ---------------------------------------------------------------------------


def test_advance_from_preflight_transitions_to_meridian_scoping(
    client, tmp_outputs
):
    run_id = "r-adv-preflight"
    _seed_run_state(tmp_outputs, run_id, phase="preflight")

    response = client.post(f"/runs/{run_id}/advance")

    assert response.status_code == 200, response.text
    body = response.json()
    # The advance succeeded; the state moved to meridian_scoping.
    assert body["advanced_to"] == "meridian_scoping"
    assert body["state"]["phase"] == "meridian_scoping"
    # The reply is non-empty (Meridian's first chat prompt).
    assert body["reply"]
    # Possibilities chip list.
    assert isinstance(body["possibilities"], list)


def test_advance_from_meridian_scoping_without_force_returns_409(
    client, tmp_outputs
):
    """In meridian_scoping the user MUST chat (not advance) — 409 otherwise."""
    run_id = "r-adv-meridian"
    _seed_run_state(tmp_outputs, run_id, phase="meridian_scoping")

    response = client.post(f"/runs/{run_id}/advance")

    assert response.status_code == 409, response.text
    body = response.json()
    # The 409 detail is a dict with the gate message — the
    # cockpit reads ``detail.message`` to render the toast.
    detail = body["detail"]
    if isinstance(detail, dict):
        message = detail.get("message", "")
    else:
        message = detail
    assert "waiting for user input" in message.lower() or \
        "meridian_scoping" in message.lower()


def test_advance_from_meridian_scoping_with_force_bypasses_to_forge(
    client, tmp_outputs
):
    """force=true advances meridian_scoping → forge_eda (skipping chat)."""
    run_id = "r-adv-meridian-force"
    _seed_run_state(tmp_outputs, run_id, phase="meridian_scoping")
    _seed_phase_artifacts(tmp_outputs, run_id, current_phase="foundry_modelling")
    # The chain walks forge → foundry → report. Seeding
    # ``eda_report.json`` + ``forecast_harness_report.json`` +
    # ``foundry_report.json`` at ``foundry_modelling`` makes each
    # step's idempotency branch fire. The user-facing semantics
    # are unchanged: force=true bypasses the chat gate; the
    # pipeline output is whatever the artifacts dictate.

    response = client.post(f"/runs/{run_id}/advance", json={"force": True})

    assert response.status_code == 200, response.text
    body = response.json()
    # force=true bypassed the chat gate and chained the remaining
    # phases — the final phase is report_ready (per the CB4 spec).
    assert body["advanced_to"] == "report_ready"
    assert body["state"]["phase"] == "report_ready"


def test_advance_from_forge_eda_chains_to_report_ready(client, tmp_outputs):
    """forge_eda → drive all remaining phases in one call (EDA is fast)."""
    run_id = "r-adv-forge"
    _seed_run_state(tmp_outputs, run_id, phase="forge_eda")
    _seed_phase_artifacts(tmp_outputs, run_id, current_phase="forge_eda")

    response = client.post(f"/runs/{run_id}/advance")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["advanced_to"] == "report_ready"
    assert body["state"]["phase"] == "report_ready"


def test_advance_from_foundry_modelling_transitions_to_report_ready(
    client, tmp_outputs
):
    run_id = "r-adv-foundry"
    _seed_run_state(tmp_outputs, run_id, phase="foundry_modelling")
    _seed_phase_artifacts(tmp_outputs, run_id, current_phase="foundry_modelling")

    response = client.post(f"/runs/{run_id}/advance")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["advanced_to"] == "report_ready"
    assert body["state"]["phase"] == "report_ready"


def test_advance_from_report_ready_is_a_noop(client, tmp_outputs):
    run_id = "r-adv-ready"
    _seed_run_state(tmp_outputs, run_id, phase="report_ready")
    _seed_phase_artifacts(tmp_outputs, run_id, current_phase="report_ready")

    response = client.post(f"/runs/{run_id}/advance")

    assert response.status_code == 200, response.text
    body = response.json()
    # Already there — no advance.
    assert body["advanced_to"] == "report_ready"
    assert body["state"]["phase"] == "report_ready"
    assert "already" in body["reply"].lower() or "report_ready" in body["reply"].lower()


def test_advance_for_unknown_run_returns_404(client):
    response = client.post("/runs/r-does-not-exist/advance")

    assert response.status_code == 404, response.text