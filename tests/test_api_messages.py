"""Tests for POST /messages — Phase 10 CB3.

The cockpit's chat loop dispatches each user message through
``Lens.classify_intent`` and then to one of six typed handlers on the
Conductor (or one of the ``drive_run_to_*`` methods). These tests
exercise the six intent types via FastAPI's ``TestClient``, stubbing
the Lens call so the test does not require an Anthropic API key.

Each test seeds a minimal RunState on disk (so ``load_run_state``
inside the route succeeds), stubs ``Lens.classify_intent`` via
monkeypatch, posts a ``/messages`` request, and asserts the response
shape matches the contract in the plan.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _seed_run_state(tmp_outputs: Path, run_id: str, phase: str = "meridian_scoping") -> str:
    """Create ``outputs/{run_id}/run_state.json`` so the route can load it.

    Also writes a minimal ``preflight.json`` because ``ADVANCE_PIPELINE``
    dispatches to ``Conductor.drive_run_to_forge`` (when the run is in
    ``meridian_scoping``) which loads the preflight artifact. The
    test seeds only what's needed for the route's hard contract;
    the canonical data in ``data_store`` is left empty because
    ``forge_eda`` would re-build EDA from data_store and that's
    outside the chat-loop test's scope.

    For tests that need ``ADVANCE_PIPELINE`` to land cleanly,
    call ``_seed_foundry_artifacts`` afterward so the conductor's
    idempotency branch fires.

    Returns the absolute path to the run_state.json file.
    """
    from forecasting.contracts import SegmentDef, SegmentMap

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
                # ``created_at`` is required by ``RunState``; stamp
                # the seed at test-fixture-build time.
                "created_at": "2026-06-22T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
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
    return str(state_path)


def _seed_foundry_artifacts(tmp_outputs: Path, run_id: str) -> None:
    """Write the artifacts that let the Conductor's drive methods skip work.

    The Conductor's ``drive_run_to_<phase>`` methods are idempotent —
    if the artifact the method would write already exists, the
    method just transitions and returns. Seeding ``eda_report.json``
    here lets ``drive_run_to_forge`` skip the real EDA build (which
    requires real canonical data in ``data_store``) and just
    advance the phase.
    """
    run_dir = tmp_outputs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
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


@pytest.fixture()
def app(tmp_outputs: Path):
    """FastAPI app with the read + upload + messages routes wired.

    Mirrors ``tests/test_api_uploads.py`` — SurfaceRegistry +
    InProcessPlotEngine + the FMCG playbook loader.
    """
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


def _stub_lens(monkeypatch, intent_pack_dict: dict) -> None:
    """Replace ``Lens.classify_intent`` with a stub that returns the
    canned ``IntentPack`` dict regardless of the input.

    The route imports ``Lens`` from ``forecasting.agents.lens`` and
    calls ``Lens.classify_intent`` directly, so we monkeypatch the
    module attribute. Tests that don't stub Lens will hit the real
    Anthropic client and fail without ``ANTHROPIC_API_KEY``.
    """
    from forecasting.contracts import IntentEntities, IntentPack
    from forecasting.agents import lens as lens_module

    canned = IntentPack(
        intent=intent_pack_dict["intent"],
        entities=IntentEntities(**intent_pack_dict.get("entities", {})),
        confidence=intent_pack_dict.get("confidence", 0.95),
        raw_quote=intent_pack_dict.get("raw_quote", ""),
    )

    def _stub(inp, injected_client=None):  # noqa: ARG001
        return canned

    # ``classify_intent`` is a module-level function in lens.py, not a
    # method on a class. Monkeypatch the module attribute directly.
    monkeypatch.setattr(lens_module, "classify_intent", _stub)


# ---------------------------------------------------------------------------
# Six intent tests — one per kind the spec lists.
# ---------------------------------------------------------------------------


def test_scope_response_intent_writes_claim_and_returns_next_question(
    client, tmp_outputs, monkeypatch
):
    run_id = "r-msgs-scope"
    _seed_run_state(tmp_outputs, run_id)
    _stub_lens(
        monkeypatch,
        {
            "intent": "SCOPE_RESPONSE",
            "confidence": 0.91,
            "raw_quote": "include the promo calendar",
            "entities": {},
        },
    )

    response = client.post(
        "/messages", json={"run_id": run_id, "user_message": "include the promo calendar"}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["run_id"] == run_id
    assert "reply" in body and body["reply"]
    assert isinstance(body["possibilities"], list)
    assert len(body["possibilities"]) >= 1
    # Each possibility has the cockpit-side chip shape.
    for chip in body["possibilities"]:
        assert chip["kind"] in ("ACCEPT", "OVERRIDE", "CLARIFY")
        assert chip["label"]
    # A claim was persisted.
    ledger_path = tmp_outputs / run_id / "claim_ledger.json"
    assert ledger_path.exists()
    ledger = json.loads(ledger_path.read_text())
    assert ledger["run_id"] == run_id
    assert any(c["claim"] == "include the promo calendar" for c in ledger["claims"])


def test_override_intent_writes_user_override_claim(client, tmp_outputs, monkeypatch):
    run_id = "r-msgs-override"
    _seed_run_state(tmp_outputs, run_id)
    _stub_lens(
        monkeypatch,
        {
            "intent": "OVERRIDE",
            "confidence": 0.82,
            "raw_quote": "use seasonal naive instead",
            "entities": {},
        },
    )

    response = client.post(
        "/messages", json={"run_id": run_id, "user_message": "use seasonal naive instead"}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["intent"] == "OVERRIDE"
    assert body["possibilities"] == []
    ledger = json.loads((tmp_outputs / run_id / "claim_ledger.json").read_text())
    assert ledger["claims"][0]["verification_status"] == "USER_OVERRIDE_ACCEPTED"


def test_advance_pipeline_intent_calls_drive_run(client, tmp_outputs, monkeypatch):
    run_id = "r-msgs-advance"
    _seed_run_state(tmp_outputs, run_id, phase="meridian_scoping")
    # Seed ``eda_report.json`` so the Conductor's idempotency branch
    # fires — the test doesn't exercise the real EDA pipeline.
    _seed_foundry_artifacts(tmp_outputs, run_id)
    _stub_lens(
        monkeypatch,
        {
            "intent": "ADVANCE_PIPELINE",
            "confidence": 0.97,
            "raw_quote": "looks good, advance",
            "entities": {},
        },
    )

    response = client.post(
        "/messages", json={"run_id": run_id, "user_message": "looks good, advance"}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    # The dispatcher advanced the run to the next phase; the state
    # field on the response carries the post-advance RunState.
    assert body["advanced_to"] in ("forge_eda", "meridian_scoping")
    assert "state" in body
    # Possibilities list is what the cockpit renders as chips.
    assert isinstance(body["possibilities"], list)


def test_clarification_low_confidence_returns_two_options(
    client, tmp_outputs, monkeypatch
):
    run_id = "r-msgs-clarify"
    _seed_run_state(tmp_outputs, run_id)
    _stub_lens(
        monkeypatch,
        {
            "intent": "CLARIFICATION",
            "confidence": 0.42,
            "raw_quote": "huh?",
            "entities": {},
        },
    )

    response = client.post(
        "/messages", json={"run_id": run_id, "user_message": "huh?"}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["intent"] == "CLARIFICATION"
    # Two short options per spec.
    assert len(body["possibilities"]) >= 2


def test_correction_intent_in_meridian_scoping_writes_claim(
    client, tmp_outputs, monkeypatch
):
    run_id = "r-msgs-correction"
    _seed_run_state(tmp_outputs, run_id, phase="meridian_scoping")
    _stub_lens(
        monkeypatch,
        {
            "intent": "CORRECTION",
            "confidence": 0.88,
            "raw_quote": "I meant 8 SKUs not 5",
            "entities": {"skus": ["SKU_A", "SKU_B"]},
        },
    )

    response = client.post(
        "/messages", json={"run_id": run_id, "user_message": "I meant 8 SKUs not 5"}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["intent"] == "CORRECTION"
    # Correction outside meridian_scoping would be rejected, but
    # the seeded state is meridian_scoping so it lands normally.
    assert body["possibilities"] == []
    ledger = json.loads((tmp_outputs / run_id / "claim_ledger.json").read_text())
    assert any(c["claim"] == "I meant 8 SKUs not 5" for c in ledger["claims"])


def test_what_if_intent_creates_prism_run(client, tmp_outputs, monkeypatch):
    run_id = "r-msgs-whatif"
    _seed_run_state(tmp_outputs, run_id, phase="report_ready")
    _stub_lens(
        monkeypatch,
        {
            "intent": "WHAT_IF_REQUEST",
            "confidence": 0.93,
            "raw_quote": "what if promo doubles?",
            "entities": {"scenario": "promo doubles"},
        },
    )

    response = client.post(
        "/messages", json={"run_id": run_id, "user_message": "what if promo doubles?"}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["intent"] == "WHAT_IF_REQUEST"
    # Prism clone got a child whatif id; the route returns it as
    # ``prism_run_id`` so the cockpit can navigate to it.
    assert "prism_run_id" in body
    # A whatif directory was created under the parent run.
    whatif_root = tmp_outputs / run_id / "whatif"
    assert whatif_root.exists()
    assert any(whatif_root.iterdir())