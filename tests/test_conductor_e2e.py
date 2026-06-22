"""Full-chain end-to-end test for the Phase 10 cockpit driver.

The plan's CB5 acceptance criterion: one big test that uploads a
synthetic FMCG CSV, advances the run through preflight →
meridian_scoping → forge_eda → foundry_modelling → report_ready
via the real Conductor + real EDA + real forecast harness +
real ensemble, and asserts each phase boundary landed cleanly.

This test exercises the real pipeline (no stubs for the
heavy machinery) — it's the integration seam between the
Phase 10 routes (cb1 / cb3 / cb4) and the existing Phase 2-7
backends. The Lens call inside ``/messages`` is stubbed via
``monkeypatch`` (we don't need a live Anthropic key for a
deterministic test).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _csv_bytes(n_weeks: int = 16) -> bytes:
    """A small FMCG CSV — 2 SKUs × 16 weeks is enough to exercise
    EDA + a single fold-less harness pass.

    Uses ISO dates (``2024-01-01`` + week offset) because the
    feature factory's ``pd.to_datetime(result["date"],
    errors="raise")`` rejects week-string dates with a
    UserWarning-then-error path. The values alternate so the
    seasonal_naive family has something to lock onto; the trend
    is monotonic so moving average has signal too.
    """
    import datetime as _dt
    rows = []
    base = _dt.date(2024, 1, 1)
    for sku in ["SKU_A", "SKU_B"]:
        for w in range(n_weeks):
            week_date = base + _dt.timedelta(weeks=w)
            demand = float(((w % 8) + 1) * (1 if sku == "SKU_A" else 2))
            rows.append(f"{week_date.isoformat()},{sku},NORTH,{demand}")
    return ("date,sku,region,demand\n" + "\n".join(rows)).encode()


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


def _stub_lens(monkeypatch, intent_kind: str) -> None:
    """Stub Lens.classify_intent to return the given intent.

    The route imports ``Lens`` from ``forecasting.agents.lens`` and
    calls ``Lens.classify_intent`` directly, so we monkeypatch the
    module attribute (it's a function, not a class method).
    """
    from forecasting.contracts import IntentEntities, IntentPack
    from forecasting.agents import lens as lens_module

    canned = IntentPack(
        intent=intent_kind,
        entities=IntentEntities(),
        confidence=0.95,
        raw_quote="e2e stub",
    )

    def _stub(inp, injected_client=None):  # noqa: ARG001
        return canned

    monkeypatch.setattr(lens_module, "classify_intent", _stub)


def test_full_chain_upload_through_report_ready(client, tmp_outputs, monkeypatch):
    """The full Phase 10 driver loop — upload → advance → chat → advance → report.

    Eight assertions along the way, matching the plan's CB5.4
    acceptance criterion. Each phase boundary is exercised via
    the real ``/uploads`` + ``/runs/{id}/advance`` + ``/messages``
    routes against the real Conductor + EDA + forecast harness +
    ensemble. Lens is stubbed so the chat turns don't need a
    live Anthropic key.
    """
    # 1) Upload a synthetic FMCG CSV — the cockpit's entry point.
    files = {"file": ("input.csv", _csv_bytes(), "text/csv")}
    data = {"domain": "fmcg"}
    upload_response = client.post("/uploads", files=files, data=data)
    assert upload_response.status_code == 200, upload_response.text
    body = upload_response.json()
    run_id = body["run_id"]
    assert body["state"]["current_step"] == "preflight"

    # 2) Advance preflight → meridian_scoping. The reply is
    # Meridian's first chat prompt and carries the
    # "Get started" possibility chip.
    advance_response = client.post(f"/runs/{run_id}/advance")
    assert advance_response.status_code == 200, advance_response.text
    advance_body = advance_response.json()
    assert advance_body["advanced_to"] == "meridian_scoping"
    assert advance_body["reply"]
    assert isinstance(advance_body["possibilities"], list)
    assert len(advance_body["possibilities"]) >= 1
    assert advance_body["state"]["phase"] == "meridian_scoping"

    # 3) Stub Lens and send 3 SCOPE_RESPONSE messages to fill the
    # Claim Ledger. Each one should land a new USER_OVERRIDE_ACCEPTED
    # claim and return a non-empty reply + possibilities list.
    _stub_lens(monkeypatch, "SCOPE_RESPONSE")
    scope_messages = [
        "include the promo calendar",
        "use 4-week forecast horizon",
        "target MASE < 0.9 per segment",
    ]
    for user_message in scope_messages:
        msg_response = client.post(
            "/messages",
            json={"run_id": run_id, "user_message": user_message},
        )
        assert msg_response.status_code == 200, msg_response.text
        msg_body = msg_response.json()
        assert msg_body["reply"]
        assert isinstance(msg_body["possibilities"], list)

    # 4) Verify the Claim Ledger captured all three scope answers.
    ledger_path = tmp_outputs / run_id / "claim_ledger.json"
    assert ledger_path.exists(), "claim_ledger.json should be written"
    ledger = json.loads(ledger_path.read_text())
    ledger_claims = [c["claim"] for c in ledger["claims"]]
    for user_message in scope_messages:
        assert user_message in ledger_claims, (
            f"expected '{user_message}' in claim ledger, got {ledger_claims}"
        )

    # 5) The user has finished chatting. Hit /advance with
    # force=True — the operator escape hatch in the /advance
    # route's chat gate. force=True is the equivalent of the
    # "Done with scope, advance" button the cockpit UI would
    # render once the Claim Ledger has at least one entry.
    # The /advance route then walks the chain via
    # ``drive_run_to_next`` (forge → foundry → report in one
    # call, since EDA is fast).
    advance_response = client.post(
        f"/runs/{run_id}/advance", json={"force": True}
    )
    assert advance_response.status_code == 200, advance_response.text
    advance_body = advance_response.json()
    assert advance_body["advanced_to"] == "report_ready"
    assert advance_body["state"]["phase"] == "report_ready"

    # 6) The conductor's drive methods ran end-to-end and wrote
    # all the expected artifacts.
    eda_path = tmp_outputs / run_id / "eda_report.json"
    assert eda_path.exists(), "EDA report should be persisted"
    eda_payload = json.loads(eda_path.read_text())
    assert eda_payload["run_id"] == run_id

    harness_path = tmp_outputs / run_id / "forecast_harness_report.json"
    assert harness_path.exists(), "harness report should be persisted"
    foundry_path = tmp_outputs / run_id / "foundry_report.json"
    assert foundry_path.exists(), "foundry report should be persisted"
    foundry_payload = json.loads(foundry_path.read_text())
    assert foundry_payload["run_id"] == run_id

    # 7) report.json + the four Phase 7 monitoring markdown
    # artifacts land on disk at report_ready.
    report_path = tmp_outputs / run_id / "report.json"
    assert report_path.exists(), "report.json should be persisted"
    report_payload = json.loads(report_path.read_text())
    assert report_payload["run_id"] == run_id
    assert report_payload["phase"] == "report_ready"

    # 8) The four Phase 7 monitoring markdown artifacts (the
    # MLOps Monitor surface reads these) were emitted at the
    # report step. They confirm the conductor's report-write
    # ran end-to-end.
    run_dir = tmp_outputs / run_id
    for name in (
        "MONITORING_REPORT.md",
        "DRIFT_REPORT.md",
        "OVERRIDE_ANALYSIS.md",
        "MODEL_HEALTH.md",
    ):
        assert (run_dir / name).exists(), f"{name} should be emitted at report_ready"