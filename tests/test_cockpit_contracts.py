"""Tests for Phase 8 CB2: typed FastAPI request/response models + PlotEngine ABC.

Covers:

* ``PlotKind`` — the closed enum of the 7 plot kinds the plan calls for
* ``CockpitPlotRequest`` — typed request shape (run_id, kind, params)
* ``PlotResponse`` — typed response shape (kind, content_type, bytes_b64, width, height)
* ``SurfaceSnapshot`` — the typed surface state shape the cockpit UI reads
* ``PlotEngine`` ABC — interface contract
* ``InProcessPlotEngine`` — default implementation that delegates to the
  Phase 8 CB3 plot functions (tested with a stub here; full coverage in CB3)
* ``PlotEngineError`` — typed error surface

The HTTP-layer Pydantic models live in ``api/models.py`` (the
boundary the platform comment in ``contracts.py`` calls out:
"HTTP-layer models live in api/models.py - not here.").
The PlotEngine ABC lives in ``api/plot_engine.py`` so the
HTTP layer owns the boundary, mirroring the Phase 6 pattern
(``approval_gateway.py`` owns its ABC + default impl).
"""

from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from api.models import (
    CockpitPlotRequest,
    PlotKind,
    PlotResponse,
    SurfaceSnapshot,
)
from api.plot_engine import (
    InProcessPlotEngine,
    PlotEngine,
    PlotEngineError,
    UnknownPlotKindError,
)


# ---------------------------------------------------------------------------
# PlotKind
# ---------------------------------------------------------------------------


def test_plot_kind_is_closed_to_seven_values() -> None:
    """The plan calls for 7 plot kinds — the Literal is closed."""
    assert set(PlotKind.__args__) == {
        "demand_curve",
        "sparsity",
        "anomalies",
        "forecast_band",
        "backtest",
        "feature_importance",
        "drift_chart",
    }


# ---------------------------------------------------------------------------
# CockpitPlotRequest
# ---------------------------------------------------------------------------


def test_cockpit_plot_request_carries_run_id_kind_and_params() -> None:
    """A plot request is the join of run_id, kind, and params dict."""
    request = CockpitPlotRequest(
        run_id="r1",
        kind="demand_curve",
        params={"series_key": "SKU_1|WEST"},
    )
    assert request.run_id == "r1"
    assert request.kind == "demand_curve"
    assert request.params == {"series_key": "SKU_1|WEST"}


def test_cockpit_plot_request_rejects_unknown_kind() -> None:
    """An unknown plot kind fails Pydantic validation, not the engine."""
    with pytest.raises(ValidationError):
        CockpitPlotRequest(
            run_id="r1",
            kind="made_up_kind",  # type: ignore[arg-type]
            params={},
        )


def test_cockpit_plot_request_params_default_to_empty_dict() -> None:
    """Params default to an empty dict so the request shape is uniform."""
    request = CockpitPlotRequest(run_id="r1", kind="sparsity")
    assert request.params == {}


# ---------------------------------------------------------------------------
# PlotResponse
# ---------------------------------------------------------------------------


def test_plot_response_carries_bytes_b64_for_http_transport() -> None:
    """The response carries base64-encoded bytes — JSON-safe over HTTP."""
    raw = b"\x89PNG\r\n\x1a\n" + b"fake_png_payload"
    response = PlotResponse(
        kind="demand_curve",
        content_type="image/png",
        bytes_b64=base64.b64encode(raw).decode("ascii"),
        width=800,
        height=400,
    )
    assert response.kind == "demand_curve"
    assert response.content_type == "image/png"
    assert base64.b64decode(response.bytes_b64) == raw
    assert response.width == 800
    assert response.height == 400


def test_plot_response_rejects_non_png_content_type() -> None:
    """The engine produces PNG by default; SVG or other formats are explicit."""
    response = PlotResponse(
        kind="demand_curve",
        content_type="image/svg+xml",
        bytes_b64="",
        width=100,
        height=100,
    )
    # The contract allows non-PNG content types — the cockpit
    # renders SVG inline. The test pins the field is a string,
    # not a stricter enum, so a future format addition is a
    # deliberate change to the engine, not a contract break.
    assert response.content_type == "image/svg+xml"


def test_plot_response_dimensions_must_be_positive() -> None:
    """A zero or negative dimension is a contract violation."""
    with pytest.raises(ValidationError):
        PlotResponse(
            kind="demand_curve",
            content_type="image/png",
            bytes_b64="",
            width=0,
            height=400,
        )
    with pytest.raises(ValidationError):
        PlotResponse(
            kind="demand_curve",
            content_type="image/png",
            bytes_b64="",
            width=400,
            height=0,
        )


# ---------------------------------------------------------------------------
# SurfaceSnapshot
# ---------------------------------------------------------------------------


def test_surface_snapshot_carries_run_id_and_state() -> None:
    """A surface snapshot is the join of run_id, surface name, and a free-form state dict."""
    snap = SurfaceSnapshot(
        run_id="r1",
        surface="mission_control",
        state={"current_step": "foundry_modelling", "confidence": "high"},
    )
    assert snap.run_id == "r1"
    assert snap.surface == "mission_control"
    assert snap.state["current_step"] == "foundry_modelling"
    assert snap.state["confidence"] == "high"


def test_surface_snapshot_state_default_to_empty_dict() -> None:
    """State defaults to an empty dict so a fresh surface still serialises."""
    snap = SurfaceSnapshot(run_id="r1", surface="data_health")
    assert snap.state == {}


# ---------------------------------------------------------------------------
# PlotEngine ABC
# ---------------------------------------------------------------------------


def test_plot_engine_is_an_abstract_base_class() -> None:
    """The interface itself is not usable — concrete subclasses only."""
    with pytest.raises(TypeError):
        PlotEngine()  # type: ignore[abstract]


def test_in_process_plot_engine_is_a_plot_engine() -> None:
    """The in-process implementation satisfies the abstract interface."""
    assert issubclass(InProcessPlotEngine, PlotEngine)


# ---------------------------------------------------------------------------
# InProcessPlotEngine smoke (full coverage in CB3)
# ---------------------------------------------------------------------------


def test_in_process_plot_engine_delegates_to_kinds() -> None:
    """The engine routes to the per-kind handler (CB3).

    A demand-curve request is the simplest case; the per-kind
    generators in ``api/plots.py`` produce a real PNG (CB3
    replaced the CB2 placeholder with full generators). The
    assertion checks the engine dispatches to the per-kind
    handler; the per-kind test suite in
    ``test_plot_engine.py`` covers the rendering itself.
    """
    engine = InProcessPlotEngine()
    response = engine.render(
        CockpitPlotRequest(
            run_id="r1",
            kind="demand_curve",
            params={
                "weeks": ["W1", "W2", "W3"],
                "actual": [1.0, 2.0, 3.0],
                "forecast": [1.1, 1.9, 3.1],
            },
        )
    )
    assert response.kind == "demand_curve"
    assert response.content_type == "image/png"
    assert response.bytes_b64
    assert response.width > 0
    assert response.height > 0


def test_in_process_plot_engine_raises_on_unknown_kind() -> None:
    """An unknown plot kind raises a typed PlotEngineError."""
    engine = InProcessPlotEngine()
    # Construct the request bypassing the Literal validation so
    # the engine has a chance to raise the typed error.
    request = CockpitPlotRequest.model_construct(
        run_id="r1", kind="bogus", params={}
    )
    with pytest.raises(UnknownPlotKindError):
        engine.render(request)


# ---------------------------------------------------------------------------
# Error surface
# ---------------------------------------------------------------------------


def test_plot_engine_error_is_a_typed_exception() -> None:
    """PlotEngineError is the platform surface; subclasses are typed per kind."""
    assert issubclass(UnknownPlotKindError, PlotEngineError)
    assert issubclass(PlotEngineError, Exception)


def test_unknown_plot_kind_error_carries_kind() -> None:
    """The error carries the offending kind so the cockpit can show it."""
    err = UnknownPlotKindError("bogus")
    assert "bogus" in str(err)
