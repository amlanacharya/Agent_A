"""HTTP-layer Pydantic models for the Phase 8 cockpit FastAPI surface.

The platform's domain contracts live in ``forecasting/contracts.py``.
HTTP-layer shapes — the request and response models the FastAPI
endpoints consume — live here, per the boundary comment in
``contracts.py``:

    "HTTP-layer models live in api/models.py - not here."

This module owns:

* ``PlotKind`` — the closed enum of the 7 plot kinds the plan
  calls for (demand curve, sparsity, anomalies, forecast band,
  backtest, feature importance, drift chart). New plot kinds
  are deliberate additions to this Literal.
* ``CockpitPlotRequest`` — typed request shape (run_id, kind,
  params). The engine consumes it; the surface router builds it.
* ``PlotResponse`` — typed response shape (kind, content_type,
  base64-encoded bytes, width, height). The base64 encoding
  keeps the response JSON-safe over HTTP.
* ``SurfaceSnapshot`` — the typed per-surface state shape the
  cockpit UI reads (Mission Control, Data Health, etc.).
  The state is a free-form dict because each surface has
  different fields; the surface name tells the UI which
  schema to render against.

Design:

* **Pure Pydantic, no I/O.** All four models are pure data
  shapes — the engine / router / UI does the I/O.
* **Closed Literal for kinds.** New plot kinds are deliberate
  additions to ``PlotKind``, not LLM-judged expansions. The
  cockpit's plot menu stays finite and reviewable.
* **Positive-dimension contract.** ``PlotResponse.width`` and
  ``height`` are ``gt=0`` so the engine cannot emit a
  zero-sized image — that would render as a broken-image
  placeholder in the UI and waste a round-trip.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# The 7 plot kinds the plan calls for. Each one has a dedicated
# pure-function generator in ``api/plots.py`` (CB3). The Literal
# is the closed set the cockpit's plot menu dispatches against.
PlotKind = Literal[
    "demand_curve",
    "sparsity",
    "anomalies",
    "forecast_band",
    "backtest",
    "feature_importance",
    "drift_chart",
]


class CockpitPlotRequest(BaseModel):
    """A typed plot request from the cockpit UI.

    ``run_id`` is the platform's run; ``kind`` is one of the
    seven ``PlotKind`` values; ``params`` is a free-form dict
    the per-kind generator reads (e.g. ``{"series_key":
    "SKU_1|WEST"}`` for a demand curve). The HTTP layer
    validates the kind; the engine validates the params.
    """

    run_id: str
    kind: PlotKind
    params: dict[str, object] = Field(default_factory=dict)


class PlotResponse(BaseModel):
    """A typed plot response from the engine to the cockpit UI.

    ``bytes_b64`` carries the image bytes base64-encoded so the
    response is JSON-safe over HTTP. The cockpit decodes the
    field client-side and renders the resulting bytes inline
    (PNG via ``<img src="data:image/png;base64,...">``,
    SVG via inline DOM injection).

    ``width`` and ``height`` are positive integers — the
    engine cannot emit a zero-sized image. ``content_type``
    is a string rather than a Literal so a future format
    addition (JPEG, WebP) is a deliberate change to the
    engine, not a contract break.
    """

    kind: PlotKind
    content_type: str
    bytes_b64: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


# The nine cockpit surfaces the plan calls for. The Literal is
# the closed set the surface router dispatches against; new
# surfaces are deliberate additions, not LLM-judged expansions.
SurfaceName = Literal[
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
]


class SurfaceSnapshot(BaseModel):
    """The typed per-surface state the cockpit UI reads.

    Each surface has a different schema for ``state`` (Mission
    Control reads ``CockpitState``, Data Health reads
    ``EDAReport``, etc.). The free-form dict captures the
    surface-specific fields without coupling the HTTP layer
    to every domain contract. The ``surface`` name tells the
    UI which schema to render against — the surface router
    (CB4-CB7) is the source of truth for that mapping.
    """

    run_id: str
    surface: SurfaceName
    state: dict[str, object] = Field(default_factory=dict)


class MessageRequest(BaseModel):
    """Chat-loop request body for POST /messages — Phase 10 CB3.

    ``run_id`` identifies the Run the cockpit is chatting with;
    ``user_message`` is the raw text the user typed (or the
    resolved label of a possibility chip). The route is the only
    place these two strings meet — the conductor never sees the
    request envelope, only the dispatched intent + the message.
    """

    run_id: str
    user_message: str


__all__ = (
    "CockpitPlotRequest",
    "MessageRequest",
    "PlotKind",
    "PlotResponse",
    "SurfaceName",
    "SurfaceSnapshot",
)
