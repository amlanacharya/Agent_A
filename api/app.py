"""Phase 8 CB8: the FastAPI app the cockpit UI consumes.

The app is a thin HTTP wrapper around the ``SurfaceRegistry``
and the ``PlotEngine``. Three endpoints:

* ``GET /surfaces`` — list the registered surface names
  (for the cockpit's navigation menu)
* ``GET /surfaces/{surface_name}/{run_id}`` — render a
  specific surface for a run
* ``POST /plots`` — render a plot via the ``PlotEngine``

The app wires the registry + the engine at construction
time (``build_cockpit_app``); the production wiring (the
``api.app`` module's main) is not in this repo's
deployment (the platform's deployment uses the in-process
surface implementations registered at startup). A future
external surface registers itself behind the same
``SurfaceRegistry`` and the app surface stays unchanged.

Design:

* **Thin HTTP wrapper, no business logic.** The endpoints
  call the registry / engine and translate the typed
  errors to HTTP status codes. The math lives in
  ``api.surfaces`` and ``api.plots``; the surface
  routing lives in ``SurfaceRegistry``.
* **Typed error mapping.** A Pydantic validation error
  is 422 (the HTTP layer does not invent the value); a
  ``UnknownSurfaceError`` is 404; an engine-side
  ``ValueError`` is 400; an ``UnknownPlotKindError``
  is 422. The mapping is in one place so a future
  error kind is a deliberate addition.
* **In-process by default.** The ``build_cockpit_app``
  factory takes the registry and the engine as
  arguments so tests can wire in-memory providers
  without monkey-patching.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from api.models import CockpitPlotRequest
from api.plot_engine import PlotEngine, UnknownPlotKindError
from api.surfaces import SurfaceRegistry, UnknownSurfaceError


def build_cockpit_app(
    *,
    registry: SurfaceRegistry,
    engine: PlotEngine,
) -> FastAPI:
    """Build the FastAPI app wired to ``registry`` + ``engine``.

    The factory takes both as arguments so tests can wire
    in-memory providers. The production wiring is in the
    platform's startup module (out-of-repo for the
    single-process POC).
    """
    app = FastAPI(title="Agent_A Cockpit", version="0.1.0")

    @app.get("/surfaces")
    def list_surfaces() -> dict[str, list[str]]:
        """List the registered surface names (for the UI menu)."""
        return {"surfaces": registry.list_surfaces()}

    @app.get("/surfaces/{surface_name}/{run_id}")
    def render_surface(surface_name: str, run_id: str) -> dict[str, object]:
        """Render the named surface for the given run."""
        try:
            snapshot = registry.render(surface_name, run_id)  # type: ignore[arg-type]
        except UnknownSurfaceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return snapshot.model_dump()

    @app.post("/plots")
    def render_plot(request: CockpitPlotRequest) -> dict[str, object]:
        """Render a plot via the engine."""
        try:
            response = engine.render(request)
        except UnknownPlotKindError as exc:
            # Pydantic Literal already catches unknown kinds at
            # the request level (HTTP 422), so this branch is
            # defensive — engine-side validation can still
            # raise if a future engine adds stricter checks.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            # Engine-side param validation (missing / malformed
            # per-kind params). 400 is the right code: the
            # request was syntactically valid Pydantic-wise
            # but the engine rejected the params.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return response.model_dump()

    return app


__all__ = ("build_cockpit_app",)
