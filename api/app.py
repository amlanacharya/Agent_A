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
  on the request body is 422 (the HTTP layer does not
  invent the value); a ``UnknownSurfaceError`` is 404;
  an engine-side ``ValueError`` is 400. The mapping is
  in one place so a future error kind is a deliberate
  addition.
* **In-process by default.** The ``build_cockpit_app``
  factory takes the registry and the engine as
  arguments so tests can wire in-memory providers
  without monkey-patching.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.models import CockpitPlotRequest
from api.plot_engine import PlotEngine
from api.surfaces import SurfaceRegistry, UnknownSurfaceError


# Default location for run outputs. CB4 of Phase 9 added this — the
# cockpit's run selector reads from this directory by default. The
# dev launcher (api/__main__.py) uses a temp dir; the platform's
# production startup module wires the real ``backend/outputs/``
# path. Keeping the default as the cwd-relative ``outputs/`` lets
# both the launcher and the production wiring use the same code
# path without an env-var dance.
DEFAULT_OUTPUTS_ROOT = Path(os.environ.get("AGENT_A_OUTPUTS_ROOT", "outputs")).resolve()


def build_cockpit_app(
    registry: SurfaceRegistry,
    engine: PlotEngine,
    outputs_root: Path | None = None,
) -> FastAPI:
    """Build the FastAPI app wired to ``registry`` + ``engine``.

    The factory takes both as arguments so tests can wire
    in-memory providers. The production wiring is in the
    platform's startup module (out-of-repo for the
    single-process POC).
    """
    app = FastAPI(title="Agent_A Cockpit", version="0.1.0")

    # Resolve the run outputs root once. If the caller passes one
    # (e.g. the dev launcher uses a temp dir), prefer it. Otherwise
    # fall back to the env-var / cwd-relative default.
    root = (outputs_root or DEFAULT_OUTPUTS_ROOT).resolve()

    @app.get("/surfaces")
    def list_surfaces() -> dict[str, list[str]]:
        """List the registered surface names (for the UI menu)."""
        return {"surfaces": registry.list_surfaces()}

    @app.get("/runs")
    def list_runs() -> dict[str, list[dict[str, object]]]:
        """List the run directories under ``outputs_root``.

        Each entry is ``{run_id, path, last_modified_iso}``. The
        cockpit's run selector (CB4) consumes this to populate
        the left-rail dropdown. A directory is a run iff it contains
        at least one of the four Phase 7 monitoring artifacts
        (MONITORING_REPORT.md / DRIFT_REPORT.md / OVERRIDE_ANALYSIS.md
        / MODEL_HEALTH.md); this avoids surfacing scratch directories
        that happen to live under ``outputs/``.
        """
        if not root.exists():
            return {"runs": []}
        artifact_names = {
            "MONITORING_REPORT.md",
            "DRIFT_REPORT.md",
            "OVERRIDE_ANALYSIS.md",
            "MODEL_HEALTH.md",
        }
        runs: list[dict[str, object]] = []
        for child in sorted(root.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            if not any((child / name).exists() for name in artifact_names):
                continue
            stat = child.stat()
            runs.append({
                "run_id": child.name,
                "path": str(child),
                "last_modified_iso": stat.st_mtime,
            })
        return {"runs": runs}

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
        """Render a plot via the engine.

        Pydantic ``Literal`` on ``CockpitPlotRequest.kind`` rejects
        unknown kinds at the request boundary (HTTP 422), so the
        engine never sees them. The only error the engine can
        raise here is ``ValueError`` for missing / malformed
        per-kind params — translated to 400.
        """
        try:
            response = engine.render(request)
        except ValueError as exc:
            # Engine-side param validation (missing / malformed
            # per-kind params). 400 is the right code: the
            # request was syntactically valid Pydantic-wise
            # but the engine rejected the params.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return response.model_dump()

    return app


__all__ = ("build_cockpit_app",)
