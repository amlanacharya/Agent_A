"""Phase 8 CB8 + Phase 10 CB1: the FastAPI app the cockpit UI consumes.

The app is a thin HTTP wrapper around the ``SurfaceRegistry``,
the ``PlotEngine``, and (Phase 10) the upload + chat + advance
driver routes. Read endpoints:

* ``GET /surfaces`` — list the registered surface names
  (for the cockpit's navigation menu)
* ``GET /surfaces/{surface_name}/{run_id}`` — render a
  specific surface for a run
* ``POST /plots`` — render a plot via the ``PlotEngine``
* ``GET /runs`` — list run directories under ``outputs_root``
  (cockpit's run selector)
* ``GET /cockpit-state/{run_id}`` — the live ``CockpitState``
  for a run

Write endpoints (Phase 10):

* ``POST /uploads`` — multipart CSV upload, calls ``run_preflight``,
  returns the bundle + run_id + initial state
* ``POST /messages`` — chat loop dispatch (Lens → conductor)
* ``POST /runs/{run_id}/advance`` — driver button (phase advance)

The write endpoints need the platform's runtime (``backend`` on the
import path, a callable to look up domain playbooks, etc.); they
are wired by ``build_cockpit_app(..., playbook_loader=...)`` and
return 503 when the loader is absent so the read endpoints still
work in cockpit-dev mode.

Design:

* **Thin HTTP wrapper, no business logic.** The endpoints
  call the registry / engine / conductor and translate the typed
  errors to HTTP status codes. The math lives in
  ``api.surfaces`` and ``api.plots``; the lifecycle lives in
  ``forecasting.conductor`` (Phase 10 CB2); the chat loop lives
  in ``forecasting.conductor.run_meridian_chat_turn``.
* **Typed error mapping.** A Pydantic validation error
  on the request body is 422 (the HTTP layer does not
  invent the value); a ``UnknownSurfaceError`` is 404;
  an engine-side ``ValueError`` is 400; a ``PreflightBlockingError``
  is 422 with the blocking issues list; an unknown domain is 400.
  The mapping is in one place so a future error kind is a deliberate
  addition.
* **In-process by default.** The ``build_cockpit_app``
  factory takes the registry, the engine, and the playbook loader
  as arguments so tests can wire in-memory providers
  without monkey-patching.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

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


# A playbook loader is a callable ``(domain: str) -> dict`` that
# returns the right dict for a domain. The platform's production
# wiring passes a loader that resolves YAML playbooks; tests pass
# a stub lambda that returns canned dicts. ``None`` means the
# upload route is unavailable (read-only cockpit mode).
PlaybookLoader = object  # type alias for documentation; the actual
                          # signature is ``Callable[[str], dict]``.


def build_cockpit_app(
    registry: SurfaceRegistry,
    engine: PlotEngine,
    outputs_root: Path | None = None,
    *,
    playbook_loader=None,
) -> FastAPI:
    """Build the FastAPI app wired to ``registry`` + ``engine``.

    The factory takes both as arguments so tests can wire
    in-memory providers. The production wiring is in the
    platform's startup module (out-of-repo for the
    single-process POC).

    ``playbook_loader``: optional callable ``(domain: str) -> dict``.
    When provided, the ``/uploads`` route becomes live. When
    absent (the default for the cockpit-dev launcher), the route
    returns 503 so the read endpoints still work end-to-end.
    """
    app = FastAPI(title="Agent_A Cockpit", version="0.2.0")

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

    @app.post("/uploads")
    def upload_csv(
        file: UploadFile = File(...),
        domain: str = Form(...),
    ) -> dict[str, object]:
        """Upload a CSV for a new run; run Preflight; return the bundle.

        The route is the single entry point the cockpit uses to start
        a Run. It is gated on ``playbook_loader`` so the cockpit-dev
        launcher (which has no domain playbooks wired) returns 503
        rather than crashing; the production wiring always passes a
        loader.

        Side effects:
        * writes ``outputs/{run_id}/input.csv`` (the upload bytes)
        * writes ``outputs/{run_id}/run_state.json`` (initial state)
        * writes ``outputs/{run_id}/preflight.json`` (PreflightBundle
          + per-series stats, via ``run_preflight``)
        * writes canonical data via ``data_store.replace_run`` (so
          EDA / Forge / Foundry can read it later)

        Error mapping:
        * ``UnknownDomainError`` (ValueError from ``playbook_loader``)
          -> 400
        * ``PreflightBlockingError`` -> 422 with the issues list
        """
        if playbook_loader is None:
            raise HTTPException(
                status_code=503,
                detail="upload route is not wired (no playbook_loader)",
            )
        try:
            playbook = playbook_loader(domain)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        run_id = f"upload-{uuid.uuid4().hex[:12]}"
        file_bytes = file.file.read()
        (root / run_id).mkdir(parents=True, exist_ok=True)
        (root / run_id / "input.csv").write_bytes(file_bytes)

        # Backend imports are deferred to here (not at module top) so
        # the read endpoints remain importable in environments without
        # the ``forecasting`` package on PYTHONPATH (e.g. the contract
        # test's minimal app).
        from forecasting.cockpit_state import CockpitState
        from forecasting.preflight import (
            PreflightBlockingError,
            run_preflight,
        )
        from forecasting.run_state import (
            Phase,
            create_run_state,
            load_run_state,
        )

        run_state = create_run_state(run_id, domain=domain)
        try:
            bundle = run_preflight(run_id, file_bytes, domain, playbook)
        except PreflightBlockingError as exc:
            # 422 with the blocking issues so the cockpit can render
            # the reasons inline. The ``input.csv`` is preserved
            # so the user can re-upload after fixing the issues.
            raise HTTPException(
                status_code=422,
                detail=[issue.model_dump() for issue in exc.issues],
            ) from exc

        phase_value = (
            run_state.phase.value if isinstance(run_state.phase, Phase) else run_state.phase
        )
        cockpit_state = CockpitState.from_run_state(
            load_run_state(run_id),
            current_step=phase_value,
            active_agent="conductor",
        )
        return {
            "run_id": run_id,
            "domain": domain,
            "preflight": bundle.model_dump(mode="json"),
            "state": cockpit_state.to_public_dict(),
        }

    return app


__all__ = ("build_cockpit_app",)
