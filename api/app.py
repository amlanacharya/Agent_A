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

from api.models import AdvanceRequest, CockpitPlotRequest, MessageRequest
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

    @app.post("/messages")
    def post_message(request: MessageRequest) -> dict[str, object]:
        """Chat-loop dispatch — Phase 10 CB3.

        The cockpit's chat box posts here. The route:

        1. Loads the run's ``RunState`` (404 if missing).
        2. Loads the conversation history from
           ``outputs/{run_id}/obs_log.json`` via
           ``conductor.load_lens_conversation_history``.
        3. Calls ``Lens.classify_intent`` to classify the user's
           message into one of the six intent kinds.
        4. Dispatches to the Conductor (or ``conductor_tools``) based
           on the intent and returns the typed ``ConductorStepResult``
           (or a tailored prism-clone payload for WHAT_IF_REQUEST).

        Error mapping:

        * Run not found -> 404
        * CORRECTION outside ``meridian_scoping`` -> 422
        * ``LensResponseError`` (malformed classifier output) -> 502
        * LifecycleError (illegal phase advance) -> 400
        """
        # Backend imports are deferred to here (same rationale as the
        # ``/uploads`` route — keep read endpoints importable in
        # environments without the ``forecasting`` package on PYTHONPATH).
        from forecasting.agents.lens import LensInput, classify_intent
        from forecasting.agents.lens import LensResponseError
        from forecasting.conductor import (
            Conductor,
            load_lens_conversation_history,
        )
        from forecasting.run_state import (
            LifecycleError,
            Phase,
            RunNotFoundError,
            load_run_state,
        )

        try:
            state = load_run_state(request.run_id)
        except (FileNotFoundError, RunNotFoundError) as exc:
            raise HTTPException(
                status_code=404,
                detail=f"run '{request.run_id}' not found",
            ) from exc

        history = load_lens_conversation_history(root, request.run_id)
        try:
            lens_input = LensInput(
                conversation_history=history,
                user_message=request.user_message,
                pipeline_state=state,
            )
            intent = classify_intent(lens_input)
        except LensResponseError as exc:
            # The Lens classifier returned malformed JSON / wrong
            # shape. 502 is the right code: the upstream (LLM)
            # returned an unusable response.
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        conductor = Conductor(
            run_id=request.run_id,
            outputs_root=root,
        )

        # Dispatch on the typed intent. Each branch returns the
        # response payload the cockpit consumes; we keep the
        # response shapes uniform (intent, run_id, reply,
        # possibilities) and add intent-specific extras
        # (``advanced_to``, ``state``, ``prism_run_id``).
        if intent.intent == "SCOPE_RESPONSE":
            result = conductor.record_scope_response(request.user_message)
            return {
                "intent": intent.intent,
                "run_id": request.run_id,
                "reply": result.reply,
                "possibilities": [p.model_dump(mode="json") for p in result.possibilities],
                "advanced_to": result.advanced_to,
                "state": result.state.model_dump(mode="json") if result.state else None,
            }

        if intent.intent == "OVERRIDE":
            # Log the override as a USER_OVERRIDE_ACCEPTED claim so
            # the audit trail captures it. The reply is plain; no
            # possibility chips for an override.
            result = conductor.record_override(request.user_message)
            return {
                "intent": intent.intent,
                "run_id": request.run_id,
                "reply": result.reply,
                "possibilities": [p.model_dump(mode="json") for p in result.possibilities],
                "advanced_to": result.advanced_to,
                "state": result.state.model_dump(mode="json") if result.state else None,
            }

        if intent.intent == "ADVANCE_PIPELINE":
            # Pick the next drive method based on the current phase.
            # Phase transitions are PREFLIGHT→MERIDIAN_SCOPING→
            # FORGE_EDA→FOUNDRY_MODELLING→REPORT_READY.
            current_phase = state.phase
            if isinstance(current_phase, Phase):
                current_phase_value = current_phase.value
            else:
                current_phase_value = current_phase

            advance_map = {
                "preflight": conductor.drive_run_to_meridian,
                "meridian_scoping": conductor.drive_run_to_forge,
                "forge_eda": conductor.drive_run_to_foundry,
                "foundry_modelling": conductor.drive_run_to_report,
            }
            drive = advance_map.get(current_phase_value)
            if drive is None:
                # Already at REPORT_READY (or HALTED) — nothing to advance.
                return {
                    "intent": intent.intent,
                    "run_id": request.run_id,
                    "reply": (
                        f"Run is already at phase {current_phase_value}; "
                        "no further advance is possible."
                    ),
                    "possibilities": [],
                    "advanced_to": current_phase_value,
                    "state": state.model_dump(mode="json"),
                }
            try:
                result = drive(state)
            except LifecycleError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {
                "intent": intent.intent,
                "run_id": request.run_id,
                "reply": result.reply,
                "possibilities": [p.model_dump(mode="json") for p in result.possibilities],
                "advanced_to": result.advanced_to,
                "state": result.state.model_dump(mode="json") if result.state else None,
            }

        if intent.intent == "CLARIFICATION":
            # Low-confidence CLARIFICATION (per plan: < 0.6) gets the
            # two-option clarification reply. Anything at-or-above
            # 0.6 falls through to a single "I'll proceed" ack.
            if intent.confidence < 0.6:
                result = conductor.author_clarification(request.user_message)
                return {
                    "intent": intent.intent,
                    "run_id": request.run_id,
                    "reply": result.reply,
                    "possibilities": [p.model_dump(mode="json") for p in result.possibilities],
                    "state": result.state.model_dump(mode="json") if result.state else None,
                }
            return {
                "intent": intent.intent,
                "run_id": request.run_id,
                "reply": f"Got it: \"{request.user_message}\". I'll proceed.",
                "possibilities": [],
            }

        if intent.intent == "CORRECTION":
            # Per the plan, CORRECTION is only valid in
            # meridian_scoping. Outside that phase the cockpit
            # shouldn't surface the correction affordance, but if it
            # does we reject the request loudly.
            current_phase = state.phase
            current_phase_value = (
                current_phase.value if isinstance(current_phase, Phase)
                else current_phase
            )
            if current_phase_value != Phase.MERIDIAN_SCOPING.value:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"CORRECTION only valid in meridian_scoping; "
                        f"run is at {current_phase_value}"
                    ),
                )
            result = conductor.record_correction(request.user_message)
            return {
                "intent": intent.intent,
                "run_id": request.run_id,
                "reply": result.reply,
                "possibilities": [p.model_dump(mode="json") for p in result.possibilities],
                "advanced_to": result.advanced_to,
                "state": result.state.model_dump(mode="json") if result.state else None,
            }

        if intent.intent == "WHAT_IF_REQUEST":
            # Delegate to conductor_tools.create_prism_run which
            # creates the whatif directory and returns a whatif_id.
            scenario = intent.entities.scenario or request.user_message
            prism = conductor._conductor_tools.create_prism_run(  # noqa: SLF001
                request.run_id,
                scenario_description=scenario,
                entities=intent.entities.model_dump(mode="python"),
            )
            return {
                "intent": intent.intent,
                "run_id": request.run_id,
                "reply": "Scenario run created — open it from the run selector.",
                "possibilities": [],
                "prism_run_id": prism["whatif_id"],
            }

        # Exhaustiveness fallback — a future IntentType addition
        # that the route doesn't handle yet is a bug; refuse
        # loudly.
        raise HTTPException(
            status_code=500,
            detail=f"unhandled intent kind: {intent.intent}",
        )

    @app.post("/runs/{run_id}/advance")
    def post_advance(run_id: str, request: AdvanceRequest | None = None) -> dict[str, object]:
        """Driver-button advance — Phase 10 CB4.

        The cockpit's "Advance to next phase" button posts here.
        The route delegates to ``Conductor.drive_run_to_next``,
        which dispatches based on the current phase and either
        advances one boundary, chains the rest of the pipeline
        (forge_eda / force=true), or refuses with a 409 gate
        (meridian_scoping without force).

        Error mapping:

        * Run not found -> 404
        * ``AdvanceGateError`` (meridian_scoping without force, or
          terminal phase) -> 409 with the gate message in ``detail``
        * ``LifecycleError`` (illegal phase advance) -> 400
        """
        from forecasting.conductor import AdvanceGateError, Conductor
        from forecasting.run_state import LifecycleError, RunNotFoundError, load_run_state

        try:
            state = load_run_state(run_id)
        except (FileNotFoundError, RunNotFoundError) as exc:
            raise HTTPException(
                status_code=404,
                detail=f"run '{run_id}' not found",
            ) from exc

        force = bool(request.force) if request is not None else False
        conductor = Conductor(run_id=run_id, outputs_root=root)
        try:
            result = conductor.drive_run_to_next(state, force=force)
        except AdvanceGateError as exc:
            # 409: the request was understood but the resource is
            # in a state that prevents the action. The client
            # should not retry until the user has addressed the
            # gate (e.g. sent a chat message to Meridian).
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "advance_gated",
                    "phase": exc.phase,
                    "message": exc.message,
                },
            ) from exc
        except LifecycleError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "run_id": run_id,
            "advanced_to": result.advanced_to,
            "reply": result.reply,
            "possibilities": [p.model_dump(mode="json") for p in result.possibilities],
            "state": result.state.model_dump(mode="json") if result.state else None,
        }

    @app.get("/cockpit-state/{run_id}")
    def get_cockpit_state(run_id: str) -> dict[str, object]:
        """Live cockpit state for a run — Phase 10 CB6.

        The RunConsole page polls this every 5s while a run is
        in flight so the left-rail shows current_step +
        active_agent + blockers without re-fetching the full
        surface. Returns the ``CockpitState.to_public_dict()``
        shape plus a top-level ``phase`` field (the UI's
        dispatch logic reads it to decide which button to
        show: advance vs. report link).

        Error mapping:

        * Run not found -> 404
        """
        from forecasting.cockpit_state import CockpitState
        from forecasting.run_state import RunNotFoundError, load_run_state

        try:
            run_state = load_run_state(run_id)
        except (FileNotFoundError, RunNotFoundError) as exc:
            raise HTTPException(
                status_code=404,
                detail=f"run '{run_id}' not found",
            ) from exc

        phase_value = (
            run_state.phase.value
            if hasattr(run_state.phase, "value")
            else run_state.phase
        )
        # ``active_agent`` is a coarse mapping: the conductor
        # owns the linear pipeline (preflight through report)
        # and ``meridian`` owns the chat loop. The cockpit
        # reads this to render the agent avatar in the left
        # rail.
        active_agent = "meridian" if phase_value == "meridian_scoping" else "conductor"
        cockpit_state = CockpitState.from_run_state(
            run_state,
            current_step=phase_value,
            active_agent=active_agent,
        )
        payload = cockpit_state.to_public_dict()
        # Add the top-level ``phase`` field the UI's dispatch
        # logic needs. ``current_step`` already carries it but
        # the spec calls for a dedicated ``phase`` key so the
        # UI doesn't have to alias.
        payload["phase"] = phase_value
        return payload

    return app


__all__ = ("build_cockpit_app",)
