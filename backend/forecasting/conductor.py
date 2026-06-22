"""Conductor orchestrator — Phase 10 CB2.

The Conductor is the single entry point that drives a Run through the
lifecycle:

    preflight → meridian_scoping → forge_eda → foundry_modelling → report_ready

Each ``drive_run_to_<phase>`` method is a thin orchestrator over the
existing platform machinery:

* ``drive_run_to_meridian`` calls ``conductor_tools.advance_to_meridian``
  and returns Meridian's first chat prompt (templated text + an
  empty possibilities list — the LLM-driven Meridian lands in Phase 10.x).
* ``drive_run_to_forge`` reads the canonical data + segment map the
  preflight persisted, calls ``eda_toolbox.build_eda_report``, and
  persists ``eda_report.json``.
* ``drive_run_to_foundry`` builds the feature table, calls
  ``forecast_harness.run_forecast_harness`` + ``ensemble.summarise_scorecards``,
  and persists ``foundry_report.json`` + ``forecast_harness_report.json``.
* ``drive_run_to_report`` assembles the typed ``FoundryReport`` view
  of the harness output, persists ``report.json``, and writes the
  four Phase 7 markdown monitoring artifacts (so the MLOps Monitor
  surface has something to render).

The Conductor is the seam between the in-process ``run_meridian_chat_turn``
helper (Phase 10 CB3) and the existing ``conductor_tools`` /
``preflight`` / ``eda_toolbox`` / ``forecast_harness`` modules. Every
phase transition goes through ``advance_phase`` — no direct
``state.phase = ...`` mutations (per Addition E in the June 22
architecture review).

Idempotent re-invocation: per ADR-0005 ("resume is idempotent
re-invocation"), each method skips work if its artifact already
exists. So a partial advance + crash + re-advance picks up where
the last successful step left off rather than re-doing the work.

Constructor injection: every external dependency (the legacy
``conductor_tools``, the EDA toolbox, the forecast harness, the
ensemble, the promotion layer, and the outputs root) is an injected
parameter. Production wiring passes the real modules; tests pass
stubs. This is the same seam pattern as ``config_escalation.Mea
sureMASE`` and the PlotEngine ABC.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from forecasting.contracts import (
    Claim,
    ClaimLedger,
    ConductorStepResult,
    EDAReport,
    EnsembleSummary,
    ForecastHarnessReport,
    FoundryReport,
    Possibility,
    SegmentMap,
    SeriesResult,
)
from forecasting.eda_toolbox import build_eda_report
from forecasting.ensemble import summarise_scorecards
from forecasting.feature_factory import build_feature_table
from forecasting.forecast_harness import run_forecast_harness
from forecasting.forecast_harness import ForecastRequest  # re-export for callers
from forecasting.run_state import (
    Phase,
    RunState,
    advance_phase,
    load_run_state,
    save_run_state,
)


class AdvanceGateError(Exception):
    """Raised when the ``/advance`` route refuses to advance a run.

    Distinct from ``LifecycleError`` (which is about an illegal
    phase transition) because the refusal here is about the
    cockpit flow control — a run that's legally at phase X but
    can't be advanced from the button until the user does
    something else first (e.g. chat with Meridian in
    ``meridian_scoping``).

    The HTTP layer maps this to 409 Conflict because the request
    was understood but the resource is in a state that prevents
    the action; the client should not retry until the user has
    addressed the gate (sent a chat message).
    """

    def __init__(self, run_id: str, phase: str, message: str) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.phase = phase
        self.message = message

# ``ConductorStepResult.state`` is typed as a forward-ref string
# (``"RunState | None"``) because importing RunState at the top of
# contracts.py would force a cycle. Resolve it now that RunState is
# in scope. Idempotent — safe to call again.
ConductorStepResult.model_rebuild()


# Default injection: the production modules. Tests override by passing
# stubs to the Conductor constructor.
DEFAULT_CONDUCTOR_TOOLS = None  # imported lazily in __init__ to avoid
                                # a circular import with conductor_tools


def _default_conductor_tools():
    """Late-bind ``conductor_tools`` to dodge the import cycle.

    ``conductor_tools`` itself imports from ``run_state``, and we
    import from ``run_state`` above; the cycle is closed at runtime
    when the test injects a stub or production wires a real
    conductor_tools module by name. This default is the lazy escape
    hatch.
    """
    import forecasting.tools.conductor_tools as conductor_tools

    return conductor_tools


class Conductor:
    """Orchestrate a single Run through the lifecycle.

    One Conductor instance per Run. Holds the run_id, the injected
    dependency stubs, and the outputs root where artifacts land.
    Constructed lazily from the FastAPI route after a Run is created
    via ``POST /uploads``.
    """

    def __init__(
        self,
        run_id: str,
        *,
        conductor_tools=None,
        eda_toolbox=None,
        forecast_harness=None,
        ensemble=None,
        promotion=None,
        outputs_root: Path | None = None,
    ) -> None:
        self.run_id = run_id
        self._conductor_tools = conductor_tools or _default_conductor_tools()
        # The toolbox / harness / ensemble modules are passed as
        # whole modules — the conductor calls their public functions
        # directly. This keeps the seam narrow (one parameter per
        # dependency) while letting tests swap in stubs at the
        # module level.
        self._eda_toolbox = eda_toolbox or _default_eda_toolbox()
        self._forecast_harness = forecast_harness or _default_forecast_harness()
        self._ensemble = ensemble or _default_ensemble()
        self._promotion = promotion  # optional; only used by report step
        self._outputs_root = (
            Path(outputs_root).resolve() if outputs_root is not None
            else _default_outputs_root()
        )
        self._run_dir = self._outputs_root / run_id

    # ------------------------------------------------------------------
    # Phase advances — each method writes its artifact BEFORE
    # transitioning, so a crash mid-write leaves the previous phase
    # intact and re-invocation skips the work.
    # ------------------------------------------------------------------

    def drive_run_to_meridian(self, state: RunState) -> ConductorStepResult:
        """Transition ``state`` PREFLIGHT → MERIDIAN_SCOPING and return
        Meridian's first chat prompt.

        The first-cut reply is templated ("Let's scope your forecast
        — I'll ask a few questions about the data and your goals")
        because the LLM-driven Meridian lands in Phase 10.x. The
        possibilities list is empty in the templated cut; the chat
        loop fills it in via ``run_meridian_chat_turn`` once the user
        sends a message.
        """
        self._conductor_tools.advance_to_meridian(self.run_id, "")
        advanced = load_run_state(self.run_id)
        return ConductorStepResult(
            reply=(
                "Hi — I'm Meridian, your forecasting assistant. "
                "Let's scope your forecast together. I'll ask a few "
                "questions about the data and your goals before "
                "running the model."
            ),
            possibilities=[
                Possibility(
                    kind="ACCEPT",
                    label="Get started",
                    payload={"ready": True},
                ),
            ],
            advanced_to=Phase.MERIDIAN_SCOPING.value,
            state=advanced,
        )

    def drive_run_to_forge(self, state: RunState) -> ConductorStepResult:
        """Transition MERIDIAN_SCOPING → FORGE_EDA, run EDA, persist artifact.

        Idempotent: if ``eda_report.json`` already exists, the method
        just transitions and returns.
        """
        eda_path = self._run_dir / "eda_report.json"
        if not eda_path.exists():
            canonical_table, segment_map = self._load_canonical_and_segments()
            report = self._eda_toolbox.build_eda_report(
                canonical_table, segment_map
            )
            self._write_json(eda_path, report.model_dump(mode="json"))
        advanced = advance_phase(state, Phase.FORGE_EDA)
        save_run_state(advanced)
        return ConductorStepResult(
            reply="EDA complete — running Forge.",
            possibilities=[],
            advanced_to=Phase.FORGE_EDA.value,
            state=advanced,
        )

    def drive_run_to_foundry(self, state: RunState) -> ConductorStepResult:
        """Transition FORGE_EDA → FOUNDRY_MODELLING, run the harness + ensemble.

        Idempotent: if ``forecast_harness_report.json`` already exists,
        just transitions.
        """
        harness_path = self._run_dir / "forecast_harness_report.json"
        if not harness_path.exists():
            # Build the feature table from the canonical data the
            # preflight persisted. Default FeatureFlags (everything
            # on) is fine for Phase 10's first cut; per-series flags
            # land with the LLM-driven Meridian in Phase 10.x.
            from forecasting.contracts import FeatureFlags

            canonical_table, _ = self._load_canonical_and_segments()
            features = build_feature_table(
                canonical_table, FeatureFlags()
            )
            request = ForecastRequest(
                run_id=self.run_id,
                # ``model_families`` omitted → use the ForecastRequest
                # default (all registered families). Passing ``None``
                # would override the default with a literal null, which
                # the contract rejects.
                horizon=1,
            )
            harness_report = self._forecast_harness.run_forecast_harness(
                request, features=features
            )
            self._write_json(
                self._run_dir / "forecast_harness_report.json",
                harness_report.model_dump(mode="json"),
            )
            ensemble_summary = self._ensemble.summarise_scorecards(
                harness_report.scorecards
            )
            # ``summarise_scorecards`` returns an ``EnsembleTracker``,
            # not a Pydantic ``EnsembleSummary``. The tracker has a
            # ``.summary()`` method that produces the cockpit-facing
            # Pydantic view; the FoundryReport's ``ensemble`` field
            # is typed as ``EnsembleSummary`` so we need that shape.
            if hasattr(ensemble_summary, "summary"):
                ensemble_summary = ensemble_summary.summary()
            self._write_json(
                self._run_dir / "ensemble_summary.json",
                ensemble_summary.model_dump(mode="json"),
            )
            # Foundry report is the cockpit-facing view: per-series
            # outcomes, narrative, overall MASE.
            foundry_report = self._build_foundry_report(harness_report, ensemble_summary)
            self._write_json(
                self._run_dir / "foundry_report.json",
                foundry_report.model_dump(mode="json"),
            )
        advanced = advance_phase(state, Phase.FOUNDRY_MODELLING)
        save_run_state(advanced)
        return ConductorStepResult(
            reply="Foundry done — assembling the report.",
            possibilities=[],
            advanced_to=Phase.FOUNDRY_MODELLING.value,
            state=advanced,
        )

    def drive_run_to_report(self, state: RunState) -> ConductorStepResult:
        """Transition FOUNDRY_MODELLING → REPORT_READY, write the report JSON.

        The report.json is the Phase 10 Report equivalent: overall
        MASE, per-series outcomes, Foundry narrative. The four Phase 7
        markdown monitoring artifacts are also written here so the
        MLOps Monitor surface has something to render (Phase 11+
        will trigger monitoring on a Schedule rather than at report time).
        """
        report_path = self._run_dir / "report.json"
        if report_path.exists():
            # Idempotency: the report is already on disk. Just
            # transition and return — skip the foundry_report
            # load so a re-advance doesn't crash if the foundry
            # artifact is missing for any reason.
            advanced = advance_phase(state, Phase.REPORT_READY)
            save_run_state(advanced)
            return ConductorStepResult(
                reply="Report ready — open the cockpit to review.",
                possibilities=[],
                advanced_to=Phase.REPORT_READY.value,
                state=advanced,
            )
        foundry_report = self._load_foundry_report()
        report_payload = {
            "run_id": self.run_id,
            "phase": Phase.REPORT_READY.value,
            "overall_mase": foundry_report.overall_mase,
            "target_met_fraction": foundry_report.target_met_fraction,
            "series_count": len(foundry_report.series_results),
            "narrative": foundry_report.narrative,
            "series_results": [
                r.model_dump(mode="json")
                for r in foundry_report.series_results
            ],
        }
        self._write_json(report_path, report_payload)
        # Emit stub Phase 7 markdown so /runs surfaces this Run
        # immediately. Phase 11 swaps these for real drift
        # detection on a Schedule.
        for name in (
            "MONITORING_REPORT.md",
            "DRIFT_REPORT.md",
            "OVERRIDE_ANALYSIS.md",
            "MODEL_HEALTH.md",
        ):
            (self._run_dir / name).write_text(
                f"# {name}\n\n_(emitted at report_ready for run {self.run_id}; Phase 11 wires real monitoring)_\n",
                encoding="utf-8",
            )
        advanced = advance_phase(state, Phase.REPORT_READY)
        save_run_state(advanced)
        return ConductorStepResult(
            reply="Report ready — open the cockpit to review.",
            possibilities=[],
            advanced_to=Phase.REPORT_READY.value,
            state=advanced,
        )

    def drive_run_to_next(
        self,
        state: RunState,
        *,
        force: bool = False,
    ) -> ConductorStepResult:
        """Advance the run by one phase boundary (the ``/advance`` button).

        Dispatch table (mirrors the CB4 plan spec):

        * ``preflight`` → ``drive_run_to_meridian``
        * ``meridian_scoping`` → raise ``AdvanceGateError`` unless
          ``force=True`` (the user must chat first; ``force`` is the
          escape hatch the operator can flip from the cockpit dev
          console)
        * ``forge_eda`` → chain ``drive_run_to_forge`` →
          ``drive_run_to_foundry`` → ``drive_run_to_report`` (EDA is
          fast; the user clicked "advance", so they trust the rest
          of the pipeline)
        * ``foundry_modelling`` → ``drive_run_to_report``
        * ``report_ready`` → noop ``ConductorStepResult``
        * ``halted`` → ``AdvanceGateError`` (a halted run is
          terminal; the cockpit must start a new run)

        Each underlying ``drive_run_to_<phase>`` method is itself
        idempotent (skips work if its artifact already exists), so a
        partial advance + crash + re-advance resumes cleanly. The
        returned ``ConductorStepResult`` is the **last** drive
        step's result — that's the state the cockpit needs to
        render (the final phase + the final reply).
        """
        phase_value = state.phase
        if hasattr(phase_value, "value"):
            phase_value = phase_value.value

        if phase_value == "preflight":
            return self.drive_run_to_meridian(state)

        if phase_value == "meridian_scoping":
            if not force:
                raise AdvanceGateError(
                    run_id=self.run_id,
                    phase=phase_value,
                    message=(
                        "Run is in meridian_scoping — the user must "
                        "answer Meridian's questions via POST /messages "
                        "before advancing."
                    ),
                )
            # force=True: bypass the chat gate and chain to
            # report_ready. This is the operator escape hatch.
            return self._chain_to_report_ready("meridian_scoping")

        if phase_value == "forge_eda":
            # Per the CB4 plan: forge_eda → drive the remaining
            # phases in one call. EDA is fast, so the user clicked
            # advance expecting the whole pipeline to finish.
            return self._chain_to_report_ready("forge_eda")

        if phase_value == "foundry_modelling":
            return self.drive_run_to_report(state)

        if phase_value == "report_ready":
            return ConductorStepResult(
                reply="Run is already at report_ready — nothing to advance.",
                possibilities=[],
                advanced_to="report_ready",
                state=load_run_state(self.run_id),
            )

        # ``halted`` and any future terminal state land here.
        raise AdvanceGateError(
            run_id=self.run_id,
            phase=phase_value,
            message=(
                f"Run is at {phase_value}; cannot advance from a "
                "terminal state."
            ),
        )

    def _chain_to_report_ready(self, starting_phase: str) -> ConductorStepResult:
        """Drive the pipeline from ``starting_phase`` through to report_ready.

        Used by ``drive_run_to_next`` for the chained case (when
        the user clicks advance from ``forge_eda`` or
        ``meridian_scoping`` with ``force=True``). Each drive
        method is idempotent — re-invocation skips the work if
        the artifact already exists. The returned result is the
        **last** step's result, so the cockpit sees the final
        phase + the final reply.

        ``starting_phase`` decides which boundary to start at:

        * ``meridian_scoping`` (force=true) → drive forge → foundry → report
        * ``forge_eda`` → drive foundry → report
        * ``foundry_modelling`` → drive report
        * ``report_ready`` → noop
        """
        if starting_phase == "meridian_scoping":
            state = load_run_state(self.run_id)
            self.drive_run_to_forge(state)
        if starting_phase in ("meridian_scoping", "forge_eda"):
            state = load_run_state(self.run_id)
            self.drive_run_to_foundry(state)
        state = load_run_state(self.run_id)
        return self.drive_run_to_report(state)

    # ------------------------------------------------------------------
    # Chat-loop entry points — called by ``POST /messages`` after
    # ``Lens.classify_intent`` picks an intent. The dispatch logic
    # lives in the HTTP route; these methods are the typed
    # conductor-side handlers for each non-advance intent.
    # ------------------------------------------------------------------

    def record_scope_response(
        self,
        user_message: str,
    ) -> ConductorStepResult:
        """Persist the user's scoping answer as a Claim and ask the next question.

        The chat-loop payload (Lens entities + raw_quote) is folded into
        the Claim's ``applies_to`` and ``evidence_ref`` so the cockpit
        audit panel can show what the user said, when, and where it
        applied. The reply is templated because the agentic Meridian
        LLM call lands in Phase 10.x — the first cut just acknowledges
        the answer and invites the next one.
        """
        claim = Claim(
            claim_id=str(uuid.uuid4()),
            claim=user_message,
            verification_status="USER_OVERRIDE_ACCEPTED",
            evidence_type="user_confirmed",
            applies_to="run",
            downstream_impact="shapes the Meridian scope",
            must_surface_in_report=True,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._append_claim(claim)
        advanced = load_run_state(self.run_id)
        return ConductorStepResult(
            reply=(
                f"Got it — noted: \"{user_message}\". "
                "What else should I factor into the scope? "
                "(e.g. promo calendar, stockout handling, target metric)"
            ),
            possibilities=[
                Possibility(
                    kind="ACCEPT",
                    label="That's the full scope",
                    payload={"ready_to_advance": True},
                ),
                Possibility(
                    kind="OVERRIDE",
                    label="Revise a segment",
                    payload={"action": "open_segments"},
                ),
            ],
            advanced_to=advanced.phase,
            state=advanced,
        )

    def author_clarification(
        self,
        user_message: str,
    ) -> ConductorStepResult:
        """Generate a low-confidence clarification with two short options.

        Called when Lens returns ``CLARIFICATION`` with ``confidence < 0.6``.
        The two ``Possibility`` chips let the user pick the most likely
        interpretation without re-typing the message.
        """
        advanced = load_run_state(self.run_id)
        return ConductorStepResult(
            reply=(
                f"I want to make sure I understood: \"{user_message}\". "
                "Which of these is closest?"
            ),
            possibilities=[
                Possibility(
                    kind="ACCEPT",
                    label="Yes — proceed",
                    payload={"clarification": "proceed"},
                ),
                Possibility(
                    kind="CLARIFY",
                    label="Let me rephrase",
                    payload={"clarification": "rephrase"},
                ),
            ],
            advanced_to=advanced.phase,
            state=advanced,
        )

    def record_override(self, user_message: str) -> ConductorStepResult:
        """Persist a user OVERRIDE as a Claim; no advance, no chips.

        Called from ``POST /messages`` when Lens returns
        ``OVERRIDE``. The audit trail captures the override verbatim;
        the cockpit shows a short acknowledgement reply.
        """
        claim = Claim(
            claim_id=str(uuid.uuid4()),
            claim=user_message,
            verification_status="USER_OVERRIDE_ACCEPTED",
            evidence_type="user_confirmed",
            applies_to="run",
            downstream_impact="overrides an agent recommendation",
            must_surface_in_report=True,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._append_claim(claim)
        advanced = load_run_state(self.run_id)
        return ConductorStepResult(
            reply="Logged override — proceeding with your adjustment.",
            possibilities=[],
            advanced_to=advanced.phase,
            state=advanced,
        )

    def record_correction(self, user_message: str) -> ConductorStepResult:
        """Persist a user CORRECTION as a Claim (meridian_scoping only).

        Called from ``POST /messages`` when Lens returns ``CORRECTION``.
        The route rejects ``CORRECTION`` outside ``meridian_scoping``
        (422) so this helper assumes the phase guard has already
        passed.
        """
        claim = Claim(
            claim_id=str(uuid.uuid4()),
            claim=user_message,
            verification_status="USER_OVERRIDE_ACCEPTED",
            evidence_type="user_confirmed",
            applies_to="run",
            downstream_impact="corrects a prior scope statement",
            must_surface_in_report=True,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._append_claim(claim)
        advanced = load_run_state(self.run_id)
        return ConductorStepResult(
            reply="Correction logged — I'll update the scope.",
            possibilities=[],
            advanced_to=advanced.phase,
            state=advanced,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_claim(self, claim: Claim) -> None:
        """Append a Claim to ``outputs/{run_id}/claim_ledger.json``.

        The ledger is the auditable record of every user-stated scope,
        override, and correction. Phase 11 will fan the ledger out to
        the Learning Journal surface; for Phase 10 it's enough that
        the cockpit can read it back if it needs to render a
        conversation recap.
        """
        ledger_path = self._run_dir / "claim_ledger.json"
        if ledger_path.exists():
            ledger = ClaimLedger.model_validate_json(ledger_path.read_text())
        else:
            ledger = ClaimLedger(run_id=self.run_id)
        ledger.claims.append(claim)
        self._write_json(ledger_path, ledger.model_dump(mode="json"))

    def _load_canonical_and_segments(self) -> tuple[pd.DataFrame, SegmentMap]:
        """Load the canonical demand table + segment map from preflight artifacts.

        The preflight persists ``preflight.json`` with the bundle
        + segment_map.model_dump() under ``outputs/{run_id}/``. We
        read the segment map from there; the canonical table itself
        is held in ``data_store`` in-memory (one DataFrame per
        series, keyed by series_key). Preflight stores each frame
        with just ``["date", "demand"]`` columns — the series_key
        is the dict key, NOT a column. We tag each frame with its
        series_key before concatenating so downstream consumers
        (``build_feature_table``, ``run_forecast_harness``) see
        the canonical ``series_key`` / ``date`` / ``demand``
        contract.
        """
        from forecasting.data_store import get_series, get_series_keys
        from forecasting.contracts import SegmentMap

        preflight_path = self._run_dir / "preflight.json"
        if not preflight_path.exists():
            raise FileNotFoundError(
                f"preflight.json missing for run {self.run_id}; "
                "the run never reached Preflight or preflight.json was deleted"
            )
        payload = json.loads(preflight_path.read_text())
        segment_map = SegmentMap.model_validate(payload["segment_map"])

        # Reconstruct the canonical table from the data_store,
        # tagging each frame with its series_key so the concat
        # has the canonical ``series_key`` / ``date`` / ``demand``
        # columns.
        keys = get_series_keys(self.run_id)
        frames = []
        for key in keys:
            frame = get_series(self.run_id, key)
            # Defensive copy + column tagging. ``frame`` is
            # already a copy per data_store.get_series's contract
            # (``copy(deep=True)``).
            tagged = frame.assign(series_key=key)
            # ``assign`` appends the column at the end; the
            # canonical feature factory doesn't care about order
            # but the column must be present.
            frames.append(tagged)
        canonical_table = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )
        return canonical_table, segment_map

    def _load_foundry_report(self) -> FoundryReport:
        foundry_path = self._run_dir / "foundry_report.json"
        return FoundryReport.model_validate_json(foundry_path.read_text())

    def _build_foundry_report(
        self,
        harness: ForecastHarnessReport,
        ensemble: EnsembleSummary,
    ) -> FoundryReport:
        """Compose the cockpit-facing ``FoundryReport`` from the harness output."""
        per_series_results = [
            SeriesResult(
                series_key=series.series_key,
                sb_class=series.sb_class,
                mase_target=series.mase_target,
                results=series.results,
                best_model=series.best_model,
                target_met=series.target_met,
            )
            for series in harness.series_results
        ]
        overall_mase = (
            sum(r.results[0].mase for r in per_series_results if r.results)
            / max(len(per_series_results), 1)
        )
        target_met_fraction = (
            sum(1 for r in per_series_results if r.target_met)
            / max(len(per_series_results), 1)
        )
        return FoundryReport(
            run_id=self.run_id,
            series_results=per_series_results,
            overall_mase=overall_mase,
            target_met_fraction=target_met_fraction,
            narrative=f"Foundry modelled {len(per_series_results)} series.",
            ensemble=ensemble,
        )

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _default_eda_toolbox():
    """Late-bind ``eda_toolbox`` (same pattern as conductor_tools)."""
    from forecasting import eda_toolbox

    return eda_toolbox


def _default_forecast_harness():
    """Late-bind ``forecast_harness``."""
    from forecasting import forecast_harness

    return forecast_harness


def _default_ensemble():
    """Late-bind ``ensemble``."""
    from forecasting import ensemble

    return ensemble


def _default_outputs_root():
    """Default outputs root — same as ``run_state.OUTPUTS_ROOT``."""
    from forecasting.run_state import OUTPUTS_ROOT

    return OUTPUTS_ROOT


# ---------------------------------------------------------------------------
# Module-level chat-loop helpers (10.3.2 + 10.3.3)
# ---------------------------------------------------------------------------


def load_lens_conversation_history(
    outputs_root: Path | str | None,
    run_id: str,
) -> list["LensConversationTurn"]:
    """Adapter: read ``outputs/{run_id}/obs_log.json`` into Lens history.

    The ``obs_log`` is a JSON array of ``{event, ts, ...}`` entries
    written by ``conductor_tools._append_obs``. Only ``event="message"``
    entries are Lens-conversation material; HALT events are skipped
    (Lens doesn't classify halts). Per the 10.3.3 spec, an absent
    obs_log returns an empty list — Lens classifies on
    ``pipeline_state`` alone in that case.

    Lives in ``conductor.py`` (not ``lens.py``) because it's a
    conductor-side concern: the cockpit hands the conductor a
    raw ``user_message``, and the conductor loads the prior
    history to assemble the LensInput.
    """
    from forecasting.agents.lens import ConversationTurn as LensConversationTurn

    if outputs_root is None:
        return []
    obs_path = Path(outputs_root) / run_id / "obs_log.json"
    if not obs_path.exists():
        return []
    try:
        raw_log = json.loads(obs_path.read_text())
    except json.JSONDecodeError:
        return []
    if not isinstance(raw_log, list):
        return []
    history: list[LensConversationTurn] = []
    for entry in raw_log:
        if not isinstance(entry, dict):
            continue
        if entry.get("event") != "message":
            continue
        # ``role`` and ``agent`` are optional on the obs entry;
        # default to ``user`` and ``None`` respectively.
        role = entry.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        content = entry.get("content", "")
        history.append(
            LensConversationTurn(
                role=role,
                content=content,
                agent=entry.get("agent"),
            )
        )
    return history


def run_meridian_chat_turn(
    conductor: "Conductor",
    user_message: str,
    conversation_history: list | None = None,
) -> ConductorStepResult:
    """Templated Meridian reply for one user message.

    The first cut of the chat loop (Phase 10.3.2) does NOT call
    Lens.classify_intent here — the HTTP route does the Lens call
    and dispatches to ``Conductor.record_scope_response``,
    ``Conductor.author_clarification``, or one of the
    ``drive_run_to_*`` methods based on the intent. This helper is
    the **default reply** for SCOPE_RESPONSE messages that the
    cockpit routes to it (the ``record_scope_response`` method
    already covers the standard case; this helper exists so the
    chat loop has a single entry point and a single
    ``ConductorStepResult`` contract).

    The reply text is templated; the LLM-driven Meridian lands in
    Phase 10.x. ``conversation_history`` is accepted but unused in
    this first cut — the future LLM Meridian will consume it.
    """
    advanced = load_run_state(conductor.run_id)
    return ConductorStepResult(
        reply=(
            f"Understood: \"{user_message}\". "
            "I'll fold that into the scope."
        ),
        possibilities=[
            Possibility(
                kind="ACCEPT",
                label="Continue",
                payload={"ready": True},
            ),
        ],
        advanced_to=advanced.phase,
        state=advanced,
    )


__all__ = (
    "AdvanceGateError",
    "Conductor",
    "ConductorStepResult",
    "Possibility",
    "run_meridian_chat_turn",
    "load_lens_conversation_history",
)
