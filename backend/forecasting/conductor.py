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
from pathlib import Path

import pandas as pd

from forecasting.contracts import (
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
        self._conductor_tools.advance_to_meridian(self.run_id)
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
        if not report_path.exists():
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_canonical_and_segments(self) -> tuple[pd.DataFrame, SegmentMap]:
        """Load the canonical demand table + segment map from preflight artifacts.

        The preflight persists ``preflight.json`` with the bundle
        + segment_map.model_dump() under ``outputs/{run_id}/``. We
        read the segment map from there (the canonical table itself
        is held in ``data_store`` in-memory; for Phase 10 we
        re-read the per-series frames from there).
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

        # Reconstruct the canonical table from the data_store.
        frames = [get_series(self.run_id, key) for key in get_series_keys(self.run_id)]
        canonical_table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
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


__all__ = ("Conductor", "ConductorStepResult", "Possibility")
