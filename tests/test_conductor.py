"""Tests for the Conductor orchestrator (Phase 10 CB2).

The Conductor is the single entry point that drives a Run through
the lifecycle: preflight → meridian_scoping → forge_eda →
foundry_modelling → report_ready. Each ``drive_run_to_<phase>``
method is pure: take a RunState, return a ConductorStepResult.

These tests use stubs for ``conductor_tools``, ``eda_toolbox``,
``forecast_harness``, ``ensemble``, ``promotion``, and the
file-system writes. The goal is to verify the orchestration
shape (which methods are called, in what order, with what args)
without re-running the real backend.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from forecasting.contracts import SegmentDef, SegmentMap
from forecasting.run_state import Phase, RunState, create_run_state


# ---------------------------------------------------------------------------
# Test doubles — the conductor's dependencies.
# ---------------------------------------------------------------------------


@dataclass
class StubConductorTools:
    """Captures calls to the legacy conductor_tools functions."""

    advance_to_meridian_calls: list[str] = field(default_factory=list)
    log_halt_calls: list[tuple[str, str]] = field(default_factory=list)

    def advance_to_meridian(self, run_id: str) -> dict[str, Any]:
        self.advance_to_meridian_calls.append(run_id)
        return {"run_id": run_id, "phase": "meridian_scoping"}


@dataclass
class StubEdaToolbox:
    """Captures EDA report generations.

    Records ``call_count`` rather than ``args[0]`` because ``args[0]``
    is a ``pandas.DataFrame`` (truthy-ambiguous) — we only need to
    confirm the conductor called EDA once for this run, not inspect
    the DataFrame contents (that's the real ``eda_toolbox``'s job,
    covered by its own tests).
    """

    eda_calls: list[dict] = field(default_factory=list)
    next_report: Any = None

    def build_eda_report(self, *args, **kwargs) -> Any:
        self.eda_calls.append({"n_args": len(args), "kwargs_keys": sorted(kwargs)})
        return self.next_report


@dataclass
class StubForecastHarness:
    """Captures forecast harness invocations."""

    harness_calls: list[dict] = field(default_factory=list)
    next_report: Any = None

    def run_forecast_harness(self, request, features=None):
        self.harness_calls.append({"run_id": request.run_id})
        return self.next_report


@dataclass
class StubEnsemble:
    """Captures ensemble summary calls."""

    ensemble_calls: list[list] = field(default_factory=list)
    next_summary: Any = None

    def summarise_scorecards(self, scorecards):
        self.ensemble_calls.append(list(scorecards))
        return self.next_summary


@dataclass
class StubPromotion:
    """Captures promotion-decision formatters."""

    promotion_calls: list[dict] = field(default_factory=list)

    def format_promotion_decision(self, **kwargs):
        self.promotion_calls.append(kwargs)
        return "### promotion block"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def run_state(tmp_outputs: Path) -> RunState:
    state = create_run_state("r-conductor-test", domain="fmcg")
    return state


@pytest.fixture()
def stubs():
    return {
        "conductor_tools": StubConductorTools(),
        "eda_toolbox": StubEdaToolbox(),
        "forecast_harness": StubForecastHarness(),
        "ensemble": StubEnsemble(),
        "promotion": StubPromotion(),
    }


def _make_conductor(run_state, stubs, tmp_outputs):
    """Build a Conductor wired to the test doubles."""
    from forecasting.conductor import Conductor

    return Conductor(
        run_id=run_state.run_id,
        conductor_tools=stubs["conductor_tools"],
        eda_toolbox=stubs["eda_toolbox"],
        forecast_harness=stubs["forecast_harness"],
        ensemble=stubs["ensemble"],
        promotion=stubs["promotion"],
        outputs_root=tmp_outputs,
    )


def _seed_preflight(tmp_outputs: Path, run_id: str) -> None:
    """Write the minimum preflight artifacts a later-phase Conductor call needs.

    The Conductor's ``_load_canonical_and_segments`` requires:

    * ``preflight.json`` containing a ``segment_map`` key (validated
      against ``SegmentMap``),
    * at least one series registered in ``forecasting.data_store``.

    This helper stands in for what the production preflight pipeline
    would have written by the time the run reaches MERIDIAN_SCOPING.
    """
    run_dir = tmp_outputs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
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
        json.dumps({"run_id": run_id, "segment_map": segment_map.model_dump(mode="json")}),
        encoding="utf-8",
    )

    from forecasting.data_store import store_series

    store_series(
        run_id,
        "sku_a|north",
        pd.DataFrame(
            {
                "series_key": ["sku_a|north"] * 20,
                "date": pd.date_range("2024-01-01", periods=20, freq="W"),
                "demand": [10.0] * 20,
            }
        ),
    )


def _seed_foundry_report(tmp_outputs: Path, run_id: str) -> None:
    """Write a minimum ``foundry_report.json`` so ``drive_run_to_report`` can load it."""
    run_dir = tmp_outputs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "foundry_report.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "series_results": [],
                "overall_mase": 0.0,
                "target_met_fraction": 0.0,
                "narrative": "stub foundry",
                "ensemble": {
                    "weights": {},
                    "frequently_promoted": [],
                    "never_surfaced": [],
                    "retired": [],
                },
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Tests — orchestration shape per the plan's CB2.3 acceptance criteria.
# ---------------------------------------------------------------------------


def test_drive_run_to_meridian_calls_advance_to_meridian_once(
    run_state, stubs, tmp_outputs
):
    conductor = _make_conductor(run_state, stubs, tmp_outputs)
    result = conductor.drive_run_to_meridian(run_state)

    assert len(stubs["conductor_tools"].advance_to_meridian_calls) == 1
    assert stubs["conductor_tools"].advance_to_meridian_calls[0] == run_state.run_id
    assert result.reply  # non-empty
    assert isinstance(result.possibilities, list)


def test_drive_run_to_forge_persists_eda_report(run_state, stubs, tmp_outputs):
    from forecasting.contracts import EDAReport

    _seed_preflight(tmp_outputs, run_state.run_id)
    stubs["eda_toolbox"].next_report = EDAReport(
        run_id=run_state.run_id,
        series_profiles=[],
        segment_profiles=[],
        feature_config={},
        narrative="stub EDA",
    )

    conductor = _make_conductor(run_state, stubs, tmp_outputs)
    # Move into meridian_scoping first, then forge_eda is the next phase.
    meridian_state = RunState.model_copy(
        run_state, update={"phase": Phase.MERIDIAN_SCOPING.value}
    )
    result = conductor.drive_run_to_forge(meridian_state)

    # EDA was called once for our run.
    assert len(stubs["eda_toolbox"].eda_calls) == 1
    # Artifact persisted.
    eda_path = tmp_outputs / run_state.run_id / "eda_report.json"
    assert eda_path.exists()
    payload = json.loads(eda_path.read_text())
    assert payload["narrative"] == "stub EDA"
    # Phase advanced.
    assert result.advanced_to == Phase.FORGE_EDA
    assert result.state.phase == Phase.FORGE_EDA.value


def test_drive_run_to_foundry_transitions_forge_eda_to_foundry_modelling(
    run_state, stubs, tmp_outputs
):
    _seed_preflight(tmp_outputs, run_state.run_id)
    conductor = _make_conductor(run_state, stubs, tmp_outputs)
    forge_state = RunState.model_copy(
        run_state, update={"phase": Phase.FORGE_EDA.value}
    )
    # Stub out the harness + ensemble so the call doesn't blow up.
    from forecasting.contracts import EnsembleSummary, ForecastHarnessReport

    stubs["forecast_harness"].next_report = ForecastHarnessReport(
        run_id=run_state.run_id,
        horizon=1,
        series_results=[],
        scorecards=[],
        ensemble=EnsembleSummary(weights={}, frequently_promoted=[], never_surfaced=[], retired=[]),
    )
    stubs["ensemble"].next_summary = EnsembleSummary(
        weights={}, frequently_promoted=[], never_surfaced=[], retired=[]
    )

    result = conductor.drive_run_to_foundry(forge_state)

    assert result.advanced_to == Phase.FOUNDRY_MODELLING
    assert result.state.phase == Phase.FOUNDRY_MODELLING.value
    # Both harness + ensemble were invoked.
    assert len(stubs["forecast_harness"].harness_calls) == 1
    assert len(stubs["ensemble"].ensemble_calls) == 1
    # Foundry report persisted.
    foundry_path = tmp_outputs / run_state.run_id / "foundry_report.json"
    assert foundry_path.exists()


def test_drive_run_to_report_persists_report_json(run_state, stubs, tmp_outputs):
    _seed_foundry_report(tmp_outputs, run_state.run_id)
    conductor = _make_conductor(run_state, stubs, tmp_outputs)
    foundry_state = RunState.model_copy(
        run_state, update={"phase": Phase.FOUNDRY_MODELLING.value}
    )
    result = conductor.drive_run_to_report(foundry_state)

    assert result.advanced_to == Phase.REPORT_READY
    assert result.state.phase == Phase.REPORT_READY.value
    report_path = tmp_outputs / run_state.run_id / "report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["run_id"] == run_state.run_id
    assert payload["phase"] == Phase.REPORT_READY.value
