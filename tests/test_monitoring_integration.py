"""Tests for Phase 7 CB7: full-chain monitoring integration.

End-to-end test that ties Phase 7's three monitoring engines
(``data_drift``, ``model_drift``, ``business_outcomes``) and
the four report writers (``monitoring_reports``) into one
chain. The chain mirrors the production path:

1. The scheduler fires a ``monitoring`` tick.
2. The runner reads the previous run's snapshot from disk
   and the current run's scorecards / approval audit log.
3. The runner calls ``detect_data_drift``, ``detect_model_drift``,
   and ``summarise_business_outcomes`` to build a fresh
   ``MonitorSnapshot``.
4. The runner writes the four markdown artifacts to
   ``outputs/{run_id}/``.
5. The snapshot is also persisted as JSON so the cockpit
   can read it back without re-running the math.

The tests assert:

* The chain produces a valid ``MonitorSnapshot`` from a
  realistic set of inputs.
* All four markdown artifacts are written to disk.
* The artifacts are not empty / not broken.
* The persisted JSON snapshot round-trips through Pydantic
  without loss.
* The scheduler's ``monitoring`` trigger kind is wired to
  the runner (a future integration can plug in behind the
  same ScheduledJobKind literal without changing the rest
  of the platform).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from forecasting.approval_gateway import InProcessApprovalGateway
from forecasting.business_outcomes import summarise_business_outcomes
from forecasting.contracts import (
    BusinessOutcomesReport,
    DataDriftReport,
    ModelDriftReport,
    ModelScorecard,
    MonitorSnapshot,
    ScheduledJobKind,
    SchemaMapping,
    SegmentDef,
    SegmentMap,
)
from forecasting.data_drift import detect_data_drift
from forecasting.model_drift import detect_model_drift
from forecasting.monitoring_reports import (
    DRIFT_REPORT_FILENAME,
    MODEL_HEALTH_FILENAME,
    MONITORING_REPORT_FILENAME,
    OVERRIDE_ANALYSIS_FILENAME,
    format_drift_report,
    format_model_health,
    format_monitoring_report,
    format_override_analysis,
    write_drift_report,
    write_model_health,
    write_monitoring_report,
    write_override_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _scorecard(
    *,
    series_key: str,
    mase: float,
    bias: float = 0.0,
    forecast: list[float] | None = None,
    actual: list[float] | None = None,
) -> ModelScorecard:
    """Build a tiny ModelScorecard with sensible defaults."""
    return ModelScorecard(
        model_family="naive",
        series_key=series_key,
        fold_cutoff="2026-01-01",
        horizon=1,
        forecast=forecast if forecast is not None else [10.0],
        actual=actual if actual is not None else [10.0 + mase * 2],
        mae=2.0,
        rmse=2.0,
        mase=mase,
        bias=bias,
    )


def _segment_map() -> SegmentMap:
    """Build a SegmentMap with two segments for the chain test."""
    return SegmentMap(
        run_id="r1",
        segments=[
            SegmentDef(
                segment_id="G1",
                label="smooth demand",
                series_keys=["A", "B"],
                provisional=True,
            ),
            SegmentDef(
                segment_id="G2",
                label="intermittent",
                series_keys=["C"],
                provisional=True,
            ),
        ],
        provisional=True,
        derived_by="test:helper",
    )


# ---------------------------------------------------------------------------
# Full chain
# ---------------------------------------------------------------------------


def test_full_monitoring_chain_produces_all_four_artifacts(tmp_path: Path) -> None:
    """End-to-end: build a snapshot from inputs, write all four reports.

    The chain mirrors the production path: the runner reads the
    previous run's snapshot, the current run's scorecards, and
    the current run's approval audit log; calls the three
    monitoring engines; assembles a MonitorSnapshot; and writes
    the four markdown artifacts to ``output_dir``.
    """
    output_dir = tmp_path / "outputs" / "r-monitor"
    output_dir.mkdir(parents=True)

    # 1. Build the inputs the engines consume.
    previous_schema = SchemaMapping(
        date_col="week_start",
        demand_col="demand_qty",
        grain_cols=["sku_id", "location_id"],
        extra_cols=["promo_flag", "inventory_qty"],
    )
    current_schema = SchemaMapping(
        date_col="week_start",
        demand_col="demand_qty",
        grain_cols=["sku_id", "location_id"],
        extra_cols=["promo_flag"],  # inventory_qty dropped
    )
    previous_scorecards = [
        _scorecard(series_key="A", mase=0.80),
        _scorecard(series_key="B", mase=0.85),
        _scorecard(series_key="C", mase=1.20),
    ]
    current_scorecards = [
        _scorecard(series_key="A", mase=0.85),
        _scorecard(series_key="B", mase=0.90),
        _scorecard(series_key="C", mase=1.40),
    ]
    audit_root = tmp_path / "audit"
    gateway = InProcessApprovalGateway(audit_root=audit_root)
    request = gateway.raise_request(
        run_id="r-monitor",
        kind="replenishment_recommendation",
        title="release",
        summary="release",
        requested_by="agent",
    )
    gateway.acknowledge(
        request_id=request.request_id,
        decision="APPROVE",
        approver="planner",
        reason="ok",
    )
    events = gateway.read_audit_log("r-monitor")
    planner_overrides = ["planner reduced SKU_1 order by 50%"]

    # 2. Call the three monitoring engines.
    data_report = detect_data_drift(
        run_id="r-monitor",
        previous_run_id="r-prev",
        previous_schema=previous_schema,
        previous_keys=["A", "B", "C"],
        current_schema=current_schema,
        current_keys=["A", "B", "C", "D"],
        previous_df=None,  # no canonical frame; the engine handles None
        current_df=None,
    )
    model_report = detect_model_drift(
        run_id="r-monitor",
        previous_run_id="r-prev",
        previous_scorecards=previous_scorecards,
        current_scorecards=current_scorecards,
        segment_map=_segment_map(),
    )
    business_report = summarise_business_outcomes(
        run_id="r-monitor",
        scorecards=current_scorecards,
        events=events,
        planner_overrides=planner_overrides,
    )
    snapshot = MonitorSnapshot(
        run_id="r-monitor",
        previous_run_id="r-prev",
        data=data_report,
        model=model_report,
        business=business_report,
    )

    # 3. Write the four markdown artifacts.
    write_monitoring_report(snapshot, output_dir)
    write_drift_report(snapshot, output_dir)
    write_override_analysis(snapshot, output_dir)
    write_model_health(snapshot, output_dir)

    # 4. Persist the snapshot as JSON for the cockpit.
    snapshot_path = output_dir / "monitor_snapshot.json"
    snapshot_path.write_text(snapshot.model_dump_json())

    # 5. Assert the four artifacts exist and are non-empty.
    for filename in (
        MONITORING_REPORT_FILENAME,
        DRIFT_REPORT_FILENAME,
        OVERRIDE_ANALYSIS_FILENAME,
        MODEL_HEALTH_FILENAME,
    ):
        path = output_dir / filename
        assert path.exists(), f"missing artifact: {filename}"
        text = path.read_text()
        assert "r-monitor" in text, f"artifact {filename} missing run id"
        assert len(text) > 200, f"artifact {filename} is suspiciously short"

    # 6. The JSON snapshot round-trips cleanly.
    rebuilt = MonitorSnapshot.model_validate_json(snapshot_path.read_text())
    assert rebuilt == snapshot


def test_full_monitoring_chain_data_drift_surfaces_schema_drop(tmp_path: Path) -> None:
    """The chain correctly surfaces the schema drop in DRIFT_REPORT.md."""
    output_dir = tmp_path / "outputs" / "r-monitor"
    output_dir.mkdir(parents=True)

    previous_schema = SchemaMapping(
        date_col="week_start",
        demand_col="demand_qty",
        grain_cols=["sku_id"],
        extra_cols=["inventory_qty"],
    )
    current_schema = SchemaMapping(
        date_col="week_start",
        demand_col="demand_qty",
        grain_cols=["sku_id"],
        extra_cols=[],
    )
    data_report = detect_data_drift(
        run_id="r-monitor",
        previous_run_id="r-prev",
        previous_schema=previous_schema,
        previous_keys=["A"],
        current_schema=current_schema,
        current_keys=["A", "D"],
        previous_df=None,
        current_df=None,
    )
    snapshot = MonitorSnapshot(
        run_id="r-monitor",
        previous_run_id="r-prev",
        data=data_report,
        model=ModelDriftReport(
            run_id="r-monitor",
            previous_run_id="r-prev",
            mase_previous=0.0,
            mase_current=0.0,
            mase_delta=0.0,
            bias_previous=0.0,
            bias_current=0.0,
            bias_delta=0.0,
        ),
        business=BusinessOutcomesReport(
            run_id="r-monitor",
            expected_stockouts=0.0,
            expected_overstock=0.0,
            service_level=1.0,
        ),
    )
    write_drift_report(snapshot, output_dir)
    text = (output_dir / DRIFT_REPORT_FILENAME).read_text()
    # The schema drop is surfaced in the drift report.
    assert "COLUMN_DROPPED" in text
    assert "inventory_qty" in text
    # The new key is surfaced too.
    assert "D" in text


def test_full_monitoring_chain_model_drift_surfaces_segment_regression(tmp_path: Path) -> None:
    """The chain correctly surfaces the per-segment regression in MODEL_HEALTH.md."""
    output_dir = tmp_path / "outputs" / "r-monitor"
    output_dir.mkdir(parents=True)

    previous_scorecards = [
        _scorecard(series_key="A", mase=0.80),
    ]
    current_scorecards = [
        _scorecard(series_key="A", mase=1.20),  # big regression
    ]
    model_report = detect_model_drift(
        run_id="r-monitor",
        previous_run_id="r-prev",
        previous_scorecards=previous_scorecards,
        current_scorecards=current_scorecards,
        segment_map=_segment_map(),
    )
    snapshot = MonitorSnapshot(
        run_id="r-monitor",
        previous_run_id="r-prev",
        data=DataDriftReport(run_id="r-monitor", previous_run_id="r-prev"),
        model=model_report,
        business=BusinessOutcomesReport(
            run_id="r-monitor",
            expected_stockouts=0.0,
            expected_overstock=0.0,
            service_level=1.0,
        ),
    )
    write_model_health(snapshot, output_dir)
    text = (output_dir / MODEL_HEALTH_FILENAME).read_text()
    # G1 is the segment for series A; the delta should be visible.
    assert "G1" in text
    assert "0.8000" in text
    assert "1.2000" in text


def test_full_monitoring_chain_business_outcomes_uses_audit_log(tmp_path: Path) -> None:
    """The chain correctly surfaces approval patterns from a real audit log."""
    output_dir = tmp_path / "outputs" / "r-monitor"
    output_dir.mkdir(parents=True)

    audit_root = tmp_path / "audit"
    gateway = InProcessApprovalGateway(audit_root=audit_root)
    for _ in range(3):
        request = gateway.raise_request(
            run_id="r-monitor",
            kind="replenishment_recommendation",
            title="release",
            summary="release",
            requested_by="agent",
        )
        gateway.acknowledge(
            request_id=request.request_id,
            decision="APPROVE",
            approver="planner",
            reason="ok",
        )
    events = gateway.read_audit_log("r-monitor")
    business_report = summarise_business_outcomes(
        run_id="r-monitor",
        scorecards=[],
        events=events,
        planner_overrides=[],
    )
    snapshot = MonitorSnapshot(
        run_id="r-monitor",
        previous_run_id="r-prev",
        data=DataDriftReport(run_id="r-monitor", previous_run_id="r-prev"),
        model=ModelDriftReport(
            run_id="r-monitor",
            previous_run_id="r-prev",
            mase_previous=0.0,
            mase_current=0.0,
            mase_delta=0.0,
            bias_previous=0.0,
            bias_current=0.0,
            bias_delta=0.0,
        ),
        business=business_report,
    )
    write_override_analysis(snapshot, output_dir)
    text = (output_dir / OVERRIDE_ANALYSIS_FILENAME).read_text()
    # Three APPROVE decisions should surface in the approval table.
    assert "APPROVE" in text
    assert "3" in text


# ---------------------------------------------------------------------------
# Scheduler trigger kind coverage
# ---------------------------------------------------------------------------


def test_scheduled_job_kind_includes_monitoring() -> None:
    """The ScheduledJobKind Literal includes ``monitoring`` and
    ``drift_investigation`` from the Phase 6 contract — the
    scheduling seam is already in place for the Phase 7
    monitoring layer to plug in behind.
    """
    # The Literal is closed; importing the value through
    # ScheduledJobKind is a no-op, so we just assert the
    # string is allowed.
    assert "monitoring" in ScheduledJobKind.__args__  # type: ignore[attr-defined]
    assert "drift_investigation" in ScheduledJobKind.__args__  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# MonitorSnapshot persistence
# ---------------------------------------------------------------------------


def test_monitor_snapshot_persists_to_json() -> None:
    """The snapshot round-trips through JSON so the cockpit can
    read the previous run's snapshot without re-running the
    math.
    """
    snapshot = MonitorSnapshot(
        run_id="r1",
        previous_run_id="r0",
        data=DataDriftReport(
            run_id="r1",
            previous_run_id="r0",
            missing_feeds=["SKU_2|EAST"],
        ),
        model=ModelDriftReport(
            run_id="r1",
            previous_run_id="r0",
            mase_previous=0.80,
            mase_current=0.92,
            mase_delta=0.12,
            bias_previous=0.0,
            bias_current=0.0,
            bias_delta=0.0,
        ),
        business=BusinessOutcomesReport(
            run_id="r1",
            expected_stockouts=2.5,
            expected_overstock=4.0,
            service_level=0.94,
            approval_patterns={"APPROVE": 3},
        ),
    )
    raw = snapshot.model_dump_json()
    rebuilt = MonitorSnapshot.model_validate_json(raw)
    assert rebuilt == snapshot
    # The rebuilt snapshot's approval patterns survive the trip.
    assert rebuilt.business.approval_patterns == {"APPROVE": 3}


# ---------------------------------------------------------------------------
# Empty-chain safety
# ---------------------------------------------------------------------------


def test_full_monitoring_chain_handles_empty_inputs(tmp_path: Path) -> None:
    """An empty chain (no previous, no current) still produces four valid artifacts.

    The platform's first run, or a run where the harness
    didn't finish, is a valid input to the monitoring engine.
    The four artifacts render with ``(none)`` markers; the
    snapshot round-trips through JSON.
    """
    output_dir = tmp_path / "outputs" / "r-monitor"
    output_dir.mkdir(parents=True)
    snapshot = MonitorSnapshot(
        run_id="r-monitor",
        previous_run_id="r-prev",
        data=DataDriftReport(run_id="r-monitor", previous_run_id="r-prev"),
        model=ModelDriftReport(
            run_id="r-monitor",
            previous_run_id="r-prev",
            mase_previous=0.0,
            mase_current=0.0,
            mase_delta=0.0,
            bias_previous=0.0,
            bias_current=0.0,
            bias_delta=0.0,
        ),
        business=BusinessOutcomesReport(
            run_id="r-monitor",
            expected_stockouts=0.0,
            expected_overstock=0.0,
            service_level=0.0,
        ),
    )
    write_monitoring_report(snapshot, output_dir)
    write_drift_report(snapshot, output_dir)
    write_override_analysis(snapshot, output_dir)
    write_model_health(snapshot, output_dir)
    for filename in (
        MONITORING_REPORT_FILENAME,
        DRIFT_REPORT_FILENAME,
        OVERRIDE_ANALYSIS_FILENAME,
        MODEL_HEALTH_FILENAME,
    ):
        text = (output_dir / filename).read_text()
        assert "r-monitor" in text
        # Every artifact uses ``(none)`` for empty sections.
        assert "(none)" in text
