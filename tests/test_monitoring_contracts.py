"""Tests for Phase 7 CB2: monitoring contracts.

Covers the typed shape of the four monitoring contracts that
CB3-CB6 consume:

* ``DataDriftReport`` — schema, missing, distribution, new keys
* ``ModelDriftReport`` — error, bias, interval, segment degradation
* ``BusinessOutcomesReport`` — stockouts, overstock, service level,
  planner overrides, approval patterns
* ``MonitorSnapshot`` — the typed envelope the platform persists
  per monitoring run

All four are pure Pydantic; the tests exercise shape, defaults,
and the closed enums the writers downstream rely on.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from forecasting.contracts import (
    BusinessOutcomesReport,
    DataDriftReport,
    DistributionShift,
    ModelDriftReport,
    MonitorSnapshot,
    NewSeriesKeys,
    SchemaChange,
    SegmentDegradation,
)


# ---------------------------------------------------------------------------
# DataDriftReport
# ---------------------------------------------------------------------------


def test_data_drift_report_carries_all_four_signal_kinds() -> None:
    """The four signals the plan calls for are present on the report."""
    report = DataDriftReport(
        run_id="r1",
        previous_run_id="r0",
        schema_changes=[
            SchemaChange(
                kind="COLUMN_DROPPED",
                column="promo_flag",
                detail="present in r0, absent in r1",
            )
        ],
        missing_feeds=["inventory_qty"],
        distribution_shifts=[
            DistributionShift(
                column="demand_qty",
                metric="mean",
                previous=10.0,
                current=14.0,
                pct_change=0.40,
            )
        ],
        new_keys=NewSeriesKeys(
            new_skus=["SKU_NEW_1"],
            new_locations=["LOC_EAST"],
        ),
    )
    assert report.run_id == "r1"
    assert report.previous_run_id == "r0"
    assert report.schema_changes[0].kind == "COLUMN_DROPPED"
    assert report.missing_feeds == ["inventory_qty"]
    assert report.distribution_shifts[0].pct_change == pytest.approx(0.40)
    assert report.new_keys.new_skus == ["SKU_NEW_1"]


def test_data_drift_report_defaults_to_empty() -> None:
    """A report with no drift is valid and serialises cleanly."""
    report = DataDriftReport(run_id="r1", previous_run_id="r0")
    assert report.schema_changes == []
    assert report.missing_feeds == []
    assert report.distribution_shifts == []
    assert report.new_keys.new_skus == []
    assert report.new_keys.new_locations == []


def test_schema_change_kind_is_closed() -> None:
    """The Literal keeps the cockpit's drift widget finite."""
    with pytest.raises(ValidationError):
        SchemaChange(
            kind="UNKNOWN_KIND",  # type: ignore[arg-type]
            column="promo_flag",
            detail="x",
        )


def test_data_drift_report_serialises_to_dict() -> None:
    """The report survives ``model_dump`` so the writer can render it."""
    report = DataDriftReport(
        run_id="r1",
        previous_run_id="r0",
        missing_feeds=["x"],
    )
    dumped = report.model_dump()
    assert dumped["run_id"] == "r1"
    assert dumped["missing_feeds"] == ["x"]


# ---------------------------------------------------------------------------
# ModelDriftReport
# ---------------------------------------------------------------------------


def test_model_drift_report_carries_mase_bias_and_segments() -> None:
    """Error, bias, and segment degradation are all top-level fields."""
    report = ModelDriftReport(
        run_id="r1",
        previous_run_id="r0",
        mase_previous=0.85,
        mase_current=0.92,
        mase_delta=0.07,
        bias_previous=-0.02,
        bias_current=0.05,
        bias_delta=0.07,
        interval_calibration=None,
        segment_degradation=[
            SegmentDegradation(
                segment_id="G1",
                mase_previous=0.80,
                mase_current=0.95,
                mase_delta=0.15,
            )
        ],
    )
    assert report.mase_delta == pytest.approx(0.07)
    assert report.interval_calibration is None
    assert report.segment_degradation[0].segment_id == "G1"


def test_model_drift_report_interval_calibration_can_be_set_later() -> None:
    """The seam is open — passing a value works without a default code path."""
    report = ModelDriftReport(
        run_id="r1",
        previous_run_id="r0",
        mase_previous=0.85,
        mase_current=0.92,
        mase_delta=0.07,
        bias_previous=0.0,
        bias_current=0.0,
        bias_delta=0.0,
        interval_calibration=0.78,
        segment_degradation=[],
    )
    assert report.interval_calibration == pytest.approx(0.78)


def test_model_drift_report_defaults_to_empty_segments() -> None:
    """A run with no segments is valid (e.g. a single-segment pilot)."""
    report = ModelDriftReport(
        run_id="r1",
        previous_run_id="r0",
        mase_previous=0.85,
        mase_current=0.85,
        mase_delta=0.0,
        bias_previous=0.0,
        bias_current=0.0,
        bias_delta=0.0,
    )
    assert report.segment_degradation == []


# ---------------------------------------------------------------------------
# BusinessOutcomesReport
# ---------------------------------------------------------------------------


def test_business_outcomes_report_carries_all_five_signal_kinds() -> None:
    """The five signals the plan calls for are all on the report."""
    report = BusinessOutcomesReport(
        run_id="r1",
        expected_stockouts=2.3,
        expected_overstock=4.1,
        service_level=0.94,
        planner_overrides=["rejected medium-tier recommendation for SKU_1"],
        approval_patterns={
            "APPROVE": 12,
            "REJECT": 1,
            "DEFER": 0,
        },
    )
    assert report.expected_stockouts == pytest.approx(2.3)
    assert report.expected_overstock == pytest.approx(4.1)
    assert report.service_level == pytest.approx(0.94)
    assert len(report.planner_overrides) == 1
    assert report.approval_patterns["APPROVE"] == 12


def test_business_outcomes_report_service_level_bounded_zero_one() -> None:
    """Service level is a probability — must be in [0, 1]."""
    with pytest.raises(ValidationError):
        BusinessOutcomesReport(
            run_id="r1",
            expected_stockouts=0.0,
            expected_overstock=0.0,
            service_level=1.5,
        )


def test_business_outcomes_report_defaults_to_empty() -> None:
    """A run with no decisions yet is valid (e.g. first monitoring tick)."""
    report = BusinessOutcomesReport(
        run_id="r1",
        expected_stockouts=0.0,
        expected_overstock=0.0,
        service_level=0.0,
    )
    assert report.planner_overrides == []
    assert report.approval_patterns == {}


# ---------------------------------------------------------------------------
# MonitorSnapshot — the typed envelope
# ---------------------------------------------------------------------------


def test_monitor_snapshot_envelopes_all_three_reports() -> None:
    """A snapshot is the join of all three reports plus a timestamp."""
    now = datetime.now(timezone.utc).isoformat()
    snapshot = MonitorSnapshot(
        run_id="r1",
        previous_run_id="r0",
        generated_at=now,
        data=DataDriftReport(run_id="r1", previous_run_id="r0"),
        model=ModelDriftReport(
            run_id="r1",
            previous_run_id="r0",
            mase_previous=0.85,
            mase_current=0.85,
            mase_delta=0.0,
            bias_previous=0.0,
            bias_current=0.0,
            bias_delta=0.0,
        ),
        business=BusinessOutcomesReport(
            run_id="r1",
            expected_stockouts=0.0,
            expected_overstock=0.0,
            service_level=0.0,
        ),
    )
    assert snapshot.run_id == "r1"
    assert snapshot.previous_run_id == "r0"
    assert snapshot.generated_at == now
    assert snapshot.data.run_id == "r1"
    assert snapshot.model.mase_delta == 0.0
    assert snapshot.business.expected_stockouts == 0.0


def test_monitor_snapshot_default_generated_at_is_set() -> None:
    """``generated_at`` defaults to a string the writers can render."""
    snapshot = MonitorSnapshot(
        run_id="r1",
        previous_run_id="r0",
        data=DataDriftReport(run_id="r1", previous_run_id="r0"),
        model=ModelDriftReport(
            run_id="r1",
            previous_run_id="r0",
            mase_previous=0.85,
            mase_current=0.85,
            mase_delta=0.0,
            bias_previous=0.0,
            bias_current=0.0,
            bias_delta=0.0,
        ),
        business=BusinessOutcomesReport(
            run_id="r1",
            expected_stockouts=0.0,
            expected_overstock=0.0,
            service_level=0.0,
        ),
    )
    assert isinstance(snapshot.generated_at, str)
    assert len(snapshot.generated_at) > 0


# ---------------------------------------------------------------------------
# Shape contracts the writers rely on
# ---------------------------------------------------------------------------


def test_data_drift_report_round_trip_through_json() -> None:
    """Snapshot writers persist JSON; round-trip must be lossless."""
    original = DataDriftReport(
        run_id="r1",
        previous_run_id="r0",
        schema_changes=[
            SchemaChange(kind="COLUMN_ADDED", column="promo_flag", detail="x")
        ],
        missing_feeds=["inventory_qty"],
        distribution_shifts=[
            DistributionShift(
                column="demand_qty",
                metric="mean",
                previous=10.0,
                current=14.0,
                pct_change=0.40,
            )
        ],
        new_keys=NewSeriesKeys(new_skus=["A"], new_locations=["B"]),
    )
    rebuilt = DataDriftReport.model_validate_json(original.model_dump_json())
    assert rebuilt == original


def test_monitor_snapshot_round_trip_through_json() -> None:
    """The full envelope survives JSON round-trip."""
    original = MonitorSnapshot(
        run_id="r1",
        previous_run_id="r0",
        data=DataDriftReport(run_id="r1", previous_run_id="r0"),
        model=ModelDriftReport(
            run_id="r1",
            previous_run_id="r0",
            mase_previous=0.85,
            mase_current=0.92,
            mase_delta=0.07,
            bias_previous=-0.02,
            bias_current=0.05,
            bias_delta=0.07,
        ),
        business=BusinessOutcomesReport(
            run_id="r1",
            expected_stockouts=1.2,
            expected_overstock=3.4,
            service_level=0.94,
            planner_overrides=["x"],
            approval_patterns={"APPROVE": 1},
        ),
    )
    rebuilt = MonitorSnapshot.model_validate_json(original.model_dump_json())
    assert rebuilt == original
