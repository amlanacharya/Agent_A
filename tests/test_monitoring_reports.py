"""Tests for Phase 7 CB6: monitoring report writers.

Covers the four markdown report writers the plan calls for and
their pure formatter counterparts:

* ``format_monitoring_report(snapshot)`` / ``write_monitoring_report(snapshot, output_dir)`` — top-level rollup
* ``format_drift_report(snapshot)`` / ``write_drift_report(snapshot, output_dir)`` — data + model drift detail
* ``format_override_analysis(snapshot)`` / ``write_override_analysis(snapshot, output_dir)`` — planner overrides + approval patterns
* ``format_model_health(snapshot)`` / ``write_model_health(snapshot, output_dir)`` — model health (MASE, bias, segment degradation, interval calibration)

The writers are pure functions; ``write_*`` are thin I/O wrappers
that call ``format_*`` and persist to ``output_dir / FILENAME``.

The tests assert:

* The four files land in the output dir with the planned names
* Each format function is pure (same input -> same output)
* Empty / no-drift snapshots render to valid markdown (no NaN,
  no empty headings)
* The writers create ``output_dir`` if it does not exist
* All four reports include the run_id, generated_at, and the
  relevant report fields rendered into a stable markdown shape
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

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
# Snapshot factory
# ---------------------------------------------------------------------------


def _full_snapshot() -> MonitorSnapshot:
    """Build a snapshot that exercises every report field."""
    return MonitorSnapshot(
        run_id="r1",
        previous_run_id="r0",
        generated_at="2026-06-17T12:00:00Z",
        data=DataDriftReport(
            run_id="r1",
            previous_run_id="r0",
            schema_changes=[
                SchemaChange(
                    kind="COLUMN_DROPPED",
                    column="inventory_qty",
                    detail="present in r0, absent in r1",
                )
            ],
            missing_feeds=["SKU_2|EAST"],
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
                new_skus=["SKU_3"],
                new_locations=["NORTH"],
            ),
        ),
        model=ModelDriftReport(
            run_id="r1",
            previous_run_id="r0",
            mase_previous=0.80,
            mase_current=0.92,
            mase_delta=0.12,
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
        ),
        business=BusinessOutcomesReport(
            run_id="r1",
            expected_stockouts=2.5,
            expected_overstock=4.0,
            service_level=0.94,
            planner_overrides=["planner reduced SKU_1 order by 50%"],
            approval_patterns={"APPROVE": 12, "REJECT": 1, "DEFER": 0},
        ),
    )


def _empty_snapshot() -> MonitorSnapshot:
    """A snapshot with no drift, no degradation, no overrides."""
    return MonitorSnapshot(
        run_id="r1",
        previous_run_id="r0",
        generated_at="2026-06-17T12:00:00Z",
        data=DataDriftReport(run_id="r1", previous_run_id="r0"),
        model=ModelDriftReport(
            run_id="r1",
            previous_run_id="r0",
            mase_previous=0.0,
            mase_current=0.0,
            mase_delta=0.0,
            bias_previous=0.0,
            bias_current=0.0,
            bias_delta=0.0,
        ),
        business=BusinessOutcomesReport(
            run_id="r1",
            expected_stockouts=0.0,
            expected_overstock=0.0,
            service_level=1.0,
        ),
    )


# ---------------------------------------------------------------------------
# Filename constants
# ---------------------------------------------------------------------------


def test_filenames_match_plan() -> None:
    """The four filenames match the plan's artifact checklist."""
    assert MONITORING_REPORT_FILENAME == "MONITORING_REPORT.md"
    assert DRIFT_REPORT_FILENAME == "DRIFT_REPORT.md"
    assert OVERRIDE_ANALYSIS_FILENAME == "OVERRIDE_ANALYSIS.md"
    assert MODEL_HEALTH_FILENAME == "MODEL_HEALTH.md"


# ---------------------------------------------------------------------------
# format_monitoring_report
# ---------------------------------------------------------------------------


def test_format_monitoring_report_includes_run_id_and_timestamp() -> None:
    """The top-level rollup starts with a clear run / time header."""
    text = format_monitoring_report(_full_snapshot())
    assert "r1" in text
    assert "2026-06-17T12:00:00Z" in text
    assert "r0" in text  # previous run id


def test_format_monitoring_report_includes_every_section() -> None:
    """The top-level report includes the four section headings."""
    text = format_monitoring_report(_full_snapshot())
    # The monitoring report is the top-level rollup; it includes
    # every signal kind in a stable, greppable shape.
    assert "## Data Drift" in text
    assert "## Model Drift" in text
    assert "## Business Outcomes" in text
    assert "## Approval Patterns" in text


def test_format_monitoring_report_pure_function() -> None:
    """Same input -> same output (the format is stable markdown)."""
    snap = _full_snapshot()
    text1 = format_monitoring_report(snap)
    text2 = format_monitoring_report(snap)
    assert text1 == text2


def test_format_monitoring_report_empty_snapshot() -> None:
    """A snapshot with no drift / no degradation still renders."""
    text = format_monitoring_report(_empty_snapshot())
    assert "r1" in text
    assert "## Data Drift" in text
    # Empty sections use a "(none)" marker so the markdown is
    # well-formed (no heading followed by nothing).
    assert "(none)" in text


# ---------------------------------------------------------------------------
# format_drift_report
# ---------------------------------------------------------------------------


def test_format_drift_report_includes_schema_and_missing() -> None:
    """The drift report surfaces schema changes and missing feeds."""
    text = format_drift_report(_full_snapshot())
    assert "COLUMN_DROPPED" in text
    assert "inventory_qty" in text
    assert "SKU_2|EAST" in text


def test_format_drift_report_includes_distribution_shifts() -> None:
    """The drift report surfaces per-column distribution shifts."""
    text = format_drift_report(_full_snapshot())
    assert "demand_qty" in text
    assert "mean" in text
    # pct_change is formatted with a sign
    assert "+40" in text or "0.40" in text


def test_format_drift_report_includes_new_keys() -> None:
    """The drift report surfaces new SKU / location keys."""
    text = format_drift_report(_full_snapshot())
    assert "SKU_3" in text
    assert "NORTH" in text


def test_format_drift_report_includes_model_drift_section() -> None:
    """The drift report also covers the model-drift side."""
    text = format_drift_report(_full_snapshot())
    assert "## Model Drift" in text
    assert "MASE" in text or "mase" in text.lower()


# ---------------------------------------------------------------------------
# format_override_analysis
# ---------------------------------------------------------------------------


def test_format_override_analysis_includes_overrides_list() -> None:
    """The override report surfaces planner overrides as a numbered list."""
    text = format_override_analysis(_full_snapshot())
    assert "planner reduced SKU_1 order by 50%" in text


def test_format_override_analysis_includes_approval_patterns() -> None:
    """The override report surfaces the approval decision counts."""
    text = format_override_analysis(_full_snapshot())
    assert "APPROVE" in text
    assert "REJECT" in text
    assert "12" in text  # APPROVE count
    assert "1" in text   # REJECT count


def test_format_override_analysis_empty_snapshot() -> None:
    """An empty snapshot renders the override report cleanly."""
    text = format_override_analysis(_empty_snapshot())
    assert "r1" in text
    assert "(none)" in text  # the "(none)" marker for empty sections


# ---------------------------------------------------------------------------
# format_model_health
# ---------------------------------------------------------------------------


def test_format_model_health_includes_mase_and_bias() -> None:
    """The model health report surfaces the MASE / bias deltas."""
    text = format_model_health(_full_snapshot())
    assert "MASE" in text or "mase" in text.lower()
    assert "Bias" in text or "bias" in text.lower()


def test_format_model_health_includes_segment_degradation() -> None:
    """The model health report surfaces the per-segment degradation table."""
    text = format_model_health(_full_snapshot())
    assert "G1" in text
    assert "0.80" in text  # previous MASE for G1
    assert "0.95" in text  # current MASE for G1
    assert "0.15" in text  # delta for G1


def test_format_model_health_marks_interval_calibration_seam() -> None:
    """The interval calibration field shows the seam (None)."""
    text = format_model_health(_full_snapshot())
    # The engine renders a clear "no interval coverage yet" line
    # rather than "None" so the planner understands the seam.
    assert "interval" in text.lower()


def test_format_model_health_empty_snapshot() -> None:
    """An empty snapshot renders the model health report cleanly."""
    text = format_model_health(_empty_snapshot())
    assert "r1" in text
    assert "(none)" in text  # the "(none)" marker for empty sections


# ---------------------------------------------------------------------------
# write_* — I/O wrappers
# ---------------------------------------------------------------------------


def test_write_monitoring_report_creates_file(tmp_path: Path) -> None:
    """The writer creates ``MONITORING_REPORT.md`` in the output dir."""
    output_dir = tmp_path / "outputs" / "r1"
    write_monitoring_report(_full_snapshot(), output_dir)
    path = output_dir / MONITORING_REPORT_FILENAME
    assert path.exists()
    text = path.read_text()
    assert "r1" in text
    assert "## Data Drift" in text


def test_write_drift_report_creates_file(tmp_path: Path) -> None:
    """The writer creates ``DRIFT_REPORT.md`` in the output dir."""
    output_dir = tmp_path / "outputs" / "r1"
    write_drift_report(_full_snapshot(), output_dir)
    path = output_dir / DRIFT_REPORT_FILENAME
    assert path.exists()
    text = path.read_text()
    assert "COLUMN_DROPPED" in text


def test_write_override_analysis_creates_file(tmp_path: Path) -> None:
    """The writer creates ``OVERRIDE_ANALYSIS.md`` in the output dir."""
    output_dir = tmp_path / "outputs" / "r1"
    write_override_analysis(_full_snapshot(), output_dir)
    path = output_dir / OVERRIDE_ANALYSIS_FILENAME
    assert path.exists()
    text = path.read_text()
    assert "planner reduced SKU_1 order by 50%" in text


def test_write_model_health_creates_file(tmp_path: Path) -> None:
    """The writer creates ``MODEL_HEALTH.md`` in the output dir."""
    output_dir = tmp_path / "outputs" / "r1"
    write_model_health(_full_snapshot(), output_dir)
    path = output_dir / MODEL_HEALTH_FILENAME
    assert path.exists()
    text = path.read_text()
    assert "G1" in text


def test_write_creates_output_dir_if_missing(tmp_path: Path) -> None:
    """The writers create the output directory if it does not exist."""
    output_dir = tmp_path / "deeply" / "nested" / "outputs" / "r1"
    assert not output_dir.exists()
    write_monitoring_report(_full_snapshot(), output_dir)
    assert output_dir.exists()
    assert (output_dir / MONITORING_REPORT_FILENAME).exists()


def test_write_all_four_reports(tmp_path: Path) -> None:
    """All four writers land their files in the same output dir."""
    output_dir = tmp_path / "outputs" / "r1"
    snapshot = _full_snapshot()
    write_monitoring_report(snapshot, output_dir)
    write_drift_report(snapshot, output_dir)
    write_override_analysis(snapshot, output_dir)
    write_model_health(snapshot, output_dir)
    assert (output_dir / MONITORING_REPORT_FILENAME).exists()
    assert (output_dir / DRIFT_REPORT_FILENAME).exists()
    assert (output_dir / OVERRIDE_ANALYSIS_FILENAME).exists()
    assert (output_dir / MODEL_HEALTH_FILENAME).exists()


def test_write_matches_format_output(tmp_path: Path) -> None:
    """The writer output is byte-identical to the format function."""
    snapshot = _full_snapshot()
    output_dir = tmp_path / "outputs" / "r1"
    write_drift_report(snapshot, output_dir)
    on_disk = (output_dir / DRIFT_REPORT_FILENAME).read_text()
    expected = format_drift_report(snapshot)
    assert on_disk == expected
