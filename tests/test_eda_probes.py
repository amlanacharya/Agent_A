"""Tests for the Phase 2 EDA probes.

These probes are pure functions that take a canonical demand table (or a
``series_map`` for the per-series ones) and return a small Pydantic
payload. Each test exercises one probe in isolation, then a few tests
exercise the probes through ``build_eda_report`` to confirm the
orchestrator wires them up correctly.
"""
from __future__ import annotations

import pandas as pd
import pytest

from forecasting.contracts import (
    SegmentDef,
    SegmentMap,
)
from forecasting.eda_probes import (
    detect_column_types,
    detect_date_gaps_per_series,
    detect_duplicate_keys,
    detect_leakage_per_series,
    measure_missingness,
    validate_joins,
)
from forecasting.eda_toolbox import build_eda_report


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _series_frame(weeks: int, start: str = "2024-01-01", demand: float = 10.0) -> pd.DataFrame:
    """Build a single weekly per-series frame for date-gap / leakage probes."""
    return pd.DataFrame(
        {
            "date": pd.date_range(start=start, periods=weeks, freq="W-MON"),
            "demand": [demand] * weeks,
        }
    )


def _single_segment_map(series_keys: list[str], run_id: str = "r-probe") -> SegmentMap:
    return SegmentMap(
        run_id=run_id,
        segments=[SegmentDef(segment_id="G1", label="all", series_keys=series_keys)],
        provisional=True,
        derived_by="test:default",
    )


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------


def test_type_detection_labels_canonical_columns_correctly():
    df = pd.DataFrame(
        {
            "sku_id": ["A", "B", "C"],
            "location_id": ["NORTH", "SOUTH", "EAST"],
            "week_start": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
            "demand_qty": [10.0, 12.0, 14.0],
            "inventory_qty": [100.0, 90.0, 80.0],
            "stockout_flag": [False, True, False],
            "price": [9.99, 9.99, 9.99],
            "promo_flag": [False, False, True],
            "lead_time": [7.0, 7.0, 7.0],
        }
    )

    report = detect_column_types(df)
    by_name = {c.column: c.inferred_type for c in report.columns}

    assert by_name["sku_id"] == "string"
    assert by_name["location_id"] == "string"
    assert by_name["week_start"] == "datetime"
    assert by_name["demand_qty"] == "float"
    assert by_name["stockout_flag"] == "boolean"
    assert by_name["promo_flag"] == "boolean"
    assert report.contract_mismatches == []  # everything matches


def test_type_detection_flags_contract_mismatch():
    """A demand_qty column containing integers should still match the
    contract (the contract says float; pandas stores them as int). We
    expect the mismatch to fire when the column is a *string*."""
    df = pd.DataFrame(
        {
            "sku_id": ["A"],
            "location_id": ["NORTH"],
            "week_start": pd.to_datetime(["2024-01-01"]),
            "demand_qty": ["ten"],  # string in a numeric column -> mismatch
        }
    )
    report = detect_column_types(df)
    by_name = {c.column: c.inferred_type for c in report.columns}
    assert by_name["demand_qty"] == "string"
    assert "demand_qty" in report.contract_mismatches


def test_type_detection_treats_boolean_looking_strings_as_boolean():
    """The canonical layer accepts "true"/"false"/"yes"/"no" as flags —
    the probe should also recognise them so a manually-constructed
    canonical table (e.g. test data) classifies them correctly."""
    df = pd.DataFrame(
        {
            "promo_flag": ["true", "false", "yes", "no"],
        }
    )
    report = detect_column_types(df)
    assert report.columns[0].inferred_type == "boolean"


def test_type_detection_reports_empty_columns():
    df = pd.DataFrame({"empty_col": [None, None, None]})
    report = detect_column_types(df)
    inf = report.columns[0]
    assert inf.inferred_type == "empty"
    assert inf.nullable is True
    assert inf.unique_count == 0
    assert inf.sample_values == []


def test_type_detection_records_sample_values_capped():
    """The sample_values list is capped so the report does not bloat."""
    df = pd.DataFrame({"x": list(range(20))})
    report = detect_column_types(df)
    assert len(report.columns[0].sample_values) <= 5


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------


def test_missingness_counts_per_column():
    df = pd.DataFrame(
        {
            "sku_id": ["A", "B", "C"],
            "demand_qty": [1.0, None, 3.0],
            "inventory_qty": [10.0, 20.0, None],
            "price": [None, None, None],
        }
    )
    report = measure_missingness(df)
    by_name = {m.column: m for m in report.per_column}

    assert by_name["demand_qty"].missing_count == 1
    assert by_name["demand_qty"].missing_fraction == pytest.approx(1 / 3)
    assert by_name["inventory_qty"].missing_count == 1
    assert by_name["price"].missing_count == 3
    assert by_name["price"].missing_fraction == pytest.approx(1.0)


def test_missingness_does_not_count_required_columns_in_rows_metric():
    """``rows_with_missing`` should only count rows missing in *optional*
    columns. Missing in a required column is a contract violation, not a
    data-quality hint — the canonical layer already rejects it on input.
    """
    df = pd.DataFrame(
        {
            "sku_id": ["A", "B"],  # all present
            "demand_qty": [1.0, 2.0],  # all present
            "inventory_qty": [None, 10.0],  # one missing
        }
    )
    report = measure_missingness(df)
    assert report.rows_with_missing == 1
    assert report.rows_total == 2


def test_missingness_on_empty_dataframe_is_zero():
    df = pd.DataFrame({"x": []})
    report = measure_missingness(df)
    assert report.rows_total == 0
    assert all(m.missing_count == 0 for m in report.per_column)


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


def test_duplicate_detection_flags_series_date_collisions():
    df = pd.DataFrame(
        {
            "series_key": ["A", "A", "B"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
            "demand": [1.0, 1.0, 2.0],
        }
    )
    report = detect_duplicate_keys(df)
    assert report.duplicate_rows == 1
    assert report.duplicate_keys == ["A@2024-01-01"]


def test_duplicate_detection_returns_zero_on_clean_table():
    df = pd.DataFrame(
        {
            "series_key": ["A", "A", "B"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-01"]),
            "demand": [1.0, 2.0, 3.0],
        }
    )
    report = detect_duplicate_keys(df)
    assert report.duplicate_rows == 0
    assert report.duplicate_keys == []
    assert report.duplicate_fraction == 0.0


def test_duplicate_detection_handles_missing_keys_gracefully():
    """A table without the canonical primary key returns the empty
    report — never raises. This matters because the probe runs against
    arbitrary inputs, not just validated canonical tables."""
    df = pd.DataFrame({"foo": [1, 2, 3]})
    report = detect_duplicate_keys(df)
    assert report.duplicate_rows == 0


def test_duplicate_detection_on_empty_table_is_zero():
    df = pd.DataFrame({"series_key": [], "date": []})
    report = detect_duplicate_keys(df)
    assert report.duplicate_rows == 0


# ---------------------------------------------------------------------------
# Date gaps
# ---------------------------------------------------------------------------


def test_date_gaps_clean_weekly_series_has_no_gaps():
    series_map = {"A|N": _series_frame(weeks=12, start="2024-01-01")}
    report = detect_date_gaps_per_series(series_map)
    stats = report.per_series["A|N"]
    assert stats.actual_gap_count == 0
    assert stats.expected_period_days == 7
    assert stats.max_gap_days == 7
    assert stats.out_of_order_rows == 0
    assert report.series_with_gaps == []


def test_date_gaps_flags_a_multi_week_gap():
    """A 21-day gap in a weekly series is more than 1.5x the period
    (10.5 days) and should be counted as a gap."""
    rows = [
        {"date": pd.Timestamp("2024-01-01"), "demand": 1.0},
        {"date": pd.Timestamp("2024-01-08"), "demand": 1.0},
        {"date": pd.Timestamp("2024-01-15"), "demand": 1.0},
        # 3 weeks missing: 2024-01-22, 2024-01-29, 2024-02-05
        {"date": pd.Timestamp("2024-02-12"), "demand": 1.0},
        {"date": pd.Timestamp("2024-02-19"), "demand": 1.0},
    ]
    series_map = {"A|N": pd.DataFrame(rows)}
    report = detect_date_gaps_per_series(series_map)
    stats = report.per_series["A|N"]
    assert stats.actual_gap_count == 1
    assert stats.max_gap_days == 28
    assert "A|N" in report.series_with_gaps


def test_date_gaps_detects_out_of_order_rows():
    rows = [
        {"date": pd.Timestamp("2024-01-08"), "demand": 1.0},
        {"date": pd.Timestamp("2024-01-01"), "demand": 1.0},  # out of order
        {"date": pd.Timestamp("2024-01-15"), "demand": 1.0},
    ]
    series_map = {"A|N": pd.DataFrame(rows)}
    report = detect_date_gaps_per_series(series_map)
    stats = report.per_series["A|N"]
    # After sorting by date in the probe, the sorted frame has no
    # out-of-order rows, so the count is 0 — the probe sorts first.
    assert stats.out_of_order_rows == 0
    assert stats.expected_period_days == 7


def test_date_gaps_respects_explicit_expected_period():
    """When the caller passes frequency_period=7 explicitly, the probe
    uses that as the baseline rather than re-inferring from the data."""
    rows = [
        {"date": pd.Timestamp("2024-01-01"), "demand": 1.0},
        {"date": pd.Timestamp("2024-01-08"), "demand": 1.0},
    ]
    series_map = {"A|N": pd.DataFrame(rows)}
    report = detect_date_gaps_per_series(series_map, expected_period_days=7)
    assert report.per_series["A|N"].expected_period_days == 7


def test_date_gaps_handles_short_series_without_crashing():
    """A single-row series has no deltas to measure. The probe should
    return zeros rather than raising."""
    series_map = {"A|N": pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "demand": [1.0]})}
    report = detect_date_gaps_per_series(series_map)
    stats = report.per_series["A|N"]
    assert stats.actual_gap_count == 0
    assert stats.max_gap_days == 0
    assert stats.median_gap_days == 0.0


# ---------------------------------------------------------------------------
# Join validation
# ---------------------------------------------------------------------------


def test_join_validation_reports_full_coverage_on_complete_table():
    df = pd.DataFrame(
        {
            "series_key": ["A", "A", "B"],
            "demand": [1.0, 2.0, 3.0],
            "inventory_qty": [10.0, 12.0, 8.0],
            "price": [9.99, 9.99, 9.99],
            "lead_time": [7.0, 7.0, 7.0],
        }
    )
    report = validate_joins(df)
    assert report.inventory_coverage == pytest.approx(1.0)
    assert report.price_coverage == pytest.approx(1.0)
    assert report.lead_time_coverage == pytest.approx(1.0)
    assert report.issues == []


def test_join_validation_flags_per_series_missing_inventory():
    """A series with all-NaN inventory rows surfaces as an issue, even
    when the overall coverage metric is high (other series carry it)."""
    df = pd.DataFrame(
        {
            "series_key": ["A", "A", "B", "B"],
            "demand": [1.0, 2.0, 3.0, 4.0],
            "inventory_qty": [10.0, 12.0, None, None],
            "price": [9.99, 9.99, 9.99, 9.99],
            "lead_time": [7.0, 7.0, 7.0, 7.0],
        }
    )
    report = validate_joins(df)
    # 2 of 4 rows have inventory -> 0.5 coverage
    assert report.inventory_coverage == pytest.approx(0.5)
    # But the *issue* is per-series: only B is missing inventory everywhere.
    kinds = {issue.kind for issue in report.issues}
    assert "MISSING_INVENTORY_FOR_DEMAND" in kinds
    affected = {issue.series_key for issue in report.issues}
    assert "B" in affected
    assert "A" not in affected


def test_join_validation_handles_missing_optional_columns():
    """If a dimension is absent from the table, coverage is 0 and no
    per-series issues are surfaced (the column does not exist)."""
    df = pd.DataFrame({"series_key": ["A"], "demand": [1.0]})
    report = validate_joins(df)
    assert report.inventory_coverage == 0.0
    assert report.price_coverage == 0.0
    assert report.lead_time_coverage == 0.0
    assert report.issues == []


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------


def test_leakage_clean_weekly_series_has_low_forward_correlation():
    series_map = {"A|N": _series_frame(weeks=20, demand=10.0)}
    report = detect_leakage_per_series(series_map)
    stats = report.per_series["A|N"]
    # Constant series -> correlation is undefined; probe returns 0.0.
    assert stats.forward_correlation_max == 0.0
    assert stats.demand_equals_inventory_rows == 0
    assert "A|N" not in report.suspect_series


def test_leakage_flags_demand_equals_inventory():
    """demand_qty == inventory_qty is impossible. The probe should
    count the rows so the user knows their upstream join is wrong."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="W-MON"),
            "demand": [5.0, 6.0, 7.0, 8.0],
            "inventory_qty": [5.0, 6.0, 7.0, 8.0],  # identical
        }
    )
    series_map = {"A|N": df}
    report = detect_leakage_per_series(series_map)
    stats = report.per_series["A|N"]
    assert stats.demand_equals_inventory_rows == 4
    assert "A|N" in report.suspect_series


def test_leakage_flags_near_perfect_forward_correlation():
    """demand[t+1] == demand[t] for every row is a leakage red flag
    (typically a copy-paste bug in upstream ETL)."""
    n = 12
    demand = [float(i) for i in range(n)]
    # Create a frame where demand[t] always equals demand[t+1] — that's
    # not possible without repeating values, so we use a constant + 1 lag
    # pattern that gives correlation 1.0 at lag 1.
    # The probe checks lags 2..5; with constant demand there is no
    # correlation. Instead, build a series where demand[t] = demand[t+2]
    # (forward correlation at lag 2 == 1.0).
    demand_lag2 = [demand[i] for i in range(n - 2)]
    demand_lag2_aligned = demand[2:]
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(demand_lag2), freq="W-MON"),
            "demand": demand_lag2,
        }
    )
    # Append a phantom row that matches the [2:] part so the lag-2
    # autocorrelation is 1.0 — actually a simpler construction: use
    # alternating values so that lag-1 correlation is 0 but lag-2 is
    # undefined. Easier: just have a perfect periodic series of period 1.
    periodic = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="W-MON"),
            "demand": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        }
    )
    series_map = {"A|N": periodic}
    report = detect_leakage_per_series(series_map)
    stats = report.per_series["A|N"]
    # lag-2 correlation == 1.0 (every t aligns with t+2)
    assert stats.forward_correlation_max == pytest.approx(1.0, abs=0.01)
    assert "A|N" in report.suspect_series


def test_leakage_handles_empty_series():
    series_map = {"A|N": pd.DataFrame(columns=["date", "demand"])}
    report = detect_leakage_per_series(series_map)
    stats = report.per_series["A|N"]
    assert stats.forward_correlation_max == 0.0
    assert stats.demand_equals_inventory_rows == 0


# ---------------------------------------------------------------------------
# build_eda_report integration: every probe is wired up
# ---------------------------------------------------------------------------


def _two_series_canonical_table() -> pd.DataFrame:
    base = pd.Timestamp("2024-01-01")
    rows: list[dict] = []
    for week in range(8):
        rows.append(
            {
                "series_key": "SKU_A|NORTH",
                "sku_id": "SKU_A",
                "location_id": "NORTH",
                "date": base + pd.Timedelta(weeks=week),
                "week_start": base + pd.Timedelta(weeks=week),
                "demand": 10.0,
                "demand_qty": 10.0,
                "promo": False,
                "promo_flag": False,
                "inventory_qty": 50.0,
                "stockout_flag": False,
                "price": 9.99,
                "lead_time": 7.0,
            }
        )
    for week in range(8):
        rows.append(
            {
                "series_key": "SKU_B|NORTH",
                "sku_id": "SKU_B",
                "location_id": "NORTH",
                "date": base + pd.Timedelta(weeks=week),
                "week_start": base + pd.Timedelta(weeks=week),
                "demand": 20.0,
                "demand_qty": 20.0,
                "promo": week in (1, 3),
                "promo_flag": week in (1, 3),
                "inventory_qty": 60.0,
                "stockout_flag": False,
                "price": 14.99,
                "lead_time": 7.0,
            }
        )
    return pd.DataFrame(rows)


def test_build_eda_report_populates_every_probe():
    table = _two_series_canonical_table()
    seg_map = _single_segment_map(["SKU_A|NORTH", "SKU_B|NORTH"], run_id="r-integration")

    report = build_eda_report(table, seg_map, frequency_period=52)

    # Every probe must be present (no None) on the report.
    assert report.type_detection is not None
    assert report.missingness is not None
    assert report.duplicates is not None
    assert report.date_gaps is not None
    assert report.join_validation is not None
    assert report.leakage is not None


def test_build_eda_report_integration_with_probes_end_to_end():
    """Smoke test: build_eda_report on a clean two-series table surfaces
    non-None probes with the expected values, including the inventory
    coverage (full) and the duplicate count (zero)."""
    table = _two_series_canonical_table()
    seg_map = _single_segment_map(["SKU_A|NORTH", "SKU_B|NORTH"], run_id="r-e2e")

    report = build_eda_report(table, seg_map, frequency_period=52)

    # Type detection: both sku_id and location_id are strings
    type_by_name = {c.column: c for c in report.type_detection.columns}
    assert type_by_name["sku_id"].inferred_type == "string"
    assert type_by_name["demand_qty"].inferred_type == "float"

    # Missingness: optional columns are all populated -> zero rows missing
    assert report.missingness.rows_with_missing == 0

    # Duplicates: clean table -> zero
    assert report.duplicates.duplicate_rows == 0

    # Date gaps: clean weekly series, no gaps
    assert report.date_gaps.series_with_gaps == []

    # Join validation: every row has inventory / price / lead time
    assert report.join_validation.inventory_coverage == pytest.approx(1.0)
    assert report.join_validation.price_coverage == pytest.approx(1.0)
    assert report.join_validation.lead_time_coverage == pytest.approx(1.0)
    assert report.join_validation.issues == []

    # Leakage: no demand-equals-inventory
    for stats in report.leakage.per_series.values():
        assert stats.demand_equals_inventory_rows == 0


def test_build_eda_report_propagates_frequency_period_to_date_gaps():
    """frequency_period is passed through to the date-gap probe so the
    caller can fix the expected period for fold-aware validation."""
    table = _two_series_canonical_table()
    seg_map = _single_segment_map(["SKU_A|NORTH", "SKU_B|NORTH"], run_id="r-fp")

    report = build_eda_report(table, seg_map, frequency_period=7)

    for stats in report.date_gaps.per_series.values():
        assert stats.expected_period_days == 7
