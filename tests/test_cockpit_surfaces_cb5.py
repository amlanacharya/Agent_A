"""Tests for Phase 8 CB5: Data Health + Canonical Table Builder surfaces.

Covers two more cockpit surfaces, both reading existing
platform state:

* ``DataHealthSurface`` — reads the Phase 2 ``EDAReport``
  (the per-segment + per-series profile, the 6 EDA probes
  the plan calls for, the narrative) and surfaces it as
  the data-health surface state.
* ``CanonicalTableBuilderSurface`` — reads the canonical
  DataFrame (the post-Preflight demand table the platform
  feeds into EDA) and surfaces its head + row count +
  segment list. The cockpit shows the first 50 rows as a
  preview + the per-segment series count.

Both surfaces use the same provider-injection pattern as
CB4 (Mission Control + MLOps Monitor). The provider is
the seam: the production wiring reads from disk; tests
pass an in-memory provider.
"""

from __future__ import annotations

import pandas as pd
import pytest

from api.models import SurfaceSnapshot
from api.surfaces import (
    CanonicalTableBuilderSurface,
    DataHealthSurface,
)


def _eda_report() -> "object":
    """Build a small EDAReport for the tests."""
    from forecasting.contracts import (
        EDAReport,
        MissingnessReport,
        SegmentProfile,
        SeriesDemandProfile,
    )
    return EDAReport(
        run_id="r1",
        segment_profiles=[
            SegmentProfile(
                segment_id="G1",
                series_count=2,
                demand_class_distribution={"SMOOTH": 1, "ERRATIC": 1},
                median_adi=1.2,
                median_cv2=0.5,
                forecastability_breakdown={"forecastable": 2},
            )
        ],
        series_profiles=[
            SeriesDemandProfile(
                series_key="SKU_1",
                sb_class="SMOOTH",
                adi=1.0,
                cv2=0.4,
                trend_strength=0.1,
                seasonal_strength=0.2,
                recommended_models=["croston"],
            ),
            SeriesDemandProfile(
                series_key="SKU_2",
                sb_class="ERRATIC",
                adi=1.4,
                cv2=0.8,
                trend_strength=0.2,
                seasonal_strength=0.1,
                recommended_models=["croston", "ets"],
            ),
        ],
        feature_config={
            "SKU_1": {"use_lag_features": True},
            "SKU_2": {"use_lag_features": True},
        },
        narrative="Two-series run; G1 is mixed SMOOTH/ERRATIC.",
        missingness=MissingnessReport(
            per_column=[],
            per_row_count=0,
            rows_with_missing=0,
            rows_total=0,
        ),
    )


# ---------------------------------------------------------------------------
# DataHealthSurface
# ---------------------------------------------------------------------------


def test_data_health_surface_summarises_eda_report() -> None:
    """The surface surfaces a summary of the EDA report's headline numbers."""
    report = _eda_report()
    surface = DataHealthSurface(eda_report_provider=lambda rid: report)
    snapshot = surface.render("r1")
    assert isinstance(snapshot, SurfaceSnapshot)
    assert snapshot.surface == "data_health"
    assert snapshot.run_id == "r1"
    state = snapshot.state
    assert state["series_count"] == 2
    assert state["segment_count"] == 1
    assert state["narrative"] == "Two-series run; G1 is mixed SMOOTH/ERRATIC."


def test_data_health_surface_includes_segment_profiles() -> None:
    """The per-segment profiles surface in the state dict for the cockpit."""
    report = _eda_report()
    surface = DataHealthSurface(eda_report_provider=lambda rid: report)
    snapshot = surface.render("r1")
    assert snapshot.state["segment_profiles"][0]["segment_id"] == "G1"
    assert snapshot.state["segment_profiles"][0]["series_count"] == 2


def test_data_health_surface_includes_demand_class_breakdown() -> None:
    """The per-segment demand-class breakdown surfaces in the state."""
    report = _eda_report()
    surface = DataHealthSurface(eda_report_provider=lambda rid: report)
    snapshot = surface.render("r1")
    by_class = snapshot.state["demand_class_breakdown"]
    assert by_class["SMOOTH"] == 1
    assert by_class["ERRATIC"] == 1


def test_data_health_surface_handles_missing_eda_report() -> None:
    """A run with no EDA report yet surfaces a 'no data' placeholder."""
    surface = DataHealthSurface(eda_report_provider=lambda rid: None)
    snapshot = surface.render("r-empty")
    assert snapshot.state["series_count"] == 0
    assert snapshot.state["narrative"] == ""


# ---------------------------------------------------------------------------
# CanonicalTableBuilderSurface
# ---------------------------------------------------------------------------


def test_canonical_table_builder_surfaces_head_and_row_count() -> None:
    """The surface shows the first 50 rows of the canonical table + row count."""
    df = pd.DataFrame({
        "week_start": ["2024-W01", "2024-W02", "2024-W03"],
        "sku_id": ["SKU_1", "SKU_1", "SKU_2"],
        "location_id": ["WEST", "WEST", "EAST"],
        "demand_qty": [10.0, 12.0, 8.0],
    })
    surface = CanonicalTableBuilderSurface(canonical_table_provider=lambda rid: df)
    snapshot = surface.render("r1")
    assert snapshot.surface == "canonical_table_builder"
    state = snapshot.state
    assert state["row_count"] == 3
    assert state["column_count"] == 4
    assert state["columns"] == ["week_start", "sku_id", "location_id", "demand_qty"]
    # The head is a list of records (JSON-safe over HTTP).
    assert len(state["head"]) == 3
    assert state["head"][0]["sku_id"] == "SKU_1"


def test_canonical_table_builder_surfaces_segment_list() -> None:
    """The per-segment series count is surfaced for the cockpit's segment widget."""
    from forecasting.contracts import SegmentDef, SegmentMap

    df = pd.DataFrame({
        "week_start": ["2024-W01"] * 4,
        "sku_id": ["SKU_1", "SKU_1", "SKU_2", "SKU_3"],
        "location_id": ["WEST", "WEST", "EAST", "EAST"],
        "demand_qty": [10.0, 12.0, 8.0, 9.0],
    })
    segment_map = SegmentMap(
        run_id="r1",
        segments=[
            SegmentDef(
                segment_id="G1",
                label="G1",
                series_keys=["SKU_1"],
                provisional=True,
            ),
            SegmentDef(
                segment_id="G2",
                label="G2",
                series_keys=["SKU_2", "SKU_3"],
                provisional=True,
            ),
        ],
        provisional=True,
        derived_by="test:helper",
    )
    surface = CanonicalTableBuilderSurface(
        canonical_table_provider=lambda rid: df,
        segment_map_provider=lambda rid: segment_map,
    )
    snapshot = surface.render("r1")
    segments = snapshot.state["segments"]
    assert {s["segment_id"] for s in segments} == {"G1", "G2"}
    assert {s["series_count"] for s in segments} == {1, 2}


def test_canonical_table_builder_head_is_capped_at_50_rows() -> None:
    """The head is the first 50 rows — a larger table does not bloat the response."""
    df = pd.DataFrame({
        "week_start": [f"2024-W{i:02d}" for i in range(1, 200)],
        "sku_id": ["SKU_1"] * 199,
        "demand_qty": [float(i) for i in range(1, 200)],
    })
    surface = CanonicalTableBuilderSurface(canonical_table_provider=lambda rid: df)
    snapshot = surface.render("r1")
    assert snapshot.state["row_count"] == 199
    assert len(snapshot.state["head"]) == 50


def test_canonical_table_builder_handles_empty_table() -> None:
    """An empty canonical table surfaces row_count=0 + empty head, no crash."""
    df = pd.DataFrame({"week_start": [], "demand_qty": []})
    surface = CanonicalTableBuilderSurface(canonical_table_provider=lambda rid: df)
    snapshot = surface.render("r-empty")
    assert snapshot.state["row_count"] == 0
    assert snapshot.state["head"] == []


def test_canonical_table_builder_handles_missing_table() -> None:
    """A run with no canonical table yet surfaces a 'no data' placeholder."""
    surface = CanonicalTableBuilderSurface(canonical_table_provider=lambda rid: None)
    snapshot = surface.render("r-empty")
    assert snapshot.state["row_count"] == 0
    assert snapshot.state["columns"] == []
