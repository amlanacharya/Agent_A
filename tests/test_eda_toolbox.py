"""Tests for the EDA toolbox.

The toolbox takes the canonical demand table + provisional segment map
(post-Phase 1 / pre-Phase 2 handoff) and assembles an EDAReport with
per-series demand profiles, per-series FeatureFlags, and a deterministic
narrative. Most of the per-series statistics are delegated to
``forecasting.tools.preflight_stats``; this module is the thin orchestrator
that maps them onto the EDAReport / SeriesDemandProfile contract.
"""
from __future__ import annotations

import pandas as pd
import pytest

from forecasting.contracts import (
    EDAReport,
    FeatureFlags,
    SBClass,
    SegmentDef,
    SegmentMap,
    SeriesDemandProfile,
)
from forecasting.eda_toolbox import (
    _FOURIER_SEASONAL_THRESHOLD,
    _build_feature_flags,
    build_eda_report,
)


def _canonical_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _single_segment_map(series_keys: list[str], run_id: str = "r1") -> SegmentMap:
    return SegmentMap(
        run_id=run_id,
        segments=[SegmentDef(segment_id="G1", label="all series", series_keys=series_keys)],
        provisional=True,
        derived_by="test:default",
    )


def test_series_profile_carries_sb_class_from_adi_cv2():
    """A constant-demand series is SMOOTH; high-variance demand is ERRATIC/LUMPY."""
    rows = []
    base = pd.Timestamp("2024-01-01")
    # SKU_A: constant 10 across 12 weeks -> ADI=1, CV2=0 -> SMOOTH
    for week in range(12):
        rows.append(
            {
                "series_key": "SKU_A|NORTH",
                "date": base + pd.Timedelta(weeks=week),
                "demand": 10.0,
                "promo": False,
            }
        )
    # SKU_B: every-other-week spikes, otherwise zero. ADI>=1.32 (lots of
    # zeros), CV2>0.49 (huge variance) -> INTERMITTENT or LUMPY.
    spike_pattern = [0, 50, 0, 0, 80, 0, 0, 30, 0, 0, 100, 0]
    for week in range(12):
        rows.append(
            {
                "series_key": "SKU_B|NORTH",
                "date": base + pd.Timedelta(weeks=week),
                "demand": float(spike_pattern[week]),
                "promo": False,
            }
        )
    table = _canonical_table(rows)
    seg_map = _single_segment_map(["SKU_A|NORTH", "SKU_B|NORTH"])

    report = build_eda_report(table, seg_map)
    by_key = {p.series_key: p for p in report.series_profiles}

    assert by_key["SKU_A|NORTH"].sb_class == "SMOOTH"
    assert by_key["SKU_A|NORTH"].adi == pytest.approx(1.0)
    assert by_key["SKU_A|NORTH"].cv2 == pytest.approx(0.0)
    # SKU_B has both lots of zeros and high-variance spikes -> not SMOOTH.
    assert by_key["SKU_B|NORTH"].sb_class in ("ERRATIC", "INTERMITTENT", "LUMPY")
    assert by_key["SKU_B|NORTH"].sb_class != "SMOOTH"
    assert by_key["SKU_B|NORTH"].adi > 1.32  # many zeros -> high ADI


def _series_with_class(sb_class: SBClass) -> pd.DataFrame:
    """Build a tiny canonical table that classifies into the given SB class.

    Uses enough data to be a real signal rather than a constant series (which
    always classifies as SMOOTH). Each shape is hand-picked to land in the
    target class per the Syntetos-Boylan thresholds in
    ``forecasting.tools.preflight_stats._sb`` (ADI<1.32, CV2<0.49 -> SMOOTH,
    etc.).
    """
    base = pd.Timestamp("2024-01-01")
    rows: list[dict] = []
    if sb_class == "SMOOTH":
        # Constant 10 -> ADI=1.0, CV2=0.0
        for week in range(12):
            rows.append({"date": base + pd.Timedelta(weeks=week), "demand": 10.0, "promo": False})
    elif sb_class == "INTERMITTENT":
        # Lots of zeros, modest non-zero variance -> ADI>=1.32, CV2<0.49
        pattern = [0, 5, 0, 0, 6, 0, 0, 5, 0, 0, 6, 0]
        for week in range(12):
            rows.append({"date": base + pd.Timedelta(weeks=week), "demand": float(pattern[week]), "promo": False})
    elif sb_class == "ERRATIC":
        # No zeros, high variance -> ADI<1.32, CV2>=0.49
        pattern = [50, 5, 80, 2, 60, 1, 90, 3, 70, 4, 55, 2]
        for week in range(12):
            rows.append({"date": base + pd.Timedelta(weeks=week), "demand": float(pattern[week]), "promo": False})
    elif sb_class == "LUMPY":
        # Both many zeros and high variance -> ADI>=1.32, CV2>=0.49
        pattern = [0, 200, 0, 0, 50, 0, 0, 300, 0, 0, 80, 0]
        for week in range(12):
            rows.append({"date": base + pd.Timedelta(weeks=week), "demand": float(pattern[week]), "promo": False})
    return _canonical_table(rows)


def test_recommended_models_varies_by_sb_class():
    expected_per_class: dict[SBClass, list[str]] = {
        "SMOOTH": ["croston", "sba"],
        "INTERMITTENT": ["croston", "sba"],
        "ERRATIC": ["croston", "sba", "ets"],
        "LUMPY": ["croston"],
    }
    observed: dict[SBClass, list[str]] = {}
    for sb_class in ("SMOOTH", "INTERMITTENT", "ERRATIC", "LUMPY"):
        df = _series_with_class(sb_class)
        df = df.assign(series_key=f"SKU_{sb_class}|NORTH")
        table = _canonical_table(df.to_dict(orient="records"))
        seg_map = _single_segment_map([f"SKU_{sb_class}|NORTH"], run_id=f"r-{sb_class}")
        report = build_eda_report(table, seg_map)
        profile = report.series_profiles[0]
        assert profile.sb_class == sb_class, f"test data must classify as {sb_class}"
        observed[sb_class] = list(profile.recommended_models)

    assert observed == expected_per_class


def test_feature_config_enables_fourier_when_seasonality_above_threshold():
    flags = _build_feature_flags(seasonal_strength=_FOURIER_SEASONAL_THRESHOLD + 0.01, has_promo=False, frequency_period=52)
    assert flags.use_fourier is True
    assert flags.frequency_period == 52


def test_feature_config_disables_fourier_when_seasonality_below_threshold():
    flags = _build_feature_flags(seasonal_strength=_FOURIER_SEASONAL_THRESHOLD - 0.01, has_promo=False, frequency_period=52)
    assert flags.use_fourier is False
    # frequency_period is still propagated even when Fourier is off, so that
    # downstream consumers can use it for other seasonal features.
    assert flags.frequency_period == 52


def test_feature_config_lag_features_default_on():
    flags = _build_feature_flags(seasonal_strength=0.0, has_promo=False, frequency_period=None)
    assert flags.use_lag_features is True


def _two_series_table_with_promo() -> pd.DataFrame:
    """Canonical table: SKU_A has no promo, SKU_B has promo on weeks 2 and 5."""
    base = pd.Timestamp("2024-01-01")
    rows: list[dict] = []
    for week in range(12):
        rows.append(
            {"series_key": "SKU_A|NORTH", "date": base + pd.Timedelta(weeks=week), "demand": 10.0, "promo": False}
        )
        rows.append(
            {"series_key": "SKU_B|NORTH", "date": base + pd.Timedelta(weeks=week), "demand": 10.0, "promo": week in (1, 4)}
        )
    return _canonical_table(rows)


def test_feature_config_enables_promo_indicator_when_any_promo_present():
    table = _two_series_table_with_promo()
    seg_map = _single_segment_map(["SKU_A|NORTH", "SKU_B|NORTH"], run_id="r-promo")

    report = build_eda_report(table, seg_map)

    assert report.feature_config["SKU_A|NORTH"].use_promo_indicator is False
    assert report.feature_config["SKU_B|NORTH"].use_promo_indicator is True


def test_feature_config_keyed_by_series_key():
    table = _two_series_table_with_promo()
    seg_map = _single_segment_map(["SKU_A|NORTH", "SKU_B|NORTH"], run_id="r-keyed")

    report = build_eda_report(table, seg_map)

    assert set(report.feature_config.keys()) == {"SKU_A|NORTH", "SKU_B|NORTH"}
    for key, flags in report.feature_config.items():
        assert isinstance(flags, FeatureFlags)


def test_feature_config_propagates_frequency_period_when_supplied():
    table = _two_series_table_with_promo()
    seg_map = _single_segment_map(["SKU_A|NORTH", "SKU_B|NORTH"], run_id="r-fp")

    report = build_eda_report(table, seg_map, frequency_period=52)

    assert report.feature_config["SKU_A|NORTH"].frequency_period == 52
    assert report.feature_config["SKU_B|NORTH"].frequency_period == 52


def test_narrative_mentions_segment_count_and_class_distribution():
    rows = _two_series_table_with_promo().to_dict(orient="records")
    seg_map = SegmentMap(
        run_id="r-narr",
        segments=[
            SegmentDef(segment_id="G1", label="region=NORTH", series_keys=["SKU_A|NORTH", "SKU_B|NORTH"]),
        ],
        provisional=True,
        derived_by="test:segment_by_region",
    )

    report = build_eda_report(_canonical_table(rows), seg_map)

    assert report.narrative  # non-empty
    # Should mention the number of segments and the series count.
    assert "1 segment" in report.narrative
    assert "2 series" in report.narrative
    # Both SMOOTH (constant demand) -> class distribution should appear.
    assert "SMOOTH" in report.narrative


def test_segment_profiles_are_populated():
    rows = _two_series_table_with_promo().to_dict(orient="records")
    seg_map = SegmentMap(
        run_id="r-seg",
        segments=[
            SegmentDef(segment_id="G1", label="region=NORTH", series_keys=["SKU_A|NORTH", "SKU_B|NORTH"]),
        ],
        provisional=True,
        derived_by="test:default",
    )

    report = build_eda_report(_canonical_table(rows), seg_map)

    assert len(report.segment_profiles) == 1
    profile = report.segment_profiles[0]
    assert profile.segment_id == "G1"
    assert profile.series_count == 2
    # Both series are SMOOTH (constant demand) -> distribution should reflect that.
    assert profile.demand_class_distribution.get("SMOOTH") == 2


def test_escalation_hook_fires_for_degenerate_series(tmp_outputs, run_id):
    """A series with very few observations cannot be characterised by the
    standard preflight stats — the EDA layer should escalate, not silently
    drop. The tracker is a real ``EscalationTracker`` writing to the
    per-run escalation directory (``run_dir / "escalations/eda.json"``)."""
    base = pd.Timestamp("2024-01-01")
    # Two series: one healthy, one with only 2 rows.
    rows = [
        {"series_key": "SKU_HEALTHY|N", "date": base, "demand": 10.0, "promo": False},
        {"series_key": "SKU_HEALTHY|N", "date": base + pd.Timedelta(weeks=1), "demand": 12.0, "promo": False},
        {"series_key": "SKU_HEALTHY|N", "date": base + pd.Timedelta(weeks=2), "demand": 11.0, "promo": False},
        {"series_key": "SKU_HEALTHY|N", "date": base + pd.Timedelta(weeks=3), "demand": 13.0, "promo": False},
        {"series_key": "SKU_DEGEN|N", "date": base, "demand": 5.0, "promo": False},
        {"series_key": "SKU_DEGEN|N", "date": base + pd.Timedelta(weeks=1), "demand": 6.0, "promo": False},
    ]
    seg_map = SegmentMap(
        run_id=run_id,
        segments=[SegmentDef(segment_id="G1", label="all", series_keys=["SKU_HEALTHY|N", "SKU_DEGEN|N"])],
        provisional=True,
        derived_by="test:default",
    )
    from forecasting.code_escalation import EscalationTracker

    tracker = EscalationTracker(run_id=run_id, layer="eda")

    report = build_eda_report(_canonical_table(rows), seg_map, escalation=tracker)

    # Both series still appear in the profile (we don't drop them silently).
    keys = {p.series_key for p in report.series_profiles}
    assert "SKU_DEGEN|N" in keys
    # Escalation was recorded for the degenerate series.
    assert tracker.attempts >= 1
    assert tracker.status == "code_escalation"


def test_build_eda_report_emits_one_series_profile_per_series():
    rows = []
    for week in range(1, 9):  # 8 weeks of stable demand
        rows.append(
            {
                "series_key": "SKU_A|NORTH",
                "date": pd.Timestamp(f"2024-W{week:02d}-1", isoformat=True) if False else pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=week - 1),
                "demand": 10.0,
                "promo": False,
            }
        )
        rows.append(
            {
                "series_key": "SKU_B|NORTH",
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=week - 1),
                "demand": 5.0,
                "promo": False,
            }
        )
    table = _canonical_table(rows)
    seg_map = _single_segment_map(["SKU_A|NORTH", "SKU_B|NORTH"])

    report = build_eda_report(table, seg_map)

    assert isinstance(report, EDAReport)
    assert report.run_id == "r1"
    assert len(report.series_profiles) == 2
    keys = {p.series_key for p in report.series_profiles}
    assert keys == {"SKU_A|NORTH", "SKU_B|NORTH"}
    for profile in report.series_profiles:
        assert isinstance(profile, SeriesDemandProfile)
