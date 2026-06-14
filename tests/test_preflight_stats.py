import pandas as pd
import warnings

from forecasting.contracts import SchemaMapping, SegmentDef, SegmentMap
from forecasting.tools.preflight_stats import (
    aggregate_segment_profiles,
    assign_provisional_segments,
    collect_segment_exceptions,
    compute_adi_cv2_per_series,
    detect_seasonality_strength,
    detect_spikes_per_series,
    detect_structural_break_candidates,
    detect_trend_strength,
    detect_zero_runs_per_series,
)


def _sm(vals: list[float], key: str = "SKU|NORTH") -> dict:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-06", periods=len(vals), freq="W"),
            "demand": vals,
        }
    )
    return {key: df}


def test_smooth_series_classified_smooth():
    stats = compute_adi_cv2_per_series(_sm([10.0] * 52))
    assert stats["SKU|NORTH"].sb_class == "SMOOTH"


def test_intermittent_series_not_smooth():
    vals = [0, 0, 0, 5] * 13
    stats = compute_adi_cv2_per_series(_sm(vals))
    assert stats["SKU|NORTH"].sb_class in ("INTERMITTENT", "LUMPY")


def test_adi_gt_one_for_intermittent():
    vals = [0, 0, 5] * 17
    stats = compute_adi_cv2_per_series(_sm(vals))
    assert stats["SKU|NORTH"].adi > 1.0


def test_zero_run_detects_max_run():
    vals = [5, 5, 0, 0, 0, 0, 5, 5]
    stats = detect_zero_runs_per_series(_sm(vals))
    assert stats["SKU|NORTH"].max_zero_run == 4


def test_spike_detected():
    base = [10.0] * 50
    base[25] = 500.0
    stats = detect_spikes_per_series(_sm(base))
    assert stats["SKU|NORTH"].spike_count >= 1


def test_no_spike_for_flat():
    stats = detect_spikes_per_series(_sm([10.0] * 50))
    assert stats["SKU|NORTH"].spike_count == 0


def test_trend_up_detected():
    vals = [float(i) for i in range(1, 53)]
    stats = detect_trend_strength(_sm(vals))
    assert stats["SKU|NORTH"].direction == "up"
    assert stats["SKU|NORTH"].trend_strength > 0.8


def test_seasonality_low_for_flat():
    stats = detect_seasonality_strength(_sm([10.0] * 52))
    assert stats["SKU|NORTH"].seasonal_strength < 0.3


def test_aggregate_profiles_sum_equals_series_count():
    series_map = {**_sm([10.0] * 20, "A|N"), **_sm([0, 0, 5] * 7, "B|N")}
    adi_cv2 = compute_adi_cv2_per_series(series_map)
    profiles = aggregate_segment_profiles(series_map, adi_cv2, {})
    assert sum(p.series_count for p in profiles) == 2


def test_adi_accounts_for_trailing_zeros():
    stats = compute_adi_cv2_per_series(_sm([5.0, 0.0, 0.0]))
    assert stats["SKU|NORTH"].adi > 1.0


def test_adi_accounts_for_leading_and_middle_zeros():
    stats = compute_adi_cv2_per_series(_sm([0.0, 5.0, 0.0, 0.0, 5.0]))
    assert stats["SKU|NORTH"].adi == 2.5


def test_non_finite_values_emit_no_warnings_and_return_finite_stats():
    series_map = _sm([1.0, 2.0, float("inf")])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        adi = compute_adi_cv2_per_series(series_map)["SKU|NORTH"]
        zr = detect_zero_runs_per_series(series_map)["SKU|NORTH"]
        sp = detect_spikes_per_series(series_map)["SKU|NORTH"]
        tr = detect_trend_strength(series_map)["SKU|NORTH"]
        se = detect_seasonality_strength(series_map)["SKU|NORTH"]
        br = detect_structural_break_candidates(series_map)
    assert len(captured) == 0
    assert adi.adi < float("inf")
    assert adi.cv2 < float("inf")
    assert zr.zero_fraction < float("inf")
    assert sp.max_spike_ratio < float("inf")
    assert tr.trend_strength < float("inf")
    assert se.seasonal_strength < float("inf")
    assert isinstance(br, list)


def test_spike_detection_handles_empty_series():
    df = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "demand": pd.Series(dtype=float)})
    stats = detect_spikes_per_series({"EMPTY|N": df})
    assert stats["EMPTY|N"].spike_count == 0
    assert stats["EMPTY|N"].max_spike_ratio == 0.0


def test_aggregate_profiles_ignores_missing_segment_members():
    series_map = _sm([10.0] * 8, "A|N")
    adi_cv2 = compute_adi_cv2_per_series(series_map)
    seg_map = SegmentMap(
        run_id="r1",
        segments=[SegmentDef(segment_id="G1", label="mix", series_keys=["A|N", "MISSING|N"], provisional=True)],
        provisional=True,
        derived_by="test",
    )
    profiles = aggregate_segment_profiles(series_map, adi_cv2, seg_map)
    assert profiles[0].series_count == 1
    assert profiles[0].example_keys == ["A|N"]


def test_assign_provisional_segments_keeps_collision_suffixed_keys_distinct():
    schema = SchemaMapping(date_col="date", demand_col="demand", grain_cols=["sku"], extra_cols=[])
    series_map = {
        "A1": pd.DataFrame({"date": pd.date_range("2020-01-06", periods=4, freq="W"), "demand": [1.0, 2.0, 3.0, 4.0]}),
        "A1|HABCDEF12": pd.DataFrame(
            {"date": pd.date_range("2020-01-06", periods=4, freq="W"), "demand": [1.0, 1.0, 1.0, 1.0]}
        ),
    }
    seg_map = assign_provisional_segments(series_map, schema, {"segment_by": ["sku"], "max_segments": 12})
    all_keys = [k for seg in seg_map.segments for k in seg.series_keys]
    assert sorted(all_keys) == sorted(series_map.keys())
    assert len(seg_map.segments) == 2
