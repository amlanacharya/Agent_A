import pandas as pd

from forecasting.tools.preflight_stats import (
    aggregate_segment_profiles,
    collect_segment_exceptions,
    compute_adi_cv2_per_series,
    detect_seasonality_strength,
    detect_spikes_per_series,
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
