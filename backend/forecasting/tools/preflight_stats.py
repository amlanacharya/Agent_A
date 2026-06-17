from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from forecasting.contracts import (
    AdiCv2Stats,
    BreakCandidate,
    PromoAlignmentStats,
    SBClass,
    SchemaMapping,
    SeasonalityStats,
    SegmentDef,
    SegmentMap,
    SegmentProfile,
    SeriesException,
    SpikeStats,
    TrendStats,
    ZeroRunStats,
)
from forecasting.stats_utils import autocorr


def compute_adi_cv2_per_series(series_map: dict[str, pd.DataFrame]) -> dict[str, AdiCv2Stats]:
    result: dict[str, AdiCv2Stats] = {}
    for key, df in series_map.items():
        demand = _extract_numeric_demand(df, finite_only=True)
        non_zero = demand[demand > 0]
        if len(non_zero) == 0:
            result[key] = AdiCv2Stats(series_key=key, adi=999.0, cv2=0.0, sb_class="LUMPY")
            continue
        adi = float(len(demand) / len(non_zero))
        mu = float(np.mean(non_zero))
        std = float(np.std(non_zero, ddof=1)) if len(non_zero) > 1 else 0.0
        cv2 = (std / mu) ** 2 if mu > 0 else 0.0
        result[key] = AdiCv2Stats(series_key=key, adi=adi, cv2=cv2, sb_class=_sb(adi, cv2))
    return result


def detect_zero_runs_per_series(series_map: dict[str, pd.DataFrame]) -> dict[str, ZeroRunStats]:
    result: dict[str, ZeroRunStats] = {}
    for key, df in series_map.items():
        demand = _extract_numeric_demand(df, finite_only=True)
        mask = demand == 0
        result[key] = ZeroRunStats(
            series_key=key,
            max_zero_run=_max_run(mask),
            zero_fraction=float(mask.sum()) / len(demand) if len(demand) else 0.0,
        )
    return result


def detect_spikes_per_series(series_map: dict[str, pd.DataFrame]) -> dict[str, SpikeStats]:
    result: dict[str, SpikeStats] = {}
    for key, df in series_map.items():
        demand = _extract_numeric_demand(df, finite_only=True)
        if len(demand) == 0:
            result[key] = SpikeStats(series_key=key, spike_count=0, max_spike_ratio=0.0)
            continue
        q1, q3 = np.percentile(demand, [25, 75])
        threshold = q3 + 3 * (q3 - q1)
        spikes = demand[demand > threshold]
        max_ratio = float(spikes.max() / q3) if len(spikes) and q3 > 0 else 0.0
        result[key] = SpikeStats(series_key=key, spike_count=len(spikes), max_spike_ratio=max_ratio)
    return result


def detect_trend_strength(series_map: dict[str, pd.DataFrame]) -> dict[str, TrendStats]:
    result: dict[str, TrendStats] = {}
    for key, df in series_map.items():
        demand = _extract_numeric_demand(df, finite_only=True)
        nz_idx = np.where(demand > 0)[0]
        if len(nz_idx) < 3:
            result[key] = TrendStats(series_key=key, trend_strength=0.0, direction="flat")
            continue
        x, y = nz_idx.astype(float), demand[nz_idx]
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        thr = 0.01 * y.mean()
        direction = "up" if slope > thr else ("down" if slope < -thr else "flat")
        result[key] = TrendStats(series_key=key, trend_strength=round(r2, 4), direction=direction)
    return result


def detect_seasonality_strength(series_map: dict[str, pd.DataFrame]) -> dict[str, SeasonalityStats]:
    result: dict[str, SeasonalityStats] = {}
    for key, df in series_map.items():
        demand = _extract_numeric_demand(df, finite_only=True)
        best_str, best_per = 0.0, None
        for lag in [52, 12, 4]:
            if len(demand) > lag * 2:
                ac = autocorr(demand, lag)
                if ac > best_str:
                    best_str, best_per = ac, lag
        result[key] = SeasonalityStats(
            series_key=key,
            seasonal_strength=round(max(0.0, best_str), 4),
            dominant_period=best_per,
        )
    return result


def measure_promo_alignment(
    series_map: dict[str, pd.DataFrame], schema: SchemaMapping
) -> dict[str, PromoAlignmentStats]:
    _ = schema
    result: dict[str, PromoAlignmentStats] = {}
    for key, df in series_map.items():
        has_promo = any(c in df.columns for c in ["promo", "promotion"])
        result[key] = PromoAlignmentStats(series_key=key, has_promo_col=has_promo)
    return result


def detect_structural_break_candidates(series_map: dict[str, pd.DataFrame]) -> list[BreakCandidate]:
    candidates: list[BreakCandidate] = []
    for key, df in series_map.items():
        demand, finite_positions = _extract_numeric_demand(df, finite_only=True, return_positions=True)
        if len(demand) < 16:
            continue
        mu = demand.mean()
        cusum = np.cumsum(demand - mu)
        idx = int(np.argmax(np.abs(cusum)))
        strength = abs(cusum[idx]) / (demand.std() + 1e-9)
        if strength > 3.0:
            if "date" in df.columns:
                src_idx = int(finite_positions[idx])
                date_val = str(df["date"].iloc[src_idx])
            else:
                date_val = str(idx)
            candidates.append(
                BreakCandidate(
                    series_key=key,
                    break_period=date_val,
                    confidence=round(min(1.0, strength / 10.0), 3),
                )
            )
    return candidates


def assign_provisional_segments(
    series_map: dict[str, pd.DataFrame],
    schema: SchemaMapping,
    playbook: dict,
) -> SegmentMap:
    seg_by = [c for c in playbook.get("segment_by", []) if c in schema.grain_cols]
    max_segments = playbook.get("max_segments", 12)

    def _single(reason: str) -> SegmentMap:
        return SegmentMap(
            run_id="",
            segments=[
                SegmentDef(
                    segment_id="G1",
                    label="all series",
                    series_keys=sorted(series_map.keys()),
                    provisional=True,
                )
            ],
            provisional=True,
            derived_by=reason,
        )

    if not seg_by:
        return _single("default:single_segment")

    buckets: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for key in series_map.keys():
        key_parts = key.split("|")
        parts = dict(zip(schema.grain_cols, key_parts[: len(schema.grain_cols)]))
        bucket_key = tuple(parts.get(c, "") for c in seg_by)
        if len(key_parts) > len(schema.grain_cols):
            bucket_key = (*bucket_key, f"__key__={key}")
        buckets[bucket_key].append(key)

    if len(buckets) > max_segments:
        return _single(f"default:single_segment (>{max_segments} groups)")

    segments: list[SegmentDef] = []
    for i, (bkey, keys) in enumerate(sorted(buckets.items()), start=1):
        label = ", ".join(f"{c}={v}" for c, v in zip(seg_by, bkey))
        segments.append(
            SegmentDef(
                segment_id=f"G{i}",
                label=label,
                series_keys=sorted(keys),
                provisional=True,
            )
        )
    return SegmentMap(
        run_id="",
        segments=segments,
        provisional=True,
        derived_by=f"playbook:segment_by={'+'.join(seg_by)}",
    )


def aggregate_segment_profiles(
    series_map: dict[str, pd.DataFrame],
    adi_cv2: dict[str, AdiCv2Stats],
    segment_map: SegmentMap | dict[str, Any] | None,
) -> list[SegmentProfile]:
    _ = series_map
    resolved_segment_map = _coerce_segment_map(adi_cv2=adi_cv2, segment_map=segment_map)
    profiles: list[SegmentProfile] = []
    for seg in resolved_segment_map.segments:
        keys = [k for k in seg.series_keys if k in adi_cv2 and k in series_map]
        dist: dict[str, int] = defaultdict(int)
        fc: dict[str, int] = defaultdict(int)
        adis: list[float] = []
        cv2s: list[float] = []
        for k in keys:
            stats = adi_cv2[k]
            dist[stats.sb_class] += 1
            fc["forecastable" if stats.sb_class in ("SMOOTH", "ERRATIC") else "caution"] += 1
            adis.append(stats.adi)
            cv2s.append(stats.cv2)
        profiles.append(
            SegmentProfile(
                segment_id=seg.segment_id,
                series_count=len(keys),
                demand_class_distribution=dict(dist),
                median_adi=round(float(np.median(adis)), 4) if adis else 0.0,
                median_cv2=round(float(np.median(cv2s)), 4) if cv2s else 0.0,
                forecastability_breakdown=dict(fc),
                example_keys=keys[:3],
            )
        )
    return profiles


def collect_segment_exceptions(
    adi_cv2: dict[str, AdiCv2Stats],
    zero_runs: dict[str, ZeroRunStats],
    spikes: dict[str, SpikeStats],
    segment_map: SegmentMap | dict[str, Any] | None,
) -> list[SeriesException]:
    resolved_segment_map = _coerce_segment_map(adi_cv2=adi_cv2, segment_map=segment_map)
    seg_of = {k: seg.segment_id for seg in resolved_segment_map.segments for k in seg.series_keys}
    exc: list[SeriesException] = []
    for key, zr in zero_runs.items():
        if zr.zero_fraction > 0.8:
            exc.append(
                SeriesException(
                    series_key=key,
                    segment_id=seg_of.get(key, "G1"),
                    exception_type="HIGH_ZERO_FRACTION",
                    detail=f"zero_fraction={zr.zero_fraction:.2f}",
                )
            )
        if zr.max_zero_run >= 8:
            exc.append(
                SeriesException(
                    series_key=key,
                    segment_id=seg_of.get(key, "G1"),
                    exception_type="ZERO_RUN",
                    detail=f"max_zero_run={zr.max_zero_run}",
                )
            )
    for key, sp in spikes.items():
        if sp.spike_count > 0:
            exc.append(
                SeriesException(
                    series_key=key,
                    segment_id=seg_of.get(key, "G1"),
                    exception_type="SPIKE",
                    detail=f"spike_count={sp.spike_count}, max_ratio={sp.max_spike_ratio:.1f}x",
                )
            )
    return exc


def _coerce_segment_map(
    adi_cv2: dict[str, AdiCv2Stats], segment_map: SegmentMap | dict[str, Any] | None
) -> SegmentMap:
    if isinstance(segment_map, SegmentMap):
        return segment_map
    if isinstance(segment_map, dict) and segment_map:
        return SegmentMap.model_validate(segment_map)
    return SegmentMap(
        run_id="",
        segments=[
            SegmentDef(
                segment_id="G1",
                label="all series",
                series_keys=sorted(adi_cv2.keys()),
                provisional=True,
            )
        ],
        provisional=True,
        derived_by="default:single_segment",
    )


def _sb(adi: float, cv2: float) -> SBClass:
    if adi < 1.32 and cv2 < 0.49:
        return "SMOOTH"
    if adi >= 1.32 and cv2 < 0.49:
        return "INTERMITTENT"
    if adi < 1.32 and cv2 >= 0.49:
        return "ERRATIC"
    return "LUMPY"


def _max_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _extract_numeric_demand(
    df: pd.DataFrame, finite_only: bool = True, return_positions: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    raw = pd.to_numeric(df["demand"], errors="coerce")
    arr = raw.to_numpy(dtype=float, copy=False)
    if finite_only:
        mask = np.isfinite(arr)
    else:
        mask = ~np.isnan(arr)
    clean = arr[mask]
    if return_positions:
        return clean, np.flatnonzero(mask)
    return clean
