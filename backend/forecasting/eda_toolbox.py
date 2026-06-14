"""EDA toolbox — post-canonical orchestration of preflight stats.

This module is the thin EDA-layer entry point that the Forge agent calls
during ``forge_eda``. It takes a canonical demand table (already validated
by ``forecasting.canonical_data.build_canonical_table``) and a provisional
``SegmentMap``, then assembles an ``EDAReport`` with:

* per-segment ``SegmentProfile`` aggregates (delegated to
  ``forecasting.tools.preflight_stats.aggregate_segment_profiles``);
* per-series ``SeriesDemandProfile`` records (ADI/CV²/SB class, trend,
  seasonality, recommended models);
* a per-series ``FeatureFlags`` recommendation driving the Feature Factory;
* a short deterministic narrative summarising the segments and class mix;
* an optional ``EscalationTracker`` hook for degenerate series the standard
  toolbox cannot characterise.

The function is read-only on its inputs. It is pure except for the
optional ``escalation`` side effect, which writes through the tracker.
"""
from __future__ import annotations

import pandas as pd

from forecasting.code_escalation import EscalationTracker
from forecasting.contracts import (
    EDAReport,
    FeatureFlags,
    SBClass,
    SegmentMap,
    SeriesDemandProfile,
)
from forecasting.tools.preflight_stats import (
    aggregate_segment_profiles,
    compute_adi_cv2_per_series,
    detect_seasonality_strength,
    detect_trend_strength,
)


def build_eda_report(
    canonical_table: pd.DataFrame,
    segment_map: SegmentMap,
    *,
    frequency_period: int | None = None,
    escalation: EscalationTracker | None = None,
) -> EDAReport:
    """Assemble an EDAReport from a canonical demand table + segment map.

    See module docstring for the contract.
    """
    _validate_inputs(canonical_table, segment_map)
    series_keys = _series_keys_for_segments(segment_map)
    series_map = _canonical_to_series_map(canonical_table, series_keys)
    adi_cv2 = compute_adi_cv2_per_series(series_map)
    trend = detect_trend_strength(series_map)
    seasonality = detect_seasonality_strength(series_map)
    series_profiles = [
        SeriesDemandProfile(
            series_key=key,
            sb_class=adi_cv2[key].sb_class,
            adi=adi_cv2[key].adi,
            cv2=adi_cv2[key].cv2,
            trend_strength=trend[key].trend_strength,
            seasonal_strength=seasonality[key].seasonal_strength,
            recommended_models=_recommend_models(adi_cv2[key].sb_class),
        )
        for key in series_keys
        if key in adi_cv2
    ]
    feature_config = {
        key: _build_feature_flags(
            seasonal_strength=seasonality[key].seasonal_strength,
            has_promo=_series_has_promo(canonical_table, key),
            frequency_period=frequency_period,
        )
        for key in series_keys
    }
    segment_profiles = aggregate_segment_profiles(series_map, adi_cv2, segment_map)
    narrative = _build_narrative(segment_profiles, series_profiles)
    if escalation is not None:
        _maybe_escalate_degenerate(series_map, escalation)
    return EDAReport(
        run_id=segment_map.run_id,
        segment_profiles=segment_profiles,
        series_profiles=series_profiles,
        feature_config=feature_config,
        narrative=narrative,
    )


# Per-sb_class model-recommendation rule. Stable placeholder strings; the
# Foundry in Phase 4 replaces them with concrete model IDs. The rule's
# purpose today is to gate which family the harness should even try
# (e.g., LUMPY -> only Croston-class; ERRATIC -> adds ETS), not to be a
# final selection.
_MODEL_RECOMMENDATIONS: dict[SBClass, list[str]] = {
    "SMOOTH": ["croston", "sba"],
    "INTERMITTENT": ["croston", "sba"],
    "ERRATIC": ["croston", "sba", "ets"],
    "LUMPY": ["croston"],
}


def _recommend_models(sb_class: SBClass) -> list[str]:
    return list(_MODEL_RECOMMENDATIONS.get(sb_class, []))


# Threshold above which a series' seasonal autocorrelation is strong
# enough to justify enabling Fourier terms in its FeatureFlags. Conservative
# default — the gain on noisy series rarely justifies the extra columns.
_FOURIER_SEASONAL_THRESHOLD: float = 0.3


def _build_feature_flags(
    *,
    seasonal_strength: float,
    has_promo: bool,
    frequency_period: int | None,
) -> FeatureFlags:
    """Assemble per-series FeatureFlags from EDA measurements.

    Rule: Fourier terms are enabled only when the series shows meaningful
    seasonal autocorrelation; the promo indicator is enabled whenever the
    series has any non-zero promo flag observed; lag features stay on by
    default (matches the FeatureFlags dataclass).
    """
    return FeatureFlags(
        use_fourier=seasonal_strength > _FOURIER_SEASONAL_THRESHOLD,
        use_lag_features=True,
        use_promo_indicator=has_promo,
        frequency_period=frequency_period,
    )


def _validate_inputs(canonical_table: pd.DataFrame, segment_map: SegmentMap) -> None:
    if "series_key" not in canonical_table.columns:
        raise ValueError("canonical_table must contain a 'series_key' column")
    if "demand" not in canonical_table.columns:
        raise ValueError("canonical_table must contain a 'demand' column")


def _series_keys_for_segments(segment_map: SegmentMap) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for segment in segment_map.segments:
        for key in segment.series_keys:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def _canonical_to_series_map(
    canonical_table: pd.DataFrame, series_keys: list[str]
) -> dict[str, pd.DataFrame]:
    """Group the canonical table by series_key into the per-series DataFrame
    map that ``forecasting.tools.preflight_stats`` already consumes."""
    if not series_keys:
        return {}
    grouped = {
        key: group.sort_values("date").reset_index(drop=True)
        for key, group in canonical_table.groupby("series_key", sort=False)
        if key in set(series_keys)
    }
    return grouped


def _series_has_promo(canonical_table: pd.DataFrame, series_key: str) -> bool:
    """Return True if any row for ``series_key`` has a truthy ``promo`` value.

    The canonical table stores the promo flag as an object-typed column
    (bool / 0/1 / "true"/"false"); we coerce to bool and OR-reduce. Falsey
    strings and zeros stay False.
    """
    if "promo" not in canonical_table.columns:
        return False
    rows = canonical_table.loc[canonical_table["series_key"] == series_key, "promo"]
    if rows.empty:
        return False
    coerced = rows.map(_coerce_promo_flag)
    return bool(coerced.any())


def _coerce_promo_flag(value: object) -> bool:
    """Normalise a single ``promo`` cell to bool.

    Mirrors ``forecasting.canonical_data._optional_flag``: bool stays bool,
    numeric 1 is True, the string "true"/"yes"/"y"/"1" is True, everything
    else is False. Unparseable inputs do not raise — they are treated as
    no-promo, since a non-bool promo is a data-quality issue handled at
    canonical-table construction.
    """
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return False
    if isinstance(value, (int, float)):
        return value == 1
    text = str(value).strip().lower()
    return text in {"true", "yes", "y", "1"}


# Minimum number of observations needed to characterise a series with the
# standard preflight stats. Below this, ADI / CV² / seasonality are too
# noisy to be informative and the EDA layer should escalate.
_MIN_OBSERVATIONS_FOR_PROFILE: int = 4


def _maybe_escalate_degenerate(
    series_map: dict[str, pd.DataFrame], tracker: EscalationTracker
) -> None:
    """Record an escalation attempt when any series has too few observations.

    Per the governance contract, the tracker caps at
    ``MAX_CODE_ATTEMPTS`` (3) per layer; once exhausted, the EDA call must
    raise ``EscalationLimitReached`` to halt the run. We let the tracker
    surface that itself — callers that need the bounded-retry workflow
    should use the ``Custom adapter workflow`` slice.
    """
    for key, df in series_map.items():
        if len(df) < _MIN_OBSERVATIONS_FOR_PROFILE:
            tracker.request_code_attempt(
                reason=f"series {key!r} has {len(df)} observations (< {_MIN_OBSERVATIONS_FOR_PROFILE}); standard EDA stats are not reliable"
            )
            return  # one attempt per call — bounded retry is handled by the caller


def _build_narrative(segment_profiles, series_profiles) -> str:
    """Render a short, deterministic narrative for the EDA report.

    Format: "<n> segment(s) covering <m> series: <class>=<count> (...).
    Intended for human review and as a fallback when an LLM summariser is
    not in the loop. Stays stable so tests can assert on its contents.
    """
    n_segments = len(segment_profiles)
    n_series = sum(profile.series_count for profile in segment_profiles)
    distribution: dict[str, int] = {}
    for profile in series_profiles:
        distribution[profile.sb_class] = distribution.get(profile.sb_class, 0) + 1
    class_summary = ", ".join(
        f"{sb_class}={count}" for sb_class, count in sorted(distribution.items())
    ) or "no classified series"
    segment_word = "segment" if n_segments == 1 else "segments"
    series_word = "series" if n_series == 1 else "series"
    return f"{n_segments} {segment_word} covering {n_series} {series_word}: {class_summary}."


__all__ = ("build_eda_report",)
