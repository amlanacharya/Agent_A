from __future__ import annotations

import io
import json
import uuid
from pathlib import Path

import pandas as pd

from forecasting.contracts import BlockingIssue, DataQualityWarning, PreflightBundle
from forecasting.data_store import replace_run
from forecasting.run_state import run_dir
from forecasting.tools.preflight_schema import (
    build_series_keys,
    detect_frequency_and_grain,
    map_schema,
    profile_uploaded_data,
)
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
    measure_promo_alignment,
)


class PreflightBlockingError(Exception):
    def __init__(self, issues: list[BlockingIssue]):
        super().__init__(f"Preflight blocked: {[issue.code for issue in issues]}")
        self.issues = issues


def _parse_csv(file_bytes: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input did not parse into a tabular CSV DataFrame")
        if df.empty and len(df.columns) == 0:
            raise ValueError("CSV is empty or has no columns")
        if df.empty and all(str(col).startswith("Unnamed:") for col in df.columns):
            raise ValueError("CSV has no usable header/rows")
        return df
    except Exception as exc:
        raise PreflightBlockingError([BlockingIssue(code="UNPARSEABLE_FILE", message=str(exc))]) from exc


def run_preflight(run_id: str, file_bytes: bytes, domain: str, playbook: dict) -> PreflightBundle:
    _ = domain
    output_dir = run_dir(run_id)
    df = _parse_csv(file_bytes)

    quality = profile_uploaded_data(df)
    if quality.blocking_issues:
        raise PreflightBlockingError(quality.blocking_issues)

    schema = map_schema(df, playbook)
    grain = detect_frequency_and_grain(df, schema)
    series_map = build_series_keys(df, schema, playbook)

    quality.series_count = len(series_map)
    min_series = int(playbook.get("min_series", 1))
    if len(series_map) < min_series:
        quality.blocking_issues.append(
            BlockingIssue(
                code="BELOW_MIN_SERIES",
                message=f"{len(series_map)} series < playbook min_series={min_series}",
            )
        )
        raise PreflightBlockingError(quality.blocking_issues)

    adi_cv2 = compute_adi_cv2_per_series(series_map)
    zero_runs = detect_zero_runs_per_series(series_map)
    spikes = detect_spikes_per_series(series_map)
    promo_align = measure_promo_alignment(series_map, schema)
    trend = detect_trend_strength(series_map)
    seasonality = detect_seasonality_strength(series_map)
    break_candidates = detect_structural_break_candidates(series_map)

    segment_map = assign_provisional_segments(series_map, schema, playbook)
    segment_map.run_id = run_id
    segment_profiles = aggregate_segment_profiles(series_map, adi_cv2, segment_map)
    segment_exceptions = collect_segment_exceptions(adi_cv2, zero_runs, spikes, segment_map)

    _add_stat_warnings(quality, grain, playbook)

    bundle = PreflightBundle(
        run_id=run_id,
        data_quality_report=quality,
        schema_mapping=schema,
        grain_report=grain,
        segment_profiles=segment_profiles,
        segment_exceptions=segment_exceptions,
        segments=segment_map.segments,
        domain_playbook=playbook,
    )

    output = {
        "bundle": bundle.model_dump(),
        "segment_map": segment_map.model_dump(),
        "break_candidates": [candidate.model_dump() for candidate in break_candidates],
        "per_series": {
            "adi_cv2": {k: v.model_dump() for k, v in adi_cv2.items()},
            "zero_runs": {k: v.model_dump() for k, v in zero_runs.items()},
            "spikes": {k: v.model_dump() for k, v in spikes.items()},
            "promo_align": {k: v.model_dump() for k, v in promo_align.items()},
            "trend": {k: v.model_dump() for k, v in trend.items()},
            "seasonality": {k: v.model_dump() for k, v in seasonality.items()},
        },
    }

    _write_preflight_json_atomic(output_dir, output)
    replace_run(run_id, series_map)

    return bundle


def _add_stat_warnings(quality, grain, playbook: dict) -> None:
    min_history_periods = int(playbook.get("min_history_periods", 12))
    if grain.min_periods < min_history_periods:
        quality.warnings.append(
            DataQualityWarning(
                code="SHORT_HISTORY",
                message=f"Some series have fewer than {min_history_periods} periods (min={grain.min_periods})",
            )
        )


def _write_preflight_json_atomic(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "preflight.json"
    temp_path = output_dir / f".preflight.{uuid.uuid4().hex}.tmp"
    try:
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(final_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
