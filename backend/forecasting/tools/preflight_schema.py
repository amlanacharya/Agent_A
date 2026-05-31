from __future__ import annotations

import hashlib
import re

import pandas as pd

from forecasting.contracts import (
    BlockingIssue,
    DataQualityReport,
    DataQualityWarning,
    GrainReport,
    SchemaMapping,
)


def profile_uploaded_data(df: pd.DataFrame) -> DataQualityReport:
    blocking: list[BlockingIssue] = []
    warnings: list[DataQualityWarning] = []

    demand_col = _find_demand_col(df)
    if demand_col is None:
        blocking.append(BlockingIssue(code="MISSING_DEMAND_COLUMN", message="No numeric demand column found"))
    else:
        demand_numeric = pd.to_numeric(df[demand_col], errors="coerce").dropna()
        if len(demand_numeric) > 0 and (demand_numeric == 0).all():
            blocking.append(BlockingIssue(code="ALL_ZERO_DEMAND", message="All demand values are zero"))

    if _find_date_col(df) is None:
        blocking.append(BlockingIssue(code="MISSING_DATE_COLUMN", message="No parseable date column found"))

    return DataQualityReport(
        blocking_issues=blocking,
        warnings=warnings,
        row_count=len(df),
        series_count=0,
    )


def map_schema(df: pd.DataFrame, playbook: dict) -> SchemaMapping:
    date_col = _find_date_col(df)
    if date_col is None:
        fallback_date = playbook.get("time_col", "date")
        if fallback_date in df.columns:
            date_col = fallback_date
        else:
            raise ValueError(f"No parseable date column found and playbook date column '{fallback_date}' is missing")

    demand_col = _find_demand_col(df)
    if demand_col is None:
        fallback_demand = playbook.get("demand_col", "demand")
        if fallback_demand in df.columns:
            demand_col = fallback_demand
        else:
            raise ValueError(f"No demand column found and playbook demand column '{fallback_demand}' is missing")
    grain_cols = [col for col in playbook.get("common_grains", []) if col in df.columns]
    extra_cols = [col for col in df.columns if col not in [date_col, demand_col] + grain_cols]
    return SchemaMapping(
        date_col=date_col,
        demand_col=demand_col,
        grain_cols=grain_cols,
        extra_cols=extra_cols,
    )


def detect_frequency_and_grain(df: pd.DataFrame, schema: SchemaMapping) -> GrainReport:
    dates = _parse_dates(df[schema.date_col]).dropna().sort_values().unique()
    detected_frequency: str = "unknown"
    gaps_detected = False
    min_periods = max_periods = median_periods = 0

    if len(dates) > 1:
        deltas = pd.Series(dates).diff().dropna().dt.days
        median_delta = int(deltas.median())
        if 6 <= median_delta <= 8:
            detected_frequency = "weekly"
        elif 28 <= median_delta <= 32:
            detected_frequency = "monthly"
        elif median_delta == 1:
            detected_frequency = "daily"
        gaps_detected = bool((deltas > median_delta * 1.5).any())

    counts = df.groupby(schema.grain_cols)[schema.date_col].count() if schema.grain_cols else pd.Series([len(df)])
    if len(counts):
        min_periods = int(counts.min())
        max_periods = int(counts.max())
        median_periods = int(counts.median())

    return GrainReport(
        detected_frequency=detected_frequency,
        min_periods=min_periods,
        max_periods=max_periods,
        median_periods=median_periods,
        gaps_detected=gaps_detected,
    )


def build_series_keys(df: pd.DataFrame, schema: SchemaMapping, playbook: dict) -> dict[str, pd.DataFrame]:
    _ = playbook
    series_map: dict[str, pd.DataFrame] = {}
    raw_key_by_series_key: dict[str, tuple] = {}
    grain_cols = schema.grain_cols

    if grain_cols:
        for group_keys, sub_df in df.groupby(grain_cols, dropna=False):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            key = _normalise_key(group_keys)
            if key in raw_key_by_series_key and raw_key_by_series_key[key] != group_keys:
                key = _collision_safe_key(key, group_keys, raw_key_by_series_key)
            sub = sub_df[[schema.date_col, schema.demand_col]].copy()
            sub.columns = ["date", "demand"]
            raw_key_by_series_key[key] = group_keys
            series_map[key] = sub.reset_index(drop=True)
    else:
        sub = df[[schema.date_col, schema.demand_col]].copy()
        sub.columns = ["date", "demand"]
        series_map["SERIES|ALL"] = sub.reset_index(drop=True)

    return series_map


def _parse_dates(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.strip()
    if s.str.match(r"^\d{4}-W\d{1,2}$").mean() > 0.8:
        iso = s.str.replace(r"^(\d{4})-W(\d{1,2})$", r"\1-W\2-1", regex=True)
        return pd.to_datetime(iso, format="%G-W%V-%u", errors="coerce")
    try:
        return pd.to_datetime(s, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        pass
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.notna().sum() > len(s) * 0.8:
            return parsed
    return pd.Series(pd.NaT, index=values.index)


def _find_date_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        try:
            parsed = _parse_dates(df[col])
            if parsed.notna().sum() > len(df) * 0.8:
                return col
        except (TypeError, ValueError, OverflowError):
            pass
    return None


def _find_demand_col(df: pd.DataFrame) -> str | None:
    for name in ["demand", "qty", "quantity", "sales", "units"]:
        match = next((col for col in df.columns if col.lower() == name), None)
        if match:
            return match
    numeric = df.select_dtypes(include="number").columns.tolist()
    return numeric[0] if numeric else None


def _normalise_key(values: tuple) -> str:
    parts = []
    for value in values:
        if pd.isna(value):
            normalized = "NULL"
        else:
            normalized = str(value).upper().replace(" ", "_")
        normalized = re.sub(r"[^A-Z0-9_]", "", normalized)
        parts.append(normalized)
    return "|".join(parts)


def _collision_safe_key(base_key: str, raw_values: tuple, existing: dict[str, tuple]) -> str:
    raw = "|".join(str(v) for v in raw_values)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest().upper()
    for width in (8, 12, 16, 20, 24, 28, 32, 36, 40):
        key = f"{base_key}|H{digest[:width]}"
        if key not in existing or existing[key] == raw_values:
            return key
    raise ValueError(f"Unable to create collision-safe series key for grain tuple {raw_values!r}")
