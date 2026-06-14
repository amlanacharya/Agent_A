"""Canonical demand forecasting table construction and validation."""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass

import pandas as pd


CANONICAL_COLUMNS = (
    "sku_id",
    "location_id",
    "week_start",
    "demand_qty",
    "inventory_qty",
    "stockout_flag",
    "price",
    "promo_flag",
    "lead_time",
)

OPTIONAL_NUMERIC_COLUMNS = ("inventory_qty", "price", "lead_time")
MODEL_ALIAS_COLUMNS = ("series_key", "date", "demand", "promo")


@dataclass(frozen=True)
class CanonicalColumnMapping:
    sku_id: str
    location_id: str
    week_start: str
    demand_qty: str
    inventory_qty: str | None = None
    stockout_flag: str | None = None
    price: str | None = None
    promo_flag: str | None = None
    lead_time: str | None = None


class CanonicalSchemaError(ValueError):
    """Raised when a raw or canonical table violates the canonical contract."""


def build_canonical_table(df: pd.DataFrame, mapping: CanonicalColumnMapping) -> pd.DataFrame:
    _validate_mapping_sources(df, mapping)

    result = pd.DataFrame(index=df.index)
    result["sku_id"] = df[mapping.sku_id]
    result["location_id"] = df[mapping.location_id]
    result["week_start"] = _parse_week_start(df[mapping.week_start])
    result["demand_qty"] = _coerce_required_numeric(df[mapping.demand_qty], "demand_qty")

    result["inventory_qty"] = _optional_numeric(df, mapping.inventory_qty, "inventory_qty")
    result["stockout_flag"] = _optional_flag(df, mapping.stockout_flag)
    result["price"] = _optional_numeric(df, mapping.price, "price")
    result["promo_flag"] = _optional_flag(df, mapping.promo_flag)
    result["lead_time"] = _optional_numeric(df, mapping.lead_time, "lead_time")

    _validate_core_canonical_columns(result)

    result = result.sort_values(["sku_id", "location_id", "week_start"], kind="mergesort").reset_index(drop=True)
    result["series_key"] = _series_keys(result["sku_id"], result["location_id"])
    result["date"] = result["week_start"]
    result["demand"] = result["demand_qty"]
    result["promo"] = result["promo_flag"]
    validate_canonical_table(result)
    return result[list(CANONICAL_COLUMNS) + ["series_key", "date", "demand", "promo"]]


def validate_canonical_table(df: pd.DataFrame) -> None:
    missing = [column for column in (*CANONICAL_COLUMNS, *MODEL_ALIAS_COLUMNS) if column not in df.columns]
    if missing:
        raise CanonicalSchemaError(f"canonical table is missing required columns: {', '.join(missing)}")

    _validate_core_canonical_columns(df)

    if not pd.to_datetime(df["date"], errors="coerce").equals(_parse_week_start(df["week_start"])):
        raise CanonicalSchemaError("date alias must match week_start")
    if not pd.to_numeric(df["demand"], errors="coerce").equals(_coerce_required_numeric(df["demand_qty"], "demand_qty")):
        raise CanonicalSchemaError("demand alias must match demand_qty")
    if not df["promo"].equals(df["promo_flag"]):
        raise CanonicalSchemaError("promo alias must match promo_flag")


def _validate_core_canonical_columns(df: pd.DataFrame) -> None:
    missing = [column for column in CANONICAL_COLUMNS if column not in df.columns]
    if missing:
        raise CanonicalSchemaError(f"canonical table is missing required columns: {', '.join(missing)}")

    for column in ("sku_id", "location_id"):
        if df[column].isna().any():
            raise CanonicalSchemaError(f"{column} cannot contain missing values")

    _parse_week_start(df["week_start"])
    _coerce_required_numeric(df["demand_qty"], "demand_qty")
    for column in OPTIONAL_NUMERIC_COLUMNS:
        _coerce_optional_numeric(df[column], column)
    for column in ("stockout_flag", "promo_flag"):
        _validate_flag(df[column], column)


def _validate_mapping_sources(df: pd.DataFrame, mapping: CanonicalColumnMapping) -> None:
    requested = [source for source in mapping.__dict__.values() if source is not None]
    missing = [source for source in requested if source not in df.columns]
    if missing:
        raise CanonicalSchemaError(f"source data is missing mapped columns: {', '.join(missing)}")


def _parse_week_start(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    if text.str.match(r"^\d{4}-W\d{1,2}$").mean() > 0.8:
        iso_week = text.str.replace(r"^(\d{4})-W(\d{1,2})$", r"\1-W\2-1", regex=True)
        parsed = pd.to_datetime(iso_week, format="%G-W%V-%u", errors="coerce")
    else:
        try:
            parsed = pd.to_datetime(text, format="mixed", errors="coerce")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(text, errors="coerce")

    if parsed.isna().any():
        raise CanonicalSchemaError("week_start contains invalid or missing date values")
    if not parsed.dt.weekday.eq(0).all():
        raise CanonicalSchemaError("week_start must contain Monday week-start dates")
    return parsed


def _coerce_required_numeric(values: pd.Series, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise CanonicalSchemaError(f"{column} contains invalid or missing numeric values")
    return numeric.astype(float)


def _coerce_optional_numeric(values: pd.Series, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & numeric.isna()
    if invalid.any():
        raise CanonicalSchemaError(f"{column} contains invalid numeric values")
    return numeric.astype(float)


def _optional_numeric(df: pd.DataFrame, source: str | None, column: str) -> pd.Series:
    if source is None:
        return pd.Series(pd.NA, index=df.index, dtype="object")
    return _coerce_optional_numeric(df[source], column)


def _optional_flag(df: pd.DataFrame, source: str | None) -> pd.Series:
    if source is None:
        return pd.Series([False] * len(df), index=df.index, dtype="object")

    values = df[source]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(object)

    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").fillna("").str.strip().str.lower()
    true_text = text.isin({"true", "yes", "y", "1"})
    false_text = text.isin({"false", "no", "n", "0", ""})
    numeric_valid = numeric.isin([0, 1])
    invalid = values.notna() & ~numeric_valid & ~true_text & ~false_text
    if invalid.any():
        raise CanonicalSchemaError(f"{source} contains invalid flag values")
    return (numeric.fillna(0).ne(0) | true_text).astype(object)


def _validate_flag(values: pd.Series, column: str) -> None:
    if pd.api.types.is_bool_dtype(values):
        return

    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").fillna("").str.strip().str.lower()
    text_valid = text.isin({"true", "false", "yes", "no", "y", "n", "1", "0", ""})
    numeric_valid = numeric.isin([0, 1])
    invalid = values.notna() & ~text_valid & ~numeric_valid
    if invalid.any():
        raise CanonicalSchemaError(f"{column} contains invalid flag values")


def _series_keys(sku_ids: pd.Series, location_ids: pd.Series) -> list[str]:
    raw_tuples = [(_raw_key_part(sku), _raw_key_part(location)) for sku, location in zip(sku_ids, location_ids)]
    base_by_raw = {raw: "|".join(_normalise_key_part(value) for value in raw) for raw in raw_tuples}
    raw_values_by_base: dict[str, set[tuple[str, str]]] = {}
    for raw, base in base_by_raw.items():
        raw_values_by_base.setdefault(base, set()).add(raw)

    key_by_raw: dict[tuple[str, str], str] = {}
    for base, raw_values in raw_values_by_base.items():
        if len(raw_values) == 1:
            key_by_raw[next(iter(raw_values))] = base
            continue
        for raw in sorted(raw_values):
            digest = hashlib.sha1("|".join(raw).encode("utf-8")).hexdigest().upper()[:12]
            key_by_raw[raw] = f"{base}|H{digest}"

    return [key_by_raw[raw] for raw in raw_tuples]


def _raw_key_part(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def _normalise_key_part(value: object) -> str:
    normalized = str(value).upper().replace(" ", "_")
    return re.sub(r"[^A-Z0-9_]", "", normalized)
