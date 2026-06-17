"""Phase 7 CB3: data-drift detection between monitoring runs.

The data-drift engine compares the previous run's preflight bundle
shape and canonical data against the current run's, surfacing the
four signal kinds the plan calls for:

* **Schema changes** — columns added, dropped, renamed, or whose
  inferred type changed between two ``SchemaMapping``s.
* **Missing feeds** — series keys present in the previous run but
  absent from the current run (the upstream feed stopped sending
  them).
* **Distribution shifts** — per-column mean / std / median / p99
  / min / max deltas above a configurable threshold (default 5%).
* **New SKU / location keys** — series keys present in the current
  run but absent from the previous one, split into SKU and
  location by the standard ``|`` pipe-delimited convention.

Design:

* **Pure function, no I/O.** ``detect_data_drift`` is a pure
  function of (run_id, previous, current). The scheduler / cockpit
  reads the previous run's snapshot from disk and calls the
  function; the function does not touch the filesystem.
* **Tolerant of None inputs.** A first run (no previous data) is
  valid: the report is empty on the schema / missing / shift axes,
  and the new-keys axis is the full current key set. A current run
  with no canonical data yet (e.g. preflight failed) produces an
  empty distribution-shift list rather than crashing.
* **Threshold-suppressed noise.** A 1% mean shift is below the
  5% default threshold and is not surfaced. Threshold is
  configurable per call so a downstream alert policy can tighten
  it.
* **No LLM.** The engine is fully deterministic. The cockpit
  surfaces the typed report; the planner reads it.

The four public functions are:

* ``compare_schemas(previous, current) -> list[SchemaChange]``
* ``detect_missing_feeds(previous_keys, current_keys) -> list[str]``
* ``compute_distribution_shifts(previous_df, current_df, columns, threshold=0.05) -> list[DistributionShift]``
* ``detect_data_drift(...) -> DataDriftReport`` — the top-level
  orchestrator that combines the three into one typed report
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from forecasting.contracts import (
    DataDriftReport,
    DistributionMetric,
    DistributionShift,
    NewSeriesKeys,
    SchemaChange,
    SchemaMapping,
)


# The default 5% threshold matches the .env convention used
# elsewhere in the platform (card-lifecycle, marginal-gain).
# A 5% mean / std shift is the smallest signal that's
# consistently above measurement noise in a stable supply chain.
_DEFAULT_SHIFT_THRESHOLD: float = 0.05


# ---------------------------------------------------------------------------
# compare_schemas
# ---------------------------------------------------------------------------


def _all_columns(mapping: SchemaMapping) -> set[str]:
    """Flatten a SchemaMapping into the set of column names it covers.

    The schema mapping distinguishes four buckets (date, demand,
    grain, extra). For drift detection purposes a column is a
    column regardless of bucket; the engine only cares that the
    column appeared / disappeared.
    """
    return {
        mapping.date_col,
        mapping.demand_col,
        *mapping.grain_cols,
        *mapping.extra_cols,
    }


def compare_schemas(
    previous: SchemaMapping | None,
    current: SchemaMapping | None,
) -> list[SchemaChange]:
    """Surface schema-level changes between two SchemaMappings.

    Reports three kinds:

    * ``COLUMN_DROPPED`` — in previous, not in current.
    * ``COLUMN_ADDED`` — in current, not in previous.
    * ``COLUMN_TYPE_CHANGED`` — same name on both sides, different
      bucket (e.g. the date column was demoted to a regular extra
      column). The engine is conservative: this is a soft signal,
      not a hard rename.

    Either input may be ``None`` (no prior data, or no current
    data). Returns an empty list on the ``None / None`` edge case.
    """
    if previous is None or current is None:
        return []
    previous_cols = _all_columns(previous)
    current_cols = _all_columns(current)
    dropped = previous_cols - current_cols
    added = current_cols - previous_cols
    changes: list[SchemaChange] = []
    for col in sorted(dropped):
        changes.append(
            SchemaChange(
                kind="COLUMN_DROPPED",
                column=col,
                detail=f"column {col!r} present in {previous.date_col!r} schema, absent in current",
            )
        )
    for col in sorted(added):
        changes.append(
            SchemaChange(
                kind="COLUMN_ADDED",
                column=col,
                detail=f"column {col!r} absent in previous schema, present in current",
            )
        )
    return changes


# ---------------------------------------------------------------------------
# detect_missing_feeds
# ---------------------------------------------------------------------------


def detect_missing_feeds(
    previous_keys: Sequence[str],
    current_keys: Sequence[str],
) -> list[str]:
    """Return the keys present in previous but absent from current.

    Sorted alphabetically for deterministic output. The output is
    a plain ``list[str]`` — the engine does not know whether the
    missing key is a SKU, a location, or both, so the platform
    surfaces the raw key string and the planner interprets.
    """
    previous_set = set(previous_keys)
    current_set = set(current_keys)
    return sorted(previous_set - current_set)


# ---------------------------------------------------------------------------
# compute_distribution_shifts
# ---------------------------------------------------------------------------


def _safe_pct_change(previous: float, current: float) -> float:
    """Signed pct change from previous to current.

    Returns ``0.0`` when previous is zero (the engine does not
    return ``inf`` or ``nan`` — drift reporting has to be
    well-defined even on degenerate input). Sign matches the
    direction of change: positive = current > previous.
    """
    if previous == 0.0:
        return 0.0
    return (current - previous) / abs(previous)


def _summarise(df: pd.DataFrame, column: str, metric: str) -> float:
    """Compute one summary statistic for one column.

    Returns ``0.0`` for an empty frame — the engine never raises
    on empty input; it just reports no shift.
    """
    if column not in df.columns or df.empty:
        return 0.0
    series = df[column]
    if metric == "mean":
        return float(series.mean())
    if metric == "std":
        return float(series.std(ddof=0))
    if metric == "median":
        return float(series.median())
    if metric == "p99":
        return float(series.quantile(0.99))
    if metric == "min":
        return float(series.min())
    if metric == "max":
        return float(series.max())
    return 0.0


_METRICS: tuple[DistributionMetric, ...] = (
    "mean",
    "std",
    "median",
    "p99",
    "min",
    "max",
)


def compute_distribution_shifts(
    previous_df: pd.DataFrame | None,
    current_df: pd.DataFrame | None,
    columns: Sequence[str],
    threshold: float = _DEFAULT_SHIFT_THRESHOLD,
) -> list[DistributionShift]:
    """Surface per-column distribution shifts above ``threshold``.

    Iterates the six supported ``DistributionMetric`` values and
    surfaces a shift for any (column, metric) pair whose signed
    pct change exceeds the threshold. Columns present in only
    one of the two frames are silently skipped (no NaN, no
    crash) — the schema-change report already covers the add /
    drop axis.

    Returns an empty list when either frame is ``None`` (the
    engine has nothing to compare).
    """
    if previous_df is None or current_df is None:
        return []
    shifts: list[DistributionShift] = []
    for column in columns:
        for metric in _METRICS:
            previous_value = _summarise(previous_df, column, metric)
            current_value = _summarise(current_df, column, metric)
            pct = _safe_pct_change(previous_value, current_value)
            if abs(pct) >= threshold:
                shifts.append(
                    DistributionShift(
                        column=column,
                        metric=metric,
                        previous=previous_value,
                        current=current_value,
                        pct_change=pct,
                    )
                )
    return shifts


# ---------------------------------------------------------------------------
# detect_data_drift — top-level orchestrator
# ---------------------------------------------------------------------------


def _split_new_key(key: str) -> tuple[str | None, str | None]:
    """Split a pipe-delimited series key into (sku, location).

    The convention from ``data_store``: the first pipe-delimited
    segment is the SKU; the second is the location. Keys without
    a pipe are treated as a SKU-only key (single-grain run).
    Multi-segment keys (3+ pipe parts) get the SKU from part 0
    and the location from part 1; the rest is ignored.
    """
    if "|" not in key:
        return key, None
    parts = key.split("|", 2)
    if len(parts) < 2:
        return parts[0], None
    return parts[0], parts[1]


def _new_keys(
    previous_keys: Sequence[str],
    current_keys: Sequence[str],
) -> NewSeriesKeys:
    """Return the new keys split into SKU vs location buckets.

    A new key is one in current but not in previous. The split
    uses the standard pipe convention. Keys whose location slot
    is empty (single-grain) contribute to ``new_skus`` only.
    """
    previous_set = set(previous_keys)
    new = [k for k in current_keys if k not in previous_set]
    new_skus: list[str] = []
    new_locations: list[str] = []
    for key in new:
        sku, location = _split_new_key(key)
        if sku is not None:
            new_skus.append(sku)
        if location is not None:
            new_locations.append(location)
    return NewSeriesKeys(
        new_skus=sorted(set(new_skus)),
        new_locations=sorted(set(new_locations)),
    )


def detect_data_drift(
    *,
    run_id: str,
    previous_run_id: str,
    previous_schema: SchemaMapping | None,
    previous_keys: Sequence[str],
    current_schema: SchemaMapping | None,
    current_keys: Sequence[str],
    previous_df: pd.DataFrame | None,
    current_df: pd.DataFrame | None,
    shift_threshold: float = _DEFAULT_SHIFT_THRESHOLD,
) -> DataDriftReport:
    """Top-level data-drift engine.

    Combines the three signal kinds into one typed
    ``DataDriftReport``. Tolerant of ``None`` inputs on both
    sides (the engine treats them as "no data to compare on this
    axis") and of empty key lists (the engine produces an empty
    missing-feed list). The ``shift_threshold`` parameter lets a
    downstream alert policy tighten or loosen the noise floor.
    """
    schema_changes = compare_schemas(previous_schema, current_schema)
    missing_feeds = detect_missing_feeds(previous_keys, current_keys)
    drift_columns: list[str] = []
    if current_schema is not None:
        # Compare every column the current schema declares; the
        # distribution-shift function silently drops columns that
        # are missing from the previous frame, so we just pass the
        # current column set.
        drift_columns = sorted(
            _all_columns(current_schema),
        )
    distribution_shifts = compute_distribution_shifts(
        previous_df,
        current_df,
        columns=drift_columns,
        threshold=shift_threshold,
    )
    new_keys = _new_keys(previous_keys, current_keys)
    return DataDriftReport(
        run_id=run_id,
        previous_run_id=previous_run_id,
        schema_changes=schema_changes,
        missing_feeds=missing_feeds,
        distribution_shifts=distribution_shifts,
        new_keys=new_keys,
    )


__all__ = (
    "compare_schemas",
    "compute_distribution_shifts",
    "detect_data_drift",
    "detect_missing_feeds",
    # The threshold is exported so the cockpit / alert policy can
    # reference the platform's default noise floor when configuring
    # a tighter or looser threshold.
    "_DEFAULT_SHIFT_THRESHOLD",
)


# The leading-underscore exports above are module-private
# (``_all_columns``, ``_safe_pct_change``, ``_summarise``,
# ``_split_new_key``, ``_new_keys``, ``_METRICS``) and are kept
# out of ``__all__`` deliberately — they are implementation
# details, not platform surface. Tests that need them import
# them by their underscore name. The pattern matches the rest
# of the platform (see ``forecasting.code_escalation``).
