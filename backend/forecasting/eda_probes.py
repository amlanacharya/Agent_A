"""Standard EDA sub-checks (Phase 2 — the rest of the EDA toolbox).

Each public function in this module is a *probe*: a read-only analysis of
the canonical demand table that returns a small Pydantic payload. The
``eda_toolbox.build_eda_report`` orchestrator composes them into the
``EDAReport``.

Probes follow the contract the plan documents for the EDA toolbox:

* type detection — per-column inferred dtype, contract mismatch surface
* missingness — per-column NaN/None counts, rows with any missing optional
* duplicates — ``(series_key, date)`` collisions in the canonical table
* date gaps — per-series gap analysis (gaps, out-of-order rows)
* join validation — coverage of optional dimensions (inventory, price,
  lead time) and "orphan" series
* leakage checks — per-series forward correlation + impossible equality
  probes (demand == inventory)

Probes are pure (no I/O, no side effects, no escalation). The toolbox is
the only layer that talks to ``EscalationTracker`` — it observes probe
results and decides whether to escalate.

Known follow-up duplications (intentional, deferred from the Phase 2
refactor review — see the 7-change list in the Phase 2 completion log):

* ``_autocorr`` here mirrors ``forecasting.tools.preflight_stats._autocorr``.
  Extracting a shared helper would require a new public util module and
  is out of scope for the Phase 2 EDA sub-checks.
* The boolean-string set ``{"true", "false", "yes", "no", "y", "n", "1",
  "0"}`` in ``_label_for_series`` is also accepted by
  ``forecasting.canonical_data._optional_flag`` / ``_validate_flag``. A
  canonical export from ``canonical_data`` would let both layers share
  it.
* ``_REQUIRED_NON_NULL_COLUMNS`` is built from names already exported by
  ``forecasting.canonical_data`` (``CANONICAL_COLUMNS`` +
  ``MODEL_ALIAS_COLUMNS``); the tuple is inlined here to avoid a
  circular import through ``contracts`` until ``canonical_data`` grows a
  dedicated export.
* ``run_all_probes`` is a test convenience aggregator. ``build_eda_report``
  is the production orchestrator; ``run_all_probes`` has no callers
  today and is a candidate for removal if/when the test suite drops its
  few uses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.canonical_data import CANONICAL_COLUMNS, OPTIONAL_NUMERIC_COLUMNS
from forecasting.contracts import (
    ColumnTypeInference,
    DateGapsReport,
    DuplicateReport,
    InferredColumnType,
    JoinValidationIssue,
    JoinValidationReport,
    LeakageReport,
    MissingnessReport,
    MissingnessStats,
    SeriesDateGapStats,
    SeriesLeakageStats,
    TypeDetectionReport,
)


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------

# Columns the canonical contract expects to be a specific dtype. We do not
# raise on mismatch — we surface it on the report so the user (or the
# governance layer) can decide what to do. The contract is the source of
# truth, but the canonical layer already coerces on the way in, so a
# mismatch on the *output* canonical table means the upstream feed sent
# something the canonical layer had to coerce.
_CONTRACT_EXPECTED_TYPES: dict[str, InferredColumnType] = {
    "sku_id": "string",
    "location_id": "string",
    "week_start": "datetime",
    "demand_qty": "float",
    "inventory_qty": "float",
    "stockout_flag": "boolean",
    "price": "float",
    "promo_flag": "boolean",
    "lead_time": "float",
}

# Soft cap on the per-column sample values we surface. Five is enough to
# show a human reviewer what's going on without bloating the report.
_SAMPLE_VALUE_CAP: int = 5


def detect_column_types(df: pd.DataFrame) -> TypeDetectionReport:
    """Infer a dtype label for every column in ``df``.

    The output is deterministic for a given input — no random sampling, no
    pre-existing typing that would survive an unrelated reordering. The
    inferred label is a coarse bucket (``integer`` / ``float`` /
    ``boolean`` / ``string`` / ``datetime`` / ``empty`` / ``mixed``) chosen
    to match what the rest of the EDA pipeline cares about, not what
    pandas would call the dtype internally (a pandas ``int64`` is reported
    as ``integer`` even when the canonical contract says it should be
    ``float`` — see ``contract_mismatches``).
    """
    inferences: list[ColumnTypeInference] = []
    mismatches: list[str] = []
    for column in df.columns:
        inf = _infer_column(column, df[column])
        inferences.append(inf)
        expected = _CONTRACT_EXPECTED_TYPES.get(column)
        if expected is not None and inf.inferred_type != expected:
            mismatches.append(column)
    return TypeDetectionReport(columns=inferences, contract_mismatches=mismatches)


def _infer_column(name: str, series: pd.Series) -> ColumnTypeInference:
    non_null = series.dropna()
    if len(non_null) == 0:
        return ColumnTypeInference(
            column=name, inferred_type="empty", nullable=True, unique_count=0, sample_values=[]
        )

    label = _label_for_series(non_null)
    samples = [str(v) for v in non_null.head(_SAMPLE_VALUE_CAP).tolist()]
    return ColumnTypeInference(
        column=name,
        inferred_type=label,
        nullable=bool(series.isna().any()),
        unique_count=int(non_null.nunique()),
        sample_values=samples,
    )


def _label_for_series(non_null: pd.Series) -> InferredColumnType:
    """Coarse-bucket the dtype of a non-null column.

    Priority order matters:
    * booleans win first because they are technically ``bool`` in numpy
      but pandas may store them as object — we want the more useful label;
    * datetimes next because pandas' ``is_datetime64_any_dtype`` is cheap
      and the rest of the labels hinge on the values being scalar;
    * the float/integer split is driven by the *pandas* dtype, not by
      whether every observed value happens to be a whole number. A
      ``float64`` column is labelled ``float`` even when all observed
      values are integral — the column's contract (e.g. ``demand_qty``
      may legitimately hold fractional units in the future) is what
      matters, not the data we happen to see today;
    * everything else is string except the mixed case.
    """
    if pd.api.types.is_bool_dtype(non_null):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(non_null):
        return "datetime"

    coerced_numeric = pd.to_numeric(non_null, errors="coerce")
    if coerced_numeric.notna().all():
        if pd.api.types.is_float_dtype(non_null):
            return "float"
        if _is_integer_like(coerced_numeric):
            return "integer"
        return "float"

    # Last-ditch: try the "boolean-looking string" set (the canonical
    # layer accepts "true"/"false"/"yes"/"no" etc. as flags). If the column
    # is nothing but those, label it boolean — the user almost certainly
    # means it as one.
    text = non_null.astype("string").str.strip().str.lower()
    if text.isin({"true", "false", "yes", "no", "y", "n", "1", "0"}).all():
        return "boolean"

    return "string"


def _is_integer_like(values: pd.Series) -> bool:
    """Return True iff every value in ``values`` is a whole number.

    Used by ``_label_for_series`` to split the "numeric" bucket. A float
    that is exactly an integer (e.g. ``10.0``) is still ``integer``; the
    ``mixed`` bucket is reserved for columns that contain numbers AND
    strings.
    """
    finite = values[np.isfinite(values.to_numpy())]
    if len(finite) == 0:
        return False
    return bool((finite % 1 == 0).all())


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------

# Columns the canonical contract *requires* to be non-null. The canonical
# layer rejects inputs that violate this, so a non-zero missing fraction
# here means someone constructed a canonical table by hand (e.g. tests).
# We still report it — the EDA report is observational, not gating.
_REQUIRED_NON_NULL_COLUMNS: tuple[str, ...] = (
    "sku_id",
    "location_id",
    "week_start",
    "demand_qty",
    "series_key",
    "date",
    "demand",
)


def measure_missingness(df: pd.DataFrame) -> MissingnessReport:
    per_column: list[MissingnessStats] = []
    for column in df.columns:
        values = df[column]
        missing_count = int(values.isna().sum())
        per_column.append(
            MissingnessStats(
                column=column,
                missing_count=missing_count,
                missing_fraction=missing_count / len(df) if len(df) else 0.0,
            )
        )

    # Rows-with-missing only counts columns that are *not* in the required
    # non-null set, because the canonical layer already guarantees the
    # required columns are populated. This keeps the metric focused on
    # optional dimensions (inventory, price, lead_time, promo).
    optional = [c for c in df.columns if c not in _REQUIRED_NON_NULL_COLUMNS]
    if optional:
        rows_with_missing = int(df[optional].isna().any(axis=1).sum())
    else:
        rows_with_missing = 0

    return MissingnessReport(
        per_column=per_column,
        rows_with_missing=rows_with_missing,
        rows_total=len(df),
    )


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

# The (series_key, date) tuple is the canonical primary key. Duplicates on
# this key mean the upstream aggregation did not de-duplicate, or the
# canonical layer was bypassed.
_CANONICAL_PRIMARY_KEY: tuple[str, str] = ("series_key", "date")


def detect_duplicate_keys(df: pd.DataFrame) -> DuplicateReport:
    if df.empty or any(col not in df.columns for col in _CANONICAL_PRIMARY_KEY):
        return DuplicateReport(duplicate_rows=0, duplicate_keys=[], duplicate_fraction=0.0)

    # Coerce the date column to ``datetime64[ns]`` so the index values we
    # get back from ``groupby`` are Timestamps (not Python ``date``
    # objects) and render in the canonical ``YYYY-MM-DD`` form.
    date_series = pd.to_datetime(df["date"], errors="coerce")
    keys_frame = pd.DataFrame(
        {
            "series_key": df["series_key"].astype(str).to_numpy(),
            "date": date_series.to_numpy(),
        }
    )
    grouped = keys_frame.groupby(["series_key", "date"], dropna=False).size()
    duplicates = grouped[grouped > 1]

    # Render duplicate keys as ``"<series_key>@<YYYY-MM-DD>"`` — the same
    # shape Meridian and the cockpit surface to the user.
    def _format(index_value: tuple[str, object]) -> str:
        series, date = index_value
        if pd.isna(date):
            return f"{series}@<NaT>"
        ts = pd.Timestamp(date)
        return f"{series}@{ts.strftime('%Y-%m-%d')}"

    duplicate_keys = sorted(_format(idx) for idx in duplicates.index)
    # Sum of (count - 1) over duplicate groups: that's how many extra rows
    # there are beyond the first occurrence of each key.
    duplicate_rows = int((duplicates - 1).sum())
    return DuplicateReport(
        duplicate_rows=duplicate_rows,
        duplicate_keys=duplicate_keys,
        duplicate_fraction=duplicate_rows / len(df) if len(df) else 0.0,
    )


# ---------------------------------------------------------------------------
# Date gaps
# ---------------------------------------------------------------------------

# When the gap exceeds 1.5x the expected period we count it as a *gap*
# (not a single missing row). At 1.5x we still tolerate a single dropped
# week before flagging.
_GAP_MULTIPLIER: float = 1.5


def detect_date_gaps_per_series(
    series_map: dict[str, pd.DataFrame],
    *,
    expected_period_days: int | None = None,
) -> DateGapsReport:
    per_series: dict[str, SeriesDateGapStats] = {}
    series_with_gaps: list[str] = []

    for key, df in series_map.items():
        stats = _gap_stats_for_series(key, df, expected_period_days=expected_period_days)
        per_series[key] = stats
        if stats.actual_gap_count > 0:
            series_with_gaps.append(key)

    return DateGapsReport(per_series=per_series, series_with_gaps=series_with_gaps)


def _gap_stats_for_series(
    key: str,
    df: pd.DataFrame,
    *,
    expected_period_days: int | None,
) -> SeriesDateGapStats:
    if df.empty or "date" not in df.columns:
        return SeriesDateGapStats(
            series_key=key,
            expected_period_days=expected_period_days,
            actual_gap_count=0,
            max_gap_days=0,
            median_gap_days=0.0,
            out_of_order_rows=0,
        )

    dates = pd.to_datetime(df["date"], errors="coerce").dropna().sort_values().reset_index(drop=True)
    raw_diffs = dates.diff()
    out_of_order = int((raw_diffs.dt.days < 0).sum())

    inferred_period = expected_period_days
    if inferred_period is None and len(dates) > 1:
        inferred_period = _infer_period_days(dates)

    if len(dates) < 2 or inferred_period is None:
        return SeriesDateGapStats(
            series_key=key,
            expected_period_days=inferred_period,
            actual_gap_count=0,
            max_gap_days=0,
            median_gap_days=0.0,
            out_of_order_rows=out_of_order,
        )

    deltas = raw_diffs.dt.days.dropna().astype(int)
    gap_threshold = int(round(inferred_period * _GAP_MULTIPLIER))
    actual_gap_count = int((deltas > gap_threshold).sum())
    return SeriesDateGapStats(
        series_key=key,
        expected_period_days=int(inferred_period),
        actual_gap_count=actual_gap_count,
        max_gap_days=int(deltas.max()),
        median_gap_days=float(deltas.median()),
        out_of_order_rows=out_of_order,
    )


def _infer_period_days(dates: pd.Series) -> int | None:
    """Pick a representative period from sorted unique dates.

    Returns the *mode* of the per-row deltas rounded to the nearest whole
    day. We pick the mode (not the median) so that one stray gap does
    not distort the period estimate — a 7-day weekly series with a single
    14-day gap still has 7 as the modal delta.
    """
    if len(dates) < 2:
        return None
    deltas = dates.diff().dropna().dt.days
    counts = deltas.value_counts()
    if counts.empty:
        return None
    return int(counts.idxmax())


# ---------------------------------------------------------------------------
# Join validation
# ---------------------------------------------------------------------------


def validate_joins(df: pd.DataFrame) -> JoinValidationReport:
    issues: list[JoinValidationIssue] = []
    coverage: dict[str, float] = {}
    for column in OPTIONAL_NUMERIC_COLUMNS:
        if column not in df.columns:
            coverage[column] = 0.0
            continue
        coverage[column] = float(df[column].notna().mean()) if len(df) else 0.0

    # Per-series coverage: a series with a demand row count of N and an
    # inventory value count of 0 is a *missing-inventory* issue. We only
    # surface the issue (the metric is in ``inventory_coverage``).
    if "series_key" in df.columns:
        _dimension_kinds = [
            (c, k)
            for c, k in (
                ("inventory_qty", "MISSING_INVENTORY_FOR_DEMAND"),
                ("price", "MISSING_PRICE_FOR_DEMAND"),
                ("lead_time", "MISSING_LEAD_TIME_FOR_DEMAND"),
            )
            if c in df.columns
        ]
        if _dimension_kinds:
            for series_key, group in df.groupby("series_key", sort=False):
                for column, issue_kind in _dimension_kinds:
                    if group[column].isna().all():
                        issues.append(
                            JoinValidationIssue(
                                kind=issue_kind,  # type: ignore[arg-type]
                                series_key=str(series_key),
                                detail=f"series {series_key!r} has no {column} values across {len(group)} demand rows",
                            )
                        )

    return JoinValidationReport(
        issues=issues,
        inventory_coverage=coverage.get("inventory_qty", 0.0),
        price_coverage=coverage.get("price", 0.0),
        lead_time_coverage=coverage.get("lead_time", 0.0),
    )


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------

# Forward windows (in periods) we suspect of leaking. A weekly series with
# near-1 correlation between demand[t] and demand[t+1] is fine (that's just
# week-to-week carry-over). demand[t] == demand[t+4] across all rows is
# almost certainly a copy-paste bug in the upstream pipeline.
_FORWARD_LAGS: tuple[int, ...] = (2, 3, 4, 5)
_MAX_FORWARD_LAG: int = max(_FORWARD_LAGS)
# Threshold above which a forward correlation is suspicious. 0.95 is
# generous — anything below that is plausible serial correlation.
_LEAKAGE_CORR_THRESHOLD: float = 0.95


def detect_leakage_per_series(
    series_map: dict[str, pd.DataFrame],
) -> LeakageReport:
    per_series: dict[str, SeriesLeakageStats] = {}
    suspect: list[str] = []

    for key, df in series_map.items():
        stats = _leakage_stats_for_series(key, df)
        per_series[key] = stats
        if stats.demand_equals_inventory_rows > 0 or stats.forward_correlation_max > _LEAKAGE_CORR_THRESHOLD:
            suspect.append(key)

    return LeakageReport(per_series=per_series, suspect_series=suspect)


def _leakage_stats_for_series(key: str, df: pd.DataFrame) -> SeriesLeakageStats:
    if df.empty or "demand" not in df.columns:
        return SeriesLeakageStats(
            series_key=key, forward_correlation_max=0.0, demand_equals_inventory_rows=0
        )

    demand = pd.to_numeric(df["demand"], errors="coerce").to_numpy(dtype=float)
    max_corr = 0.0
    finite = np.isfinite(demand)
    clean = demand[finite]
    if len(clean) > _MAX_FORWARD_LAG:
        for lag in _FORWARD_LAGS:
            if len(clean) > lag:
                c = _autocorr(clean, lag)
                if abs(c) > abs(max_corr):
                    max_corr = c

    deq = 0
    if {"demand", "inventory_qty"}.issubset(df.columns):
        demand_num = pd.to_numeric(df["demand"], errors="coerce")
        inv_num = pd.to_numeric(df["inventory_qty"], errors="coerce")
        deq = int(((demand_num == inv_num) & demand_num.notna() & inv_num.notna()).sum())

    return SeriesLeakageStats(
        series_key=key,
        forward_correlation_max=round(max_corr, 4),
        demand_equals_inventory_rows=deq,
    )


def _autocorr(x: np.ndarray, lag: int) -> float:
    """Pearson correlation between ``x[:-lag]`` and ``x[lag:]``.

    Mirrors the helper in ``preflight_stats`` — duplicated here so this
    module stays self-contained and testable. Returns 0.0 on degenerate
    input (constant series, lag >= length) rather than NaN so the report
    stays serialisable.
    """
    if lag <= 0 or lag >= len(x):
        return 0.0
    a = x[:-lag]
    b = x[lag:]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Convenience aggregator (used by tests; build_eda_report is the public API)
# ---------------------------------------------------------------------------


def run_all_probes(
    canonical_table: pd.DataFrame,
    series_map: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """Run every probe in one call. Convenience for unit tests."""
    return {
        "type_detection": detect_column_types(canonical_table),
        "missingness": measure_missingness(canonical_table),
        "duplicates": detect_duplicate_keys(canonical_table),
        "date_gaps": detect_date_gaps_per_series(series_map),
        "join_validation": validate_joins(canonical_table),
        "leakage": detect_leakage_per_series(series_map),
    }


__all__ = (
    "detect_column_types",
    "measure_missingness",
    "detect_duplicate_keys",
    "detect_date_gaps_per_series",
    "validate_joins",
    "detect_leakage_per_series",
    "run_all_probes",
)
