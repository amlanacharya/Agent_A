"""Canonical feature table generation for forecasting models."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from forecasting.contracts import FeatureFlags


REQUIRED_CANONICAL_COLUMNS = ("series_key", "date", "demand")
TIME_DEPENDENT_FEATURE_COLUMNS = ("lag_1", "lag_2", "rolling_mean_4")
# Columns produced by the stockout / availability family.
STOCKOUT_FEATURE_COLUMNS = (
    "stockout_rolling_count_4",
    "days_since_stockout",
    "inventory_cover_ratio",
)
# Columns produced by the hierarchy family. Computed on the parent grain
# (sku_id aggregated across location_id) and broadcast to each child
# (sku_id, location_id) row.
HIERARCHY_FEATURE_COLUMNS = ("parent_lag_1", "parent_rolling_mean_4")
# Columns produced by the lifecycle / cold-start family. These are
# computed *as of* the fold cutoff (or the row's own date when there is
# no fold), so the "history available to a model" is the history the
# model would have at prediction time.
LIFECYCLE_FEATURE_COLUMNS = (
    "history_length",
    "days_since_first_obs",
    "cold_start_flag",
)
# Columns produced by the intermittency family. All time-dependent and
# fold-aware.
INTERMITTENCY_FEATURE_COLUMNS = (
    "rolling_adi_8",
    "rolling_cv2_8",
    "trailing_zero_run",
)


class FeatureFactoryError(ValueError):
    """Raised when canonical feature generation cannot proceed."""


def validate_canonical_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_CANONICAL_COLUMNS if column not in df.columns]
    if missing:
        raise FeatureFactoryError(f"canonical data is missing required columns: {', '.join(missing)}")


def build_feature_table(
    df: pd.DataFrame,
    flags: FeatureFlags,
    fold_cutoffs: Sequence[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Build the canonical feature table.

    When ``fold_cutoffs`` is provided, time-dependent features (lags and
    rolling windows) are computed independently for each cutoff using only
    the data with ``date <= cutoff`` and then assigned to the rows in the band
    ``(prev_cutoff, current_cutoff]`` (or to all rows, when only one cutoff
    is supplied). Rows with no cutoff ``<=`` their date receive NaN for the
    time-dependent features — preventing walk-forward validation from
    peeking at future rows.

    Time-independent features (promo indicator, Fourier terms) are always
    computed from the full input frame because they do not depend on the
    demand history.
    """
    validate_canonical_columns(df)
    _validate_fold_cutoffs(fold_cutoffs)

    result = df.copy(deep=True)
    try:
        result["date"] = pd.to_datetime(result["date"], errors="raise")
    except Exception as exc:
        raise FeatureFactoryError("canonical date column contains invalid date values") from exc

    result = result.sort_values(["series_key", "date"], kind="mergesort").reset_index(drop=True)

    for _enabled, _fn, _cols in [
        (flags.use_lag_features,           _compute_time_dependent_features,  TIME_DEPENDENT_FEATURE_COLUMNS),
        (flags.use_stockout_features,      _compute_stockout_features,        STOCKOUT_FEATURE_COLUMNS),
        (flags.use_hierarchy_features,     _compute_hierarchy_features,       HIERARCHY_FEATURE_COLUMNS),
        (flags.use_lifecycle_features,     _compute_lifecycle_features,       LIFECYCLE_FEATURE_COLUMNS),
        (flags.use_intermittency_features, _compute_intermittency_features,   INTERMITTENCY_FEATURE_COLUMNS),
    ]:
        if _enabled:
            result[list(_cols)] = _fn(result, fold_cutoffs)[list(_cols)]

    if flags.use_promo_indicator:
        if "promo" not in result.columns:
            raise FeatureFactoryError("promo column is required when use_promo_indicator is enabled")
        result["promo_indicator"] = _promo_to_indicator(result["promo"])

    if flags.use_fourier:
        if flags.fourier_terms < 1:
            raise FeatureFactoryError("fourier_terms must be at least 1 when use_fourier is enabled")
        if flags.frequency_period is not None and flags.frequency_period < 1:
            raise FeatureFactoryError("frequency_period must be at least 1 when set")
        fourier = _add_fourier_features(result, flags.fourier_terms, flags.frequency_period)
        result[list(fourier.columns)] = fourier

    return result


def _validate_fold_cutoffs(fold_cutoffs: Sequence[pd.Timestamp] | None) -> None:
    if fold_cutoffs is None or len(fold_cutoffs) == 0:
        return
    parsed = [pd.Timestamp(c) for c in fold_cutoffs]
    for previous, current in zip(parsed, parsed[1:]):
        if current <= previous:
            raise FeatureFactoryError(
                "fold_cutoffs must be strictly ascending and contain no duplicates"
            )


def _compute_time_dependent_features(
    df: pd.DataFrame, fold_cutoffs: Sequence[pd.Timestamp] | None
) -> pd.DataFrame:
    """Compute lag and rolling features, optionally fold-aware.

    Returns a DataFrame indexed the same way as ``df`` (after sort) with the
    three time-dependent feature columns. Rows that have no fold cutoff
    ``<=`` their date are NaN'd out (no leakage).
    """
    if not fold_cutoffs:
        grouped_demand = df.groupby("series_key", sort=False)["demand"]
        return pd.DataFrame(
            {
                "lag_1": grouped_demand.shift(1),
                "lag_2": grouped_demand.shift(2),
                "rolling_mean_4": grouped_demand.transform(
                    lambda series: series.shift(1).rolling(window=4, min_periods=1).mean()
                ),
            },
            index=df.index,
        )

    out = pd.DataFrame(index=df.index, columns=list(TIME_DEPENDENT_FEATURE_COLUMNS), dtype=float)
    for band_mask, prefix in _iter_fold_bands(df, fold_cutoffs):
        if prefix is None:
            continue
        prefix_grouped = prefix.groupby("series_key", sort=False)["demand"]
        prefix_features = pd.DataFrame(
            {
                "lag_1": prefix_grouped.shift(1),
                "lag_2": prefix_grouped.shift(2),
                "rolling_mean_4": prefix_grouped.transform(
                    lambda series: series.shift(1).rolling(window=4, min_periods=1).mean()
                ),
            },
            index=prefix.index,
        )
        # Reindex the prefix features onto the full df index so the band
        # rows line up; rows outside the prefix keep NaN automatically.
        aligned = prefix_features.reindex(df.index)
        out.loc[band_mask, "lag_1"] = aligned.loc[band_mask, "lag_1"].to_numpy()
        out.loc[band_mask, "lag_2"] = aligned.loc[band_mask, "lag_2"].to_numpy()
        out.loc[band_mask, "rolling_mean_4"] = aligned.loc[band_mask, "rolling_mean_4"].to_numpy()
    return out


def _iter_fold_bands(
    df: pd.DataFrame, fold_cutoffs: Sequence[pd.Timestamp]
) -> list[tuple[pd.Series, pd.DataFrame | None]]:
    """Yield ``(band_mask, prefix)`` pairs in fold-cutoff order.

    A band is the slice of rows assigned to a particular fold cutoff
    (rows with date in ``(prev_cutoff, current_cutoff]``). The prefix is
    the rows available at the cutoff date — i.e. rows with
    ``date <= current_cutoff`` — and is what each family should compute
    features on so that fold-aware features never see the future.

    The trailing "future" band (rows strictly after the last cutoff) has
    prefix=None: features there must remain NaN, the harness will
    compute inference-time features separately.
    """
    cutoffs = sorted(pd.Timestamp(c) for c in fold_cutoffs)
    bands: list[tuple[pd.Timestamp | None, pd.Timestamp]] = [(None, cutoffs[0])]
    for previous, current in zip(cutoffs, cutoffs[1:]):
        bands.append((previous, current))
    bands.append((cutoffs[-1], None))

    pairs: list[tuple[pd.Series, pd.DataFrame | None]] = []
    for lower_inclusive, upper in bands:
        # Define the row positions that belong to this band.
        if upper is None:
            band_mask = df["date"] > lower_inclusive  # type: ignore[operator]
        else:
            band_mask = df["date"] <= upper
        if lower_inclusive is not None:
            band_mask &= df["date"] > lower_inclusive
        if not band_mask.any():
            continue
        # Trailing "future" band (no upper cutoff) - rows here are
        # strictly after every cutoff, so no fold-aware features are
        # defined. Leave them NaN; the forecast harness will compute
        # inference-time features for these rows out of band.
        if upper is None:
            pairs.append((band_mask, None))
            continue
        # Compute features on the PREFIX (rows available up to and
        # including the cutoff). This is the fold-aware view: a row in
        # this band may only see demand from rows at or before ``upper``.
        prefix_mask = df["date"] <= upper
        prefix = df.loc[prefix_mask]
        pairs.append((band_mask, prefix))
    return pairs


# ---------------------------------------------------------------------------
# Stockout / availability family
# ---------------------------------------------------------------------------


def _coerce_stockout_flag(values: pd.Series) -> pd.Series:
    """Return a 0/1 int Series for stockout_flag values (handles bool/int)."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(int)
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.fillna(0).clip(lower=0).astype(int)


def _compute_stockout_features(
    df: pd.DataFrame, fold_cutoffs: Sequence[pd.Timestamp] | None
) -> pd.DataFrame:
    """Compute the stockout / availability feature family.

    - ``stockout_rolling_count_4``: count of stockout-flagged weeks in
      the prior 4 weeks (lag-1 window, so the current week is excluded).
    - ``days_since_stockout``: number of days since the most recent
      stockout event for the same series. NaN when no prior stockout
      has been observed. Fold-aware.
    - ``inventory_cover_ratio``: inventory_qty divided by the prior
      4-week mean demand. NaN when there is no prior demand or when
      inventory is missing. Fold-aware.

    All three features are NaN for rows that have no fold cutoff ``<=``
    their date (no leakage).
    """
    if "stockout_flag" not in df.columns:
        raise FeatureFactoryError(
            "stockout_flag column is required when use_stockout_features is enabled"
        )
    if "inventory_qty" not in df.columns:
        raise FeatureFactoryError(
            "inventory_qty column is required when use_stockout_features is enabled"
        )

    stockout_int = _coerce_stockout_flag(df["stockout_flag"])
    inventory = pd.to_numeric(df["inventory_qty"], errors="coerce").astype(float)

    out = pd.DataFrame(index=df.index, columns=list(STOCKOUT_FEATURE_COLUMNS), dtype=float)

    if not fold_cutoffs:
        grouped_stockout = stockout_int.groupby(df["series_key"], sort=False)
        out["stockout_rolling_count_4"] = grouped_stockout.transform(
            lambda s: s.shift(1).rolling(window=4, min_periods=1).sum()
        )
        # days_since_stockout: for each row, days since the most recent
        # stockout event strictly before this row's date.
        out["days_since_stockout"] = _days_since_event(df, stockout_int, value=1)
        # inventory_cover_ratio: inventory / rolling 4 of demand.
        grouped_demand = df.groupby("series_key", sort=False)["demand"]
        demand_roll = grouped_demand.transform(
            lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
        )
        out["inventory_cover_ratio"] = inventory / demand_roll.replace(0, np.nan)
        return out

    for band_mask, prefix in _iter_fold_bands(df, fold_cutoffs):
        if prefix is None:
            continue
        prefix_grouped_stockout = stockout_int.groupby(prefix["series_key"], sort=False)
        prefix_grouped_demand = prefix["demand"].groupby(prefix["series_key"], sort=False)
        prefix_features = pd.DataFrame(
            {
                "stockout_rolling_count_4": prefix_grouped_stockout.transform(
                    lambda s: s.shift(1).rolling(window=4, min_periods=1).sum()
                ),
                "days_since_stockout": _days_since_event(prefix, stockout_int, value=1),
                "inventory_cover_ratio": (
                    inventory.reindex(prefix.index)
                    / prefix_grouped_demand.transform(
                        lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
                    ).replace(0, np.nan)
                ),
            },
            index=prefix.index,
        )
        aligned = prefix_features.reindex(df.index)
        for column in STOCKOUT_FEATURE_COLUMNS:
            out.loc[band_mask, column] = aligned.loc[band_mask, column].to_numpy()
    return out


def _days_since_event(
    df: pd.DataFrame, flags: pd.Series, value: int
) -> pd.Series:
    """Return the number of days since the most recent ``flags == value`` event.

    Iterates over each series in ``df["series_key"]`` and computes the
    time delta in days to the most recent prior occurrence of ``value``.
    Returns NaN when no prior occurrence exists. ``flags`` must be
    aligned with ``df`` (same index) and the result is indexed by
    ``df.index``.
    """
    flags_aligned = flags.reindex(df.index)
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for _, indices in df.groupby("series_key", sort=False).indices.items():
        sub_dates = df["date"].iloc[indices]
        sub_flags = flags_aligned.iloc[indices]
        # Vectorised per-series: a "last event date" running maximum,
        # then the delta in days.
        last_event = sub_dates.where(sub_flags == value).ffill()
        delta = (sub_dates - last_event).dt.days
        result.iloc[indices] = delta.to_numpy()
    return result


# ---------------------------------------------------------------------------
# Hierarchy family
# ---------------------------------------------------------------------------


def _parent_keys(df: pd.DataFrame) -> pd.Series:
    """Return the parent-grain key for each row.

    Hierarchy v1: parent = sku_id (aggregated across location_id). The
    series_key in our canonical table is ``sku|loc``, so the parent is
    everything to the left of the first ``|``. If sku_id is missing we
    fall back to the existing series_key (degenerate single-level
    hierarchy) so the family still runs.
    """
    if "sku_id" in df.columns:
        return df["sku_id"].fillna("__missing__").astype(str)
    return df["series_key"]


def _compute_hierarchy_features(
    df: pd.DataFrame, fold_cutoffs: Sequence[pd.Timestamp] | None
) -> pd.DataFrame:
    """Compute parent-grain (sku-level) lag-1 and rolling-4 demand.

    The parent demand at week W is the sum of demand across all children
    of the same parent at week W. From that aggregated parent series we
    compute lag-1 and rolling-4, then broadcast the parent value back to
    each child row. Fold-aware — rows after the last cutoff are NaN'd
    out.
    """
    parent = _parent_keys(df)
    out = pd.DataFrame(index=df.index, columns=list(HIERARCHY_FEATURE_COLUMNS), dtype=float)

    if not fold_cutoffs:
        # Aggregate to (parent, date) first so children of the same
        # parent on the same date sum to a single row in the parent
        # series. This is what makes "all children see the same parent
        # value" hold.
        parent_agg = (
            df.assign(_parent=parent).groupby(["_parent", "date"], sort=False)["demand"].sum()
        )
        parent_gb = parent_agg.groupby(level="_parent", sort=False)
        parent_lag_1 = parent_gb.shift(1)
        parent_rolling = parent_gb.transform(
            lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
        )
        # Broadcast (parent, date) -> each child row that shares the key.
        keys = list(zip(parent, df["date"]))
        out["parent_lag_1"] = parent_lag_1.reindex(keys).to_numpy()
        out["parent_rolling_mean_4"] = parent_rolling.reindex(keys).to_numpy()
        return out

    for band_mask, prefix in _iter_fold_bands(df, fold_cutoffs):
        if prefix is None:
            continue
        prefix_parent = parent.reindex(prefix.index)
        prefix_agg = (
            prefix.assign(_parent=prefix_parent)
            .groupby(["_parent", "date"], sort=False)["demand"]
            .sum()
        )
        prefix_gb = prefix_agg.groupby(level="_parent", sort=False)
        agg_features = pd.DataFrame(
            {
                "parent_lag_1": prefix_gb.shift(1),
                "parent_rolling_mean_4": prefix_gb.transform(
                    lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
                ),
            }
        )
        # Reindex (parent, date) -> prefix index, then to full df index.
        keys = list(zip(prefix_parent, prefix["date"]))
        aligned_band = pd.DataFrame(
            {
                "parent_lag_1": agg_features["parent_lag_1"].reindex(keys).to_numpy(),
                "parent_rolling_mean_4": agg_features["parent_rolling_mean_4"].reindex(keys).to_numpy(),
            },
            index=prefix.index,
        )
        aligned = aligned_band.reindex(df.index)
        for column in HIERARCHY_FEATURE_COLUMNS:
            out.loc[band_mask, column] = aligned.loc[band_mask, column].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Lifecycle / cold-start family
# ---------------------------------------------------------------------------

_COLD_START_THRESHOLD = 4


def _lifecycle_block(frame: pd.DataFrame, out: pd.DataFrame) -> None:
    """Fill ``out`` with lifecycle stats for every series in ``frame``."""
    for _, indices in frame.groupby("series_key", sort=False).indices.items():
        sub_dates = frame["date"].iloc[indices]
        first_obs = sub_dates.iloc[0]
        history_length = pd.Series(np.arange(len(sub_dates)), index=sub_dates.index, dtype=float)
        days_since_first = (sub_dates - first_obs).dt.days.astype(float)
        cold_start = (history_length < _COLD_START_THRESHOLD).astype(float)
        out.loc[sub_dates.index, "history_length"] = history_length.to_numpy()
        out.loc[sub_dates.index, "days_since_first_obs"] = days_since_first.to_numpy()
        out.loc[sub_dates.index, "cold_start_flag"] = cold_start.to_numpy()


def _compute_lifecycle_features(
    df: pd.DataFrame, fold_cutoffs: Sequence[pd.Timestamp] | None
) -> pd.DataFrame:
    """Compute lifecycle / cold-start features.

    - ``history_length``: number of prior observations for this series
      visible to the model (i.e. as of the fold cutoff for fold-aware
      runs, or strictly before the row's date for non-fold runs).
    - ``days_since_first_obs``: number of days from the first
      observation of the series to the fold cutoff (or row date).
    - ``cold_start_flag``: 1 if the row is one of the first
      ``_COLD_START_THRESHOLD`` observations for the series under the
      fold-aware view, else 0.

    Lifecycle features depend on the cutoff so they are fold-aware.
    """
    out = pd.DataFrame(index=df.index, columns=list(LIFECYCLE_FEATURE_COLUMNS), dtype=float)
    if not fold_cutoffs:
        _lifecycle_block(df, out)
        return out
    for band_mask, prefix in _iter_fold_bands(df, fold_cutoffs):
        if prefix is None:
            continue
        _lifecycle_block(prefix, out)
    return out


# ---------------------------------------------------------------------------
# Intermittency family
# ---------------------------------------------------------------------------


def _compute_intermittency_features(
    df: pd.DataFrame, fold_cutoffs: Sequence[pd.Timestamp] | None
) -> pd.DataFrame:
    """Compute intermittency features over a rolling 8-week window.

    - ``rolling_adi_8``: Average Demand Interval — mean inter-demand
      interval (in weeks) over the prior 8 weeks, computed as
      ``window / max(weeks_with_demand, 1)``. 1.0 means the series
      demanded every week; 8.0 means the series demanded once in 8
      weeks.
    - ``rolling_cv2_8``: squared coefficient of variation of demand
      values in the prior 8 weeks (variance / mean²). NaN when the
      mean is zero (the whole window is zero — the variance is zero but
      the CV² is undefined).
    - ``trailing_zero_run``: length of the current run of consecutive
      zero-demand weeks ending at the most recent observation *before*
      the current row.

    All three features are time-dependent and fold-aware.
    """
    out = pd.DataFrame(index=df.index, columns=list(INTERMITTENCY_FEATURE_COLUMNS), dtype=float)

    if not fold_cutoffs:
        for column, values in _intermittency_block(df).items():
            out[column] = values
        return out

    for band_mask, prefix in _iter_fold_bands(df, fold_cutoffs):
        if prefix is None:
            continue
        block = _intermittency_block(prefix)
        for column in INTERMITTENCY_FEATURE_COLUMNS:
            out.loc[band_mask, column] = block[column].reindex(df.index).loc[band_mask].to_numpy()
    return out


def _intermittency_block(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the intermittency block for a single series-key frame.

    Assumes the input has a single set of series, sorted by date.
    """
    out = pd.DataFrame(index=df.index, columns=list(INTERMITTENCY_FEATURE_COLUMNS), dtype=float)
    for _, indices in df.groupby("series_key", sort=False).indices.items():
        sub = df["demand"].iloc[indices]
        # Window of last 8 weeks (lag-1 so the current week is excluded).
        # min_periods=1 means we get a value as soon as there is one
        # prior observation.
        shifted = sub.shift(1)
        recent = shifted.rolling(window=8, min_periods=1)
        count = recent.count()
        nonzero = recent.apply(lambda w: float((w > 0).sum()), raw=True)
        mean = recent.mean()
        std = recent.std()
        # ADI: weeks-in-window / count-of-nonzero-weeks. NaN-safe via
        # the count == 0 branch (returns NaN).
        adi = np.where(nonzero > 0, 8.0 / nonzero, np.nan)
        # CV²: std² / mean². NaN when mean is zero (degenerate).
        cv2 = np.where(mean > 0, (std ** 2) / (mean ** 2), np.nan)
        # Trailing zero-run: the number of consecutive prior zero weeks
        # ending immediately before the current row.
        is_zero = (shifted.fillna(0) == 0).astype(int)
        # group by cumsum of non-zero: each block of zeros gets its own
        # group, so a cumcount within the block gives the run length.
        run = is_zero.groupby((is_zero == 0).cumsum()).cumcount()
        run = run.where(is_zero == 1, 0)
        out.loc[sub.index, "rolling_adi_8"] = adi
        out.loc[sub.index, "rolling_cv2_8"] = cv2
        out.loc[sub.index, "trailing_zero_run"] = run.to_numpy()
    return out


def _promo_to_indicator(promo: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(promo) or pd.api.types.is_bool_dtype(promo):
        return promo.fillna(0).ne(0).astype(int)

    numeric = pd.to_numeric(promo, errors="coerce")
    text_true = promo.astype("string").fillna("").str.strip().str.lower().isin({"true", "yes", "y"})
    return (numeric.fillna(0).ne(0) | text_true).astype(int)


def _add_fourier_features(df: pd.DataFrame, terms: int, frequency_period: int | None) -> pd.DataFrame:
    row_number = df.groupby("series_key", sort=False).cumcount() + 1
    if frequency_period is None:
        period = df.groupby("series_key", sort=False)["series_key"].transform("size")
    else:
        period = frequency_period
    cols = {}
    for term in range(1, terms + 1):
        angle = 2 * np.pi * term * row_number / period
        cols[f"sin_{term}"] = np.sin(angle)
        cols[f"cos_{term}"] = np.cos(angle)
    return pd.DataFrame(cols, index=df.index)
