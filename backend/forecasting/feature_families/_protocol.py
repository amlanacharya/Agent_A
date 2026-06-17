"""FeatureFamily protocol and shared fold-band scaffold.

Each *family* in the Feature Factory is an adapter: a small class that
takes the canonical demand table plus an optional list of fold cutoffs
and returns a DataFrame containing only that family's columns. Families
declare the columns they produce (so the factory can validate the
output) and a flag-driven ``enabled_by`` predicate (so adding a new
family is one file and one registry entry — no edits to the factory).

The fold-band scaffold lives here so every family gets the same
walk-forward safety guarantee for free: rows strictly after the last
cutoff receive NaN, and the prefix (the rows visible at the cutoff)
is the only data the family sees when computing time-dependent values.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence, runtime_checkable

import pandas as pd

from forecasting.contracts import FeatureFlags


class FeatureFactoryError(ValueError):
    """Raised when canonical feature generation cannot proceed.

    Defined here (and re-exported from :mod:`forecasting.feature_factory`)
    so the per-family modules do not have to import the factory itself
    just to raise a well-typed error — a one-way import keeps the
    dependency graph acyclic.
    """


# ---------------------------------------------------------------------------
# Fold-band scaffold
# ---------------------------------------------------------------------------


def iter_fold_bands(
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


def apply_family_to_fold_bands(
    df: pd.DataFrame,
    fold_cutoffs: Sequence[pd.Timestamp] | None,
    *,
    columns: tuple[str, ...],
    compute_prefix: "Callable[[pd.DataFrame], pd.DataFrame]",
) -> pd.DataFrame:
    """Run a fold-aware family: compute on the prefix, align to the band.

    Returns a DataFrame indexed the same way as ``df`` with the family's
    columns. Rows that have no fold cutoff ``<=`` their date are NaN'd
    out (no leakage).

    ``compute_prefix`` is the family-specific work — it takes the
    prefix (rows available at the cutoff) and returns a DataFrame with
    the family's columns, indexed the same way as the prefix.
    """
    out = pd.DataFrame(index=df.index, columns=list(columns), dtype=float)
    if not fold_cutoffs:
        # No fold cutoffs: compute on the full frame and align.
        full = compute_prefix(df)
        for column in columns:
            out[column] = full[column].to_numpy()
        return out
    for band_mask, prefix in iter_fold_bands(df, fold_cutoffs):
        if prefix is None:
            continue
        block = compute_prefix(prefix)
        # Reindex the prefix features onto the full df index so the band
        # rows line up; rows outside the prefix keep NaN automatically.
        aligned = block.reindex(df.index)
        for column in columns:
            out.loc[band_mask, column] = aligned.loc[band_mask, column].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Family protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FeatureFamily(Protocol):
    """The contract every FeatureFactory family must satisfy.

    A family:

    * has a stable ``name`` (the canonical column prefix / log key)
    * declares the ``columns`` it produces, in order
    * reports whether it is enabled for a given ``FeatureFlags``
    * ``compute``s its columns on ``df``, fold-aware when ``fold_cutoffs``
      is supplied. The result must be indexed the same way as ``df``.
    """

    name: str
    columns: tuple[str, ...]

    def enabled_by(self, flags: FeatureFlags) -> bool: ...

    def compute(
        self,
        df: pd.DataFrame,
        fold_cutoffs: Sequence[pd.Timestamp] | None,
    ) -> pd.DataFrame: ...

