"""Hierarchy family: parent-grain lag and rolling features.

Aggregates demand to the parent grain (v1: ``sku_id`` aggregated
across ``location_id``) and broadcasts the parent's lag-1 and rolling-4
back to each child row. All children of the same parent see the same
parent value on the same date.

Fold-aware — rows after the last cutoff are NaN'd out.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from forecasting.contracts import FeatureFlags

from forecasting.feature_families._protocol import (
    FeatureFamily,
    iter_fold_bands,
)


COLUMNS: tuple[str, ...] = ("parent_lag_1", "parent_rolling_mean_4")


def _parent_keys(df: pd.DataFrame) -> pd.Series:
    """Return the parent-grain key for each row.

    Hierarchy v1: parent = sku_id (aggregated across location_id). If
    sku_id is missing we fall back to the existing series_key
    (degenerate single-level hierarchy) so the family still runs.
    """
    if "sku_id" in df.columns:
        return df["sku_id"].fillna("__missing__").astype(str)
    return df["series_key"]


def _compute_on_parent_frame(prefix: pd.DataFrame, parent: pd.Series) -> pd.DataFrame:
    """Compute parent-grain lag/rolling for one fold's prefix."""
    prefix_parent = parent.reindex(prefix.index)
    parent_agg = (
        prefix.assign(_parent=prefix_parent)
        .groupby(["_parent", "date"], sort=False)["demand"]
        .sum()
    )
    parent_gb = parent_agg.groupby(level="_parent", sort=False)
    agg_features = pd.DataFrame(
        {
            "parent_lag_1": parent_gb.shift(1),
            "parent_rolling_mean_4": parent_gb.transform(
                lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
            ),
        }
    )
    keys = list(zip(prefix_parent, prefix["date"]))
    return pd.DataFrame(
        {
            "parent_lag_1": agg_features["parent_lag_1"].reindex(keys).to_numpy(),
            "parent_rolling_mean_4": agg_features["parent_rolling_mean_4"].reindex(keys).to_numpy(),
        },
        index=prefix.index,
    )


class HierarchyFamily:
    """Parent-grain (sku-level) lag-1 and rolling-4 demand; fold-aware."""

    name: str = "hierarchy"
    columns: tuple[str, ...] = COLUMNS

    def enabled_by(self, flags: FeatureFlags) -> bool:
        return bool(flags.use_hierarchy_features)

    def compute(
        self,
        df: pd.DataFrame,
        fold_cutoffs: Sequence[pd.Timestamp] | None,
    ) -> pd.DataFrame:
        parent = _parent_keys(df)
        out = pd.DataFrame(index=df.index, columns=list(self.columns), dtype=float)

        if not fold_cutoffs:
            full = _compute_on_parent_frame(df, parent)
            out["parent_lag_1"] = full["parent_lag_1"].to_numpy()
            out["parent_rolling_mean_4"] = full["parent_rolling_mean_4"].to_numpy()
            return out

        for band_mask, prefix in iter_fold_bands(df, fold_cutoffs):
            if prefix is None:
                continue
            block = _compute_on_parent_frame(prefix, parent)
            aligned = block.reindex(df.index)
            for column in self.columns:
                out.loc[band_mask, column] = aligned.loc[band_mask, column].to_numpy()
        return out


assert isinstance(HierarchyFamily(), FeatureFamily)
