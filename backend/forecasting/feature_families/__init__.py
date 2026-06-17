"""Feature families for the canonical Feature Factory.

Each family is a small adapter following the ``FeatureFamily`` protocol
defined in :mod:`forecasting.feature_families._protocol`. The factory
iterates ``all_families()`` to produce the canonical feature table;
adding a new family is one file and one entry below.

The five time-dependent families (lag/rolling, stockout, hierarchy,
lifecycle, intermittency) all share the same fold-band scaffold via
:func:`apply_family_to_fold_bands` and :func:`iter_fold_bands`. The
promo indicator and Fourier features are not families in this sense —
they are time-independent and live on ``build_feature_table`` directly
so the public flag surface stays unchanged.
"""

from __future__ import annotations

from forecasting.feature_families._protocol import (
    FeatureFamily,
    apply_family_to_fold_bands,
    iter_fold_bands,
)
from forecasting.feature_families.hierarchy import HierarchyFamily
from forecasting.feature_families.intermittency import IntermittencyFamily
from forecasting.feature_families.lifecycle import LifecycleFamily
from forecasting.feature_families.stockout import StockoutFamily
from forecasting.feature_families.time_dependent import TimeDependentFamily


# The canonical, ordered list of feature families. The order is the
# order in which their columns are concatenated into the result frame
# — kept stable so existing tests and downstream consumers see
# deterministic output.
ALL_FAMILIES: tuple[FeatureFamily, ...] = (
    TimeDependentFamily(),
    StockoutFamily(),
    HierarchyFamily(),
    LifecycleFamily(),
    IntermittencyFamily(),
)


def all_families() -> tuple[FeatureFamily, ...]:
    """Return the canonical tuple of registered families.

    Returns a fresh tuple each call so callers cannot accidentally
    mutate the module-level state.
    """
    return ALL_FAMILIES


__all__ = (
    "FeatureFamily",
    "apply_family_to_fold_bands",
    "iter_fold_bands",
    "all_families",
    "TimeDependentFamily",
    "StockoutFamily",
    "HierarchyFamily",
    "LifecycleFamily",
    "IntermittencyFamily",
)
