"""Shared helpers for per-series statistics and flag coercion.

The Phase 1 ``preflight_stats`` and Phase 2 ``eda_probes`` modules
were shipped in different phases but share some underlying math
(autocorrelation on a 1-D array) and a similar boolean-string
vocabulary. This module is the one place those helpers live; both
EDA layers import from here.

Keeping the helpers in a small utility module — rather than folding
the two EDA modules into a single namespace — matches the
``CONTEXT.MD`` glossary that treats "preflight" and "EDA" as the
same logical layer but reflects the actual shipping sequence (Phase 1
shipped first, Phase 2 added the probes on top). The
deletion-test-friendly "if I delete the helper from one module, does
the other stop working?" answer is now "no" — the helper is a
named, owned concept.

The Phase 2 follow-up consolidation notes called out three known
duplications; this module addresses the two that are safe to share
without changing the EDA namespaces:

* :func:`autocorr` — Pearson correlation between ``x[:-lag]`` and
  ``x[lag:]``, returning 0.0 on degenerate input (constant series,
  ``lag >= length``).
* :func:`BOOLEAN_FLAG_TEXT` — the boolean-looking string vocabulary
  the canonical layer accepts and the probes use to label columns.
  Adding a new value to this vocabulary is one edit; before this
  module, the two layers had to be edited in lockstep.
"""

from __future__ import annotations

import numpy as np


# Boolean-looking text values the canonical layer accepts as flag
# inputs and the EDA probes use to label a column as ``boolean``. Empty
# string is the canonical layer's "value missing after stripping"
# sentinel; the probes drop it because they are looking at non-null
# values.
BOOLEAN_FLAG_TEXT: frozenset[str] = frozenset(
    {"true", "false", "yes", "no", "y", "n", "1", "0"}
)


def autocorr(x: np.ndarray, lag: int) -> float:
    """Pearson correlation between ``x[:-lag]`` and ``x[lag:]``.

    Mirrors the helper that lived in both ``eda_probes`` and
    ``preflight_stats``; consolidated here so a fix (e.g. the lag=0
    edge case) is one edit, not two. Returns 0.0 on degenerate
    input (constant series, ``lag >= length``) rather than NaN so
    reports stay JSON-serialisable.
    """
    if lag <= 0 or lag >= len(x):
        return 0.0
    a = x[:-lag]
    b = x[lag:]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


__all__ = (
    "BOOLEAN_FLAG_TEXT",
    "autocorr",
)
