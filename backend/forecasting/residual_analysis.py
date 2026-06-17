"""Residual decomposition for the Phase 4.1 two-path escalation loop.

CB2 of Phase 4.1: the deterministic math that backs the
``propose_feature_changes`` tool (CB3) with ``Claim(evidence_type=pattern)``
evidence. Pure functions, no I/O, no LLM. The proposal tool consumes
``ResidualDecomposition`` and ranks ``Proposal`` candidates by the
severity of the patterns emitted here.

Design rules:

* **Closed pattern set, named in :mod:`forecasting.contracts`.** The
  decomposition never returns a free-text residual narrative; it
  returns ``ResidualPatternHit`` rows from a closed Literal. Free-text
  would force the LLM to do structured extraction at proposal time,
  which is the wrong place for that work.
* **Severity in [0, 1].** A normalised score so the proposal tool
  can rank by max hit severity without re-reading residual stats.
  Thresholds are constants below; ``.env``-configurable thresholds
  land in CB4 alongside the marginal-gain stop condition.
* **Context fields are optional.** A scorecard with no canonical
  demand slice (the most common case for unit tests) gets the pure
  stats block; the contextual ``promo_*`` / ``stockout_*`` /
  ``parent_*`` means stay ``None``. The proposal tool ranks the
  pure patterns only when context is absent.
* **No I/O, no escalation.** The probe is a pure function. The
  proposal tool decides what to do with the result.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from forecasting.contracts import (
    ModelScorecard,
    ResidualDecomposition,
    ResidualPattern,
    ResidualPatternHit,
    ResidualStats,
)
from forecasting.stats_utils import autocorr


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Severity thresholds for the closed pattern set. A pattern is emitted
# when its measured value crosses the threshold; severity is the value
# clipped to [0, 1]. The thresholds are deliberately conservative — the
# proposal tool is the place to be aggressive about recommending
# changes; the decomposition is the place to be conservative about
# claiming a pattern is present.
#
# BIASED: |mean residual| / mean actual demand. 0.2 = forecast is off
# by 20% of the demand level on average (a meaningful bias).
# AUTOCORR: max |autocorrelation| at lags 1/2/4/8. 0.3 is a noticeable
# residual structure.
# PROMO/STOCKOUT/PARENT_GAP: |mean_residual_in_group -
# mean_residual_out_group| / overall_residual_std. 0.5 is half a
# standard-deviation shift between groups.
# HETERO: residual_std / mean_actual. 1.0 = residual std equals the
# demand level (the forecast is no better than predicting the mean).
_BIAS_THRESHOLD = 0.2
_AUTOCORR_THRESHOLD = 0.3
_GROUP_GAP_THRESHOLD = 0.5
_HETERO_THRESHOLD = 1.0


# ---------------------------------------------------------------------------
# Pure stats
# ---------------------------------------------------------------------------


def _residual_stats_from_arrays(
    series_key: str,
    forecast: Sequence[float],
    actual: Sequence[float],
) -> ResidualStats:
    """Compute the pure-residual-stats block (no context).

    Returns a ``ResidualStats`` with the optional contextual fields
    (``promo_*`` / ``stockout_*`` / ``parent_*``) all ``None``. The
    caller is responsible for filling them in when context is
    available — see :func:`_augment_stats_with_context`.
    """
    forecast_arr = np.asarray(forecast, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    if forecast_arr.shape != actual_arr.shape:
        raise ValueError(
            f"forecast and actual length mismatch for {series_key}: "
            f"{forecast_arr.shape} vs {actual_arr.shape}"
        )
    n = int(forecast_arr.shape[0])
    residuals = actual_arr - forecast_arr
    residual_mean = float(residuals.mean()) if n else 0.0
    residual_std = float(residuals.std()) if n > 1 else 0.0
    mae = float(np.mean(np.abs(residuals))) if n else 0.0

    def _ac(lag: int) -> float | None:
        if n <= lag:
            return None
        return autocorr(residuals, lag)

    return ResidualStats(
        series_key=series_key,
        n=n,
        residual_mean=residual_mean,
        residual_std=residual_std,
        mae=mae,
        autocorr_lag_1=_ac(1),
        autocorr_lag_2=_ac(2),
        autocorr_lag_4=_ac(4),
        autocorr_lag_8=_ac(8),
    )


def _augment_stats_with_context(
    stats: ResidualStats,
    forecast: Sequence[float],
    actual: Sequence[float],
    canonical_slice: pd.DataFrame | None,
    parent_residual_mean: float | None,
) -> ResidualStats:
    """Fill the contextual fields on a ``ResidualStats`` in place.

    The ``canonical_slice`` is a DataFrame with one row per forecast
    step, in order, carrying the promo / stockout flag columns for
    the same weeks. ``parent_residual_mean`` is the mean residual at
    the parent (sku_id aggregated across location_id) grain, when
    known.

    Both inputs are optional. When ``canonical_slice`` is None, the
    promo / stockout means stay None; when ``parent_residual_mean``
    is None, the parent field stays None.
    """
    updates: dict[str, float] = {}
    if canonical_slice is not None and len(canonical_slice) == len(forecast):
        residuals = np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float)
        if "promo" in canonical_slice.columns:
            promo_mask = _flag_mask(canonical_slice["promo"])
            non_promo_mask = ~promo_mask
            if promo_mask.any():
                updates["promo_residual_mean"] = float(residuals[promo_mask].mean())
            if non_promo_mask.any():
                updates["non_promo_residual_mean"] = float(residuals[non_promo_mask].mean())
        if "stockout_flag" in canonical_slice.columns:
            stockout_mask = _flag_mask(canonical_slice["stockout_flag"])
            non_stockout_mask = ~stockout_mask
            if stockout_mask.any():
                updates["stockout_residual_mean"] = float(residuals[stockout_mask].mean())
            if non_stockout_mask.any():
                updates["non_stockout_residual_mean"] = float(residuals[non_stockout_mask].mean())
    if parent_residual_mean is not None:
        updates["parent_residual_mean"] = float(parent_residual_mean)
    if not updates:
        return stats
    return stats.model_copy(update=updates)


def _flag_mask(series: pd.Series) -> np.ndarray:
    """Coerce a promo / stockout flag column to a boolean mask.

    Handles int (0/1), bool, and string ("true"/"false") inputs.
    The canonical layer normalises these upstream so this is
    defence-in-depth, not a primary path.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return numeric.to_numpy(dtype=bool)


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def _detect_patterns(stats: ResidualStats) -> list[ResidualPatternHit]:
    """Run the closed pattern set against the stats and emit hits.

    Each pattern is checked against its threshold; a hit is emitted
    only when the threshold is crossed. Severity is the measured
    value clipped to [0, 1] — a value above the threshold has
    severity 1.0, signalling "definitely present" to the proposal
    tool. A value just below the threshold has severity just below
    1.0; a value well below has low severity.
    """
    hits: list[ResidualPatternHit] = []

    # BIASED: |mean residual| / mean actual. We need the mean actual
    # demand for the normalisation; the scorecard doesn't carry it
    # directly, so we recover it from residual + mean: actual_mean =
    # forecast_mean + residual_mean. The decomposition function
    # passes the canonical slice in the context, which has the
    # actual demand column — but to keep this function pure we use
    # the stats block. We approximate: |residual_mean| / |actual_mean|
    # is the relative bias only if we know actual_mean. We can
    # recover it from the scorecard via the forecast + residual
    # relationship ONLY if we re-derive; instead we use a different
    # normalisation: |residual_mean| / (residual_std + eps). This
    # gives a bias-vs-noise ratio, which is what the proposal tool
    # cares about: "is the bias large compared to the natural
    # noise, or is the model just noisy?". A pure-bias model has
    # a high ratio; a noisy model has a low ratio.
    if stats.residual_std > 0:
        bias_ratio = abs(stats.residual_mean) / stats.residual_std
        # Map the ratio to a [0, 1] severity: a ratio of 1.0 = bias
        # equals one std → severity 1.0; a ratio of 0.2 = bias is
        # 20% of one std → severity 0.2. We cap at 1.0.
        severity = min(1.0, bias_ratio)
        if severity >= _BIAS_THRESHOLD:
            hits.append(
                ResidualPatternHit(
                    pattern="BIASED_RESIDUAL",
                    severity=severity,
                    detail=(
                        f"mean residual {stats.residual_mean:.3f} / std "
                        f"{stats.residual_std:.3f} = ratio {bias_ratio:.2f}"
                    ),
                )
            )

    # AUTOCORRELATED: max |autocorrelation| at lags {1, 2, 4, 8}.
    ac_values = [
        v for v in (stats.autocorr_lag_1, stats.autocorr_lag_2, stats.autocorr_lag_4, stats.autocorr_lag_8) if v is not None
    ]
    if ac_values:
        max_ac = max(abs(v) for v in ac_values)
        if max_ac >= _AUTOCORR_THRESHOLD:
            hits.append(
                ResidualPatternHit(
                    pattern="AUTOCORRELATED_RESIDUAL",
                    severity=min(1.0, max_ac),
                    detail=f"max |autocorr| across lags 1/2/4/8 = {max_ac:.2f}",
                )
            )

    # PROMO / STOCKOUT / PARENT_GAP: |mean_in - mean_out| / std.
    # These need std > 0 to normalise; if the residual is constant
    # the patterns are undefined and we skip them.
    if stats.residual_std > 0:
        for pattern, in_mean, out_mean, label in (
            (
                "PROMO_RESIDUAL_SPIKE",
                stats.promo_residual_mean,
                stats.non_promo_residual_mean,
                "promo",
            ),
            (
                "STOCKOUT_RESIDUAL_SPIKE",
                stats.stockout_residual_mean,
                stats.non_stockout_residual_mean,
                "stockout",
            ),
        ):
            if in_mean is None or out_mean is None:
                continue
            gap = abs(in_mean - out_mean) / stats.residual_std
            if gap >= _GROUP_GAP_THRESHOLD:
                hits.append(
                    ResidualPatternHit(
                        pattern=pattern,
                        severity=min(1.0, gap),
                        detail=(
                            f"{label}-week mean residual {in_mean:.3f} vs "
                            f"non-{label} {out_mean:.3f}, gap {gap:.2f} std"
                        ),
                    )
                )
        if stats.parent_residual_mean is not None:
            gap = abs(stats.residual_mean - stats.parent_residual_mean) / stats.residual_std
            if gap >= _GROUP_GAP_THRESHOLD:
                hits.append(
                    ResidualPatternHit(
                        pattern="PARENT_CHILD_RESIDUAL_GAP",
                        severity=min(1.0, gap),
                        detail=(
                            f"child mean residual {stats.residual_mean:.3f} vs "
                            f"parent {stats.parent_residual_mean:.3f}, gap {gap:.2f} std"
                        ),
                    )
                )

    # HETEROSCEDASTIC: residual_std vs MAE. A well-calibrated model
    # has std close to MAE * sqrt(pi/2) (the relationship for a
    # half-normal error distribution). A std >> MAE means long-tail
    # errors — the model is right on average but occasionally
    # catastrophic. We use residual_std / mae as the ratio; values
    # much above 1.0 are suspect.
    if stats.mae > 0:
        hetero_ratio = stats.residual_std / stats.mae
        if hetero_ratio >= _HETERO_THRESHOLD:
            hits.append(
                ResidualPatternHit(
                    pattern="HETEROSCEDASTIC_RESIDUAL",
                    severity=min(1.0, hetero_ratio / 2.0),
                    detail=(
                        f"residual std {stats.residual_std:.3f} / mae {stats.mae:.3f} "
                        f"= ratio {hetero_ratio:.2f}"
                    ),
                )
            )

    return hits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decompose_residuals(
    scorecard: ModelScorecard,
    *,
    canonical_slice: pd.DataFrame | None = None,
    parent_residual_mean: float | None = None,
) -> ResidualDecomposition:
    """Run the residual decomposition for one ``ModelScorecard``.

    Parameters
    ----------
    scorecard
        The output of a single (model, series, fold) backtest. Only
        the ``forecast`` and ``actual`` arrays are consumed; the
        other fields (``model_family``, ``mase``, etc.) are not read
        here. The decomposition is family-agnostic — same code path
        for naive, XGBoost, anything.
    canonical_slice
        Optional DataFrame with one row per forecast step, in
        order, carrying ``promo`` and ``stockout_flag`` columns for
        the same weeks. When ``None``, the contextual patterns
        (``PROMO_RESIDUAL_SPIKE``, ``STOCKOUT_RESIDUAL_SPIKE``) are
        not emitted.
    parent_residual_mean
        Optional mean residual at the parent (sku_id aggregated
        across location_id) grain, when known. When ``None``, the
        ``PARENT_CHILD_RESIDUAL_GAP`` pattern is not emitted.

    Returns
    -------
    ResidualDecomposition
        The pure stats block plus the pattern hits. The
        ``fold_cutoff`` is copied from the scorecard so the
        proposal tool can aggregate across folds.
    """
    stats = _residual_stats_from_arrays(
        scorecard.series_key, scorecard.forecast, scorecard.actual
    )
    stats = _augment_stats_with_context(
        stats, scorecard.forecast, scorecard.actual,
        canonical_slice=canonical_slice,
        parent_residual_mean=parent_residual_mean,
    )
    patterns = _detect_patterns(stats)
    return ResidualDecomposition(
        series_key=scorecard.series_key,
        fold_cutoff=scorecard.fold_cutoff,
        stats=stats,
        patterns=patterns,
    )


__all__ = (
    "decompose_residuals",
)
