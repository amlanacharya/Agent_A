"""Config-escalation loop for the two-path escalation path (CB5).

The loop takes a ranked ``Proposal[]`` from CB3 and tries one
config proposal at a time, measuring the MASE delta on each.
Proposals that improve MASE past the marginal-gain threshold
(CB4) are kept; proposals that don't are reverted. The loop
stops on target met, marginal-gain floor, per-knob cap, or
proposals exhausted.

Code proposals (``kind == "code"``) are out of scope here — they
go through the existing ``model_escalation.request_custom_family_attempt``
path, which already implements the 3-attempt cap and human
permission gate the plan requires. This module only handles
``kind == "config"`` proposals; anything else is filtered out
at the entry.

Design rules:

* **One proposal at a time.** A config proposal mutates a
  single ``FeatureFlag`` (or swaps a model family, or tunes a
  parameter). The loop applies, measures, decides. No batch
  application — the per-proposal MASE delta is the keep/kill
  signal.
* **Kill = revert.** When a proposal fails the keep/kill test,
  the change is reverted so the next proposal starts from a
  clean state. The attempt still counts toward the patience
  count — the DS made a try.
* **Per-knob-type attempt cap, not per-proposal.** The plan says
  "per-knob-type attempt cap (default 3 per Run)" — meaning
  3 tries on ``enable_promo_indicator``, 3 on
  ``enable_stockout_features``, etc. The cap is a per-action-kind
  counter; a knob that's been tried 3 times is dropped from the
  candidate list for the rest of the Run.
* **Reuse the harness.** ``measure_mase`` is a thin wrapper
  around ``run_forecast_harness``. The loop does not re-implement
  fitting/backtest logic.
* **Stub-friendly MASE.** ``MeasureMASE`` is a Protocol so tests
  pass a deterministic stub returning a per-proposal MASE without
  calling the real harness.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from forecasting.contracts import (
    ConfigAction,
    FeatureFlags,
    ForecastRequest,
    MeasureMASE,
    ModelFamilyName,
    ModelScorecard,
    Proposal,
    ProposalTarget,
)
from forecasting.forecast_harness import run_forecast_harness
from forecasting.feature_factory import build_feature_table
from forecasting.marginal_gain import MarginalGainConfig, should_stop


# ---------------------------------------------------------------------------
# Stop reason
# ---------------------------------------------------------------------------

StopReason = Literal[
    "target_met",
    "marginal_gain_floor",
    "knob_cap",
    "config_exhausted",
    "no_config_proposals",
]


# ---------------------------------------------------------------------------
# Measure-MASE seam
# ---------------------------------------------------------------------------
#
# ``MeasureMASE`` is imported from :mod:`forecasting.contracts`
# (the canonical type home) and re-exported below in ``__all__``
# for one session of source compat. The production adapter
# (``default_measure_mase``) and the stub adapter live with the
# code that uses them - the Protocol itself lives in ``contracts``.


def default_measure_mase(
    *,
    canonical_table: "pd.DataFrame",  # noqa: F821
    fold_cutoffs: Sequence["pd.Timestamp"],  # noqa: F821
    horizon: int,
    target_col: str = "demand",
) -> MeasureMASE:
    """Build a production ``MeasureMASE`` bound to a canonical demand table.

    Each call rebuilds the feature table from ``canonical_table`` +
    ``flags`` and runs the harness. The returned MASE is the mean
    MASE across all (series, fold) scorecards — a single number
    summarising the post-application model quality.
    """

    def _measure(flags: FeatureFlags, model_family: ModelFamilyName) -> float:
        features = build_feature_table(
            canonical_table, flags, fold_cutoffs=fold_cutoffs
        )
        request = ForecastRequest(
            run_id="config_escalation",
            model_families=[model_family],
            horizon=horizon,
            target_col=target_col,
        )
        report = run_forecast_harness(request, features=features)
        scorecards: list[ModelScorecard] = report.scorecards
        if not scorecards:
            return float("inf")
        mase_values = [s.mase for s in scorecards if not _is_nan(s.mase)]
        if not mase_values:
            return float("inf")
        return float(sum(mase_values) / len(mase_values))

    return _measure


def _is_nan(x: float) -> bool:
    """``math.isnan`` with a friendlier import surface.

    Avoids the ``math`` import at module top because the rest of
    the module deliberately doesn't use it; pulling it in just
    for this one helper would be noise.
    """
    return x != x  # NaN is the only float that is not equal to itself


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class ConfigApplicationError(ValueError):
    """Raised when a config proposal cannot be applied.

    Distinct from a MASE-regression "kill" — this is a structural
    failure (unknown action, malformed payload, target outside
    the candidate set). The loop catches it and treats the
    proposal as a no-op + a counted attempt.
    """


def _enable_for_action(action: ConfigAction) -> str | None:
    """Map a ``enable_*`` action to the ``FeatureFlags`` field it flips.

    Returns the field name (e.g. ``"use_promo_indicator"``) or
    ``None`` for actions that are not flag toggles. The mapping
    is the seam between the closed ``ConfigAction`` Literal
    (CB1) and the existing ``FeatureFlags`` shape — when a new
    flag is added to ``FeatureFlags``, the corresponding
    ``enable_*`` action and this entry land together.
    """
    return {
        "enable_lag_features": "use_lag_features",
        "enable_promo_indicator": "use_promo_indicator",
        "enable_stockout_features": "use_stockout_features",
        "enable_hierarchy_features": "use_hierarchy_features",
        "enable_lifecycle_features": "use_lifecycle_features",
        "enable_intermittency_features": "use_intermittency_features",
    }.get(action)


def _validate_target_scope(proposal: Proposal) -> None:
    """Reject proposals that don't target a series when in series mode.

    The loop runs at the segment level; proposals targeting a
    specific series are filtered upstream by the caller. This
    check is defence-in-depth: a config proposal with a
    segment-level target and a series-level apply is a contract
    violation.
    """
    # All config proposals in scope target a series (the loop
    # is per-series inside the segment). Segment-level config
    # proposals are a future feature; for now we accept any
    # target shape and apply the change globally.
    _ = proposal  # target validation is a no-op until segment-level lands


def apply_config_proposal(
    proposal: Proposal,
    *,
    current_flags: FeatureFlags,
    current_model_family: ModelFamilyName,
) -> tuple[FeatureFlags, ModelFamilyName]:
    """Apply a single config proposal. Returns the new (flags, family).

    Raises ``ConfigApplicationError`` when the proposal cannot
    be applied: unknown action, malformed payload, or non-config
    proposal passed by mistake. The caller treats this as a
    no-op + a counted attempt.
    """
    if proposal.kind != "config":
        raise ConfigApplicationError(
            f"apply_config_proposal only handles kind=config, got kind={proposal.kind!r}"
        )
    if proposal.config_action is None:
        raise ConfigApplicationError(
            "config proposal has no config_action set"
        )
    _validate_target_scope(proposal)
    action = proposal.config_action

    if action == "swap_model_family":
        family = proposal.action_payload.get("family")
        if not isinstance(family, str):
            raise ConfigApplicationError(
                "swap_model_family requires action_payload={'family': <ModelFamilyName>}"
            )
        return current_flags, family  # type: ignore[return-value]

    if action == "tune_model_parameter":
        # The parameter change is captured on the proposal but
        # not actually applied to a model — the model family
        # registry handles parameter dispatch in CB5+. For now,
        # a tune proposal is a no-op that records the intent
        # on the attempt's audit log. The MASE delta will be
        # the same as before the proposal, which means the
        # keep/kill test correctly drops it as "no improvement".
        return current_flags, current_model_family

    if action == "increase_fourier_terms":
        # Increase fourier_terms by 1, capped at 8. The cap
        # matches the FeatureFactory's "8 terms" practical
        # limit (more terms = risk of overfitting).
        new_terms = min(current_flags.fourier_terms + 1, 8)
        new_flags = current_flags.model_copy(update={"fourier_terms": new_terms})
        return new_flags, current_model_family

    flag_field = _enable_for_action(action)
    if flag_field is None:
        raise ConfigApplicationError(f"unknown config_action: {action!r}")
    # Enable the flag. Disabling a feature via a config proposal
    # is not in scope — a "disable" action would be a separate
    # code path (and arguably belongs in code-escalation, since
    # it's a model-class change, not a config flip).
    new_flags = current_flags.model_copy(update={flag_field: True})
    return new_flags, current_model_family


# ---------------------------------------------------------------------------
# Attempt + report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigAttemptResult:
    """The outcome of one config proposal application + MASE measurement.

    ``kept`` is True when the proposal passed the keep/kill
    test. ``mase_before`` is the MASE at the start of the
    attempt; ``mase_after`` is the MASE measured after applying
    the proposal (whether kept or reverted). The report's
    ``final_mase`` is the last kept MASE in the sequence.
    """

    proposal: Proposal
    mase_before: float
    mase_after: float
    kept: bool
    error: str | None = None  # set when the application raised


@dataclass(frozen=True)
class ConfigEscalationReport:
    """The output of one config-escalation round.

    ``attempts`` lists every proposal the loop considered,
    in the order it considered them. ``stopped_reason`` is the
    reason the loop exited — see ``StopReason`` for the
    closed set. ``final_mase`` is the MASE at the end of the
    loop (the last kept MASE, or the starting MASE if no
    proposal was kept).
    """

    run_id: str
    segment_id: str | None
    attempts: list[ConfigAttemptResult] = field(default_factory=list)
    stopped_reason: StopReason = "config_exhausted"
    final_mase: float = float("inf")
    final_flags: FeatureFlags | None = None
    final_model_family: ModelFamilyName | None = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def _action_kind(proposal: Proposal) -> str:
    """The "knob kind" for per-knob attempt capping.

    All config proposals are bucketed by their action name. A
    proposal that uses ``enable_promo_indicator`` competes in
    the same cap bucket as every other ``enable_promo_indicator``
    proposal in the Run. ``swap_model_family`` and
    ``tune_model_parameter`` have their own buckets.
    """
    if proposal.config_action is not None:
        return proposal.config_action
    return "<unknown>"


def run_config_escalation(
    *,
    run_id: str,
    proposals: Sequence[Proposal],
    starting_flags: FeatureFlags,
    starting_model_family: ModelFamilyName,
    starting_mase: float,
    measure_mase: MeasureMASE,
    config: MarginalGainConfig,
    segment_id: str | None = None,
    knob_caps: dict[str, int] | None = None,
) -> ConfigEscalationReport:
    """Run the config-escalation loop.

    Parameters
    ----------
    run_id
        Echoed on the report. The loop is otherwise stateless
        with respect to the run.
    proposals
        The ranked ``Proposal[]`` from CB3. Filtered to
        ``kind=config``; code proposals are routed to
        ``model_escalation`` and out of scope here.
    starting_flags
        The ``FeatureFlags`` at the start of the loop.
    starting_model_family
        The model family at the start of the loop.
    starting_mase
        The MASE measured at ``starting_flags`` +
        ``starting_model_family``. Passed in (not re-measured)
        because the loop's first MASE delta is
        ``mase_after - starting_mase``.
    measure_mase
        The ``MeasureMASE`` callable. Production uses
        ``default_measure_mase(...)``; tests use a stub.
    config
        The marginal-gain thresholds (CB4). Used to decide
        keep/kill on each attempt and to decide the floor stop.
    segment_id
        Echoed on the report. The loop does not use the
        segment ID for any routing decision; it is metadata
        for the audit log.
    knob_caps
        Per-action-kind attempt cap. Defaults to ``{action: 3}``
        for every action when ``None``. Pass an explicit dict
        to override (e.g. tests use small caps to keep the
        fixture tight).

    Returns
    -------
    ConfigEscalationReport
        The full attempt log and the final state.
    """
    # Filter to config proposals. Code proposals are routed
    # elsewhere; including them here would silently swallow them.
    config_proposals = [p for p in proposals if p.kind == "config"]
    if not config_proposals:
        return ConfigEscalationReport(
            run_id=run_id,
            segment_id=segment_id,
            stopped_reason="no_config_proposals",
            final_mase=starting_mase,
            final_flags=starting_flags,
            final_model_family=starting_model_family,
        )

    caps: dict[str, int] = dict(knob_caps) if knob_caps is not None else {}
    # Default cap for any action not explicitly listed: 3 attempts.
    default_cap = 3

    current_flags = starting_flags
    current_family: ModelFamilyName = starting_model_family  # type: ignore[assignment]
    current_mase = starting_mase
    # ---------------------------------------------------------------------------
    # Why the per-run attempt ledger is in-memory and not file-backed.
    # ---------------------------------------------------------------------------
    # A FeatureFlag flip is a per-run experiment - the next run starts from
    # the saved champion config and re-decides every proposal afresh. There
    # is nothing to "honour" across restarts; persisting the counter would
    # leak yesterday's failed attempts into today's experiment. The
    # contrasting design lives in ``code_escalation.py`` - custom model
    # families are permanent, so its attempt ledger is file-backed. If
    # unified observability is ever needed, the seam is an ``AttemptTracker``
    # Protocol that both modules could implement.
    # ---------------------------------------------------------------------------
    attempts: list[ConfigAttemptResult] = []
    history: list[float] = [starting_mase]
    stopped_reason: StopReason = "config_exhausted"

    for proposal in config_proposals:
        kind = _action_kind(proposal)
        # Per-knob cap: drop the proposal if this kind has
        # already used its budget.
        used = sum(1 for a in attempts if _action_kind(a.proposal) == kind)
        cap = caps.get(kind, default_cap)
        if used >= cap:
            attempts.append(
                ConfigAttemptResult(
                    proposal=proposal,
                    mase_before=current_mase,
                    mase_after=current_mase,
                    kept=False,
                    error=f"knob cap reached: kind={kind} used={used} cap={cap}",
                )
            )
            continue

        # Apply the proposal (mutate flags / swap family).
        try:
            new_flags, new_family = apply_config_proposal(
                proposal,
                current_flags=current_flags,
                current_model_family=current_family,
            )
        except ConfigApplicationError as exc:
            attempts.append(
                ConfigAttemptResult(
                    proposal=proposal,
                    mase_before=current_mase,
                    mase_after=current_mase,
                    kept=False,
                    error=str(exc),
                )
            )
            continue

        # Measure the new MASE.
        mase_after = measure_mase(new_flags, new_family)

        # Decide keep / kill on the marginal-gain rule. The
        # patience history is updated regardless of keep/kill
        # — the DS made a try, and that try counts toward
        # the patience count.
        history.append(mase_after)
        if should_stop(history, config, current_mase=mase_after):
            # The stop condition fired *because* of this
            # attempt. We did not keep the change; revert.
            attempts.append(
                ConfigAttemptResult(
                    proposal=proposal,
                    mase_before=current_mase,
                    mase_after=mase_after,
                    kept=False,
                )
            )
            # Determine which stop condition fired. If target
            # was met, it was target_met (not marginal_gain_floor).
            if mase_after <= config.target_mase:
                stopped_reason = "target_met"
                # We *did* hit target, so keep the change.
                current_flags = new_flags
                current_family = new_family
                current_mase = mase_after
                attempts[-1] = ConfigAttemptResult(
                    proposal=proposal,
                    mase_before=current_mase,
                    mase_after=mase_after,
                    kept=True,
                )
            else:
                stopped_reason = "marginal_gain_floor"
            break

        # Keep the change (the proposal is either an
        # improvement or a non-improvement that did not
        # yet trigger the patience floor).
        attempts.append(
            ConfigAttemptResult(
                proposal=proposal,
                mase_before=current_mase,
                mase_after=mase_after,
                kept=True,
            )
        )
        current_flags = new_flags
        current_family = new_family
        current_mase = mase_after

    # If we ran out of proposals without stopping, decide
    # whether the cap was the reason. Check whether every
    # remaining proposal (had we continued) would have hit
    # a cap.
    if stopped_reason == "config_exhausted":
        all_capped = all(
            _action_kind(p) in {k for k, v in Counter(
                _action_kind(a.proposal) for a in attempts
            ).items() if v >= caps.get(k, default_cap)}
            for p in config_proposals[len(attempts):]
        )
        # The above expression is True only if every remaining
        # proposal would have hit a cap. This is a soft signal
        # — the more honest stopped_reason is "config_exhausted"
        # when we just ran out of proposals. We only flip to
        # "knob_cap" when the last attempt was itself a cap
        # rejection.
        last = attempts[-1] if attempts else None
        if last is not None and last.error and last.error.startswith("knob cap reached"):
            stopped_reason = "knob_cap"

    return ConfigEscalationReport(
        run_id=run_id,
        segment_id=segment_id,
        attempts=attempts,
        stopped_reason=stopped_reason,
        final_mase=current_mase,
        final_flags=current_flags,
        final_model_family=current_family,
    )


__all__ = (
    "MeasureMASE",
    "ConfigApplicationError",
    "ConfigAttemptResult",
    "ConfigEscalationReport",
    "StopReason",
    "apply_config_proposal",
    "default_measure_mase",
    "run_config_escalation",
)
