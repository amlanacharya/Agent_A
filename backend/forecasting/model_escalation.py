"""Model creation escalation path (Phase 4).

The plan is explicit: a custom / new model family is **escalation**,
not default behaviour. The flow is:

1. Standard model families (``naive``, ``seasonal_naive``,
   ``moving_average``, ``exponential_smoothing``, ``croston``,
   ``xgboost_global``, ``aggregate_allocate``) all fail for the
   segment under consideration. Failure modes include "no family
   produced a usable forecast", "every family has worse-than-naive
   MASE", or "the data shape does not match any registered family".
2. The agent **explains the gap** - it records the failure reasons
   in an ``EscalationTracker`` (shared with the EDA / adapter
   escalation paths) and surfaces the explanation to the cockpit.
3. A human **grants coding permission** - the escalation transitions
   from ``toolbox`` to ``code_escalation``. The harness never
   auto-promotes custom model families.
4. The agent gets a **maximum of three** attempts. Each attempt
   must pass four review gates before it is accepted:

   - **data_contract** - the model produces a properly-shaped
     forecast vector (length == horizon, no NaN / inf, numeric).
   - **backtest** - the model's MASE on a held-out fold is finite
     and non-negative.
   - **robustness** - the model is stable across the fold cutoffs
     (no extreme variance / blow-ups).
   - **review** - a human has signed off on the candidate.

5. A successful model is **promoted** - it joins the governed
   ``ModelFamilyName`` registry. A failed model produces an
   exact ``ModelFailureReport`` (blocker + evidence + failed
   gates) and the run halts with the failure surfaced in the
   cockpit.

The escalation layer is the only legal way to add a new family
short of editing the registry directly. Custom families are
tracked separately from the governed ones so the cockpit can
distinguish "registered model" from "experimental model".
"""

from __future__ import annotations

import math
from typing import Iterable, Literal

from forecasting.code_escalation import (
    EscalationLimitReached,
    EscalationTracker,
    FailureReport,
    ProblemKind,
)
from forecasting.contracts import (
    ModelFamilyName,
    ModelFailureReport,
    ModelScorecard,
    RobustnessCheck,
)


# ---------------------------------------------------------------------------
# The four review gates. Order matters: a candidate must pass
# every gate in this order to be promoted. The escalation layer
# records the first failure and aborts the attempt.
# ---------------------------------------------------------------------------

Gate = Literal["data_contract", "backtest", "robustness", "review"]
GATE_ORDER: tuple[Gate, ...] = ("data_contract", "backtest", "robustness", "review")
PROBLEM_KIND: ProblemKind = "model_limitation"


# ---------------------------------------------------------------------------
# Review helpers
# ---------------------------------------------------------------------------


def check_data_contract(
    *,
    forecast: list[float],
    actual: list[float] | None,
    horizon: int,
) -> RobustnessCheck:
    """Validate the data-contract gate.

    The forecast must be a horizon-long numeric vector with no NaN
    or infinity. The actuals are checked for length parity but are
    not required to be all-present (an empty actuals list means we
    are in inference mode, not a backtest).
    """
    issues: list[str] = []
    if not isinstance(forecast, list):
        issues.append("forecast must be a list")
    if len(forecast) != horizon:
        issues.append(f"forecast length {len(forecast)} != horizon {horizon}")
    for index, value in enumerate(forecast):
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            issues.append(f"forecast[{index}] is not finite")
            break
    if actual is not None and len(actual) != len(forecast):
        issues.append(f"actual length {len(actual)} != forecast length {len(forecast)}")
    return RobustnessCheck(
        check="data_contract",
        passed=not issues,
        detail="; ".join(issues) if issues else "forecast shape and finiteness ok",
    )


def check_backtest(
    *,
    forecast: list[float],
    actual: list[float],
    mase: float,
) -> RobustnessCheck:
    """Validate the backtest gate.

    MASE must be finite and non-negative. A negative MASE is
    mathematically possible (the naive baseline has been beaten so
    hard that the in-sample MAE is below zero) but it is always a
    data bug - guard against it explicitly.
    """
    issues: list[str] = []
    if math.isnan(mase) or math.isinf(mase):
        issues.append(f"MASE is not finite: {mase}")
    if mase < 0:
        issues.append(f"MASE is negative ({mase}) - the in-sample naive MAE is non-positive")
    if not actual:
        issues.append("no actuals to score against")
    if len(forecast) != len(actual):
        issues.append(f"forecast/actual length mismatch: {len(forecast)} vs {len(actual)}")
    return RobustnessCheck(
        check="backtest",
        passed=not issues,
        detail="; ".join(issues) if issues else "backtest metrics are finite and non-negative",
    )


def check_robustness(
    *,
    scorecards: Iterable[ModelScorecard],
    family: ModelFamilyName,
    max_relative_mase: float = 5.0,
) -> RobustnessCheck:
    """Validate the robustness gate.

    The model must be stable across the folds the harness ran it
    on. We compute the max MASE and the median MASE; if the max is
    more than ``max_relative_mase`` times the median, the model
    is blowing up on at least one fold and we reject it.

    The default cap is generous (5x) - the goal is to catch true
    blow-ups (MASE = 1000 on a series where the median is 2),
    not to gate against ordinary fold-to-fold variance.
    """
    issues: list[str] = []
    family_scorecards = [scorecard for scorecard in scorecards if scorecard.model_family == family]
    if not family_scorecards:
        issues.append(f"no scorecards found for {family!r}")
        return RobustnessCheck(check="robustness", passed=False, detail="; ".join(issues))
    mases = sorted(scorecard.mase for scorecard in family_scorecards)
    median = _median(mases)
    maximum = mases[-1]
    if math.isnan(median) or math.isnan(maximum):
        issues.append("MASE contains NaN values")
    elif median > 0 and maximum / median > max_relative_mase:
        issues.append(
            f"max MASE ({maximum:.3f}) is {maximum / median:.1f}x the median ({median:.3f})"
        )
    return RobustnessCheck(
        check="robustness",
        passed=not issues,
        detail="; ".join(issues) if issues else f"max/median MASE ratio is {maximum / max(median, 1e-9):.2f}x",
    )


def check_review(
    *,
    human_approved: bool,
    approver: str | None = None,
) -> RobustnessCheck:
    """Validate the review gate.

    A human must explicitly approve the candidate. The harness
    will not promote an un-reviewed custom family. ``approver``
    is recorded in the detail for the audit trail.
    """
    if not human_approved:
        return RobustnessCheck(
            check="review",
            passed=False,
            detail="human approval not granted",
        )
    return RobustnessCheck(
        check="review",
        passed=True,
        detail=f"approved by {approver or 'unknown reviewer'}",
    )


# ---------------------------------------------------------------------------
# Custom-family escalation
# ---------------------------------------------------------------------------


def request_custom_family_attempt(
    run_id: str,
    proposed_family: str,
    *,
    reason: str,
) -> tuple[EscalationTracker, int]:
    """Request a new attempt at the custom-family escalation.

    Wraps the shared ``EscalationTracker`` with the
    ``forecasting_model`` layer. Returns the tracker and the
    attempt number. Raises ``EscalationLimitReached`` if the
    three-attempt cap has been hit.
    """
    tracker = EscalationTracker(run_id=run_id, layer="forecasting_model")
    attempt = tracker.request_code_attempt(reason)
    return tracker, attempt


def record_custom_family_failure(
    tracker: EscalationTracker,
    attempt: int,
    *,
    failed_gate: Gate,
    detail: str,
) -> None:
    """Record a failed attempt at the custom-family escalation.

    ``failed_gate`` is the first gate the attempt failed (the
    escalation aborts at the first failure). ``detail`` is a
    human-readable explanation that goes into the failure report.
    """
    reason = f"{failed_gate}: {detail}"
    tracker.record_failed_attempt(attempt, reason)


def declare_custom_family_failure(
    tracker: EscalationTracker,
    *,
    proposed_family: str,
    failed_gates: list[Gate],
    evidence: list[str],
    blocker: str,
    recommended_next_action: str,
) -> ModelFailureReport:
    """Produce the final ``ModelFailureReport`` after the three-attempt cap.

    Wraps the shared ``EscalationTracker.declare_failure_report``
    with the ``model_limitation`` problem kind and a Phase 4
    failure-report shape. The tracker is responsible for
    persisting the on-disk state; the returned report is what
    the cockpit surfaces.
    """
    shared_report: FailureReport = tracker.declare_failure_report(
        blocker=blocker,
        evidence=evidence,
        problem_kind=PROBLEM_KIND,
        recommended_next_action=recommended_next_action,
    )
    return ModelFailureReport(
        run_id=shared_report.run_id,
        proposed_family=proposed_family,
        status="blocked",
        blocker=blocker,
        evidence=evidence,
        attempts=shared_report.attempts,
        failed_reasons=shared_report.failed_reasons,
        failed_gates=failed_gates,
        recommended_next_action=recommended_next_action,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[midpoint])
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def gate_was_failed(checks: list[RobustnessCheck], gate: Gate) -> bool:
    """True if any check for ``gate`` failed."""
    return any(not check.passed for check in checks if check.check == gate)


def promote_custom_family(
    proposed_family: str,
    *,
    checks: list[RobustnessCheck],
) -> ModelFamilyName | None:
    """Promote a custom family once every gate has passed.

    Returns the canonical ``ModelFamilyName`` to add to the
    registry, or ``None`` if any gate failed. The actual registry
    edit is left to the harness / EDA / cockpit layer - the
    escalation layer is the gatekeeper, not the registry.
    """
    if not all(check.passed for check in checks):
        return None
    # The promoted family takes the proposed name verbatim. The
    # registry enforces the ``ModelFamilyName`` Literal so a
    # caller passing a typo gets a clear Pydantic error.
    return proposed_family  # type: ignore[return-value]


__all__ = [
    "GATE_ORDER",
    "PROBLEM_KIND",
    "check_data_contract",
    "check_backtest",
    "check_robustness",
    "check_review",
    "request_custom_family_attempt",
    "record_custom_family_failure",
    "declare_custom_family_failure",
    "promote_custom_family",
    "gate_was_failed",
    "EscalationLimitReached",
    "Gate",
    "RobustnessCheck",
]
