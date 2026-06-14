import pytest

from forecasting.code_escalation import (
    ALLOWED_LAYERS,
    EscalationLimitReached,
    EscalationTracker,
    ProblemKind,
)
from forecasting.run_state import create_run_state


def test_tracker_starts_in_toolbox_with_zero_attempts(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    tracker = EscalationTracker(run_id=run_id, layer="eda")

    assert tracker.run_id == run_id
    assert tracker.layer == "eda"
    assert tracker.attempts == 0
    assert tracker.status == "toolbox"


@pytest.mark.parametrize("layer", sorted(ALLOWED_LAYERS))
def test_allowed_layers_can_create_trackers(layer, run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    tracker = EscalationTracker(run_id=run_id, layer=layer)

    assert tracker.layer == layer


def test_invalid_layer_is_rejected():
    with pytest.raises(ValueError, match="invalid escalation layer"):
        EscalationTracker(run_id="run-1", layer="reporting")


def test_request_code_attempt_returns_attempt_numbers_up_to_three(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    tracker = EscalationTracker(run_id=run_id, layer="forecasting_model")

    assert tracker.request_code_attempt("baseline model failed") == 1
    assert tracker.request_code_attempt("need feature transform") == 2
    assert tracker.request_code_attempt("diagnostic fallback") == 3
    assert tracker.attempts == 3


def test_request_code_attempt_after_three_raises_limit_reached(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    tracker = EscalationTracker(run_id=run_id, layer="feature_engineering")

    for reason in ["first", "second", "third"]:
        tracker.request_code_attempt(reason)

    with pytest.raises(EscalationLimitReached, match="3 code-generation attempts"):
        tracker.request_code_attempt("fourth")

    assert tracker.attempts == 3


def test_record_failed_attempt_keeps_failed_reasons_by_attempt_number(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    tracker = EscalationTracker(run_id=run_id, layer="schema_mapping")
    first = tracker.request_code_attempt("infer date column")
    second = tracker.request_code_attempt("parse custom format")

    tracker.record_failed_attempt(first, "missing fiscal calendar")
    tracker.record_failed_attempt(second, "ambiguous date tokens")

    assert tracker.failed_reasons == {
        1: "missing fiscal calendar",
        2: "ambiguous date tokens",
    }


def test_failure_report_requires_exactly_three_recorded_failed_attempts(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    tracker = EscalationTracker(run_id=run_id, layer="diagnostics")
    for reason in ["first", "second", "third"]:
        attempt = tracker.request_code_attempt(reason)
        if attempt < 3:
            tracker.record_failed_attempt(attempt, f"failed {attempt}")

    with pytest.raises(ValueError, match="exactly three recorded failed attempts"):
        tracker.declare_failure_report(
            blocker="diagnostic cannot isolate driver",
            evidence=["ranked features conflict"],
            problem_kind="model_limitation",
            recommended_next_action="ask user to narrow diagnostic scope",
        )


def test_declare_failure_report_returns_structured_blocked_report(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    tracker = EscalationTracker(run_id=run_id, layer="canonical_table")
    for reason in ["normalize item keys", "join locations", "resolve duplicates"]:
        attempt = tracker.request_code_attempt(reason)
        tracker.record_failed_attempt(attempt, f"attempt {attempt} failed")

    report = tracker.declare_failure_report(
        blocker="canonical table cannot be built deterministically",
        evidence=[
            "duplicate SKU rows have conflicting units",
            "location hierarchy is missing parent nodes",
        ],
        problem_kind="business_semantics",
        recommended_next_action="request business owner resolution for duplicate SKU units",
    )

    assert report.status == "blocked"
    assert report.run_id == run_id
    assert report.layer == "canonical_table"
    assert report.blocker == "canonical table cannot be built deterministically"
    assert report.evidence == [
        "duplicate SKU rows have conflicting units",
        "location hierarchy is missing parent nodes",
    ]
    assert report.attempts == 3
    assert report.failed_reasons == {
        1: "attempt 1 failed",
        2: "attempt 2 failed",
        3: "attempt 3 failed",
    }
    assert report.problem_kind == "business_semantics"
    assert report.recommended_next_action == (
        "request business owner resolution for duplicate SKU units"
    )
    assert tracker.status == "blocked"


@pytest.mark.parametrize(
    "problem_kind",
    ["data", "business_semantics", "platform_limitation", "model_limitation"],
)
def test_all_problem_kinds_are_allowed(problem_kind: ProblemKind, run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    tracker = EscalationTracker(run_id=run_id, layer="eda")
    for reason in ["first", "second", "third"]:
        attempt = tracker.request_code_attempt(reason)
        tracker.record_failed_attempt(attempt, reason)

    report = tracker.declare_failure_report(
        blocker="blocked",
        evidence=["evidence"],
        problem_kind=problem_kind,
        recommended_next_action="next action",
    )

    assert report.problem_kind == problem_kind


def test_invalid_problem_kind_is_rejected(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    tracker = EscalationTracker(run_id=run_id, layer="eda")
    for reason in ["first", "second", "third"]:
        attempt = tracker.request_code_attempt(reason)
        tracker.record_failed_attempt(attempt, reason)

    with pytest.raises(ValueError, match="invalid problem kind"):
        tracker.declare_failure_report(
            blocker="blocked",
            evidence=["evidence"],
            problem_kind="process",
            recommended_next_action="next action",
        )


def test_escalation_attempts_persist_across_tracker_instances(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    first_tracker = EscalationTracker(run_id=run_id, layer="feature_engineering")
    first_tracker.request_code_attempt("standard feature factory cannot express fiscal calendar")
    first_tracker.record_failed_attempt(1, "fiscal periods missing from source data")

    resumed_tracker = EscalationTracker(run_id=run_id, layer="feature_engineering")

    assert resumed_tracker.attempts == 1
    assert resumed_tracker.status == "code_escalation"
    assert resumed_tracker.failed_reasons == {
        1: "fiscal periods missing from source data",
    }
    assert resumed_tracker.request_code_attempt("try customer adapter") == 2


def test_escalation_limit_persists_across_tracker_instances(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    tracker = EscalationTracker(run_id=run_id, layer="eda")
    for reason in ["first", "second", "third"]:
        tracker.request_code_attempt(reason)

    resumed_tracker = EscalationTracker(run_id=run_id, layer="eda")

    with pytest.raises(EscalationLimitReached):
        resumed_tracker.request_code_attempt("fourth")
