"""Tests for the model escalation path (Phase 4)."""

from __future__ import annotations

import math

import pytest

from forecasting.code_escalation import EscalationLimitReached, FailureReport
from forecasting.contracts import (
    ModelFamilyName,
    ModelScorecard,
    RobustnessCheck,
)
from forecasting.model_escalation import (
    GATE_ORDER,
    PROBLEM_KIND,
    check_backtest,
    check_data_contract,
    check_review,
    check_robustness,
    declare_custom_family_failure,
    gate_was_failed,
    promote_custom_family,
    record_custom_family_failure,
    request_custom_family_attempt,
)


def _scorecard(*, family: ModelFamilyName, series_key: str, mase: float) -> ModelScorecard:
    return ModelScorecard(
        model_family=family,
        series_key=series_key,
        fold_cutoff="2024-01-01",
        horizon=2,
        forecast=[1.0, 1.0],
        actual=[1.0, 1.0],
        mae=abs(mase),
        rmse=abs(mase),
        mase=mase,
        bias=0.0,
    )


# ---------------------------------------------------------------------------
# Gate-order and problem-kind constants
# ---------------------------------------------------------------------------


def test_gate_order_is_data_contract_then_backtest_then_robustness_then_review() -> None:
    assert GATE_ORDER == ("data_contract", "backtest", "robustness", "review")


def test_problem_kind_is_model_limitation() -> None:
    assert PROBLEM_KIND == "model_limitation"


# ---------------------------------------------------------------------------
# check_data_contract
# ---------------------------------------------------------------------------


def test_check_data_contract_passes_for_valid_forecast() -> None:
    check = check_data_contract(forecast=[1.0, 2.0, 3.0], actual=[1.0, 2.0, 3.0], horizon=3)
    assert check.passed is True
    assert check.check == "data_contract"


def test_check_data_contract_fails_for_wrong_length() -> None:
    check = check_data_contract(forecast=[1.0, 2.0], actual=[1.0, 2.0, 3.0], horizon=3)
    assert check.passed is False
    assert "length" in check.detail


def test_check_data_contract_fails_for_nan() -> None:
    check = check_data_contract(forecast=[1.0, float("nan")], actual=[1.0, 2.0], horizon=2)
    assert check.passed is False
    assert "not finite" in check.detail


def test_check_data_contract_fails_for_infinity() -> None:
    check = check_data_contract(forecast=[1.0, float("inf")], actual=[1.0, 2.0], horizon=2)
    assert check.passed is False


def test_check_data_contract_allows_mismatched_actual_length() -> None:
    # Inference mode: actuals are optional, length is only
    # checked when present.
    check = check_data_contract(forecast=[1.0, 2.0], actual=None, horizon=2)
    assert check.passed is True


# ---------------------------------------------------------------------------
# check_backtest
# ---------------------------------------------------------------------------


def test_check_backtest_passes_for_finite_non_negative_mase() -> None:
    check = check_backtest(forecast=[1.0, 2.0], actual=[1.5, 2.5], mase=0.8)
    assert check.passed is True


def test_check_backtest_fails_for_nan_mase() -> None:
    check = check_backtest(forecast=[1.0, 2.0], actual=[1.5, 2.5], mase=float("nan"))
    assert check.passed is False
    assert "not finite" in check.detail


def test_check_backtest_fails_for_negative_mase() -> None:
    check = check_backtest(forecast=[1.0, 2.0], actual=[1.5, 2.5], mase=-0.5)
    assert check.passed is False
    assert "negative" in check.detail


def test_check_backtest_fails_for_empty_actuals() -> None:
    check = check_backtest(forecast=[1.0, 2.0], actual=[], mase=0.5)
    assert check.passed is False
    assert "no actuals" in check.detail


# ---------------------------------------------------------------------------
# check_robustness
# ---------------------------------------------------------------------------


def test_check_robustness_passes_for_stable_family() -> None:
    scorecards = [
        _scorecard(family="xgboost_global", series_key="A", mase=1.0),
        _scorecard(family="xgboost_global", series_key="B", mase=1.1),
        _scorecard(family="xgboost_global", series_key="C", mase=0.9),
    ]
    check = check_robustness(scorecards=scorecards, family="xgboost_global")
    assert check.passed is True


def test_check_robustness_fails_for_blowup() -> None:
    scorecards = [
        _scorecard(family="xgboost_global", series_key="A", mase=1.0),
        _scorecard(family="xgboost_global", series_key="B", mase=1.1),
        # 100x the median is way past the 5x cap.
        _scorecard(family="xgboost_global", series_key="C", mase=100.0),
    ]
    check = check_robustness(scorecards=scorecards, family="xgboost_global")
    assert check.passed is False
    assert "max MASE" in check.detail


def test_check_robustness_fails_when_family_has_no_scorecards() -> None:
    check = check_robustness(scorecards=[], family="xgboost_global")
    assert check.passed is False


def test_check_robustness_ignores_other_families() -> None:
    scorecards = [
        _scorecard(family="xgboost_global", series_key="A", mase=1.0),
        _scorecard(family="naive", series_key="A", mase=1000.0),  # not the family under test
    ]
    check = check_robustness(scorecards=scorecards, family="xgboost_global")
    assert check.passed is True


def test_check_robustness_fails_on_nan_mase() -> None:
    scorecards = [
        _scorecard(family="xgboost_global", series_key="A", mase=1.0),
        _scorecard(family="xgboost_global", series_key="B", mase=float("nan")),
    ]
    check = check_robustness(scorecards=scorecards, family="xgboost_global")
    assert check.passed is False


# ---------------------------------------------------------------------------
# check_review
# ---------------------------------------------------------------------------


def test_check_review_fails_when_not_approved() -> None:
    check = check_review(human_approved=False)
    assert check.passed is False
    assert "not granted" in check.detail


def test_check_review_passes_with_approver() -> None:
    check = check_review(human_approved=True, approver="planner@team")
    assert check.passed is True
    assert "planner@team" in check.detail


# ---------------------------------------------------------------------------
# gate_was_failed
# ---------------------------------------------------------------------------


def test_gate_was_failed_returns_true_when_any_check_failed() -> None:
    checks = [
        RobustnessCheck(check="data_contract", passed=True, detail="ok"),
        RobustnessCheck(check="backtest", passed=False, detail="bad mase"),
    ]
    assert gate_was_failed(checks, "backtest") is True


def test_gate_was_failed_returns_false_when_all_passed() -> None:
    checks = [
        RobustnessCheck(check="data_contract", passed=True, detail="ok"),
        RobustnessCheck(check="backtest", passed=True, detail="ok"),
    ]
    assert gate_was_failed(checks, "backtest") is False


def test_gate_was_failed_returns_false_when_gate_not_present() -> None:
    assert gate_was_failed([], "backtest") is False


# ---------------------------------------------------------------------------
# promote_custom_family
# ---------------------------------------------------------------------------


def test_promote_custom_family_returns_none_when_any_gate_failed() -> None:
    checks = [
        RobustnessCheck(check="data_contract", passed=True, detail="ok"),
        RobustnessCheck(check="backtest", passed=False, detail="bad"),
    ]
    assert promote_custom_family("my_lstm", checks=checks) is None


def test_promote_custom_family_returns_name_when_all_gates_passed() -> None:
    checks = [
        RobustnessCheck(check="data_contract", passed=True, detail="ok"),
        RobustnessCheck(check="backtest", passed=True, detail="ok"),
        RobustnessCheck(check="robustness", passed=True, detail="ok"),
        RobustnessCheck(check="review", passed=True, detail="ok"),
    ]
    assert promote_custom_family("my_lstm", checks=checks) == "my_lstm"


# ---------------------------------------------------------------------------
# request / record / declare lifecycle (uses shared EscalationTracker)
# ---------------------------------------------------------------------------


def test_request_returns_attempt_number_in_range_one_to_three(tmp_outputs) -> None:
    run_id = "test-esc-1"
    tracker, attempt = request_custom_family_attempt(run_id, "my_lstm", reason="gap")
    assert attempt == 1
    # The shared tracker is the one keyed on "forecasting_model".
    assert tracker.layer == "forecasting_model"


def test_three_attempts_succeed_then_fourth_raises(tmp_outputs) -> None:
    run_id = "test-esc-2"
    for expected in (1, 2, 3):
        tracker, attempt = request_custom_family_attempt(run_id, "my_lstm", reason="gap")
        assert attempt == expected
        record_custom_family_failure(
            tracker,
            attempt,
            failed_gate="data_contract",
            detail=f"attempt {expected} failed",
        )
    with pytest.raises(EscalationLimitReached):
        request_custom_family_attempt(run_id, "my_lstm", reason="cap reached")


def test_declare_failure_report_wraps_shared_failure_with_phase4_shape(tmp_outputs) -> None:
    run_id = "test-esc-3"
    failed_gates: list = []
    for _ in range(3):
        tracker, attempt = request_custom_family_attempt(run_id, "my_lstm", reason="gap")
        record_custom_family_failure(
            tracker,
            attempt,
            failed_gate="backtest",
            detail="mase blew up",
        )
        failed_gates.append("backtest")
    # Reconstruct tracker so we can call declare on it.
    from forecasting.code_escalation import EscalationTracker
    tracker = EscalationTracker(run_id=run_id, layer="forecasting_model")
    report = declare_custom_family_failure(
        tracker,
        proposed_family="my_lstm",
        failed_gates=failed_gates,
        evidence=["mase=42 on fold 1", "mase=51 on fold 2", "mase=38 on fold 3"],
        blocker="MASE exceeds threshold on every fold",
        recommended_next_action="Revert to governed families and file a learning",
    )
    assert report.status == "blocked"
    assert report.attempts == 3
    assert report.failed_gates == failed_gates
    assert report.proposed_family == "my_lstm"
    assert "mase=42 on fold 1" in report.evidence
    assert report.recommended_next_action.startswith("Revert")


def test_declare_failure_report_persists_state_to_disk(tmp_outputs) -> None:
    run_id = "test-esc-4"
    for _ in range(3):
        tracker, attempt = request_custom_family_attempt(run_id, "my_lstm", reason="gap")
        record_custom_family_failure(
            tracker,
            attempt,
            failed_gate="review",
            detail="planner rejected",
        )
    from forecasting.code_escalation import EscalationTracker
    tracker = EscalationTracker(run_id=run_id, layer="forecasting_model")
    declare_custom_family_failure(
        tracker,
        proposed_family="my_lstm",
        failed_gates=["review"],
        evidence=["rejected at 2026-06-14"],
        blocker="Reviewer did not approve",
        recommended_next_action="Gather more evidence and re-submit",
    )
    # A fresh tracker instance should see the failure report on
    # disk with status="blocked".
    fresh = EscalationTracker(run_id=run_id, layer="forecasting_model")
    assert fresh.status == "blocked"
