"""Tests for ``forecasting.config_escalation`` (Phase 4.1 CB5).

The loop iterates a ranked ``Proposal[]``, applies one config
proposal at a time, measures the MASE delta, and keeps or
reverts. Tests use a stub ``MeasureMASE`` so the harness is
not actually called — the loop's behaviour is pinned against a
deterministic per-proposal MASE sequence.

The test cases below cover every stop reason, every keep/kill
branch, and the per-knob attempt cap.
"""

from __future__ import annotations

import pytest

from forecasting.config_escalation import (
    ConfigApplicationError,
    ConfigAttemptResult,
    ConfigEscalationReport,
    apply_config_proposal,
    run_config_escalation,
)
from forecasting.contracts import (
    Claim,
    ConfigAction,
    FeatureFlags,
    ModelFamilyName,
    Proposal,
    ProposalTarget,
)
from forecasting.marginal_gain import MarginalGainConfig
from tests.stubs import StubMeasureMASE


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _flags(**overrides) -> FeatureFlags:
    """Build a FeatureFlags with sensible test defaults."""
    base = {
        "use_fourier": False,
        "use_lag_features": True,
        "use_promo_indicator": False,
        "fourier_terms": 3,
        "use_stockout_features": False,
        "use_hierarchy_features": False,
        "use_lifecycle_features": False,
        "use_intermittency_features": False,
    }
    base.update(overrides)
    return FeatureFlags(**base)


def _claim(series_key: str = "A") -> Claim:
    return Claim(
        claim_id="test-claim-id",
        claim="synthetic",
        verification_status="SUPPORTED",
        evidence_type="pattern",
        evidence_ref="abc",
        applies_to=series_key,
        downstream_impact="test",
        created_at="2026-06-17T00:00:00+00:00",
    )


def _config_proposal(
    action: ConfigAction,
    *,
    series_key: str = "A",
    payload: dict | None = None,
) -> Proposal:
    """Build a config-kind Proposal with the given action."""
    return Proposal(
        kind="config",
        config_action=action,
        action_payload=payload or {},
        target=ProposalTarget(scope="series", series_key=series_key, segment_id=None),
        expected_delta=0.1,
        rationale=f"enable {action}",
        evidence=_claim(series_key),
    )


# ---------------------------------------------------------------------------
# apply_config_proposal
# ---------------------------------------------------------------------------


def test_apply_enable_promo_indicator_flips_flag() -> None:
    """An enable_promo_indicator proposal flips the matching flag."""
    flags = _flags(use_promo_indicator=False)
    new_flags, family = apply_config_proposal(
        _config_proposal("enable_promo_indicator"),
        current_flags=flags,
        current_model_family="naive",
    )
    assert new_flags.use_promo_indicator is True
    assert family == "naive"


def test_apply_enable_stockout_features_flips_flag() -> None:
    flags = _flags(use_stockout_features=False)
    new_flags, family = apply_config_proposal(
        _config_proposal("enable_stockout_features"),
        current_flags=flags,
        current_model_family="naive",
    )
    assert new_flags.use_stockout_features is True
    assert family == "naive"


def test_apply_swap_model_family_returns_new_family() -> None:
    """A swap_model_family proposal returns the family from action_payload."""
    flags = _flags()
    new_flags, family = apply_config_proposal(
        _config_proposal("swap_model_family", payload={"family": "xgboost_global"}),
        current_flags=flags,
        current_model_family="naive",
    )
    assert family == "xgboost_global"
    assert new_flags == flags  # flags unchanged


def test_apply_swap_model_family_rejects_missing_payload() -> None:
    """A swap_model_family proposal without 'family' in payload raises."""
    with pytest.raises(ConfigApplicationError, match="swap_model_family requires"):
        apply_config_proposal(
            _config_proposal("swap_model_family", payload={}),
            current_flags=_flags(),
            current_model_family="naive",
        )


def test_apply_increase_fourier_terms_increments_and_caps() -> None:
    """increase_fourier_terms increments by 1, capped at 8."""
    flags = _flags(fourier_terms=3)
    new_flags, _ = apply_config_proposal(
        _config_proposal("increase_fourier_terms"),
        current_flags=flags,
        current_model_family="naive",
    )
    assert new_flags.fourier_terms == 4
    # Cap at 8: starting at 8, increment does not change the value.
    flags_at_cap = _flags(fourier_terms=8)
    new_flags_at_cap, _ = apply_config_proposal(
        _config_proposal("increase_fourier_terms"),
        current_flags=flags_at_cap,
        current_model_family="naive",
    )
    assert new_flags_at_cap.fourier_terms == 8


def test_apply_rejects_code_proposal() -> None:
    """A code-kind proposal raises (the loop only handles config)."""
    code = Proposal(
        kind="code",
        code_action="new_model_family",
        target=ProposalTarget(scope="series", series_key="A", segment_id=None),
        rationale="test",
        evidence=_claim("A"),
    )
    with pytest.raises(ConfigApplicationError, match="only handles kind=config"):
        apply_config_proposal(
            code, current_flags=_flags(), current_model_family="naive"
        )


# ---------------------------------------------------------------------------
# run_config_escalation — no config proposals
# ---------------------------------------------------------------------------


def test_loop_returns_no_config_proposals_when_input_empty() -> None:
    """No proposals at all -> no_config_proposals, no attempts."""
    report = run_config_escalation(
        run_id="r1",
        proposals=[],
        starting_flags=_flags(),
        starting_model_family="naive",
        starting_mase=1.5,
        measure_mase=StubMeasureMASE(value=0.0),
        config=MarginalGainConfig(),
    )
    assert report.stopped_reason == "no_config_proposals"
    assert report.attempts == []
    assert report.final_mase == 1.5


def test_loop_filters_out_code_proposals() -> None:
    """A list of only code proposals -> no_config_proposals, no attempts."""
    code = Proposal(
        kind="code",
        code_action="new_feature_family",
        target=ProposalTarget(scope="series", series_key="A", segment_id=None),
        rationale="test",
        evidence=_claim("A"),
    )
    report = run_config_escalation(
        run_id="r1",
        proposals=[code],
        starting_flags=_flags(),
        starting_model_family="naive",
        starting_mase=1.5,
        measure_mase=StubMeasureMASE(value=0.0),
        config=MarginalGainConfig(),
    )
    assert report.stopped_reason == "no_config_proposals"


# ---------------------------------------------------------------------------
# run_config_escalation — happy path
# ---------------------------------------------------------------------------


def test_loop_keeps_improving_proposal() -> None:
    """A proposal that drops MASE is kept; the new MASE is reported."""
    # Two proposals. First: enable_promo_indicator -> MASE 1.5 -> 1.3 (improvement).
    # Second: enable_stockout_features -> MASE 1.3 -> 1.25 (improvement).
    # Both are kept; final_mase = 1.25; final_flags has both flags on.
    proposals = [
        _config_proposal("enable_promo_indicator"),
        _config_proposal("enable_stockout_features"),
    ]
    measure = StubMeasureMASE(values=[1.3, 1.25])
    report = run_config_escalation(
        run_id="r1",
        proposals=proposals,
        starting_flags=_flags(),
        starting_model_family="naive",
        starting_mase=1.5,
        measure_mase=measure,
        config=MarginalGainConfig(),
    )
    assert report.final_mase == 1.25
    assert report.final_flags.use_promo_indicator is True
    assert report.final_flags.use_stockout_features is True
    assert report.stopped_reason == "config_exhausted"
    kept = [a for a in report.attempts if a.kept]
    assert len(kept) == 2


def test_loop_reverts_non_improving_proposal() -> None:
    """A proposal that doesn't improve MASE is reverted only when the
    patience floor is hit.

    With patience=3, a single non-improvement does not trigger
    the floor; the proposal is kept (the change is small but
    not noise). The test pins this: the first non-improvement
    is kept, the second is kept, the third triggers the floor
    and is reverted.

    The "revert" semantic the handoff doc describes is "the
    change is reverted when should_stop returns True", which
    is exactly the patience-floor trigger. A non-improvement
    that does not yet trigger the floor is *kept* — the DS
    takes the small win and continues.
    """
    # Three proposals, all tiny non-improvements. patience=4
    # means the 3rd attempt is the one that trips the floor
    # (history reaches 4 entries, the deltas are all below
    # threshold).
    proposals = [
        _config_proposal("enable_promo_indicator"),
        _config_proposal("enable_stockout_features"),
        _config_proposal("enable_hierarchy_features"),
    ]
    measure = StubMeasureMASE(values=[1.495, 1.494, 1.493])
    report = run_config_escalation(
        run_id="r1",
        proposals=proposals,
        starting_flags=_flags(),
        starting_model_family="naive",
        starting_mase=1.5,
        measure_mase=measure,
        config=MarginalGainConfig(min_mase_delta=0.02, patience=4, target_mase=0.5),
    )
    # First two kept (patience not yet hit). Third triggers
    # marginal_gain_floor -> reverted.
    kept = [a for a in report.attempts if a.kept]
    reverted = [a for a in report.attempts if not a.kept]
    assert len(kept) == 2
    assert len(reverted) == 1
    assert report.stopped_reason == "marginal_gain_floor"
    # The reverted proposal was the 3rd (hierarchy_features);
    # the kept flags reflect the first two changes.
    assert report.final_flags.use_promo_indicator is True
    assert report.final_flags.use_stockout_features is True
    # The hierarchy flag was the revert target; final state
    # reflects the *kept* flags only (the revert returned to
    # the pre-hierarchy state, which had use_hierarchy_features=False).
    assert report.final_flags.use_hierarchy_features is False


# ---------------------------------------------------------------------------
# run_config_escalation — stop conditions
# ---------------------------------------------------------------------------


def test_loop_stops_on_target_met() -> None:
    """When the new MASE <= target, the loop stops with target_met and keeps the change."""
    proposals = [
        _config_proposal("enable_promo_indicator"),
        _config_proposal("enable_stockout_features"),
    ]
    # First proposal hits target (1.5 -> 0.95, target = 1.0).
    measure = StubMeasureMASE(values=[0.95])
    report = run_config_escalation(
        run_id="r1",
        proposals=proposals,
        starting_flags=_flags(),
        starting_model_family="naive",
        starting_mase=1.5,
        measure_mase=measure,
        config=MarginalGainConfig(target_mase=1.0),
    )
    assert report.stopped_reason == "target_met"
    assert report.final_mase == 0.95
    assert report.final_flags.use_promo_indicator is True
    # Only the first proposal was tried (the loop stopped on target).
    assert len(report.attempts) == 1
    assert report.attempts[0].kept is True


def test_loop_stops_on_marginal_gain_floor() -> None:
    """After patience non-improvements, the loop stops with marginal_gain_floor."""
    proposals = [
        _config_proposal("enable_promo_indicator"),
        _config_proposal("enable_stockout_features"),
        _config_proposal("enable_hierarchy_features"),
    ]
    # All three proposals: MASE 1.5 -> 1.495 -> 1.494 -> 1.493.
    # patience=2, so after the 2nd non-improvement (1.494), the
    # 3rd attempt would push us past patience and stop. The
    # 3rd proposal is the one that triggers the floor.
    measure = StubMeasureMASE(values=[1.495, 1.494, 1.493])
    report = run_config_escalation(
        run_id="r1",
        proposals=proposals,
        starting_flags=_flags(),
        starting_model_family="naive",
        starting_mase=1.5,
        measure_mase=measure,
        config=MarginalGainConfig(min_mase_delta=0.02, patience=2, target_mase=0.5),
    )
    assert report.stopped_reason == "marginal_gain_floor"
    # The 3rd proposal triggered the stop, so it is the last
    # attempt and is NOT kept (the change is reverted).
    last = report.attempts[-1]
    assert last.kept is False
    # Final MASE is the starting MASE (no kept change at the end).
    assert report.final_mase == 1.5


def test_loop_stops_when_proposals_exhausted() -> None:
    """After all proposals tried without stop, stopped_reason is config_exhausted."""
    proposals = [
        _config_proposal("enable_promo_indicator"),
    ]
    measure = StubMeasureMASE(values=[1.30])  # one improvement, kept
    report = run_config_escalation(
        run_id="r1",
        proposals=proposals,
        starting_flags=_flags(),
        starting_model_family="naive",
        starting_mase=1.5,
        measure_mase=measure,
        config=MarginalGainConfig(),
    )
    assert report.stopped_reason == "config_exhausted"
    assert report.final_mase == 1.30


# ---------------------------------------------------------------------------
# run_config_escalation — per-knob cap
# ---------------------------------------------------------------------------


def test_loop_respects_per_knob_cap() -> None:
    """After 3 enable_promo_indicator attempts, the 4th is dropped as knob_cap."""
    # 4 proposals of the same kind. cap defaults to 3.
    proposals = [
        _config_proposal("enable_promo_indicator"),
        _config_proposal("enable_promo_indicator"),
        _config_proposal("enable_promo_indicator"),
        _config_proposal("enable_promo_indicator"),
    ]
    # MASE values: each proposal improves by 0.10 (kept), but
    # the 4th hits the cap and is dropped without measurement.
    measure = StubMeasureMASE(values=[1.40, 1.30, 1.20, 0.0])  # 4th value unused
    report = run_config_escalation(
        run_id="r1",
        proposals=proposals,
        starting_flags=_flags(),
        starting_model_family="naive",
        starting_mase=1.50,
        measure_mase=measure,
        config=MarginalGainConfig(),
    )
    # 3 kept + 1 cap-rejection. The cap-rejection is the last
    # attempt; stopped_reason is knob_cap.
    kept = [a for a in report.attempts if a.kept]
    capped = [a for a in report.attempts if a.error and "knob cap" in a.error]
    assert len(kept) == 3
    assert len(capped) == 1
    assert report.stopped_reason == "knob_cap"
    # The 4th proposal was not measured — the stub's calls
    # counter should be 3.
    assert measure.calls == 3


def test_loop_knob_caps_are_per_action_kind() -> None:
    """Two different actions have independent caps."""
    proposals = [
        _config_proposal("enable_promo_indicator"),
        _config_proposal("enable_promo_indicator"),
        _config_proposal("enable_promo_indicator"),
        _config_proposal("enable_stockout_features"),
        _config_proposal("enable_stockout_features"),
    ]
    # 3 promo + 2 stockout. Each kind has its own cap of 3.
    measure = StubMeasureMASE(values=[1.40, 1.30, 1.20, 1.10, 1.05])
    report = run_config_escalation(
        run_id="r1",
        proposals=proposals,
        starting_flags=_flags(),
        starting_model_family="naive",
        starting_mase=1.50,
        measure_mase=measure,
        config=MarginalGainConfig(),
    )
    # All 5 should be tried (3 promo + 2 stockout; neither
    # kind hits the cap of 3 on this run).
    kept = [a for a in report.attempts if a.kept]
    assert len(kept) == 5
    assert report.stopped_reason == "config_exhausted"


def test_loop_custom_knob_caps() -> None:
    """An explicit knob_caps dict overrides the default cap of 3."""
    proposals = [
        _config_proposal("enable_promo_indicator"),
        _config_proposal("enable_promo_indicator"),
    ]
    measure = StubMeasureMASE(values=[1.40, 1.30])
    report = run_config_escalation(
        run_id="r1",
        proposals=proposals,
        starting_flags=_flags(),
        starting_model_family="naive",
        starting_mase=1.50,
        measure_mase=measure,
        config=MarginalGainConfig(),
        knob_caps={"enable_promo_indicator": 1},
    )
    # With cap=1 for promo, the 2nd proposal is rejected.
    kept = [a for a in report.attempts if a.kept]
    capped = [a for a in report.attempts if a.error and "knob cap" in a.error]
    assert len(kept) == 1
    assert len(capped) == 1
    assert report.stopped_reason == "knob_cap"


# ---------------------------------------------------------------------------
# run_config_escalation — application errors
# ---------------------------------------------------------------------------


def test_loop_records_application_error_and_continues() -> None:
    """A proposal that raises ConfigApplicationError is recorded and the loop continues."""
    proposals = [
        # swap_model_family with empty payload -> ConfigApplicationError
        _config_proposal("swap_model_family", payload={}),
        # A normal proposal after -> still tried
        _config_proposal("enable_promo_indicator"),
    ]
    measure = StubMeasureMASE(values=[1.30])
    report = run_config_escalation(
        run_id="r1",
        proposals=proposals,
        starting_flags=_flags(),
        starting_model_family="naive",
        starting_mase=1.5,
        measure_mase=measure,
        config=MarginalGainConfig(),
    )
    assert len(report.attempts) == 2
    first = report.attempts[0]
    assert first.kept is False
    assert first.error is not None and "swap_model_family" in first.error
    second = report.attempts[1]
    assert second.kept is True
    assert report.final_mase == 1.30
    assert report.stopped_reason == "config_exhausted"
