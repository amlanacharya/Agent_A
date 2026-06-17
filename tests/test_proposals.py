"""Tests for ``forecasting.proposals`` (Phase 4.1 CB3).

CB3 is the LLM-backed judgement step in the two-path escalation
loop. The tests below use a deterministic stub LLM (no real API
calls) and pin the contract:

* free-text action names from the LLM are rejected;
* proposals without evidence are rejected;
* ranking is deterministic given the LLM output;
* the prompt is reproducible (same input -> same prompt string);
* every emitted proposal carries a pattern-backed Claim.
"""

from __future__ import annotations

import json
import math
import uuid

import numpy as np
import pandas as pd
import pytest

from forecasting.contracts import (
    Claim,
    CodeAction,
    ConfigAction,
    ModelScorecard,
    Proposal,
    ProposalKind,
    ProposalTarget,
    ResidualDecomposition,
    ResidualPattern,
    ResidualPatternHit,
    ResidualStats,
)
from forecasting.proposals import (
    ProposeLLMCallable,
    propose_feature_changes,
)
from forecasting.residual_analysis import decompose_residuals


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _scorecard(series_key: str, forecast: list[float], actual: list[float]) -> ModelScorecard:
    """Build a ModelScorecard for tests (same helper as test_residual_analysis)."""
    f = np.asarray(forecast, dtype=float)
    a = np.asarray(actual, dtype=float)
    residuals = a - f
    mae = float(np.mean(np.abs(residuals))) if len(f) else 0.0
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if len(f) else 0.0
    bias = float(np.mean(residuals)) if len(f) else 0.0
    return ModelScorecard(
        model_family="naive",
        series_key=series_key,
        fold_cutoff="2024-01-01T00:00:00",
        horizon=len(forecast),
        forecast=forecast,
        actual=actual,
        mae=mae,
        rmse=rmse,
        mase=mae,
        bias=bias,
    )


def _decomp_with_pattern(series_key: str, pattern: ResidualPattern, severity: float) -> ResidualDecomposition:
    """Build a ResidualDecomposition carrying a single pattern hit.

    Convenience for tests that want to drive the proposal tool
    with a known input, bypassing the real CB2 math.
    """
    stats = ResidualStats(
        series_key=series_key,
        n=10,
        residual_mean=0.0,
        residual_std=1.0,
        mae=0.5,
    )
    return ResidualDecomposition(
        series_key=series_key,
        fold_cutoff="2024-01-01T00:00:00",
        stats=stats,
        patterns=[ResidualPatternHit(pattern=pattern, severity=severity, detail="synthetic")],
    )


def _stub_llm(candidates: list[dict]) -> ProposeLLMCallable:
    """Build a deterministic stub LLM that returns the given candidates."""
    class _Stub:
        def __init__(self, payload: list[dict]) -> None:
            self.payload = payload
            self.calls: list[str] = []

        def propose(self, prompt: str) -> list[dict]:
            self.calls.append(prompt)
            return self.payload
    return _Stub(candidates)


def _claim(series_key: str = "A") -> Claim:
    """Build a minimal Claim for embedding in a Proposal-shaped dict."""
    return Claim(
        claim_id=str(uuid.uuid4()),
        claim="synthetic claim",
        verification_status="SUPPORTED",
        evidence_type="pattern",
        evidence_ref="abc123",
        applies_to=series_key,
        downstream_impact="test",
        created_at="2026-06-17T00:00:00+00:00",
    )


def _claim_dict(series_key: str = "A") -> dict:
    """Return ``_claim(...).model_dump()`` for use inside LLM stub payloads."""
    return _claim(series_key).model_dump()


# ---------------------------------------------------------------------------
# Empty / no-pattern inputs
# ---------------------------------------------------------------------------


def test_propose_returns_empty_list_when_no_decompositions() -> None:
    """No decompositions and LLM returns no candidates -> empty Proposal[]."""
    llm = _stub_llm([])  # LLM has nothing to emit
    assert propose_feature_changes([], [], llm=llm) == []


def test_propose_returns_empty_list_when_decompositions_have_no_patterns() -> None:
    """Decompositions present but no patterns -> empty Proposal[] (LLM still called)."""
    stats = ResidualStats(series_key="A", n=10, residual_mean=0.0, residual_std=0.0, mae=0.0)
    decomp = ResidualDecomposition(series_key="A", fold_cutoff="2024-01-01T00:00:00", stats=stats, patterns=[])
    llm = _stub_llm([])
    result = propose_feature_changes([], [decomp], llm=llm)
    assert result == []
    # The LLM was called (the prompt is the audit artifact) but it
    # had no candidates to emit.
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_propose_emits_proposal_for_top_pattern() -> None:
    """A top-1 pattern produces a Proposal that the LLM-named candidate maps to."""
    decomp = _decomp_with_pattern("A", "PROMO_RESIDUAL_SPIKE", 0.8)
    candidate = {
        "kind": "config",
        "config_action": "enable_promo_indicator",
        "expected_delta": 0.3,
        "rationale": "promo residuals spiked",
        "evidence": _claim_dict("A"),
        "target": {"scope": "series", "series_key": "A", "segment_id": None},
    }
    llm = _stub_llm([candidate])
    result = propose_feature_changes([], [decomp], llm=llm)
    assert len(result) == 1
    assert result[0].config_action == "enable_promo_indicator"
    assert result[0].target.series_key == "A"


def test_propose_emits_code_proposal() -> None:
    """A code-kind proposal is validated and emitted."""
    decomp = _decomp_with_pattern("A", "HETEROSCEDASTIC_RESIDUAL", 0.9)
    candidate = {
        "kind": "code",
        "code_action": "new_model_family",
        "expected_delta": 0.5,
        "rationale": "intermittency is the model-class problem",
        "evidence": _claim_dict("A"),
        "target": {"scope": "series", "series_key": "A", "segment_id": None},
        "action_payload": {"family_name": "lgbm"},
    }
    llm = _stub_llm([candidate])
    result = propose_feature_changes([], [decomp], llm=llm)
    assert len(result) == 1
    assert result[0].kind == "code"
    assert result[0].code_action == "new_model_family"


# ---------------------------------------------------------------------------
# Validation rejects bad LLM output
# ---------------------------------------------------------------------------


def test_propose_rejects_unknown_action() -> None:
    """A free-text action name is rejected by Pydantic Literal validation."""
    decomp = _decomp_with_pattern("A", "PROMO_RESIDUAL_SPIKE", 0.8)
    candidate = {
        "kind": "config",
        "config_action": "enable_warp_drive",  # not in ConfigAction Literal
        "expected_delta": 0.3,
        "rationale": "test",
        "evidence": _claim_dict("A"),
        "target": {"scope": "series", "series_key": "A", "segment_id": None},
    }
    llm = _stub_llm([candidate])
    result = propose_feature_changes([], [decomp], llm=llm)
    assert result == []


def test_propose_rejects_missing_evidence() -> None:
    """A proposal without a Claim is dropped (the Proposal contract requires one)."""
    decomp = _decomp_with_pattern("A", "PROMO_RESIDUAL_SPIKE", 0.8)
    candidate = {
        "kind": "config",
        "config_action": "enable_promo_indicator",
        "expected_delta": 0.3,
        "rationale": "test",
        # no evidence
        "target": {"scope": "series", "series_key": "A", "segment_id": None},
    }
    llm = _stub_llm([candidate])
    result = propose_feature_changes([], [decomp], llm=llm)
    assert result == []


def test_propose_rejects_malformed_claim_in_evidence() -> None:
    """An evidence claim with the wrong evidence_type is rejected at validate."""
    decomp = _decomp_with_pattern("A", "PROMO_RESIDUAL_SPIKE", 0.8)
    bad_claim = _claim_dict("A")
    bad_claim["evidence_type"] = "nope"  # not a valid EvidenceType value
    candidate = {
        "kind": "config",
        "config_action": "enable_promo_indicator",
        "expected_delta": 0.3,
        "rationale": "test",
        "evidence": bad_claim,
        "target": {"scope": "series", "series_key": "A", "segment_id": None},
    }
    llm = _stub_llm([candidate])
    result = propose_feature_changes([], [decomp], llm=llm)
    assert result == []


def test_propose_rejects_non_dict_candidates() -> None:
    """Non-dict entries in the LLM's list output are silently dropped."""
    decomp = _decomp_with_pattern("A", "PROMO_RESIDUAL_SPIKE", 0.8)
    llm = _stub_llm([
        "not a dict",
        42,
        None,
        {
            "kind": "config",
            "config_action": "enable_promo_indicator",
            "expected_delta": 0.3,
            "rationale": "test",
            "evidence": _claim_dict("A"),
            "target": {"scope": "series", "series_key": "A", "segment_id": None},
        },
    ])
    result = propose_feature_changes([], [decomp], llm=llm)
    # Only the last (valid) dict survives.
    assert len(result) == 1
    assert result[0].config_action == "enable_promo_indicator"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_propose_prompt_is_deterministic_given_same_input() -> None:
    """Two calls with the same input produce the same prompt string (audit artifact)."""
    decomp_a = _decomp_with_pattern("A", "PROMO_RESIDUAL_SPIKE", 0.8)
    decomp_b = _decomp_with_pattern("B", "STOCKOUT_RESIDUAL_SPIKE", 0.6)
    llm1 = _stub_llm([])
    llm2 = _stub_llm([])
    propose_feature_changes([], [decomp_a, decomp_b], llm=llm1, segment_id="G1")
    propose_feature_changes([], [decomp_a, decomp_b], llm=llm2, segment_id="G1")
    assert llm1.calls[0] == llm2.calls[0]


def test_propose_prompt_includes_segment_id_when_set() -> None:
    """Segment-level runs declare the scope in the prompt."""
    decomp = _decomp_with_pattern("A", "PROMO_RESIDUAL_SPIKE", 0.5)
    llm = _stub_llm([])
    propose_feature_changes([], [decomp], llm=llm, segment_id="G1")
    assert "segment_id=G1" in llm.calls[0]
    assert "per-series" not in llm.calls[0]


def test_propose_prompt_omits_segment_id_when_unset() -> None:
    """Series-level runs say so in the prompt."""
    decomp = _decomp_with_pattern("A", "PROMO_RESIDUAL_SPIKE", 0.5)
    llm = _stub_llm([])
    propose_feature_changes([], [decomp], llm=llm)
    assert "per-series" in llm.calls[0]
    assert "segment_id=" not in llm.calls[0]


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_propose_ranks_by_pattern_severity_x_expected_delta() -> None:
    """Two proposals with different (severity, expected_delta) come out in rank order."""
    # Series A has severity 0.9, series B has severity 0.5.
    decomp_a = _decomp_with_pattern("A", "PROMO_RESIDUAL_SPIKE", 0.9)
    decomp_b = _decomp_with_pattern("B", "PROMO_RESIDUAL_SPIKE", 0.5)
    candidate_a = {
        "kind": "config",
        "config_action": "enable_promo_indicator",
        "expected_delta": 0.3,
        "rationale": "A",
        "evidence": _claim_dict("A"),
        "target": {"scope": "series", "series_key": "A", "segment_id": None},
    }
    candidate_b = {
        "kind": "config",
        "config_action": "enable_promo_indicator",
        "expected_delta": 0.3,
        "rationale": "B",
        "evidence": _claim_dict("B"),
        "target": {"scope": "series", "series_key": "B", "segment_id": None},
    }
    # Feed B first to ensure the sort is by rank, not insertion.
    llm = _stub_llm([candidate_b, candidate_a])
    result = propose_feature_changes([], [decomp_a, decomp_b], llm=llm)
    assert len(result) == 2
    assert result[0].target.series_key == "A"  # higher rank first
    assert result[1].target.series_key == "B"


def test_propose_rank_is_stable_for_ties() -> None:
    """Two proposals with the same rank key come out in a stable order across calls."""
    decomp_a = _decomp_with_pattern("A", "PROMO_RESIDUAL_SPIKE", 0.5)
    decomp_b = _decomp_with_pattern("B", "PROMO_RESIDUAL_SPIKE", 0.5)
    cand_a = {
        "kind": "config", "config_action": "enable_promo_indicator",
        "expected_delta": 0.3, "rationale": "A", "evidence": _claim_dict("A"),
        "target": {"scope": "series", "series_key": "A", "segment_id": None},
    }
    cand_b = {
        "kind": "config", "config_action": "enable_promo_indicator",
        "expected_delta": 0.3, "rationale": "B", "evidence": _claim_dict("B"),
        "target": {"scope": "series", "series_key": "B", "segment_id": None},
    }
    llm1 = _stub_llm([cand_a, cand_b])
    llm2 = _stub_llm([cand_b, cand_a])
    r1 = propose_feature_changes([], [decomp_a, decomp_b], llm=llm1)
    r2 = propose_feature_changes([], [decomp_a, decomp_b], llm=llm2)
    # Same order regardless of LLM input order.
    assert [p.target.series_key for p in r1] == [p.target.series_key for p in r2]


# ---------------------------------------------------------------------------
# Aggregation across folds
# ---------------------------------------------------------------------------


def test_propose_takes_max_severity_across_folds() -> None:
    """When two folds report the same pattern, the max severity wins."""
    stats = ResidualStats(series_key="A", n=10, residual_mean=0.0, residual_std=1.0, mae=0.5)
    fold1 = ResidualDecomposition(
        series_key="A", fold_cutoff="2024-01-01T00:00:00", stats=stats,
        patterns=[ResidualPatternHit(pattern="PROMO_RESIDUAL_SPIKE", severity=0.4, detail="fold1")],
    )
    fold2 = ResidualDecomposition(
        series_key="A", fold_cutoff="2024-02-01T00:00:00", stats=stats,
        patterns=[ResidualPatternHit(pattern="PROMO_RESIDUAL_SPIKE", severity=0.8, detail="fold2")],
    )
    candidate = {
        "kind": "config", "config_action": "enable_promo_indicator",
        "expected_delta": 0.3, "rationale": "A", "evidence": _claim_dict("A"),
        "target": {"scope": "series", "series_key": "A", "segment_id": None},
    }
    llm = _stub_llm([candidate])
    result = propose_feature_changes([], [fold1, fold2], llm=llm)
    assert len(result) == 1
    # The rank key uses the max severity (0.8), so the prompt
    # should report severity=0.80 for the series (not 0.40 or any
    # average).
    prompt = llm.calls[0]
    # The prompt renders the pattern in its Literal form (uppercase,
    # as it appears in ResidualPattern). The max-severity rule
    # means the prompt reports 0.80 (the max across the two folds),
    # not 0.40 (the per-fold value).
    assert "PROMO_RESIDUAL_SPIKE=0.80" in prompt


# ---------------------------------------------------------------------------
# Segment-level targeting
# ---------------------------------------------------------------------------


def test_propose_segment_level_target_uses_segment_id() -> None:
    """When segment_id is set, the LLM's target_id is forwarded as segment_id."""
    decomp = _decomp_with_pattern("A", "PROMO_RESIDUAL_SPIKE", 0.7)
    candidate = {
        "kind": "config", "config_action": "enable_promo_indicator",
        "expected_delta": 0.3, "rationale": "all of G1", "evidence": _claim_dict("A"),
        "target": {"scope": "segment", "series_key": None, "segment_id": "G1"},
    }
    llm = _stub_llm([candidate])
    result = propose_feature_changes([], [decomp], llm=llm, segment_id="G1")
    assert len(result) == 1
    assert result[0].target.scope == "segment"
    assert result[0].target.segment_id == "G1"


# ---------------------------------------------------------------------------
# Every emitted proposal has an evidence Claim
# ---------------------------------------------------------------------------


def test_propose_every_emitted_proposal_has_evidence_claim() -> None:
    """Defensive sweep: across a mixed input, every emitted proposal has evidence."""
    decomp_a = _decomp_with_pattern("A", "PROMO_RESIDUAL_SPIKE", 0.7)
    decomp_b = _decomp_with_pattern("B", "STOCKOUT_RESIDUAL_SPIKE", 0.5)
    cand_a = {
        "kind": "config", "config_action": "enable_promo_indicator",
        "expected_delta": 0.3, "rationale": "A", "evidence": _claim_dict("A"),
        "target": {"scope": "series", "series_key": "A", "segment_id": None},
    }
    cand_b = {
        "kind": "config", "config_action": "enable_stockout_features",
        "expected_delta": 0.4, "rationale": "B", "evidence": _claim_dict("B"),
        "target": {"scope": "series", "series_key": "B", "segment_id": None},
    }
    cand_no_evidence = {
        "kind": "config", "config_action": "enable_hierarchy_features",
        "expected_delta": 0.2, "rationale": "C",  # no evidence
        "target": {"scope": "series", "series_key": "A", "segment_id": None},
    }
    llm = _stub_llm([cand_a, cand_b, cand_no_evidence])
    result = propose_feature_changes([], [decomp_a, decomp_b], llm=llm)
    assert len(result) == 2
    for p in result:
        # The Proposal contract requires an evidence claim; the test
        # is the defensive sweep that no proposal slipped through
        # with a missing or empty claim.
        assert p.evidence is not None
        assert p.evidence.claim_id != ""
        assert p.evidence.evidence_type == "pattern"
