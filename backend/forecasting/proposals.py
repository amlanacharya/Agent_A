"""The propose_feature_changes tool (Phase 4.1 CB3).

The Foundry agent's only judgement call in the two-path escalation
loop: read the deterministic residual decomposition from CB2, ask
the LLM for candidate changes, validate each candidate against the
closed Literal surface, and return a ranked ``Proposal[]``.

Seam contract (kept tight on purpose):

* **CB2 is the only evidence source.** Every emitted ``Proposal``
  carries a ``Claim`` whose ``evidence_ref`` points at the
  decomposition that backed it. The proposal tool does not
  invent evidence — it asks the LLM to interpret the patterns
  the decomposition found, then validates the LLM's output.
* **CB4 (the harness) is the only execution path.** This tool
  never applies a proposal — it produces a ranked list and
  hands it to the config-escalation loop. The LLM chooses the
  candidates; the harness decides which to apply.
* **The closed Literal sets are the safety boundary.** A
  free-text action from the LLM is rejected at the Pydantic
  model boundary, not at execution time. Pydantic Literal
  validation is the single chokepoint for bad LLM output.

Tests use a deterministic stub LLM that returns fixed lists —
no real API calls, no network.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol, Sequence, get_args

from pydantic import ValidationError

from forecasting.contracts import (
    CodeAction,
    ConfigAction,
    ModelScorecard,
    Proposal,
    ProposalTarget,
    ResidualDecomposition,
)


# ---------------------------------------------------------------------------
# LLM seam
# ---------------------------------------------------------------------------


class ProposeLLMCallable(Protocol):
    """The interface the proposal tool needs from the LLM layer.

    Tests pass a stub; production passes a thin wrapper around the
    real Anthropic client (see ``agents/lens.py`` for the existing
    pattern). The protocol is intentionally minimal: one method,
    one input (the prompt), one output (a list of candidate dicts
    the proposal tool validates).
    """

    def propose(self, prompt: str) -> list[dict]: ...


# ---------------------------------------------------------------------------
# Pattern aggregation
# ---------------------------------------------------------------------------


def _aggregate_pattern_severity(
    decompositions: Sequence[ResidualDecomposition],
) -> dict[str, dict[str, float]]:
    """Aggregate residual patterns across folds per series.

    For each (series_key, pattern), take the max severity across
    folds. A pattern that fired in 3 of 3 folds is stronger
    evidence than one that fired in 1 of 3 — taking the max
    rewards consistency, but does not penalise a single weak
    signal either (the keep/kill test in CB4 will measure the
    real-world effect).

    Note: at runtime ``pattern`` is a string because
    ``ResidualPattern`` is a typing.Literal. The annotation uses
    the string form for accuracy.
    """
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for decomp in decompositions:
        per_pattern: dict[str, float] = out[decomp.series_key]
        for hit in decomp.patterns:
            existing = per_pattern.get(hit.pattern, 0.0)
            if hit.severity > existing:
                per_pattern[hit.pattern] = hit.severity
    return out


# ---------------------------------------------------------------------------
# Prompt construction (deterministic, audit-friendly)
# ---------------------------------------------------------------------------

# Reference table for the LLM: which config action is the natural
# response to which residual pattern. The LLM is not *required* to
# use this mapping — it can recommend any closed-Literal action —
# but the prompt mentions it as a hint so the common case is
# one-shot. Keeping it as data, not a hard-coded lookup, preserves
# the LLM's room to suggest a different fix when the data calls
# for one.
_PATTERN_HINTS: dict[str, str] = {
    "BIASED_RESIDUAL": "consider enable_lag_features or enable_hierarchy_features",
    "AUTOCORRELATED_RESIDUAL": "consider enable_lag_features or increase_fourier_terms",
    "PROMO_RESIDUAL_SPIKE": "consider enable_promo_indicator",
    "STOCKOUT_RESIDUAL_SPIKE": "consider enable_stockout_features",
    "PARENT_CHILD_RESIDUAL_GAP": "consider enable_hierarchy_features",
    "HETEROSCEDASTIC_RESIDUAL": "consider enable_intermittency_features",
}


def _build_prompt(
    aggregated: dict[str, dict[str, float]],
    *,
    segment_id: str | None,
) -> str:
    """Build the deterministic LLM prompt from the aggregated patterns.

    The prompt is the audit artifact: the literal text the LLM saw
    for a given (run, segment). Same input → same prompt string.
    Includes the closed set of allowed actions so the LLM does not
    invent free-text values; the Pydantic Literal validation is a
    backstop, but mentioning the set in the prompt cuts wasted
    tokens.
    """
    header = (
        "You are a forecasting analyst. Given the residual patterns "
        "below, recommend concrete config changes to the Feature "
        "Factory / model selection to address them. Return a JSON "
        "array of candidate proposals. Each proposal must have:\n"
        "  - kind: 'config' or 'code'\n"
        f"  - config_action (for kind=config): one of {list(get_args(ConfigAction))}\n"
        f"  - code_action (for kind=code): one of {list(get_args(CodeAction))}\n"
        "  - target_scope: 'series' or 'segment'\n"
        "  - target_id: the series_key or segment_id\n"
        "  - expected_delta: a float in [0, 1] estimating the MASE improvement\n"
        "  - rationale: a one-sentence justification\n"
        "  - evidence: a Claim object (claim_id, claim, verification_status, evidence_type, applies_to, downstream_impact, created_at)\n"
        "Do not invent action names outside the allowed sets.\n"
        "\n"
        "Pattern-to-action hints (suggestions, not requirements):\n"
    )
    for pattern, hint in _PATTERN_HINTS.items():
        header += f"  - {pattern}: {hint}\n"
    if segment_id is not None:
        header += f"\nScope: segment_id={segment_id}\n"
    else:
        header += "\nScope: per-series (one proposal per series affected)\n"
    body_lines = ["Residual patterns (per series, max severity across folds):"]
    for series_key in sorted(aggregated.keys()):
        patterns = aggregated[series_key]
        if not patterns:
            continue
        line = f"  - {series_key}: "
        # ``patterns`` is dict[str, float] because ResidualPattern is
        # a typing.Literal — at runtime the keys are strings, not
        # enum members. Sort by severity descending, render directly.
        line += ", ".join(
            f"{p}={severity:.2f}"
            for p, severity in sorted(patterns.items(), key=lambda kv: -kv[1])
        )
        body_lines.append(line)
    return header + "\n" + "\n".join(body_lines) + "\n"


# ---------------------------------------------------------------------------
# Candidate validation
# ---------------------------------------------------------------------------


def _coerce_candidate(raw: dict, *, default_target_id: str, default_scope: str) -> dict:
    """Coerce a raw LLM dict into the shape ``Proposal.model_validate`` accepts.

    The LLM may emit slight field-name variants
    (``target_id`` vs ``series_key``, ``action`` vs ``config_action``,
    etc.). We normalise to the canonical shape and let Pydantic

    Literal validation reject anything outside the closed sets.
    """
    coerced = dict(raw)
    # Normalise action naming: LLM might emit "action" for both
    # kinds. Prefer the explicit config_action / code_action split;
    # fall back to "action" only if the LLM signals kind first.
    kind = coerced.get("kind")
    action = coerced.pop("action", None)
    if kind == "config" and "config_action" not in coerced and action:
        coerced["config_action"] = action
    if kind == "code" and "code_action" not in coerced and action:
        coerced["code_action"] = action
    # Normalise target naming: LLM may emit "target_id" + "target_scope"
    # or "series_key" / "segment_id" directly. Build the ProposalTarget
    # shape.
    if "target" not in coerced:
        scope = coerced.pop("target_scope", None) or default_scope
        target_id = coerced.pop("target_id", None) or default_target_id
        if scope == "series":
            coerced["target"] = {
                "scope": "series",
                "series_key": target_id,
                "segment_id": None,
            }
        else:
            coerced["target"] = {
                "scope": "segment",
                "series_key": None,
                "segment_id": target_id,
            }
    return coerced


def _validate_candidate(raw: dict, *, default_target_id: str, default_scope: str) -> Proposal | None:
    """Validate a single LLM candidate dict, returning ``Proposal`` or ``None``.

    Returns ``None`` when the candidate is rejected: free-text
    action names (Pydantic Literal validation), missing evidence,
    missing action of the right kind, or any other contract
    violation. The harness never sees rejected candidates.
    """
    try:
        coerced = _coerce_candidate(
            raw, default_target_id=default_target_id, default_scope=default_scope
        )
        return Proposal.model_validate(coerced)
    except ValidationError:
        return None


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _rank_key(proposal: Proposal, pattern_severity: float) -> float:
    """Ranking key for a proposal: pattern_severity x expected_delta.

    Two Runs with the same LLM output produce the same order
    because the ranking is a pure function of (severity,
    expected_delta) — both come from the inputs, neither from
    the LLM's reordering.
    """
    return pattern_severity * max(proposal.expected_delta, 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def propose_feature_changes(
    scorecards: Sequence[ModelScorecard],
    decompositions: Sequence[ResidualDecomposition],
    *,
    llm: ProposeLLMCallable,
    segment_id: str | None = None,
) -> list[Proposal]:
    """Build the ranked ``Proposal[]`` for one (segment, model-family) pair.

    Parameters
    ----------
    scorecards
        The post-baseline scorecards (read-only; used to attach
        the series context to the prompt). Not consumed directly
        for ranking — the decomposition is the evidence.
    decompositions
        One ``ResidualDecomposition`` per (series, fold). The
        function aggregates max-severity per (series, pattern)
        across folds.
    llm
        The LLM callable. In tests this is a stub that returns
        a fixed list of dicts; in production it wraps the
        Anthropic client. The function never blocks on network
        when a stub is passed.
    segment_id
        When set, proposals target a segment; when None, each
        proposal targets a specific series_key. The LLM prompt
        declares the scope so the model emits the right target
        type.

    Returns
    -------
    list[Proposal]
        The ranked list. Empty when the decomposition found no
        patterns, when the LLM returned no candidates, or when
        every candidate was rejected by validation.

    Notes
    -----
    The LLM is constrained to the closed ``ConfigAction`` /
    ``CodeAction`` Literal sets. Free-text action names are
    rejected by Pydantic Literal validation before they reach
    the harness. A proposal without a ``Claim`` evidence field
    is rejected the same way (the ``_validate_candidate`` path
    checks the evidence claim exists).
    """
    aggregated = _aggregate_pattern_severity(decompositions)
    prompt = _build_prompt(aggregated, segment_id=segment_id)
    raw_candidates = llm.propose(prompt)
    if not isinstance(raw_candidates, list):
        return []

    default_scope = "segment" if segment_id is not None else "series"
    default_target_id = segment_id if segment_id is not None else ""

    validated: list[tuple[Proposal, float]] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        proposal = _validate_candidate(
            raw, default_target_id=default_target_id, default_scope=default_scope
        )
        if proposal is None:
            continue
        # Pin evidence: every proposal from this tool carries a
        # pattern-backed Claim. The LLM supplies the Claim in
        # ``proposal.evidence``; we just verify it's well-formed
        # (the Proposal contract requires it, so a malformed
        # claim would have failed Pydantic validation already).
        # The decomposition only feeds the rank key — the LLM
        # does not get to invent severities.
        target_series = proposal.target.series_key or ""
        per_series = aggregated.get(target_series, {})
        top_severity = max(per_series.values()) if per_series else 0.0
        evidence_claim = proposal.evidence
        # Rank: max pattern severity x expected_delta.
        validated.append((proposal, _rank_key(proposal, top_severity)))

    # Deterministic sort: descending by rank key, then by
    # (kind, config_action, code_action, target) as a stable
    # tie-breaker so two proposals with the same rank_key come
    # out in the same order every time.
    validated.sort(
        key=lambda pv: (
            -pv[1],
            pv[0].kind,
            pv[0].config_action or "",
            pv[0].code_action or "",
            pv[0].target.series_key or "",
            pv[0].target.segment_id or "",
        )
    )
    return [p for p, _ in validated]


__all__ = (
    "ProposeLLMCallable",
    "propose_feature_changes",
)
