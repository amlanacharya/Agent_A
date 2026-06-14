"""
Single source of truth for all shared domain Pydantic models.
HTTP-layer models live in api/models.py - not here.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Pre-flight contracts
# ---------------------------------------------------------------------------

class BlockingIssue(BaseModel):
    code: str
    message: str


class DataQualityWarning(BaseModel):
    code: str
    message: str
    affected_series: list[str] = Field(default_factory=list)


class DataQualityReport(BaseModel):
    blocking_issues: list[BlockingIssue]
    warnings: list[DataQualityWarning]
    row_count: int
    series_count: int


class SchemaMapping(BaseModel):
    date_col: str
    demand_col: str
    grain_cols: list[str]
    extra_cols: list[str]


class GrainReport(BaseModel):
    detected_frequency: Literal["daily", "weekly", "monthly", "unknown"]
    min_periods: int
    max_periods: int
    median_periods: int
    gaps_detected: bool


SBClass = Literal["SMOOTH", "ERRATIC", "INTERMITTENT", "LUMPY"]


class AdiCv2Stats(BaseModel):
    series_key: str
    adi: float
    cv2: float
    sb_class: SBClass


class ZeroRunStats(BaseModel):
    series_key: str
    max_zero_run: int
    zero_fraction: float


class SpikeStats(BaseModel):
    series_key: str
    spike_count: int
    max_spike_ratio: float


class PromoAlignmentStats(BaseModel):
    series_key: str
    has_promo_col: bool
    aligned_fraction: float | None = None


class TrendStats(BaseModel):
    series_key: str
    trend_strength: float
    direction: Literal["up", "down", "flat"]


class SeasonalityStats(BaseModel):
    series_key: str
    seasonal_strength: float
    dominant_period: int | None = None


class BreakCandidate(BaseModel):
    series_key: str
    break_period: str
    confidence: float


class SegmentProfile(BaseModel):
    segment_id: str  # "G1", "G2", ... - matches SegmentDef.segment_id
    series_count: int
    demand_class_distribution: dict[str, int]  # {"SMOOTH": 4, "ERRATIC": 1, ...}
    median_adi: float
    median_cv2: float
    forecastability_breakdown: dict[str, int]  # preliminary counts, not Forge's official call
    example_keys: list[str] = Field(default_factory=list)


class SegmentDef(BaseModel):
    """One segment in the (provisional) segment map. See plan_v2 §6 / CONTEXT 'Segment'."""

    segment_id: str  # "G1", "G2", ...
    label: str  # human-readable, e.g. "region=NORTH" or "all series"
    series_keys: list[str]  # member series keys
    provisional: bool = True  # True until refined/confirmed by the user in Meridian


class SegmentMap(BaseModel):
    """Series→segment grouping. Provisional from pre-flight; locked into the pack at confirmation."""

    run_id: str
    segments: list[SegmentDef]
    provisional: bool = True
    derived_by: str  # e.g. "playbook:segment_by=region" or "default:single_segment"


class SeriesException(BaseModel):
    series_key: str
    segment_id: str  # segment this outlier belongs to
    exception_type: Literal["HIGH_ZERO_FRACTION", "SPIKE", "ZERO_RUN"]
    detail: str


class PreflightBundle(BaseModel):
    """
    Aggregate-only handoff injected into Meridian's system prompt (plan_v2 correction #4).
    Per-series statistics are NOT carried here - they are persisted to preflight.json and
    read on demand by the diagnostic tools. Keeping them out of the bundle keeps individual
    series stats out of Claude's context on every Meridian turn (sentinel pattern / §5).
    """

    run_id: str
    data_quality_report: DataQualityReport
    schema_mapping: SchemaMapping
    grain_report: GrainReport
    segment_profiles: list[SegmentProfile]  # aggregate per segment - NOT per series
    segment_exceptions: list[SeriesException]  # small list of per-series outliers within segments
    segments: list[SegmentDef]  # provisional segment map
    domain_playbook: dict  # raw YAML playbook dict


# ---------------------------------------------------------------------------
# Feature flags (used by the pack + Forge feature config)
# ---------------------------------------------------------------------------

class FeatureFlags(BaseModel):
    use_fourier: bool = False
    use_lag_features: bool = True
    use_promo_indicator: bool = False
    fourier_terms: int = 3
    # Explicit seasonal period for Fourier terms. When set, Fourier cycles
    # over this period (e.g. 52 for weekly data with annual seasonality)
    # rather than over the row-count of each series. Required for
    # walk-forward validation so Fourier phases stay aligned across folds.
    frequency_period: int | None = None


# DomainContextPack is defined after Claim / Risk below - it embeds them.


# ---------------------------------------------------------------------------
# Conductor / intent contracts
# ---------------------------------------------------------------------------

IntentType = Literal[
    "SCOPE_RESPONSE",  # answering Meridian's question
    "OVERRIDE",  # contradicting a data-backed agent recommendation
    "ADVANCE_PIPELINE",  # "ok let's model", "looks good"
    "WHAT_IF_REQUEST",  # "what if promo on SKU X week 10"
    "CLARIFICATION",  # user asking a question
    "CORRECTION",  # fixing a prior statement - only valid during meridian_scoping;
    # post-confirmation treated as OVERRIDE
]


class IntentEntities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skus: list[str] = Field(default_factory=list)
    segments: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)  # ISO strings
    metrics: list[str] = Field(default_factory=list)
    scenario: str | None = None  # free-text scenario description if WHAT_IF_REQUEST


class IntentPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: IntentType
    entities: IntentEntities = Field(default_factory=IntentEntities)
    confidence: float = Field(ge=0.0, le=1.0)  # 0.0-1.0
    raw_quote: str  # verbatim fragment of user message that drove classification


# ---------------------------------------------------------------------------
# Claim ledger
# ---------------------------------------------------------------------------

VerificationStatus = Literal[
    "SUPPORTED", "CONTRADICTED", "AMBIGUOUS", "UNVERIFIABLE", "USER_OVERRIDE_ACCEPTED"
]

EvidenceType = Literal[
    "statistical_test", "association", "pattern", "user_confirmed", "unverifiable_business_input"
]


class Claim(BaseModel):
    claim_id: str  # uuid4
    claim: str  # human-readable assertion
    verification_status: VerificationStatus
    evidence_type: EvidenceType
    evidence_ref: str | None = None  # tool-call result summary backing the claim
    applies_to: str  # segment_id, series_key, or "run"
    downstream_impact: str  # what this claim affects downstream
    must_surface_in_report: bool = False  # True for USER_OVERRIDE_ACCEPTED
    created_at: str
    resolved_at: str | None = None
    resolution_note: str | None = None


class ClaimLedger(BaseModel):
    run_id: str
    claims: list[Claim] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Risk register
# ---------------------------------------------------------------------------

class Risk(BaseModel):
    risk_id: str
    description: str
    severity: Literal["low", "medium", "high"]  # matches add_risk
    source: str  # matches add_risk
    acknowledged: bool = False
    created_at: str
    acknowledged_at: str | None = None


class RiskRegister(BaseModel):
    run_id: str
    risks: list[Risk] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Domain context pack (Meridian output, Foundry/Prism input)
# Matches exactly what compile_domain_context_pack emits / writes to
# domain_context_pack.json - this model IS the validated return type of that
# tool, not a parallel structure. (review §2)
# ---------------------------------------------------------------------------

class ForecastScope(BaseModel):
    target_col: str
    grain: list[str]
    horizon: int
    mase_target: float | None = None  # user MASE override if set; else playbook/floor applies


class DomainContextPack(BaseModel):
    run_id: str
    domain: str
    forecast_scope: ForecastScope
    segments: list[SegmentDef]  # locked segment map (provisional=False)
    claim_ledger: list[Claim]
    risk_register: list[Risk]
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    override_count: int = 0
    open_risks: int = 0
    confirmed_at: str | None = None
    confirmed: bool = False


# ---------------------------------------------------------------------------
# EDA report (Forge output)
# ---------------------------------------------------------------------------

class SeriesDemandProfile(BaseModel):
    series_key: str
    sb_class: SBClass
    adi: float
    cv2: float
    trend_strength: float
    seasonal_strength: float
    recommended_models: list[str]


class EDAReport(BaseModel):
    run_id: str
    segment_profiles: list[SegmentProfile]
    series_profiles: list[SeriesDemandProfile]
    feature_config: dict[str, FeatureFlags]
    narrative: str


# ---------------------------------------------------------------------------
# Foundry results
# ---------------------------------------------------------------------------

class ModelResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    mase: float
    mae: float
    rmse: float
    forecast: list[float]
    selected: bool


class SeriesResult(BaseModel):
    series_key: str
    sb_class: SBClass
    mase_target: float
    results: list[ModelResult]
    best_model: str
    target_met: bool
    self_correction_rounds: int = 0


class FoundryReport(BaseModel):
    run_id: str
    series_results: list[SeriesResult]
    overall_mase: float
    target_met_fraction: float
    narrative: str


# ---------------------------------------------------------------------------
# Prism (what-if) contracts
# ---------------------------------------------------------------------------

class WhatIfOverride(BaseModel):
    whatif_id: str
    series_key: str
    override_type: Literal["PROMO_EVENT", "STOCKOUT", "PRICE_CHANGE", "MANUAL_UPLIFT"]
    magnitude: float
    start_period: str
    end_period: str
    description: str


class ScenarioComparison(BaseModel):
    whatif_id: str
    series_key: str
    baseline_forecast: list[float]
    scenario_forecast: list[float]
    delta_pct: float
    demand_class_changed: bool
    baseline_sb_class: SBClass
    scenario_sb_class: SBClass
