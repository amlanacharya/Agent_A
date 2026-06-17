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
    # Phase 3 feature families. All default to False so existing callers
    # keep producing the same canonical feature table.
    #
    # Stockout / availability: rolling stockout counts, days-since-last
    # stockout, inventory cover ratio derived from stockout_flag and
    # inventory_qty. Requires ``stockout`` and ``inventory`` columns on
    # the canonical input.
    use_stockout_features: bool = False
    # Hierarchy: parent-level (sku_id aggregated across location_id)
    # lag-1 and rolling-4 demand, fold-aware. Requires ``sku_id`` to be
    # present on the canonical input.
    use_hierarchy_features: bool = False
    # Lifecycle / cold-start: history length, time-since-first-observation
    # in days, and a cold-start flag. Pure row-local features (no leakage).
    use_lifecycle_features: bool = False
    # Intermittency: rolling-window ADI, CV², and trailing zero-run length.
    # All time-dependent and fold-aware.
    use_intermittency_features: bool = False


# ---------------------------------------------------------------------------
# Foundry escalation contracts
# ---------------------------------------------------------------------------
# The Phase 4.1 two-path escalation loop (config vs code) needs a typed
# output the Foundry agent produces and the harness consumes. The LLM
# call returns ``Proposal[]``; the harness iterates the list, tries
# config proposals first, and only escalates to code proposals when
# config escalation is exhausted without hitting the MASE target.
#
# The contract separates the *kind* of action (config vs code) from
# the *specific* action (which flag, which model, which family) and
# from the *target* (one series or a whole segment). Splitting them
# into Literal fields rather than a single free-text ``action`` string
# keeps every proposal machine-validatable: a config proposal that
# names a non-existent FeatureFlag is rejected at the model boundary
# (Pydantic Literal validation), not at execution time.

ProposalKind = Literal["config", "code"]

# A config proposal flips a knob that already exists. The Literal is
# the closed set of knobs the harness currently understands; new ones
# land here when the harness grows, not in the LLM prompt. ``model``
# is the model-family swap (one of the 6 governed families) and the
# rest are FeatureFlag toggles / parameter tweaks.
ConfigAction = Literal[
    "enable_lag_features",
    "enable_promo_indicator",
    "enable_stockout_features",
    "enable_hierarchy_features",
    "enable_lifecycle_features",
    "enable_intermittency_features",
    "increase_fourier_terms",
    "swap_model_family",
    "tune_model_parameter",
]

# A code proposal adds a new building block. Today the only two
# shapes the plan allows are a new feature family or a new model
# family; if a third ever lands (a new adapter, a new preflight
# probe), it joins this Literal.
CodeAction = Literal["new_feature_family", "new_model_family"]


class ProposalTarget(BaseModel):
    """The scope a single proposal applies to.

    Exactly one of ``series_key`` or ``segment_id`` is set. A
    series-level proposal is one-off (e.g. "this one stockout-heavy
    series needs the stockout family"); a segment-level proposal
    applies to every series in the segment (e.g. "all INTERMITTENT
    series in G1 would benefit from intermittency features").

    # The harness's Proposal application logic branches on which field
    # is set; the ``scope`` Literal is a redundant but explicit hint
    # for downstream rendering and audit.
    """

    scope: Literal["series", "segment"]
    series_key: str | None = None
    segment_id: str | None = None


# ---------------------------------------------------------------------------
# Residual decomposition (Phase 4.1 CB2)
# ---------------------------------------------------------------------------
# The decompose_residuals tool reads a ModelScorecard's forecast/actual
# arrays, computes per-series residual statistics, and emits a closed
# set of ``ResidualPattern``s the propose_feature_changes tool ranks
# against in CB3. The pattern set is closed and named: a free-text
# residual narrative would force the LLM to do structured extraction
# at proposal time, which is the wrong place for that work.
#
# Each pattern carries a ``severity`` in [0, 1] so the proposal tool
# can rank (config_proposal_evidence = max severity of matched
# patterns) without re-reading the residual stats. The patterns are
# the *interface* between the deterministic math (CB2) and the LLM
# judgement (CB3) — keeping the seam here means CB2 has no LLM
# dependency and CB3 has no math dependency.

ResidualPattern = Literal[
    # Forecast systematically over- or under-shoots actuals. The
    # severity is the absolute mean residual normalised by the mean
    # actual demand. A bias of 0.5 means the forecast is off by ~50%
    # of the mean demand level on average.
    "BIASED_RESIDUAL",
    # Lag-k autocorrelation of the residual is high for some k in
    # {1, 2, 4, 8}. Implies the model has missed a temporal
    # structure the existing features don't capture. Severity is the
    # max absolute autocorrelation across the lags.
    "AUTOCORRELATED_RESIDUAL",
    # Mean residual on promo-flagged weeks is materially different
    # from mean residual on non-promo weeks. Implies the model
    # under-models the promo lift. Severity is the absolute
    # difference in mean residuals divided by overall residual std.
    "PROMO_RESIDUAL_SPIKE",
    # Same as PROMO but for stockout-flagged weeks. Distinguishes
    # "model missed a stockout-week correction" from
    # "model missed a promo-week correction" — the two are
    # orthogonal and the proposal tool ranks them separately.
    "STOCKOUT_RESIDUAL_SPIKE",
    # Mean residual on this series is materially different from
    # the mean residual of the parent (sku_id aggregated across
    # location_id). Implies the child has a location-specific
    # effect the hierarchy features don't capture. Severity is the
    # absolute difference in mean residuals divided by overall
    # residual std.
    "PARENT_CHILD_RESIDUAL_GAP",
    # Residual variance is much higher than what the in-sample
    # naive MAE would predict. Implies a regime change or a
    # non-stationary driver the features don't track.
    "HETEROSCEDASTIC_RESIDUAL",
]


class ResidualStats(BaseModel):
    """The deterministic per-series residual statistics.

    All fields are pure functions of the scorecard's forecast and
    actual arrays, except ``promo_residual_mean`` /
    ``stockout_residual_mean`` / ``parent_residual_mean`` which
    require the canonical demand slice for the fold horizon and the
    parent-grain residual mean respectively. The pure math (mean,
    std, autocorr, bias) is always populated; the contextual fields
    are populated only when the caller provides the context.
    """

    series_key: str
    n: int
    # Pure residual stats
    residual_mean: float
    residual_std: float
    mae: float
    # Autocorrelation at standard lags. None when n is too short to
    # compute a given lag (lag must be < n).
    autocorr_lag_1: float | None = None
    autocorr_lag_2: float | None = None
    autocorr_lag_4: float | None = None
    autocorr_lag_8: float | None = None
    # Contextual residual means, populated when context is provided
    promo_residual_mean: float | None = None
    non_promo_residual_mean: float | None = None
    stockout_residual_mean: float | None = None
    non_stockout_residual_mean: float | None = None
    parent_residual_mean: float | None = None


class ResidualPatternHit(BaseModel):
    """One (pattern, severity) pair emitted by the decomposition.

    The decomposition never returns a free-text narrative; it
    returns a list of hits, one per pattern the residual stats
    cross the threshold for. The proposal tool consumes the hits
    list and ranks ``Proposal`` candidates by max hit severity.
    """

    pattern: ResidualPattern
    severity: float  # in [0, 1]
    detail: str  # short human-readable explanation for the audit log


class ResidualDecomposition(BaseModel):
    """The deterministic output of ``decompose_residuals`` for one series.

    One ``ResidualDecomposition`` per (series, fold) pair. The
    Foundry agent aggregates across folds when ranking proposals at
    the segment level (CB3) — see ``propose_feature_changes``.

    ``patterns`` is empty when the residual is well-explained by the
    model's features and the proposal tool has nothing to recommend.
    """

    series_key: str
    fold_cutoff: str  # ISO date string, matches ModelScorecard.fold_cutoff
    stats: ResidualStats
    patterns: list[ResidualPatternHit] = Field(default_factory=list)


class Proposal(BaseModel):
    """A single candidate change the Foundry agent proposes.

    Output of the ``propose_feature_changes`` tool (CB3 of Phase 4.1);
    consumed by the config-escalation loop (CB5) and the code-escalation
    gate (the existing ``model_escalation`` path for ``code`` proposals).

    ``expected_delta`` is the agent's prediction of the MASE improvement
    the proposal would produce, used by the harness only as a *hint* for
    ranking; the keep/kill decision is made on the *actual* MASE delta
    measured after application. Proposals with no predicted delta are
    accepted (the harness ranks them last), but the LLM is prompted to
    always provide one.

    ``evidence`` is the Claim the agent is making for the proposal
    (typically ``evidence_type=pattern`` grounded in the residual
    decomposition). On success the Claim is verifier-promoted to a
    markdown card in ``LEARNINGS.md`` per the Phase 1 promotion rule.
    """

    kind: ProposalKind
    # For config proposals, ``config_action`` is set; for code proposals,
    # ``code_action`` is set. Using a discriminated union would be more
    # strictly typed but the current shape keeps ``kind`` and the
    # action fields in one record, which matches how the LLM emits
    # proposals (one row per candidate, not a tagged union).
    config_action: ConfigAction | None = None
    code_action: CodeAction | None = None
    # Free-text parameter payload for ``tune_model_parameter`` and the
    # code-proposal cases. For ``tune_model_parameter`` it is
    # ``{"parameter": "n_estimators", "value": 200}``; for code
    # proposals it is a markdown-card-ready description of the new
    # family. The harness validates against the action's schema before
    # applying — wrong-shape payloads are dropped, not applied blind.
    action_payload: dict = Field(default_factory=dict)
    target: ProposalTarget
    expected_delta: float = 0.0
    rationale: str
    evidence: Claim


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


# ---------------------------------------------------------------------------
# EDA sub-check contracts (Phase 2 — data quality probes)
#
# Each probe operates on the canonical demand table and produces a small
# Pydantic payload. The Forge EDA layer composes them into the EDAReport.
# Probes are *advisory* (they never block the run on their own) but they
# escalate through the ``EscalationTracker`` when standard expectations
# fail.
# ---------------------------------------------------------------------------

InferredColumnType = Literal[
    "integer",
    "float",
    "boolean",
    "string",
    "datetime",
    "empty",
    "mixed",
]


class ColumnTypeInference(BaseModel):
    column: str
    inferred_type: InferredColumnType
    nullable: bool
    unique_count: int
    sample_values: list[str] = Field(default_factory=list)


class TypeDetectionReport(BaseModel):
    columns: list[ColumnTypeInference]
    # The number of columns whose inferred type did NOT match the type the
    # canonical contract expects (e.g. a "demand_qty" column containing
    # strings). High counts surface as a warning in the EDA narrative.
    contract_mismatches: list[str] = Field(default_factory=list)


class MissingnessStats(BaseModel):
    column: str
    missing_count: int
    missing_fraction: float


class MissingnessReport(BaseModel):
    per_column: list[MissingnessStats]
    # Number of rows that have at least one missing value in the optional /
    # not-required-by-contract columns. "Required" columns (sku_id,
    # location_id, week_start, demand_qty) are not allowed to have missing
    # values — see canonical_data validation.
    rows_with_missing: int
    rows_total: int


class DuplicateReport(BaseModel):
    # (series_key, date) collisions in the canonical table. A canonical
    # table MUST have a unique key per series per date; duplicates indicate
    # an upstream aggregation mistake that the canonical layer did not
    # de-duplicate.
    duplicate_rows: int
    duplicate_keys: list[str] = Field(default_factory=list)
    duplicate_fraction: float


class SeriesDateGapStats(BaseModel):
    series_key: str
    expected_period_days: int | None = None
    actual_gap_count: int
    max_gap_days: int
    median_gap_days: float
    out_of_order_rows: int


class DateGapsReport(BaseModel):
    per_series: dict[str, SeriesDateGapStats]
    # Series that have at least one gap strictly larger than 1.5x the
    # expected period.
    series_with_gaps: list[str] = Field(default_factory=list)


class JoinValidationIssue(BaseModel):
    kind: Literal[
        "MISSING_INVENTORY_FOR_DEMAND",
        "MISSING_PRICE_FOR_DEMAND",
        "MISSING_LEAD_TIME_FOR_DEMAND",
        "INVENTORY_WITHOUT_DEMAND",
    ]
    series_key: str
    detail: str


class JoinValidationReport(BaseModel):
    issues: list[JoinValidationIssue]
    # Demand rows that have a NaN inventory value (when inventory_qty is
    # expected to be populated). Keep an issue per series for traceability.
    inventory_coverage: float
    price_coverage: float
    lead_time_coverage: float


class SeriesLeakageStats(BaseModel):
    series_key: str
    # Correlation between demand[t] and demand[t+1] is expected (lag-1
    # autocorrelation is a feature). A near-1 correlation between demand[t]
    # and demand[t+2..t+5] in a weekly series is a leakage red flag — it
    # usually means a future column leaked into the past.
    forward_correlation_max: float
    # demand_qty equal to inventory_qty is impossible: inventory is the
    # stock on hand, not what was sold. Detecting it tells the user their
    # upstream join is wrong.
    demand_equals_inventory_rows: int


class LeakageReport(BaseModel):
    per_series: dict[str, SeriesLeakageStats]
    # Series with at least one suspicion — short-cut for the EDA narrative.
    suspect_series: list[str] = Field(default_factory=list)


class EDAReport(BaseModel):
    run_id: str
    segment_profiles: list[SegmentProfile]
    series_profiles: list[SeriesDemandProfile]
    feature_config: dict[str, FeatureFlags]
    narrative: str
    # Phase 2 probes. All default to empty/None so that callers (and
    # existing tests) constructed before the probes existed continue to
    # pass without modification.
    type_detection: TypeDetectionReport | None = None
    missingness: MissingnessReport | None = None
    duplicates: DuplicateReport | None = None
    date_gaps: DateGapsReport | None = None
    join_validation: JoinValidationReport | None = None
    leakage: LeakageReport | None = None


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
# Promotion (Phase 5.2 CB1)
# ---------------------------------------------------------------------------
# Champion/challenger promotion: a candidate model is run on a
# fixed ``BacktestWindow`` (the same cutoffs for every candidate,
# so the comparison is fair) and compared against the current
# champion on WAPE / per-segment WAPE / shadow-mode agreement.
# The comparison is pure: the candidate's scorecards and the
# champion's scorecards are inputs; the function returns a typed
# ``PromotionComparison`` record. The actual decision (promote
# or reject) is a separate function (CB2).
#
# Why a fixed window instead of a rolling one: the platform
# already does walk-forward validation per fold inside the
# harness. The promotion layer's role is *comparison*, not
# validation — the candidate and the champion must be scored
# on the *same* cutoffs so the WAPE delta is not contaminated
# by data drift between runs.


class BacktestWindow(BaseModel):
    """The fixed backtest window the promotion comparison uses.

    All candidates and the current champion are scored against
    these exact cutoffs. The window is a closed interval
    ``[start, end]`` with a list of cutoffs inside it; each
    cutoff is a fold-start that the harness will backtest
    from.

    ``start`` and ``end`` are ISO-8601 timestamps. ``cutoffs``
    are ISO-8601 timestamps strictly between ``start`` and
    ``end``. ``horizon`` is the forecast horizon (in the same
    time unit as the timestamps — days, weeks, etc., depending
    on the dataset).
    """

    start: str
    end: str
    cutoffs: list[str]
    horizon: int


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


# ---------------------------------------------------------------------------
# Phase 4 - Forecasting Harness contracts
#
# A "model family" is one of the governed, registered model types a
# harness can fit. The harness is what decides which families to run,
# which to surface, and how to weight them. The families themselves
# stay small and self-contained.
# ---------------------------------------------------------------------------

ModelFamilyName = Literal[
    "naive",
    "seasonal_naive",
    "moving_average",
    "exponential_smoothing",
    "croston",
    "xgboost_global",
    "aggregate_allocate",
]

# A scorecard for one model fit on one series' backtest fold. The
# forecast itself is a horizon-long vector; metrics summarise how it
# compared to the held-out window. ``mase`` is normalised by the
# series' own in-sample naive MAE so it is comparable across
# intermittent and smooth demand alike.
class ModelScorecard(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_family: ModelFamilyName
    series_key: str
    fold_cutoff: str  # ISO date of the fold cutoff used for the backtest
    horizon: int
    forecast: list[float]
    actual: list[float]
    mae: float
    rmse: float
    mase: float
    # ``bias`` is signed: positive = under-forecast, negative = over-forecast.
    bias: float


# A single robustness check the harness can run on a fitted model:
# either the data-contract check (the model returned a properly-shaped
# forecast, in the right units, with no NaN/inf), or a backtest gate
# (the backtest scorecard is in-bounds for the model's claimed
# performance). The model escalation path requires ALL of these to
# pass before a custom model is accepted.
class RobustnessCheck(BaseModel):
    check: Literal["data_contract", "backtest", "robustness", "review"]
    passed: bool
    detail: str


# The harness's unified request shape. ``fold_cutoffs`` is shared with
# the Feature Factory so the same fold bands drive both the feature
# computation and the backtest/train split. ``model_families`` lets
# callers opt in / out of specific families without changing the
# harness signature.
class ForecastRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    # Feature table produced by ``build_feature_table``. Must contain
    # ``series_key``, ``date``, ``demand`` plus any opt-in Phase 3
    # families (``use_xgboost_global`` will need at least the lag /
    # rolling / promo columns).
    feature_table: list[dict] = Field(default_factory=list)
    # Series-level target column. Defaults to ``"demand"``; provided as
    # a hook for what-if reruns that forecast a different metric.
    target_col: str = "demand"
    fold_cutoffs: list[str] = Field(default_factory=list)
    horizon: int = 1
    model_families: list[ModelFamilyName] = Field(
        default_factory=lambda: [
            "naive",
            "seasonal_naive",
            "moving_average",
            "exponential_smoothing",
            "croston",
            "xgboost_global",
            "aggregate_allocate",
        ]
    )
    # Optional per-segment demand class hints. When provided, the
    # Croston family is enabled only for INTERMITTENT / LUMPY segments
    # and the seasonal family is enabled only for SMOOTH / ERRATIC
    # segments. When absent the harness uses a class-blind fallback
    # (run all families, let the scorecard pick).
    segment_sb_class: dict[str, SBClass] = Field(default_factory=dict)

    def mase_target_for(self, series_key: str) -> float:
        """Return the MASE target a series must beat to be ``forecastable``.

        The default is 1.0 — beating the naive baseline by construction.
        Per-series overrides (set during Meridian scoping as Claims) are
        out of scope for Phase 4 and will be wired in here when the
        per-series MASE Claim lands.

        Defined as a regular method on the model (not monkey-patched at
        import time) so the policy is visible at the declaration site
        and the Pydantic v2 model surface stays self-contained.
        """
        return 1.0


# The harness's unified output. ``series_results`` is a flat list of
# per-series best-model picks; ``scorecards`` keeps the full backtest
# history so the ensemble tracker and the promotion decision can
# audit every fold.
class ForecastHarnessReport(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    horizon: int
    # Per-series best-model pick (small enough to ship in the cockpit).
    series_results: list[SeriesResult]
    # Full backtest history (larger; persisted to disk for review).
    scorecards: list[ModelScorecard]
    # Ensemble behaviour — segment weights and retire / promote / never
    # surfaced lists. ``None`` for runs that did not invoke the
    # ensemble layer (single-model fast paths).
    ensemble: "EnsembleSummary | None" = None
    # The aggregate set of robustness checks the harness ran. Phase 5
    # will read this to gate promotion.
    robustness_checks: list[RobustnessCheck] = Field(default_factory=list)
    # When a model family ran but produced no usable forecast for any
    # series (e.g. XGBoost on a 1-row history), the family name is
    # recorded here. The ensemble layer uses this list to flag
    # never-surfaced models.
    never_surfaced: list[ModelFamilyName] = Field(default_factory=list)
    # Markdown summary consumed by the cockpit / learning journal. The
    # harness is the canonical place to surface what the platform
    # did at fit time; downstream layers should not re-derive this.
    narrative: str = ""


# Per-segment ensemble metadata. ``weights`` is ``family -> weight``,
# normalised so the values across a single segment sum to 1.0.
# ``frequently_promoted`` and ``retired`` are reference snapshots of
# the ``EnsembleTracker`` history so the cockpit can show "X has been
# best in 80% of folds for INTERMITTENT segments" without re-running
# the tracker.
class EnsembleSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # ``segment_id -> family -> weight`` view. Empty for runs that did
    # not produce an ensemble (single-family fallbacks).
    weights: dict[str, dict[str, float]] = Field(default_factory=dict)
    # Families that have been best-in-fold for >= 50% of series in
    # any segment over the run's history. Surfaced in the cockpit.
    frequently_promoted: list[ModelFamilyName] = Field(default_factory=list)
    # Families that fit successfully but were never best-in-fold.
    never_surfaced: list[ModelFamilyName] = Field(default_factory=list)
    # Families that were promoted in a prior run and have been
    # replaced; we keep their scorecards around for audit but exclude
    # them from the live ensemble weights.
    retired: list[ModelFamilyName] = Field(default_factory=list)


# Produced by ``model_escalation.declare_failure_report`` after the
# three-attempt cap on a custom model family. The plan calls for the
# block to be: data contract, backtest, robustness, review. The
# attached failure report makes it easy for the cockpit to surface
# the exact gate that failed.
class ModelFailureReport(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    proposed_family: str  # free-text; not a ModelFamilyName because it may be a custom code path
    status: Literal["blocked"] = "blocked"
    blocker: str
    evidence: list[str] = Field(default_factory=list)
    attempts: int = 3
    failed_reasons: dict[int, str] = Field(default_factory=dict)
    failed_gates: list[Literal["data_contract", "backtest", "robustness", "review"]] = Field(default_factory=list)
    recommended_next_action: str


ForecastHarnessReport.model_rebuild()


# ---------------------------------------------------------------------------
# Phase 6: Approvals, Scheduling, ERP Handoff
# ---------------------------------------------------------------------------
# Phase 6 is the platform's own native layer for human approvals,
# scheduled jobs, and the ERP handoff payload. The cockpit UI is the
# surface the human interacts with; the in-process gateway and
# scheduler are the engine. The two are deliberately decoupled behind
# small ABCs (``ApprovalGateway``, ``Scheduler``) so a future
# alternative implementation (a webhook, an external workflow, an
# enterprise scheduler) can plug in behind the same interfaces
# without changing the rest of the platform. None of those
# alternatives is planned today; the seam exists for the day one
# is chosen, not as a current dependency.
#
# Keeping these contracts pure Pydantic with no I/O makes them easy
# to assert against in tests and impossible to leak platform
# internals into the cockpit UI or any future consumer.
#
# The naming follows the principle the rest of the platform uses:
#   - "kind" is a closed Literal (the set the platform actually emits)
#   - "status" is a closed Literal (the states a request / run can be in)
#   - identifiers (request_id, trigger_id, run_id) are opaque strings
#     the gateway hands back so callers don't conflate them with the
#     platform's run_id. run_id is included explicitly so an audit
#     record can join back to the platform's run.
# ---------------------------------------------------------------------------


# All seven approval kinds from the original Phase 6 plan. New kinds
# must be added to this Literal; a closed set keeps the cockpit's
# approval widget and any future consumer's mapping finite and
# reviewable.
ApprovalKind = Literal[
    "data_contract",                 # user signs off on the schema mapping
    "risky_schema_semantics",        # user accepts ambiguous column meanings
    "custom_code_permission",        # user grants the agent a code-escalation attempt
    "unforecastable_grain_fallback", # user accepts a coarser grain for an unforecastable series
    "official_forecast_publication", # user publishes a Foundry report
    "replenishment_recommendation",  # user signs off on a replenishment batch
    "erp_procurement_handoff",       # user releases the approved batch to ERP
]

ApprovalStatus = Literal["pending", "approved", "rejected"]
ApprovalDecisionValue = Literal["APPROVE", "REJECT", "DEFER"]


class ApprovalRequest(BaseModel):
    """A request the platform raises when a decision needs a human.

    The gateway holds the request until a human calls
    acknowledge() with a decision. ``run_id`` is the platform's
    run; ``request_id`` is the gateway's local id.
    """

    request_id: str
    run_id: str
    kind: ApprovalKind
    title: str
    summary: str
    payload: dict[str, object] = Field(default_factory=dict)
    requested_by: str
    requested_at: str  # ISO-8601 timestamp
    status: ApprovalStatus = "pending"
    # Once acknowledged, the decision and approver are recorded here.
    decision: ApprovalDecisionValue | None = None
    approver: str | None = None
    decided_at: str | None = None
    reason: str | None = None


class ApprovalEvent(BaseModel):
    """A single audit entry for the approval lifecycle.

    Written to ``outputs/{run_id}/approvals.jsonl`` (one event per
    state change). event_type is one of the four transitions the
    gateway can produce: request raised, decision recorded, expiry,
    or a status correction.
    """

    event_id: str
    request_id: str
    run_id: str
    event_type: Literal["raised", "decided", "expired", "corrected"]
    occurred_at: str
    actor: str
    notes: str = ""


# All six scheduled job kinds from the original Phase 6 plan.
ScheduledJobKind = Literal[
    "data_refresh",
    "validation",
    "forecast_generation",
    "review",
    "monitoring",
    "drift_investigation",
]

ScheduledJobStatus = Literal["queued", "running", "succeeded", "failed", "skipped", "awaiting_approval"]


class ScheduledJobTrigger(BaseModel):
    """A declarative trigger registered with the scheduler.

    ``cron`` is a small expression the in-process scheduler evaluates
    every tick (``"every 5m"``, ``"hourly"``, ``"daily 02:00"``).
    The Scheduler is documented separately; the trigger carries just
    the schedule shape, not the runner logic.
    """

    trigger_id: str
    kind: ScheduledJobKind
    cron: str
    enabled: bool = True
    # Optional run_id to resume a specific run; absent means create a new one.
    run_id: str | None = None
    # Free-form kwargs the runner understands for this kind (e.g.
    # ``{"window_days": 90}`` for a drift_investigation).
    params: dict[str, object] = Field(default_factory=dict)
    created_at: str
    created_by: str


class ScheduledJobRun(BaseModel):
    """The outcome of one tick of the scheduler for one trigger.

    A trigger can fire many runs; each is logged for audit. ``status``
    is the terminal-or-current state. ``awaiting_approval`` means the
    run reached an ApprovalRequest and is parked until a human acts.
    """

    run_id: str
    trigger_id: str
    kind: ScheduledJobKind
    started_at: str
    finished_at: str | None = None
    status: ScheduledJobStatus
    result_payload: dict[str, object] = Field(default_factory=dict)
    error: str | None = None


class ErpHandoffPayload(BaseModel):
    """The package released to ERP/procurement after a successful approval.

    Built only after the matching ``ApprovalRequest`` reaches
    ``approved`` status; never built from a ``pending`` or
    ``rejected`` request. ``recommendations`` is the batch of
    ReplenishmentRecommendations approved for release. ``approval``
    is the audit summary (request id, approver, reason, timestamp)
    that ERP systems persist alongside the receiving transaction
    so the chain of custody is reconstructable downstream.
    """

    handoff_id: str
    run_id: str
    released_at: str
    approval_request_id: str
    approver: str
    approval_reason: str
    approved_at: str
    # Per-series recommendations released to ERP. Typed as a list of
    # dicts (rather than importing ReplenishmentRecommendation
    # directly) to keep the boundary free of internal contract
    # coupling — the cockpit UI and any future consumer do not need
    # to know the platform's full recommendation shape, only the
    # fields ERP needs (sku, location, quantity, unit, request_date).
    recommendations: list[dict[str, object]] = Field(default_factory=list)
    # A flat audit trail joining the request, decision, and release.
    audit_trail: list[ApprovalEvent] = Field(default_factory=list)
