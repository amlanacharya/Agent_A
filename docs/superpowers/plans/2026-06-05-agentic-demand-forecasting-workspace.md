# Agentic Demand Forecasting Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an agentic demand forecasting workspace that converts messy production data into replenishment recommendations, learns from each run, and stays governed through audit, approvals, and observable MLOps.

**Architecture:** Use a governed core for canonical data, feature generation, model training, promotion, replenishment policy, monitoring, and approvals. Add bounded agentic extension paths for EDA, adapters, feature engineering, and forecasting code only when standard tools fail, capped at three attempts per layer. Store durable learning in markdown; use graph/memory systems as indexes over the artifacts.

**Tech Stack:** UiPath for orchestration and approvals; Python forecasting/ML harness; XGBoost and baseline forecasting model families; markdown knowledge artifacts; optional Graphify/mem0-style retrieval index; cockpit UI for observability.

---

## Two Candidate Plans

### Plan A: Superpower-Style Governed Tracer Plan

This plan decomposes the platform into small, independently verifiable subsystems. Each subsystem has a contract, tests, approval gates, and markdown artifacts. Agentic code is allowed only as a controlled escalation path.

Strengths:
- Strong auditability and enterprise trust.
- Clear separation between raw data, canonical data, features, models, policy, and learning.
- Handles messy production data without making the model layer chaotic.
- Fits UiPath orchestration well.
- Supports future subagent-driven implementation.

Weaknesses:
- Slower to build than a demo-first approach.
- Requires discipline around contracts and artifact hygiene.

### Plan B: Inherent Product-First Cockpit Plan

This plan starts from the futuristic Data Intelligence Cockpit and builds visible workflows first: project setup, data health, model arena, forecast review, replenishment board, and learning journal. Backend contracts evolve from what the cockpit needs.

Strengths:
- Better for demos, stakeholder alignment, and user imagination.
- Makes agent state visible early.
- Forces useful product language around decisions, confidence, and exceptions.

Weaknesses:
- Risk of beautiful UI over weak harnesses.
- Could hide core data/model correctness problems until late.
- More likely to create ad hoc backend behavior.

## Winner

Promote **Plan A with Plan B's cockpit as a first-class workstream**.

Reason: this product will fail if the harness is not trustworthy. The cockpit matters, but it must expose real state from governed subsystems, not become a decorative shell. The winning plan is therefore contract-first, evidence-first, and cockpit-visible.

---

## Promoted Plan

### Phase 1: Workspace And Knowledge Substrate

- [x] Define the project folder standard:
  - `CONTEXT.md`
  - `DATA_CONTRACT.md`
  - `LEARNINGS.md`
  - `ASSUMPTIONS.md`
  - `DECISIONS.md`
  - `RUNBOOK.md`
  - `MODEL_REGISTRY.md`
  - `PROMOTION_DECISIONS.md`
- [x] Define memory layers:
  - global product memory
  - customer memory
  - project memory
- [x] Define promotion rules:
  - auto-promote safe technical facts
  - verifier-promote modeling lessons with evidence
  - human-approve business semantics and policies

> ✅ **Phase 1 complete.** Implemented in `learning_workspace.py` (run workspace creation, 8 required artifacts, 3-tier promotion with `LearningPromotionError` guard, memory layer validation). Tests in `test_learning_workspace.py`.

### Phase 2: Data Intake, EDA, And Canonical Contract

- [x] Build the standard EDA toolbox:
  - [x] schema inference — `preflight_schema.py`
  - [x] type detection — `eda_probes.detect_column_types` (per-column dtype inference with contract mismatch surfacing)
  - [x] missingness — `eda_probes.measure_missingness` (per-column + per-row counts; required columns excluded from the rows-with-missing metric)
  - [x] duplicates — `eda_probes.detect_duplicate_keys` (`(series_key, date)` collisions)
  - [x] date gaps — `eda_probes.detect_date_gaps_per_series` (per-series expected/actual gap, out-of-order rows)
  - [x] grain detection — `preflight.py` frequency/grain detection
  - [x] SKU/location cardinality — `preflight.py` series cardinality
  - [x] demand sparsity — `eda_toolbox.py` ADI/CV² per series
  - [x] stockout distortion — `preflight.py` zero runs detection
  - [x] join validation — `eda_probes.validate_joins` (per-dimension coverage + per-series missing issues)
  - [x] leakage checks — `eda_probes.detect_leakage_per_series` (forward correlation at lags 2..5 + demand==inventory probe)
- [x] Build schema mapping into the canonical demand forecasting schema:
  - `sku_id`
  - `location_id`
  - `week_start`
  - `demand_qty`
  - `inventory_qty`
  - `stockout_flag`
  - `price`
  - `promo_flag`
  - `lead_time`
- [x] Add the custom adapter escalation path:
  - standard tool fails
  - agent explains gap
  - human grants coding permission
  - agent gets maximum three tries
  - verifier checks canonical output
  - successful adapter gets tests and markdown card
  - failed adapter produces exact failure report

> ✅ **Phase 2 complete.** EDA toolbox (`eda_probes.py`, 6 new sub-checks) wired into `build_eda_report`; canonical schema mapping (`canonical_data.py`) and adapter escalation (`code_escalation.py`, 6 layers) unchanged. All 11 EDA sub-checks now implemented. New contracts: `TypeDetectionReport`, `MissingnessReport`, `DuplicateReport`, `DateGapsReport`, `JoinValidationReport`, `LeakageReport` — added to `EDAReport` with `None` defaults so existing callers keep working. Tests: `tests/test_eda_probes.py` (27 new) + one regression-guard in `tests/test_eda_toolbox.py`; full suite 200 passing.

### Phase 3: Feature Factory

- [x] Build a versioned Feature Factory shared by all models.
- [x] Include feature families:
  - [x] lag demand — `lag_1`, `lag_2` in `feature_factory.py`
  - [x] rolling statistics — `rolling_mean_4` in `feature_factory.py`
  - [x] seasonality and calendar — Fourier sin/cos terms in `feature_factory.py`
  - [x] price and promotion — promo indicator in `feature_factory.py`
  - [x] stockout and availability — `stockout_rolling_count_4`, `days_since_stockout`, `inventory_cover_ratio` (gated by `use_stockout_features`, requires `stockout_flag` + `inventory_qty`)
  - [x] hierarchy — `parent_lag_1`, `parent_rolling_mean_4` (parent = `sku_id` aggregated across `location_id`; gated by `use_hierarchy_features`)
  - [x] lifecycle and cold-start — `history_length`, `days_since_first_obs`, `cold_start_flag` (gated by `use_lifecycle_features`)
  - [x] intermittency — `rolling_adi_8`, `rolling_cv2_8`, `trailing_zero_run` (gated by `use_intermittency_features`)
- [x] Enforce time-aware feature generation and no-leakage checks.
- [x] Allow agentic feature code only when the Feature Factory cannot express the needed transformation, capped at three tries.

> ✅ **Phase 3 complete.** All 8 feature families implemented in `feature_factory.py`; the time-aware band logic is factored into `_iter_fold_bands()` and reused by every time-dependent family. New `FeatureFlag` fields default to `False` so existing callers (and the 200 pre-existing tests) keep producing identical output. Each new family is fold-aware: rows strictly after the last cutoff get NaN, so walk-forward validation cannot peek. Hierarchy aggregates to `(parent, date)` first so all children of the same parent see the same parent value. Tests: 16 new in `test_feature_factory.py` (one for each new family, one smoke test for all 4 families composing, plus the fold-cutoff NaN guard); full suite 216 passing.

### Phase 4: Forecasting Harness

- [x] Implement governed model families:
  - naive baseline — `forecasting_models.NaiveModel`
  - seasonal naive — `forecasting_models.SeasonalNaiveModel`
  - moving average / exponential smoothing — `forecasting_models.MovingAverageModel` + `ExponentialSmoothingModel` (Holt, additive trend, alpha/beta grid-searched)
  - intermittent demand models — `forecasting_models.CrostonModel`
  - XGBoost/global ML model over canonical features — `forecasting_models.XGBoostGlobalModel` (xgboost >= 3.2, sklearn dep added; recursive lag-1 forecast)
  - aggregate-and-allocate fallback — `forecasting_models.AggregateAllocateModel` (top-down: parent-grain forecast, share-allocation to children)
- [x] Add model creation escalation:
  - only after existing model families fail — `EnsembleTracker` flags `never_surfaced` and `failed_families`; harness drops failed families per-series
  - human permission required — `check_review` gate
  - maximum three agentic attempts — `model_escalation.request_custom_family_attempt` wraps shared `EscalationTracker` with the existing 3-attempt cap
  - output must pass data contract, backtest, robustness, and review gates — `model_escalation.check_data_contract` / `check_backtest` / `check_robustness` / `check_review`; failures produce a `ModelFailureReport` after the cap
- [x] Track ensemble behavior:
  - model weights by segment — `EnsembleTracker.weights_for_segment` (proportional to win rate, with a 5% floor for protected families naive/seasonal_naive/croston)
  - frequently promoted models — `EnsembleTracker.frequently_promoted` (>= 50% best-in-fold rate)
  - models that never surface — `EnsembleTracker.never_surfaced` (ran but never won) plus the harness's `ForecastHarnessReport.never_surfaced` (fit failed entirely)
  - retired but retained model histories — `EnsembleTracker.retire` + `EnsembleSummary.retired`; scorecards stay in the audit history

> ✅ **Phase 4 complete.** Governed model families in `forecasting_models.py`; ensemble tracking in `ensemble.py`; custom-family escalation in `model_escalation.py`; the harness in `forecast_harness.py` ties them together and returns a `ForecastHarnessReport` with scorecards, robustness checks, and the ensemble summary. New contracts in `contracts.py`: `ModelFamilyName`, `ModelScorecard`, `RobustnessCheck`, `ForecastRequest`, `ForecastHarnessReport`, `EnsembleSummary`, `ModelFailureReport`. Tests: 83 new across `test_forecasting_models.py` (24), `test_ensemble.py` (17), `test_model_escalation.py` (27), `test_forecast_harness.py` (15); full suite 299 passing. Dependencies added: `xgboost>=3.2.0`, `scikit-learn>=1.5.0`.
>
> 🧹 **Phase 4 simplified (2026-06-14).** `/simplify` pass across all four phase 4 modules: removed dead `total == 0` branch in `EnsembleTracker.weights_for_segment`; tightened floor loop bound from `len(active)+1` to `len(PROTECTED_FAMILIES)+1`; merged two-pass `summarise_scorecards` into one; moved inference fit outside the per-fold cutoff loop (was re-fitting full history once per fold); pre-computed `lag_1`/`lag_2`/`rolling_mean_4` column indices before the XGBoost horizon loop; cached `history.dropna()` in `AggregateAllocateModel._fit_series`; replaced `_decode_xgboost_model`'s direct `from xgboost import` with the already-passed module parameter; replaced hand-rolled `_median` with `statistics.median`; collapsed `_series_keys_in_order` to `list(dict.fromkeys(...))`; removed `hasattr` guard on monkey-patch; cleaned up `__all__` across all four modules (removed duplicates, private symbols, and non-public imports). Full suite 299 passing.

### Phase 5: Evaluation, Promotion, And Replenishment Policy

- [ ] Define metric portfolio:
  - WAPE
  - bias
  - horizon-level error
  - segment-level error
  - interval coverage
  - stockout and overstock impact
- [ ] Build champion/challenger promotion:
  - fixed backtest windows
  - leakage checks
  - segment scorecards
  - shadow mode where needed
  - promotion decision markdown
- [ ] Build deterministic replenishment policy:
  - lead-time demand
  - safety stock
  - reorder point
  - MOQ
  - pack size
  - current inventory
  - open purchase orders
  - approval thresholds

> ❌ **Phase 5 not started.**

### Phase 6: UiPath Orchestration

- [ ] Route approvals for:
  - data contract
  - risky schema semantics
  - custom code permission
  - unforecastable grain fallback
  - official forecast publication
  - replenishment recommendations
  - ERP/procurement handoff
- [ ] Schedule runs:
  - data refresh
  - validation
  - forecast generation
  - review
  - monitoring
  - drift investigation

> ❌ **Phase 6 not started.** Approval-needed flag exists in `cockpit_state.py` but UiPath routing and scheduling are not implemented.

### Phase 7: Monitoring And Augmented MLOps

- [ ] Monitor data drift:
  - schema changes
  - missing feeds
  - distribution shifts
  - new SKUs or locations
- [ ] Monitor model drift:
  - forecast error
  - bias
  - interval calibration
  - segment degradation
- [ ] Monitor business outcomes:
  - stockouts
  - overstock
  - service level
  - planner overrides
  - approval/rejection patterns
- [ ] Generate recurring artifacts:
  - `MONITORING_REPORT.md`
  - `DRIFT_REPORT.md`
  - `OVERRIDE_ANALYSIS.md`
  - `MODEL_HEALTH.md`

> ❌ **Phase 7 not started.**

### Phase 8: Data Intelligence Cockpit

- [ ] Build cockpit surfaces:
  - Mission Control
  - Data Health
  - Canonical Table Builder
  - EDA Explorer
  - Feature Factory
  - Model Arena
  - Forecast Review
  - Replenishment Board
  - MLOps Monitor
  - Learning Journal
- [x] Show live platform state:
  - current agent step
  - tool result
  - code escalation status
  - attempt count
  - verifier gate
  - approval needed
  - confidence and blockers
- [ ] Make plots available on demand:
  - demand curves
  - sparsity
  - anomalies
  - forecast bands
  - backtests
  - feature importance
  - drift charts

> ⚠️ **Phase 8 partial.** Live state model (`cockpit_state.py`) is complete — all 7 live state fields implemented with `to_public_dict()`, `with_blocker()`, `mark_approval_needed()`. No UI surfaces or plot generation implemented yet.

## Hard Rules

- Agentic code is escalation, not default behavior.
- Each layer gets maximum three code-generation attempts.
- After three failed attempts, stop and produce a failure report.
- Canonical schema remains stable even when adapters are custom.
- Markdown is durable source of truth.
- Graph/memory systems index markdown; they do not replace it.
- Harnesses decide promotion; agents propose candidates and evidence.
- UiPath governs approvals and downstream action.
- Cockpit explains what the platform is doing in real time.

## Self-Review

Spec coverage: covers workspace, DS-STAR resilience, AutoResearch learning, UiPath orchestration, feature engineering, EDA, custom code escalation, model harness, MLOps, monitoring, drift, and cockpit.

Placeholder scan: no open TBD/TODO placeholders remain.

Scope check: this is too large for one engineering implementation plan. It should be split into subsystem plans before coding:
- workspace and markdown memory
- canonical data and EDA
- feature factory
- forecast harness
- MLOps and registry
- UiPath orchestration
- cockpit UI

## Progress Summary (as of 2026-06-14)

| Phase | Status | Notes |
|---|---|---|
| 1: Workspace & Knowledge Substrate | ✅ Complete | `learning_workspace.py` + tests |
| 2: Data Intake, EDA, Canonical Contract | ✅ Complete | `eda_probes.py` (6 new sub-checks) wired into `build_eda_report`; canonical schema + escalation unchanged. 200 tests pass. |
| 3: Feature Factory | ✅ Complete | All 8 families implemented in `feature_factory.py` (4 new: stockout/availability, hierarchy, lifecycle/cold-start, intermittency). Fold-aware band logic factored into `_iter_fold_bands()`. 16 new tests; 216 total pass. |
| 4: Forecasting Harness | ✅ Complete | 6 governed model families + ensemble + custom-family escalation. `forecasting_models.py`, `ensemble.py`, `model_escalation.py`, `forecast_harness.py`. New contracts: `ModelFamilyName`, `ModelScorecard`, `RobustnessCheck`, `ForecastRequest`, `ForecastHarnessReport`, `EnsembleSummary`, `ModelFailureReport`. 83 new tests; 299 total pass. |
| 5: Evaluation, Promotion, Replenishment | ❌ Not started | |
| 6: UiPath Orchestration | ❌ Not started | |
| 7: Monitoring & Augmented MLOps | ❌ Not started | |
| 8: Data Intelligence Cockpit | ⚠️ Partial | Live state model done; no UI surfaces or plots |
