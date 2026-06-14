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

- [ ] Build the standard EDA toolbox:
  - [x] schema inference — `preflight_schema.py`
  - [ ] type detection — not explicitly implemented
  - [ ] missingness — not explicitly implemented
  - [ ] duplicates — not implemented
  - [ ] date gaps — not implemented
  - [x] grain detection — `preflight.py` frequency/grain detection
  - [x] SKU/location cardinality — `preflight.py` series cardinality
  - [x] demand sparsity — `eda_toolbox.py` ADI/CV² per series
  - [x] stockout distortion — `preflight.py` zero runs detection
  - [ ] join validation — not implemented
  - [ ] leakage checks — not implemented
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

> ⚠️ **Phase 2 partial.** Canonical schema mapping (`canonical_data.py`) and adapter escalation (`code_escalation.py`, 6 layers including `eda`, `schema_mapping`, `canonical_table`) are complete. EDA toolbox has 5 of 11 sub-checks; missing: type detection, missingness, duplicates, date gaps, join validation, leakage checks.

### Phase 3: Feature Factory

- [x] Build a versioned Feature Factory shared by all models.
- [ ] Include feature families:
  - [x] lag demand — `lag_1`, `lag_2` in `feature_factory.py`
  - [x] rolling statistics — `rolling_mean_4` in `feature_factory.py`
  - [x] seasonality and calendar — Fourier sin/cos terms in `feature_factory.py`
  - [x] price and promotion — promo indicator in `feature_factory.py`
  - [ ] stockout and availability — not implemented
  - [ ] hierarchy — not implemented
  - [ ] lifecycle and cold-start — not implemented
  - [ ] intermittency — not implemented
- [x] Enforce time-aware feature generation and no-leakage checks.
- [x] Allow agentic feature code only when the Feature Factory cannot express the needed transformation, capped at three tries.

> ⚠️ **Phase 3 partial.** Feature Factory exists with fold-aware generation enforced. 4 of 8 feature families implemented; missing: stockout/availability, hierarchy, lifecycle/cold-start, intermittency.

### Phase 4: Forecasting Harness

- [ ] Implement governed model families:
  - naive baseline
  - seasonal naive
  - moving average / exponential smoothing
  - intermittent demand models
  - XGBoost/global ML model over canonical features
  - aggregate-and-allocate fallback
- [ ] Add model creation escalation:
  - only after existing model families fail
  - human permission required
  - maximum three agentic attempts
  - output must pass data contract, backtest, robustness, and review gates
- [ ] Track ensemble behavior:
  - model weights by segment
  - frequently promoted models
  - models that never surface
  - retired but retained model histories

> ❌ **Phase 4 not started.**

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
| 2: Data Intake, EDA, Canonical Contract | ⚠️ Partial | Canonical schema + escalation done; 6 EDA sub-checks missing |
| 3: Feature Factory | ⚠️ Partial | 4 of 8 feature families; fold-aware generation done |
| 4: Forecasting Harness | ❌ Not started | |
| 5: Evaluation, Promotion, Replenishment | ❌ Not started | |
| 6: UiPath Orchestration | ❌ Not started | |
| 7: Monitoring & Augmented MLOps | ❌ Not started | |
| 8: Data Intelligence Cockpit | ⚠️ Partial | Live state model done; no UI surfaces or plots |
