# Agent P — Architecture Design Spec
**Date:** 2026-05-30
**Version:** v2.0 (post grill-with-docs session)
**Supersedes:** plan_v1.md
**Domain:** FMCG default — multi-domain via playbooks
**Stack:** FastAPI + React (Vite) + Claude Sonnet + Python tools

---

## 1. What Changed From plan_v1.md

Seven corrections made during the design review session. Everything else carries forward unchanged.

1. **Verification statuses** — 5-status set from plan.md replaces the 4-status set in plan_v1.md: `SUPPORTED`, `CONTRADICTED`, `AMBIGUOUS`, `UNVERIFIABLE`, `USER_OVERRIDE_ACCEPTED`. `CONTRADICTED` and `UNVERIFIABLE` are distinct states with different Meridian behaviours.
2. **Evidence types** — 5 types formalised: `statistical_test`, `association`, `pattern`, `user_confirmed`, `unverifiable_business_input`. `statistical_test` and `association` require a preceding tool call. See §6.
3. **Prism re-classifies Demand Class** — `run_forge_for_scenario` must re-run `classify_demand_profiles` for affected series before re-running feature config. A scenario override can shift a series' demand class; freezing it at baseline would produce invalid model gates. See ADR-0001.
4. **Preflight bundle** — `series_profiles` is renamed `segment_profiles` (aggregate statistics per segment, not per series) plus a new `segment_exceptions` field (small list of per-series statistical outliers). Individual series data stays in `data_store`.
5. **Run State** — `pipeline_state` is a single `RunState` Pydantic model persisted to `outputs/{run_id}/run_state.json` after every mutation. Loaded at the start of every request. No in-memory-only state for anything that must survive across requests. See ADR-0003.
6. **Guard Layer limits** — all limits (token budget, max tool calls per agent, duplicate call threshold) are `.env`-configurable. Values stated in this spec are defaults only.
7. **Backend logging** — every tool call, agent invocation, guard check, and pipeline state transition emits `log.info` to the backend terminal via Python's `logging` module. Separate from `obs_log.json`.

**Also resolved:** plan_phase2.md is superseded and retired. POC_v1 (`C:\FFA\POC_v1`) is used as a statistical reference only — no files are imported or ported. See ADR-0004.

---

## 2. Named Agent Roster

| Name | Role | Model |
|---|---|---|
| **Lens** | Intent classification — reads conversation, produces typed intent_pack | Claude Haiku (fast, cheap) |
| **Conductor** | Orchestration — routes intent to domain agents, manages Run State | Claude Sonnet |
| **Meridian** | Domain Specialist — scoping conversation, claim ledger, domain_context_pack | Claude Sonnet |
| **Forge** | EDA Agent — translates confirmed pack into per-segment modelling spec | Claude Sonnet |
| **Foundry** | Model Selection + Ensemble — SKU-level model execution, self-correction | Claude Sonnet |
| **Prism** | Scenario Runner — inherits confirmed pack, re-runs Forge+Foundry with overrides | Claude Sonnet |

**Pre-flight layer** — deterministic Python, no LLM. Runs on upload before any agent.

---

## 3. Architecture

```
User message / file upload
    ↓
Lens            — classifies intent, extracts entities → intent_pack
    ↓
Conductor       — reads intent_pack + RunState → orchestration_directive
    ↓ routes to
┌──────────┬──────────┬──────────┬──────────────────────┐
│ Meridian │  Forge   │ Foundry  │  Prism               │
│ scoping  │  spec    │  models  │  (child scenario run) │
└──────────┴──────────┴──────────┴──────────────────────┘
    ↓ all output via SSE stream
React frontend
```

**Data never travels through Claude's context.** Series data is stored in `data_store` on upload. All agents receive `run_id` and call tools that read from `data_store` internally. The sentinel pattern applies to all agents.

**The flow is a loop, not a chain.** The arrows above are the happy path, but any domain agent can pause the Run for human input (an Agent Need → Checkpoint) and the Conductor can route a Scope Amendment *backward* to Meridian (a Loop-back) before continuing. Agents also narrate their reasoning as they work, streamed live. So post-confirmation is a supervised agent loop with human checkpoints — not a silent pipeline. See ADR-0005; CONTEXT 'Agent Need', 'Pause', 'Loop-back', 'Agent Narration'.

---

## 4. Lens — Intent Agent

Lens is a **structured-output call, no tool use**. It receives the last 6 conversation turns + current user message + `pipeline_state` and returns a typed `intent_pack`. No tool loop — one call, one response.

Lens weights `pipeline_state` and the last agent message heavily when classifying short or ambiguous messages (e.g., "ok", "yes"). A short message after a risk warning is `SCOPE_RESPONSE`; after a proceed question it is `ADVANCE_PIPELINE`. The 0.6 confidence threshold handles genuine ambiguity — context resolves the rest. See ADR-0002.

**Input:** `{ conversation_history[-6:], user_message, pipeline_state }`

**Output:**
```python
intent_pack = {
    "intent": Literal[
        "SCOPE_RESPONSE",      # user answering Meridian's question
        "OVERRIDE",            # user overriding an agent recommendation
        "ADVANCE_PIPELINE",    # "ok let's model now", "looks good"
        "WHAT_IF_REQUEST",     # "what if promo on SKU X week 10"
        "CLARIFICATION",       # user asking a question
        "CORRECTION",          # fixing a prior statement — only valid during meridian_scoping
                               # post-confirmation, treated as OVERRIDE; Conductor offers Scenario Run
    ],
    "entities": {
        "skus":     ["SKU_101"],
        "segments": ["G2"],
        "dates":    ["2024-10-01"],
        "metrics":  ["promo", "horizon"],
        "scenario": "20% promo on SKU_101 weeks 10–12"
    },
    "confidence": 0.91,
    "raw_quote":  "what if we run a promo on SKU_101"
}
```

Lens runs on every user message. Conductor never parses raw user language.

---

## 5. Conductor — Orchestration Agent

Conductor receives `intent_pack + RunState` and issues a single `orchestration_directive`. It does no data work — pure control-plane logic.

**Run State (authoritative, persisted to disk):**
```python
RunState = {
    "run_id":                  str,
    "phase":                   Literal["preflight", "meridian_scoping", "forge_eda",
                                       "foundry_modelling", "report_ready"],
    "pack_confirmed":          bool,
    "pack_version":            int,         # bumped on every scope amendment (loop-back)
    "meridian_turn_count":     int,
    "forge_complete":          bool,
    "foundry_complete":        bool,
    "active_whatif_runs":      list[str],   # whatif_ids
    "open_risks":              int,         # count of unacknowledged risks only
    "override_count":          int,
    "awaiting_input":          dict | None, # AgentNeed payload when paused; None when running
    "loopback_count":          int,         # scope-amendment loop-backs so far (capped)
    "tokens_used_total":       int,         # cumulative — persists across pause/resume
    "foundry_calls_total":     int,         # cumulative Foundry tool calls — persists across pause/resume
    "halt_reason":             str | None,
    "domain":                  str,
    "created_at":              str          # ISO timestamp
}
```

`awaiting_input`, `loopback_count`, `tokens_used_total`, and `foundry_calls_total` exist because a Pause resumes by *re-invoking* the agent (it skips already-done steps via persisted artifacts); the cumulative counters must live here and be seeded/written-back per invocation, or resume would silently reset the Guard. See §13 and ADR-0005.

`open_risks` counts only unacknowledged risks — a risk is closed when the user explicitly acknowledges it (becomes `ACCEPTED_RISK` Claim) or Meridian resolves it via subsequent diagnostic evidence.

**Conductor tools:**
```
get_run_state(run_id)
    → full RunState dict (loaded from run_state.json)

update_run_state(run_id, patch)
    → applies patch to RunState, persists to run_state.json

advance_to_meridian(run_id, user_message)
    → routes message + preflight_bundle to Meridian, opens SSE stream

confirm_pack_and_advance(run_id)
    → locks domain_context_pack, triggers Forge
    → only callable when phase=meridian_scoping AND pack_confirmed=False
      AND intent=ADVANCE_PIPELINE AND open_risks=0

trigger_foundry(run_id)
    → called after Forge completes

create_prism_run(run_id, scenario_description, entities)
    → creates child run, returns whatif_id
    → only callable when phase=report_ready

get_agent_status(run_id, agent)
    → "done" | "running" | "failed" | "not_started"

surface_clarification(run_id, message)
    → Conductor-authored clarification pushed to user (never exposes internal mechanics)
    → offers two options derived from closest candidate intents
    → used when intent.confidence < 0.6

log_halt(run_id, reason)
    → writes halt to obs log, pushes error SSE event

resolve_need(run_id, answer)
    → consumes the user's answer to the current awaiting_input AgentNeed:
      • USER_DECISION  → clears awaiting_input, resumes the SAME agent (which re-invokes,
                         skips completed steps, and records the outcome itself)
      • SCOPE_AMENDMENT → routes to Meridian (a loop-back); see reroute
    → only callable when awaiting_input != None

reroute(run_id, target_phase, reason)
    → moves phase backward for a scope amendment (e.g. foundry_modelling → meridian_scoping),
      increments loopback_count, bumps pack_version on re-lock
    → after Meridian re-locks the amended pack, only the affected series are re-run
      (reuses Prism's affected-series engine on the main namespace)
```

**Agentic interaction (Needs / Pauses / Loop-back).** Any domain agent may call `raise_need(kind, question, options, context)` when it hits a decision with no defensible default. This raises a resumable `PauseForInput`, writes the need to `awaiting_input`, emits `agent_needs_input`, and suspends the Run (non-terminal — *not* a Halt). Conductor routes by `kind`: `USER_DECISION` it voices itself (an extension of `surface_clarification`); `SCOPE_AMENDMENT` it routes to Meridian via `reroute`. Checkpoints are rare by design — agents take a documented default and log a Claim wherever one exists. See ADR-0005.

**Conductor hard rules (enforced in system prompt):**
- Never call `confirm_pack_and_advance` if `open_risks > 0` and user has not acknowledged risks
- Never call `trigger_foundry` before `forge_complete = True`
- Never call `create_prism_run` before `phase = report_ready`
- If `intent.confidence < 0.6`, call `surface_clarification` — do not guess
- A Halt is terminal — never attempt to resume a halted run
- When `awaiting_input != None`, the only valid action is `resolve_need` — do not advance or re-route otherwise; a free-text answer is classified by Lens first
- Never exceed the loop-back cap (`loopback_count`, default 3, `.env`) — the Guard Halts the run if it would
- A loop-back is a *scoped* amendment to the specific finding — never reopen general scoping

**Conductor output is not streamed.** Its routing decisions surface via `phase_change` and `decision_update` SSE events only. Users never see Conductor's tool call trace.

---

## 6. Meridian — Domain Specialist

Meridian conducts the scoping conversation. It operates at segment level — never interrogating individual series one-by-one. It presents evidence first and asks the user to interpret. It uses humble language and never overclaims causality.

**Conversation arc:** Phases 0–6 from plan.md §7 (unchanged). Phase 4 is a *review* step — `compile_domain_context_pack` assembles what has been incrementally built via tool calls throughout Phases 1–3, then Meridian presents a readable formatted summary (not raw JSON) for user review.

**Verification statuses (5):**
```
SUPPORTED              → data confirms the claim (add to pack with evidence)
CONTRADICTED           → data actively pushes back (challenge user with evidence, offer constrained alternatives)
                         Transient only — must resolve to SUPPORTED or USER_OVERRIDE_ACCEPTED before pack confirmation
AMBIGUOUS              → data is inconclusive (conservative assumption applied, risk logged)
UNVERIFIABLE           → data cannot speak to this (accepted with caveat, risk logged)
USER_OVERRIDE_ACCEPTED → user overrode a data-backed recommendation (consequence logged, must surface in report)
```

Meridian makes its case once on `CONTRADICTED`. If the user still insists, it accepts as `USER_OVERRIDE_ACCEPTED` and moves on. It never refuses or loops.

**Evidence types (5), in descending strength:**
```
statistical_test          → ADF, Chow test, KPSS, etc. — requires preceding tool call with result cited
association               → co-occurrence without established causality — requires preceding tool call
pattern                   → series shape consistent with interpretation — can be asserted conversationally
user_confirmed            → user asserted it, data could not corroborate or deny
unverifiable_business_input → forward-looking claim; historical data structurally cannot address — always
                              accompanied by an open Risk
```

**Pre-flight tools** (deterministic Python — run on upload, before Meridian starts):
```
profile_uploaded_data()
map_schema()
detect_frequency_and_grain()
build_series_keys()
compute_adi_cv2_per_series()
detect_zero_runs_per_series()
detect_spikes_per_series()
measure_promo_alignment()
detect_trend_strength()
detect_seasonality_strength()
detect_structural_break_candidates()
```

**Preflight bundle** (handed to Meridian via Conductor):
```python
preflight_bundle = {
    "data_quality_report":  {...},   # blocking issues + warnings
    "schema_mapping":       {...},
    "grain_report":         {...},
    "segment_profiles":     [...],   # aggregate statistics per segment — NOT per series
    "segment_exceptions":   [...],   # small list of per-series statistical outliers within segments
    "segments":             [...],
    "domain_playbook":      {...}    # loaded from YAML; guide not hard rules
}
```

Series data stays in `data_store`. Individual series never travel through Claude's context.

**Mid-level diagnostic tools** (Meridian calls dynamically in conversation):
```
summarise_demand_segments(segment_id?)
diagnose_zero_demand_policy(segment_id)
diagnose_spike_policy(segment_id)
diagnose_granularity_feasibility(min_series?)
diagnose_horizon_feasibility(horizon_periods, segment_id?)
diagnose_structural_break_candidates(date, segment_id?)
diagnose_forecastability_by_segment(segment_id)
refine_segments(segments)   # rewrite the provisional segment map when the user adjusts the cut
```

**Provisional segments:** Pre-flight produces a *provisional, suggested* segment map deterministically (playbook `segment_by` grain hint → grouping; else a single segment `G1`). The playbook is a guide, not a hard rule. Meridian presents the suggestion and the user may refine it via `refine_segments`; the map is frozen into the Domain Context Pack at confirmation. All `segment_id → series_keys` resolution goes through this map — diagnostics never substring-match the id against series keys. See CONTEXT 'Segment'.

**Pack management tools:**
```
add_claim(claim, evidence_ref, verification_status, evidence_type,
          applies_to, downstream_impact)
    → appends to claim_ledger, returns claim_id

add_risk(risk, severity, source)
    → appends to risk_register

log_override(decision, agent_recommendation, user_reason,
             consequence, severity)
    → appends claim with verification_status=USER_OVERRIDE_ACCEPTED
    → sets must_surface_in_report=True
    → NOTE: not a separate store — a filtered view of the claim_ledger

compile_domain_context_pack(run_id)
    → assembles pack from claim_ledger
    → returns pack_complete: bool, validation_errors[], validation_warnings[]
    → writes to outputs/{run_id}/domain_context_pack.json

    Validation errors (blocking — pack cannot be confirmed):
      - no forecast scope (missing target, grain, or horizon)
      - no segments defined
      - any Claim still in CONTRADICTED status
      - open_risks > 0 with none acknowledged
      - required schema fields missing

    Validation warnings (non-blocking — surfaced to user):
      - all Claims are UNVERIFIABLE or USER_OVERRIDE_ACCEPTED (no data-backed claims)
      - a segment has no segment-level rules
      - a data-available feature has no policy set
      - override count > 3
```

**Max tool calls:** default 20, `.env`-configurable.

**Structural breaks:** Candidate break dates (statistical signal from pre-flight) must be explicitly confirmed by the user before Forge uses them. Silence does not confirm. Unacknowledged candidates are treated as absent.

---

## 7. Forge — EDA Agent

Forge receives the confirmed `domain_context_pack` via its system prompt (injected before first call). No conversation — one pass of tool calls producing the `eda_report`.

**Forge tools:**
```
run_full_eda(segment_id)
classify_demand_profiles(segment_id)
    → Syntetos-Boylan matrix per series → SMOOTH/ERRATIC/INTERMITTENT/LUMPY
detect_structural_breaks(segment_id, confirmed_dates[])
    → Chow test at pack-confirmed break dates only
flag_stockouts(segment_id, threshold_weeks)
specify_feature_config(segment_id, demand_class)
design_walk_forward_folds(segment_id, horizon, break_dates[])
    → minimum 2 folds post-break required
    → if < 2 folds possible post-break: series flagged caution,
      "insufficient post-break history for reliable validation"
    → pre-break data never used as fallback
select_evaluation_metric(demand_class)
    → MASE for SMOOTH/ERRATIC/INTERMITTENT; MAD+fill_rate for LUMPY
compile_eda_report(run_id)
    → writes to outputs/{run_id}/eda_report.json
```

**Checkpoint (Agent Need):** if Forge detects a *strong structural break that is not in the pack's confirmed set*, it does not silently ignore it (silence ≠ confirm) — it calls `raise_need(kind="SCOPE_AMENDMENT", …)`, pausing the Run so Meridian can put the break to the user. All other forks (e.g. <2 post-break folds) take their documented default — flag `caution`, log a Claim — and do not pause. Forge **narrates** its work (`agent_reasoning`) as it goes.

**Max tool calls:** default 20, `.env`-configurable.

---

## 8. Foundry — Model Selection & Ensemble

Foundry runs at SKU-series level. Segment policy gates the model search space at the tool layer.

**Demand class → allowed models (gate enforced in tool dispatcher):**
```
SMOOTH        → XGBoost, LightGBM, RandomForest, Ridge, ARIMA, ETS-additive
ERRATIC       → GradientBoosting, ETS-multiplicative, Holt-Winters
INTERMITTENT  → Croston, SBA, ADIDA, TSB
LUMPY         → SBA, TSB, ADIDA, zero-inflated models
```

**MASE target — three-layer hierarchy:**
1. Universal floor: MASE < 1.0 (better than naïve) — always required
2. Playbook default: domain-specific threshold (e.g., MASE < 0.8 for FMCG)
3. User override: set as a Claim during Meridian scoping, flagged `ACCEPTED_RISK` if Meridian's evidence suggests it is unrealistic
`assess_target_feasibility` checks against whichever layer is active.

**Foundry tools:**
```
get_segment_series_list(segment_id)
train_and_evaluate(series_key, model_name, hyperparams)
    → rejects if model outside demand_class gate
walk_forward_validate(series_key, model_name, n_folds)
build_ensemble(series_key, base_models[], strategy)
    → only considered if single model plateaus AND delta > 5% on MASE
assess_target_feasibility(series_key)
    → achievable: bool; if False: theoretical_floor_mase, recommendations[]
record_series_result(series_key, result)
compile_foundry_report(run_id)
    → writes to outputs/{run_id}/foundry_report.json
```

**Self-correction** (agentic equivalent of a data scientist's iterative modelling session):
- Round 1: best single model → if MASE meets target, done
- Round 2: structural change within same family (different feature config or next model in family) — re-runs walk-forward validation as the model demands it
- Round 3: most complex model in family, then ensemble if delta > 5% MASE
- Still failing: `assess_target_feasibility()` → declare unforecastable with reasoning

**Forecastability outcomes:**
- `forecastable` — model meets MASE target
- `caution` — modellable but carries known risk (marginal history, high CV², ACCEPTED_RISK Claim, or insufficient post-break folds); can be assigned pre-modelling by Meridian/Forge or post-modelling by Foundry
- `unforecastable` — all self-correction rounds exhausted, `assess_target_feasibility` confirmed theoretical floor exceeds target

**Checkpoint (Agent Need):** after its deterministic pass, Foundry **batches** all series that fell short of the MASE target (after self-correction) into a single `raise_need(kind="USER_DECISION", …)` — *"N series fell short — drop / relax target / accept as caution?"* — never one pause per series. Picking *relax target* promotes it to a `SCOPE_AMENDMENT` (it edits `mase_target`). On resume, Foundry re-invokes, skips completed series, and records the chosen outcome. Foundry **narrates** (`agent_reasoning`) per segment / self-correction escalation.

**Max tool calls:** guard layer manages cumulative Foundry tool calls per run against `max_tool_calls_foundry` (default 500, `.env`-configurable). The counter is per-run **persisted** in Run State (`foundry_calls_total`) so it survives pause/resume — see §13.

---

## 9. Prism — Scenario Runner

Prism is a child run. It inherits the locked `domain_context_pack` from the parent run and the baseline `foundry_report`. It applies a scenario override and re-runs Forge + Foundry **for affected series only**.

**Demand Class reclassification:** If a scenario override materially changes a series' demand pattern (e.g., injecting a promo event or stockout), Prism re-runs `classify_demand_profiles` for affected series before re-running feature config and model selection. The Demand Class is not frozen at baseline — see ADR-0001.

**Prism tools:**
```
get_baseline_result(run_id, series_key?)

parse_scenario(scenario_description, intent_entities)
    → structured override:
      {
        "type": "ADD_PROMO_EVENT" | "CHANGE_HORIZON" | "EXCLUDE_SERIES" |
                "CHANGE_PRICE" | "INJECT_STOCKOUT" | "CHANGE_FEATURE_FLAG",
        "affected_series": ["SKU_101|WEST"],
        "modification": { "weeks": [10, 11, 12], "uplift_pct": 20 }
      }

apply_scenario_to_pack(run_id, whatif_id, structured_override)
    → clones domain_context_pack with override applied

classify_demand_profiles_for_scenario(whatif_id, affected_series[])
    → re-runs Syntetos-Boylan for affected series
    → required before run_forge_for_scenario if override may shift demand class

run_forge_for_scenario(whatif_id, affected_series[])
    → re-runs feature_config for affected series
    → stationarity and break detection inherited from baseline

run_foundry_for_scenario(whatif_id, affected_series[])
    → re-runs train_and_evaluate + walk_forward_validate for affected series
    → uses same model family as baseline unless demand class shifted or metric degrades > 10%

compile_comparison(whatif_id)
    → side-by-side per series: baseline vs scenario
    → writes to outputs/{run_id}/whatif/{whatif_id}/comparison.json
```

**Max tool calls:** default 20, `.env`-configurable.

---

## 10. Backend API

FastAPI. Serves the built React bundle at `/` in production. No auth (solo analyst POC).

```
POST /api/v1/runs/create
    body: { domain: str }
    → { run_id: str }

POST /api/v1/runs/{run_id}/upload
    multipart: file (CSV), domain (str)
    → PreflightResponse (data_quality, schema_mapping, grain_report,
                         segments, series_count, blocking_issues, warnings)

POST /api/v1/runs/{run_id}/message
    body: { content: str }
    → 202 Accepted (response comes via SSE stream)

GET /api/v1/runs/{run_id}/stream
    → text/event-stream (see §12)

GET /api/v1/runs/{run_id}/decisions
    → { claims[], risks[], overrides[] }

GET /api/v1/runs/{run_id}/report
    → full foundry_report JSON

GET /api/v1/runs/{run_id}/artifacts/{name}
    → file download (domain_context_pack.json, eda_report.json,
                     foundry_report.json, obs_log.json, run_state.json)

POST /api/v1/runs/{run_id}/whatif
    body: { scenario_description: str }
    → { whatif_id: str }

GET /api/v1/runs/{run_id}/whatif/{whatif_id}/stream
    → SSE (same event types as main stream, scoped to scenario run)

GET /api/v1/runs/{run_id}/whatif/{whatif_id}/compare
    → comparison JSON (baseline vs scenario per series)

GET /api/v1/runs
    → [{ run_id, domain, phase, series_count, created_at }]

GET /api/v1/runs/{run_id}
    → RunSummaryResponse (phase, claim_count, risk_count, override_count,
                          token_usage, halt_reason, final_pack_created)
```

---

## 11. Frontend Structure

**Tech stack:** React 18 + Vite + TypeScript + Tailwind CSS + shadcn/ui
**State:** Zustand (4 stores)
**Streaming:** EventSource (SSE)
**HTTP:** fetch

**Zustand stores:**
```typescript
runStore      — run_id, pipeline_state, phase, conversation history
streamStore   — partial: string (current streaming token buffer)
decisionStore — claims[], risks[], overrides[]
prismStore    — whatif_id, scenario, comparison_result
```

**Component tree:**
```
App
├── RunsSidebar
│   └── RunCard (× N)
├── MainPanel
│   ├── PhaseBar
│   ├── ConversationView          ← meridian_scoping, AND whenever awaiting_input != None
│   │   ├── MessageBubble
│   │   ├── StreamingBubble
│   │   └── CheckpointBubble       ← renders an Agent Need: question + options[] as buttons
│   ├── PipelineProgress          ← active during forge_eda + foundry_modelling
│   │   ├── ForgeProgress
│   │   ├── FoundryProgress
│   │   └── ActivityFeed           ← live agent_reasoning narration
│   ├── ReportSummary             ← shown when phase = report_ready
│   │   ├── TechnicalView         ← MASE, fold ranges, self-correction, model reasoning
│   │   ├── BusinessView          ← plain language, MAPE translation, open risks
│   │   └── PrismButton           ← hidden until report_ready
│   └── InputBar
│       └── MessageInput
├── DecisionPanel                 ← collapsible right panel
│   ├── ClaimCard (× N)
│   ├── RiskList
│   └── OverrideList
└── PrismDrawer
    ├── ScenarioInput
    ├── PrismProgress
    └── ComparisonTable
```

**Three layout zones:**
- Left 200px: RunsSidebar
- Centre flex: MainPanel
- Right 320px: DecisionPanel (collapsible, decision log only — no raw tool trace)

---

## 12. SSE Event Contract

| Event | Payload | Consumer |
|---|---|---|
| `token` | `{ content: str }` | streamStore.partial += content |
| `message_done` | `{ agent: str, full_text: str }` | runStore.history.push, streamStore.partial = "" |
| `decision_update` | `{ claim_id, claim, verification_status, evidence_type, applies_to }` | decisionStore.claims.push |
| `risk_update` | `{ risk, severity, source }` | decisionStore.risks.push |
| `override_update` | `{ decision, consequence, severity }` | decisionStore.overrides.push |
| `phase_change` | `{ phase: str }` | runStore.phase, PhaseBar re-renders |
| `forge_progress` | `{ segment_id, status: "pending"\|"running"\|"done" }` | ForgeProgress |
| `foundry_progress` | `{ done: int, total: int, by_segment: dict }` | FoundryProgress |
| `agent_reasoning` | `{ agent: str, text: str }` | append to PipelineProgress activity feed |
| `agent_needs_input` | `{ agent, kind, question, options: list, context }` | set `awaiting_input`; render checkpoint bubble in ConversationView |
| `pipeline_done` | `{ forecastable, caution, unforecastable }` | show ReportSummary |
| `error` | `{ reason: str, halt_reason: str }` | error state |

Conductor's tool call trace is never streamed — routing decisions surface as `phase_change` and `decision_update` events only. `agent_reasoning` carries only plain-language decisions/findings from domain agents (Forge/Foundry/Prism/Meridian), never raw tool JSON; Conductor never narrates.

---

## 13. Guard Layer

Shared across all agents. Enforced in the tool dispatcher. All limits `.env`-configurable.

- **Token budget:** 80,000 tokens per run (default). Checked before each API call.
- **Max tool calls:** 20 per agent invocation for Conductor, Meridian, Forge, and Prism (default). Foundry is exempt — cumulative Foundry tool calls tracked per run against `max_tool_calls_foundry` (default 500).
- **Duplicate call detection:** MD5 hash of `(tool_name, json.dumps(args, sort_keys=True))`. Hard stop after 2 identical calls (default). Tracked per agent *invocation*; safe across resume because re-invocation short-circuits already-done steps via artifacts rather than re-dispatching them.
- **Loop-back cap:** `loopback_count` (default 3, `.env`). A scope-amendment loop-back that would exceed it raises `GuardHalt`.
- **Cumulative budgets persist in Run State:** `tokens_used_total` and `foundry_calls_total` live in `run_state.json`, seeded and written back on every (re-)invocation — so a Pause/resume (or restart) cannot reset a budget. The Foundry counter is therefore per-run *persisted*, not just a per-instance field (refines the per-run fix).
- **Sentinel pattern:** Series data in `data_store` only. Tools receive `run_id`. Series never in Claude's context.
- **Pause ≠ Halt:** `PauseForInput` (raised by `raise_need`) suspends the Run *non-terminally* — it is resumable and resets nothing. `GuardHalt` (or Conductor `log_halt`) ends the Run permanently and cannot be resumed; halt reason written to `run_state.json` and surfaced via `error` SSE event.

---

## 14. Observability

**Backend terminal logging:** Every tool call, agent invocation, guard check, and pipeline state transition emits `log.info` via Python's standard `logging` module. This is the primary real-time debugging surface.

**Structured artifact logging:** Every tool call also logged as structured JSON to `outputs/{run_id}/obs_log.json`:
```json
{
    "timestamp":      "2026-05-30T10:14:32.441Z",
    "tool_name":      "diagnose_spike_policy",
    "input_args":     {"segment_id": "G2"},
    "output_summary": "...",
    "latency_ms":     342.7
}
```

**Run summary** written to `outputs/{run_id}/run_summary.json`:
```json
{
    "run_id":             "...",
    "domain":             "FMCG",
    "phase":              "report_ready",
    "claim_count":        8,
    "open_risk_count":    3,
    "override_count":     1,
    "call_count":         42,
    "token_usage":        { "input": 24300, "output": 6100 },
    "halt_reason":        null,
    "final_pack_created": true
}
```

---

## 15. File Structure

```
C:\Agent_A\                         ← code/dir (product name is "Agent P")
├── plan_v2.md                      ← this file (authoritative architecture)
├── CONTEXT.MD                      ← domain glossary (authoritative)
├── docs/
│   ├── plans/plan.md               ← implementation plan (Tasks 1–31)
│   ├── specs/spec.md               ← behavioural spec (contracts + invariants)
│   ├── adr/
│   │   ├── 0001-scenario-reclassifies-demand-class.md
│   │   ├── 0002-lens-uses-pipeline-state-as-classification-prior.md
│   │   ├── 0003-run-state-persisted-to-json-file.md
│   │   └── 0004-no-direct-port-from-poc-v1.md
│   └── superpowers/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── domain_playbooks/
│   │   ├── fmcg.yaml
│   │   └── nbfc.yaml
│   ├── outputs/
│   │   └── {run_id}/
│   │       ├── run_state.json          ← RunState (single source of truth)
│   │       ├── preflight.json
│   │       ├── domain_context_pack.json
│   │       ├── eda_report.json
│   │       ├── series_results/
│   │       │   └── {series_key}.json
│   │       ├── foundry_report.json
│   │       ├── obs_log.json
│   │       ├── run_summary.json
│   │       └── whatif/
│   │           └── {whatif_id}/
│   │               ├── modified_pack.json
│   │               └── comparison.json
│   └── forecasting/
│       ├── __init__.py
│       ├── data_store.py
│       ├── guard.py
│       ├── observability.py
│       ├── preflight.py
│       ├── providers.py
│       ├── playbooks.py
│       ├── run_state.py                ← RunState Pydantic model + load/save helpers
│       ├── agents/
│       │   ├── lens.py
│       │   ├── conductor.py
│       │   ├── meridian.py
│       │   ├── forge.py
│       │   ├── foundry.py
│       │   └── prism.py
│       ├── contracts.py
│       └── api/
│           ├── models.py
│           ├── sse.py
│           └── routers/
│               ├── runs.py
│               ├── stream.py
│               ├── message.py
│               └── whatif.py
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── stores/
│       │   ├── runStore.ts
│       │   ├── streamStore.ts
│       │   ├── decisionStore.ts
│       │   └── prismStore.ts
│       ├── components/
│       │   ├── RunsSidebar/
│       │   ├── MainPanel/
│       │   │   ├── PhaseBar/
│       │   │   ├── ConversationView/
│       │   │   ├── PipelineProgress/
│       │   │   ├── ReportSummary/
│       │   │   │   ├── TechnicalView/
│       │   │   │   └── BusinessView/
│       │   │   └── InputBar/
│       │   ├── DecisionPanel/
│       │   └── PrismDrawer/
│       ├── hooks/
│       │   └── useSSE.ts
│       └── api/
│           └── client.ts
│
└── tests/
    ├── test_guard.py
    ├── test_data_store.py
    ├── test_run_state.py
    ├── test_preflight.py
    ├── test_lens.py
    ├── test_conductor.py
    ├── test_meridian_tools.py
    ├── test_forge_tools.py
    ├── test_foundry_tools.py
    └── test_prism_tools.py
```

---

## 16. Out of Scope for POC

- Authentication / multi-user sessions
- Real-time collaboration on a run
- Model serving / deployment endpoint
- Automated retraining schedules
- External APIs (competitor data, weather, social sentiment) — Phase 2+
- LLM-as-judge eval framework for Meridian conversation quality
- Golden conversation regression tests
- Async FastAPI handlers (sync is fine for POC)
- Mid-tool-call interruption / true preemption (deferred "lever 4"). Agents pause only at Need checkpoints; the user interjects at turn boundaries, not mid-call. Revisit if real-time interruption becomes necessary (would require async). See ADR-0005.
- Mobile / responsive layout
- Dark mode
- MCP tool servers (revisit Phase 2 if tool surface grows)
- LangGraph (RunState file approach is sufficient for POC concurrency model)

---

## 17. Handoff Contracts

```python
# Pre-flight → Meridian (via Conductor)
preflight_bundle = {
    "data_quality_report": {...},
    "schema_mapping":      {...},
    "grain_report":        {...},
    "segment_profiles":    [...],   # aggregate per segment
    "segment_exceptions":  [...],   # per-series outliers only
    "segments":            [...],
    "domain_playbook":     {...}
}

# Meridian → Forge (via Conductor confirming pack)
{
    "domain_context_pack": {...},
    "claim_ledger":        [...],
    "override_count":      1,
    "open_risks":          [...],
    "user_confirmed":      True,
    "timestamp":           "ISO"
}

# Forge → Foundry (via Conductor)
{
    "eda_report":          {...},
    "domain_context_pack": {...},
    "open_risks":          [...],
    "timestamp":           "ISO"
}

# Foundry → Report Layer
{
    "per_series_results":     [...],
    "segment_summary":        {...},
    "domain_context_pack":    {...},
    "open_risks":             [...],
    "override_consequences":  [...],   # downstream_impact from USER_OVERRIDE_ACCEPTED Claims
    "timestamp":              "ISO"
}

# Prism inherits: domain_context_pack + foundry_report from parent run_id
```

Open risks and override consequences accumulate across all agents and are never dropped.

---

## 18. Series Key Format

Series keys are pipe-delimited concatenations of grain dimension values in fixed order defined by the domain playbook's `common_grains` field. The time dimension is always excluded (it is the time axis, not a key dimension).

Example: `sku_region_week` grain → `SKU_101|WEST`

Normalisation at pre-flight (immutable for the Run lifetime):
- Values uppercased
- Spaces replaced with underscores
- Characters outside `[A-Z0-9_|]` stripped

Used as `data_store` dict keys. As file path components, the `|` separator is illegal in Windows filenames, so it is percent-encoded to `%7C` through one reversible mapping in `data_store` (`key_to_filename`/`filename_to_key`) — e.g. `series_results/SKU_101%7CWEST.json`. Keys never contain `%` (stripped by normalisation), so the mapping is lossless. All `series_results/` reads and writes route through it; no `replace("|","_")` at call sites.