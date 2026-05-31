# Agent P

A multi-agent demand forecasting assistant for FMCG (and other domains). A user uploads time-series sales data, converses with the system to scope and validate the forecast, and receives a structured forecast report with optional what-if scenario analysis.

## Language

### Data & Series

**Segment**:
A named grouping of time-series (`segment_id` like `G1`, `G2`). The pre-flight layer produces a *provisional* segmentation deterministically: the domain playbook is a **guide** — if it names a `segment_by` grain dimension present in the data, series are grouped by that dimension's distinct values; otherwise all series fall into a single segment (`G1`). This suggested cut is presented to the user, who can refine it (merge, split, re-segment) during the Meridian conversation via `refine_segments`. The map stays provisional through `meridian_scoping` and is **frozen into the Domain Context Pack at confirmation**; Forge and Foundry read the locked map. A series belongs to exactly one Segment. The map (`series_key → segment_id`) is the single source of segment membership — never substring-matched against series keys.
_Avoid_: Group, category, cohort, bucket

**Series**:
A single time-indexed demand sequence identified by a unique series key (e.g., SKU × location × channel). The atomic unit of modelling in Foundry.
_Avoid_: SKU, time series (generic), row

**Series Key**:
A pipe-delimited concatenation of grain dimension values in fixed order defined by the domain playbook (e.g., `SKU_101|WEST` for `sku_region_week` grain — week is the time axis, not a key dimension). Normalised at pre-flight: values uppercased, spaces replaced with underscores, characters outside `[A-Z0-9_|]` stripped. Immutable for the lifetime of a Run. Used as a dict key in `data_store`. As a file path component (e.g. `series_results/`), the `|` separator — illegal in Windows filenames — is percent-encoded to `%7C` via a single reversible mapping defined in `data_store` (`key_to_filename` / `filename_to_key`); because keys never contain `%`, the mapping is lossless (the old `|`→`_` was lossy and collision-prone). All reads and writes route through that mapping — never `replace("|","_")` at call sites.
_Avoid_: Series ID, series identifier, SKU key

**Forecastability**:
A per-series outcome assigned by the end of a Run. One of three values: `forecastable` (sufficient history, Foundry found a model meeting the MASE target), `caution` (modellable but carries known risk — e.g., marginal history, high CV², or an ACCEPTED_RISK Claim; can be flagged pre-modelling by Meridian or Forge, or post-modelling when Foundry falls short of target without hitting the unforecastable floor), `unforecastable` (Foundry exhausted all self-correction rounds and `assess_target_feasibility` confirmed the theoretical floor exceeds the target).
_Avoid_: Forecastable flag, series status, model outcome

**Demand Class**:
A per-series classification (`SMOOTH`, `ERRATIC`, `INTERMITTENT`, `LUMPY`) derived from the Syntetos-Boylan matrix using ADI and CV². Assigned by Forge during EDA. Gates which models Foundry may use for that series. In a Scenario Run, re-evaluated for affected series if the scenario override materially changes demand pattern (e.g., injecting a stockout or promo event).
_Avoid_: Demand type, demand category, series type

### Claims & Evidence

**Claim**:
A scoping assertion about the data or domain recorded during the Meridian conversation. Can originate from Meridian (evidence-backed via tool output) or from the user (override of Meridian's recommendation). Each Claim carries one verification status and one evidence type.
_Avoid_: Decision, rule, assumption, finding

**Evidence Type**:
The strength and nature of evidence backing a Claim. Five values in descending strength: `statistical_test` (ADF, Chow test, etc. — requires a preceding tool call with result cited), `association` (two things co-occur but causality not established — requires a preceding tool call), `pattern` (series shape consistent with an interpretation — can be asserted conversationally), `user_confirmed` (user asserted it, data could not corroborate or deny), `unverifiable_business_input` (forward-looking claim historical data structurally cannot address — always accompanied by an open Risk).
_Avoid_: Evidence strength, evidence level, confidence type

**Verification Status**:
The evidential standing of a Claim. One of five values: `SUPPORTED` (data confirms it), `CONTRADICTED` (data actively pushes back — Meridian challenges the user with evidence and offers constrained alternatives; always a transient state that resolves before pack confirmation), `AMBIGUOUS` (data is inconclusive — conservative assumption applied, risk logged), `UNVERIFIABLE` (data cannot speak to this — accepted with caveat), `USER_OVERRIDE_ACCEPTED` (user overrode a data-backed recommendation — consequence logged, must surface in report). `CONTRADICTED` cannot appear in the final Domain Context Pack — it must resolve to `SUPPORTED` (user was right, data re-evaluated) or `USER_OVERRIDE_ACCEPTED` (user insisted, override logged). Meridian makes its case once and accepts the outcome.
_Avoid_: Confidence, state, flag

**Override**:
A Claim with verification status `USER_OVERRIDE_ACCEPTED` — a user decision to proceed contrary to Meridian's evidence-backed recommendation. Always carries the agent's recommendation, the user's stated reason, the downstream consequence, and a severity. Must surface in the final report. Not a separate entity — a subset of the Claim Ledger.
_Avoid_: User correction, manual decision, exception

**Risk**:
A concern flagged during the Meridian conversation about the quality or reliability of a scoping decision. Exists in two states: *open* (added to the risk register, not yet acknowledged by the user) or *closed* (user explicitly acknowledged it, becoming an `ACCEPTED_RISK` Claim; or Meridian resolved it via subsequent diagnostic evidence). `open_risks` in pipeline state counts only unacknowledged risks. Conductor blocks `confirm_pack_and_advance` while `open_risks > 0`.
_Avoid_: Warning, concern, flag, issue

**Claim Ledger**:
The append-only log of all Claims for a Run. Assembled by Meridian; compiled into the Domain Context Pack at the end of scoping. Overrides are a filtered view of the Claim Ledger — not a separate store.
_Avoid_: Decision log, claim list

### Artifacts

**Domain Context Pack**:
The confirmed problem specification — a structured artifact capturing all scoping decisions made during the Meridian conversation (segments, horizons, zero and spike policies, break dates, etc.), verified by evidence and signed off by the user. Locked (read-only) before Forge starts. Prism clones it and applies overrides to the clone; it never mutates the original.
_Avoid_: Config, spec, context, settings

### State

**Run State**:
The single authoritative source of truth for a Run's current status. A Pydantic model persisted to `outputs/{run_id}/run_state.json` after every mutation. Loaded at the start of every HTTP request. Conductor reads and writes it via `get_run_state` / `update_run_state`. No in-memory-only state for anything that must survive across requests.
_Avoid_: Pipeline state, session state, app state

### Observability

**Backend Log**:
Every tool call, agent invocation, guard check, and pipeline state transition emits a `log.info` (or equivalent) to the backend terminal at runtime. This is separate from `obs_log.json` (structured artifact) and the SSE stream (frontend events). The terminal log is the primary real-time debugging surface for the developer.
_Avoid_: Console log, debug log, print

### Output

**Report**:
The final output of a Run — a single structured JSON artifact containing all forecasting decisions, model outcomes per series, open risks, and override consequences. Available when the Run reaches `report_ready`. Rendered by the frontend `ReportSummary` component as two switchable views: the Technical View (MASE metrics, model selection reasoning, fold performance, self-correction history) and the Business View (plain language accuracy, MAPE translation with caveat, forecastable/caution/unforecastable counts, open risks, override consequences). Both views are derived from the same JSON — not separate endpoints or files.
_Avoid_: Forecast report, output, results, summary doc

**Technical View**:
The Report view for the data science audience — MASE, MAD, fold ranges, self-correction rounds, demand class, feature config, ensemble delta. Shown in the `ReportSummary` component.
_Avoid_: Internal report, DS view

**Business View**:
The Report view for the business audience — forecastable/caution/unforecastable counts, approximate accuracy in plain language, MAPE translation (clearly labelled, with caveat for zero-demand periods), open risks, override consequences. Shown in the `ReportSummary` component.
_Avoid_: External report, stakeholder view

**Override Consequence**:
The `downstream_impact` field from Override Claims, surfaced explicitly in the Report. Tells the reader what effect the user's override decision had on the forecasting outcome.
_Avoid_: Override impact, consequence, downstream effect

### Modelling

**Walk-Forward Fold**:
A time-series aware validation fold with no future leakage. Designed by Forge per segment using horizon, history length, and confirmed break dates. Post-break folds never include pre-break data. Minimum 2 folds required for valid walk-forward validation. If post-break history is too short to support 2 folds, the series is flagged `caution` with reason "insufficient post-break history" — it is still modelled but validation confidence is explicitly degraded. Pre-break history is never used as a fallback when post-break history is short.
_Avoid_: Cross-validation fold, time split, holdout

**Structural Break**:
A date at which the demand-generating process materially changed (e.g., a product relaunch, a market shock). Exists in two states: a *candidate* (statistical signal from pre-flight or Meridian diagnostics, unconfirmed) and a *confirmed break* (explicitly acknowledged by the user during Meridian scoping, written into the Domain Context Pack as a `SUPPORTED` Claim). Forge only runs the Chow test at confirmed break dates. Unacknowledged candidates are treated as absent — silence does not confirm.
_Avoid_: Change point, regime change, break point

**MASE Target**:
The acceptable MASE threshold for a series to be considered `forecastable`. Three-layer hierarchy: (1) universal floor — MASE < 1.0 (better than naïve) is always required; (2) playbook default — domain-specific threshold (e.g., MASE < 0.8 for FMCG); (3) user override — set as a Claim during Meridian scoping, flagged as `ACCEPTED_RISK` if Meridian's evidence suggests it is unrealistic. `assess_target_feasibility` checks against whichever layer is active.
_Avoid_: MASE threshold, accuracy target, error target

**Self-Correction**:
Foundry's iterative modelling loop — the agentic equivalent of a data scientist's model refinement session. Three rounds: (1) best single model in the demand class gate; (2) structural changes within the same family (different feature config, next model in family) — re-runs walk-forward validation as the model demands it; (3) most complex model in family, then ensemble if delta exceeds 5% MASE over best single. If all three rounds fail, `assess_target_feasibility` is called to confirm unforecastable status with reasoning.
_Avoid_: Retry loop, fallback, model search

### Guard & Safety

**Halt**:
Any condition that terminates a Run before `report_ready`. Raised either by the Guard Layer (token budget exceeded, duplicate tool call detected, max tool calls reached, loop-back cap exceeded) or by Conductor (open risks unacknowledged, agent failed). Halt reason is recorded in `run_summary.json` and surfaced via the `error` SSE event. **Terminal** — a halted Run cannot be resumed; the user must start a new Run. Distinct from a Pause, which is resumable.
_Avoid_: Error, failure, abort, stop

**Pause**:
A *non-terminal* suspension of a Run while it waits for human input, triggered when an agent raises an Agent Need. The Run holds at its current phase (`awaiting input`) and resumes when the user answers — nothing is reset and no work is discarded. The opposite of a Halt: a Halt ends a Run, a Pause merely waits.
_Avoid_: Stop, block, halt, freeze

**Guard Layer**:
Shared enforcement layer applied to every tool call across all agents. Checks token budget, per-agent tool call limits, and duplicate call detection before dispatching. Raises `GuardHalt` (a Python exception) when any limit is exceeded. Not an agent — purely deterministic Python. All limits (token budget, max tool calls per agent, duplicate call threshold) are `.env`-configurable; values in plan_v1.md are defaults only.
_Avoid_: Safety layer, rate limiter, budget checker

### Agentic Interaction

**Agent Need**:
A structured request an agent raises (`raise_need`) when it hits a decision it should not make alone, triggering a Pause. Carries a `kind`, a question, candidate options, and supporting context. Two kinds: a **User Decision** (a quick choice the Conductor puts to the user) and a **Scope Amendment** (a change to the confirmed scope, handled by Meridian via a Loop-back). Raised sparingly — only when no defensible default exists.
_Avoid_: Prompt, interrupt, request, ask

**Checkpoint**:
The moment a Run pauses on an Agent Need and surfaces a question to the user. A Checkpoint is how "the agents take human input when required" — distinct from Meridian's ongoing scoping conversation, which is dialogue rather than a suspension.
_Avoid_: Gate, stop point, approval step

**Loop-back**:
Conductor routing a Scope Amendment back to Meridian, which amends the confirmed Domain Context Pack, re-locks it (a new pack version), and re-runs only the affected series. Bounded — a Run may only loop back a limited number of times before it Halts. This is why the Run lifecycle is not strictly linear.
_Avoid_: Rollback, rewind, retry, revision

**Agent Narration**:
The plain-language reasoning an agent emits as it works (which segment it is starting, why it is escalating a model, what it just found), streamed to the user as a live activity feed. How the user perceives the system as reasoning rather than as a progress bar. Always decisions and findings — never raw tool output.
_Avoid_: Log, trace, commentary, output

### Domain & Configuration

**Domain Playbook**:
A YAML file encoding domain-specific defaults for a Run — minimum series count, expected schema columns, default horizon, segment inference rules, zero-run and spike thresholds. Selected at run creation. A guide, not a hard constraint: the user can override any default or introduce segment dimensions not present in the playbook during the Meridian conversation. Not a prompt — feeds Python logic and is injected into agent system prompts as structured context.
_Avoid_: Config file, domain config, domain rules, domain spec

**Preflight Bundle**:
The handoff object from the Pre-flight Layer to Meridian (via Conductor). Contains: `data_quality_report`, `schema_mapping`, `grain_report`, `segment_profiles` (aggregate statistics per segment — series count, demand class distribution, median ADI, median CV², forecastability breakdown), `segment_exceptions` (a small list of per-series statistical outliers within their segment — the only per-series detail surfaced without a tool call), `segments`, `domain_playbook`. Individual series data stays in `data_store`; consistent with the sentinel pattern.
_Avoid_: Preflight output, upload result, data bundle

### Pre-flight

**Pre-flight Layer**:
Deterministic Python (no LLM) that runs on CSV upload before any agent starts. Profiles the data, maps the schema, detects grain and frequency, builds series keys, and computes per-series statistics. Produces the preflight bundle handed to Meridian via Conductor.
_Avoid_: Validation layer, ingestion, data pipeline

**Blocking Issue**:
A data quality problem that makes the Run impossible to proceed — missing date or demand column, fewer series than the playbook minimum, all-zero data. Conductor halts immediately and presents the issue to the user. No degraded-mode fallback.
_Avoid_: Error, fatal error, critical warning

**Warning**:
A data quality problem worth surfacing but that does not stop the Run — sparse data, some series below minimum history length, resolvable schema ambiguities. Passed to Meridian as part of the preflight bundle for use in scoping conversation.
_Avoid_: Non-blocking issue, soft error

### Intent & Routing

**Intent Pack**:
The structured output of Lens — a typed classification of one user message, consumed by Conductor to decide routing. Contains intent type, extracted entities, confidence score, and a raw quote. Produced on every user message; Conductor never parses raw user language directly.
_Avoid_: Intent, message classification, parsed message

**Intent Type**:
One of six values classifying what the user is doing: `SCOPE_RESPONSE` (answering Meridian's question), `OVERRIDE` (contradicting an agent recommendation), `ADVANCE_PIPELINE` (approving progress to the next phase), `WHAT_IF_REQUEST` (requesting a Scenario Run), `CLARIFICATION` (asking a question), `CORRECTION` (fixing a prior statement — only valid during `meridian_scoping`; post-confirmation, any scoping change is treated as `OVERRIDE` and Conductor offers a Scenario Run instead).
_Avoid_: Intent class, message type, command

**Clarification**:
A Conductor-authored message pushed to the user when Lens returns `intent.confidence < 0.6`. Offers the user two options derived from the closest candidate intents. Never exposes internal mechanics (e.g., "Lens returned low confidence"). Always written by Conductor from its system prompt guidance — never routed to Meridian for phrasing.
_Avoid_: Disambiguation prompt, re-prompt, fallback message

### Meridian Conversation Arc

**Pack Validation Error**:
A blocking condition returned by `compile_domain_context_pack` that prevents pack confirmation. Five cases: no forecast scope defined (missing target, grain, or horizon); no segments defined; any Claim still in `CONTRADICTED` status; `open_risks > 0` with none acknowledged; required schema fields missing. All must be resolved before `confirm_pack_and_advance` can be called.
_Avoid_: Validation failure, compile error, pack error

**Pack Validation Warning**:
A non-blocking condition returned by `compile_domain_context_pack` — pack can be confirmed but the condition is surfaced to the user. Four cases: all Claims are `UNVERIFIABLE` or `USER_OVERRIDE_ACCEPTED` (no data-backed claims); a segment has no segment-level rules; a data-available feature has no policy set; override count exceeds 3.
_Avoid_: Soft error, pack warning, compile warning

**Pack Review (Phase 4)**:
The point in the Meridian conversation where `compile_domain_context_pack` is called to assemble and validate all incrementally-built Claims. Meridian then presents the result as a readable formatted summary — not raw JSON — so the user can review the full picture before committing. This is a review step, not a generation step; the pack has been assembled throughout Phases 1–3 via tool calls.
_Avoid_: Pack generation, pack creation, pack output

### Session & Lifecycle

**Run**:
The top-level unit of work — one CSV upload, one conversation arc, one output report. Lifecycle phases: `preflight → meridian_scoping → forge_eda → foundry_modelling → report_ready`. The progression is **not strictly linear**: a Run may Pause for a Checkpoint at any phase, and a Loop-back can move it backward (e.g. `foundry_modelling → meridian_scoping`) to amend scope before continuing.
_Avoid_: Session, job, task, execution

**Scenario Run**:
A child of a Run that re-executes partial pipeline steps (Forge + Foundry) with overrides applied. Created by Prism. Does not start a new conversation — it inherits the parent Run's confirmed domain context.
_Avoid_: What-if run, child run, Prism run