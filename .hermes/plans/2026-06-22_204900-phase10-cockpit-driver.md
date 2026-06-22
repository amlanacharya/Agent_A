# Phase 10 — Cockpit Driver (Upload → Conductor → Report)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing Phase 2–9 backend and Phase 9 cockpit together so a planner can upload a CSV in the cockpit, chat with Meridian about the scope (typing free text or picking from the accepted-possibilities list), watch Forge + Foundry run, and read the resulting report — without leaving the browser or hand-running CLI commands.

**Architecture:** Add an in-process `Conductor` orchestrator module that drives the lifecycle `preflight → meridian_scoping → forge_eda → foundry_modelling → report_ready` from a single Run ID. Wire `api/app.py` with three new HTTP endpoints (`POST /uploads`, `POST /messages`, `POST /runs/{id}/advance`) that bridge the cockpit to the conductor. Add a `run_meridian_chat_turn` helper that calls `Lens.classify_intent` + the existing `conductor_tools.advance_to_meridian` / `confirm_pack_and_advance` and produces a structured response the cockpit can render (text reply + accepted-possibilities list + state patch). Add a `cockpit/src/pages/RunConsole.tsx` that owns the chat loop + the upload form + the live Mission Control surface polling. Keep everything synchronous and in-process; async workers are Phase 11+.

**Tech Stack:** FastAPI (`api/app.py` already wired), Anthropic SDK (`Lens` already imports it), the existing `conductor_tools.py` tool set, the existing `learning_workspace.py`, React + TanStack Query (Phase 9 already set up). No new dependencies. No new external integrations.

---

## Why this phase exists

Phase 9 shipped a cockpit that reads `outputs/{run_id}/` and renders it beautifully. But there is no way to **create** a `run_id` from inside the cockpit:

- `api/app.py` exposes `GET /surfaces`, `GET /surfaces/{name}/{run_id}`, `GET /runs`, `POST /plots`. All read-only.
- `backend/forecasting/preflight.py:53 run_preflight` exists and is correct, but has zero production callers — `tests/test_preflight_orchestrator.py` is its only consumer.
- `backend/forecasting/tools/conductor_tools.py` ships 6 individual tool functions (`get_run_state`, `update_run_state`, `advance_to_meridian`, `confirm_pack_and_advance`, `create_prism_run`, `log_halt`), but no module calls them in sequence. The lifecycle is documented in `CONTEXT.md` and ADR-0005 but the orchestrator itself is not in the repo.
- `Lens` in `backend/forecasting/agents/lens.py:88 classify_intent` exists and works — but no HTTP endpoint invokes it.

This phase closes those four gaps.

---

## What's in scope

1. **`POST /uploads`** — multipart CSV upload, writes to `outputs/{run_id}/input.csv`, calls `run_preflight`, returns the preflight bundle + a `run_id` + the first `CockpitState`.
2. **`Conductor` orchestrator module** (`backend/forecasting/conductor.py`, new) — exposes `drive_run_to_meridian`, `drive_run_to_forge`, `drive_run_to_foundry`, `drive_run_to_report`. Each one consumes current `RunState`, runs the next phase's entry point synchronously, returns updated state. The conductor is the **single source of the lifecycle** that ADR-0005 promised.
3. **`POST /messages`** — user types into the cockpit. Server calls `Lens.classify_intent`, then dispatches to one of: (a) conductor's `advance_to_meridian` if `ADVANCE_PIPELINE`, (b) returns Meridian's chat response if `SCOPE_RESPONSE`, (c) returns a clarification message if `CLARIFICATION` with low confidence. Response shape: `{reply: str, possibilities: list[str], state_patch: dict, run_id: str}`.
4. **`POST /runs/{id}/advance`** — driver button. Calls `Conductor.drive_run_to_meridian` (if preflight done but not in `meridian_scoping`), then returns the chat state for the user to see the first Meridian question. Subsequent calls after `meridian_scoping` are blocked by the server until the user has answered via `/messages`.
5. **Cockpit `RunConsole` page** (`frontend/src/pages/RunConsole.tsx`, new) — replaces the placeholder Mission Control route at `/` with: upload form, live `CockpitState` panel, chat box with accepted-possibilities chips, run-advance button. Adds a `/runs/:runId/console` deep link.
6. **Conductor → Phase 5 replenishment dispatch** (small) — when Foundry hits `report_ready`, the conductor writes a replenishment batch if the run reached a replenishment tier; otherwise it returns the report and stops.

## What's out of scope

- Async workers, Celery, Redis, background-job queues. Synchronous in-process only.
- A scheduler that auto-triggers runs from S3 / SAP / cron. The CSV upload IS the trigger; live ingestion is Phase 11+.
- Auth, user accounts, multi-tenancy. The cockpit trusts the network, same as Phase 9.
- ADR-0005's full Checkpoint / Loopback machinery (a Checkpoint pauses the run on a Checkpoint and resumes on user input). The first cut drives the linear lifecycle; Checkpoints land in Phase 10.x or 11 once the linear trip works end-to-end.
- The DataFrame store / pre-aggregate persistence layer. Today's `data_store` writes CSV-only; keep that for Phase 10.

---

## Sub-checkboxes (order of execution)

The plan's headline checkboxes are coarse; below is the work breakdown the executor will tick as the work ships. **Tick the matching line, not the headline, when a unit lands.** (Reference: the `plan` skill's "Tick accuracy (variant)" pitfall.)

### Phase 10 CB1 — `/uploads` endpoint + run_preflight wired to HTTP

- [x] **10.1.1** Add `POST /uploads` route in `api/app.py` accepting `multipart/form-data` with a `file: UploadFile` field + a `domain: str` form field + an optional `playbook_name: str` field. Route writes `file_bytes` to `outputs/{run_id}/input.csv` (UUID-derived run_id), creates the `RunState` via `run_state.create_run_state(run_id)`, calls `preflight.run_preflight(run_id, file_bytes, domain, playbook)` synchronously, returns `{run_id, preflight: PreflightBundle.model_dump(), state: RunState.model_dump()}`. Error mapping: `PreflightBlockingError` → 422 with the issues list; ValueError → 400.
- [x] **10.1.2** Add a `domain_playbook_for(name)` helper in `api/app.py` (or a new `api/playbooks.py`) that returns the right YAML playbook dict (FMCG default + any future domain). Today: only `fmcg` is supported, return the FMCG default dict.
- [x] **10.1.3** Test: `tests/test_api_uploads.py` — POST a valid 12-week FMCG CSV → 200 with `run_id` + `preflight` dict + `state.phase == "preflight"`; POST malformed bytes → 422 with `blocking_issues`; POST with missing `domain` → 400.
- [x] **10.1.4** Verify: `uv run pytest tests/test_api_uploads.py -v` — 3 passed.

### Phase 10 CB2 — `Conductor` orchestrator module

- [ ] **10.2.1** Create `backend/forecasting/conductor.py` with the `Conductor` class. Constructor takes `run_id` + an injected `conductor_tools` module (so tests can stub). Methods (each pure: take `RunState`, return updated `RunState` + a `ConductorStepResult` namedtuple with `{reply, possibilities, advanced_to: Phase, state}`):
  - `drive_run_to_meridian(state)` — calls `conductor_tools.advance_to_meridian(run_id)`; returns Meridian's first chat prompt (text + possibilities).
  - `drive_run_to_forge(state)` — calls the existing `eda_toolbox.build_eda_report` synchronously; transitions `meridian_scoping → forge_eda` via `advance_phase`; persists `eda_report.json` to `outputs/{run_id}/`.
  - `drive_run_to_foundry(state)` — calls `forecast_harness.run_forecast_harness` then `ensemble.summarise_scorecards`; transitions `forge_eda → foundry_modelling`; persists `foundry_report.json` + `forecast_harness_report.json`.
  - `drive_run_to_report(state)` — calls `promotion.format_promotion_decision` if a champion/challenger decision is needed, else skips; transitions `foundry_modelling → report_ready`; persists `outputs/{run_id}/REPORT.json` (a serialized `Report`-equivalent: overall MASE, per-series outcomes, override consequences, open risks).
- [ ] **10.2.2** Add `ConductorStepResult` namedtuple (in `contracts.py` or alongside) with fields: `reply: str`, `possibilities: list[Possibility]` (where `Possibility = {kind: Literal["ACCEPT", "OVERRIDE", "CLARIFY"], label: str, payload: dict}`), `advanced_to: Phase`, `state: RunState`. The `possibilities` list is what the cockpit renders as chips/buttons next to the text reply.
- [ ] **10.2.3** Tests: `tests/test_conductor.py` — stub `conductor_tools` + the phase-specific entry points; verify `drive_run_to_meridian` calls `advance_to_meridian` once and returns a non-empty `reply`. Same for the other three methods. 4 tests minimum.
- [ ] **10.2.4** Verify: `uv run pytest tests/test_conductor.py -v` — 4 passed.

### Phase 10 CB3 — `/messages` endpoint (Lens + Conductor chat loop)

- [x] **10.3.1** Add `POST /messages` route in `api/app.py`. Body: `{run_id: str, user_message: str}`. Server: load `RunState`, call `Lens.classify_intent(LensInput(conversation_history=[...loaded from outputs/{run_id}/obs_log.json if any...], user_message=user_message, pipeline_state=run_state))`. Dispatch on `intent.intent`:
  - `SCOPE_RESPONSE` → call `Conductor.record_scope_response(run_id, user_message, lens_pack)` which writes a `Claim` to `outputs/{run_id}/claim_ledger.json` + advances the Meridian conversation. Return `{reply: meridian's next question text, possibilities: [...]}`.
  - `OVERRIDE` → write a `USER_OVERRIDE_ACCEPTED` Claim; return `{reply: "logged override", possibilities: []}`.
  - `ADVANCE_PIPELINE` → call `conductor.drive_run_to_<next_phase>`; return `{reply, possibilities, advanced_to}`.
  - `CLARIFICATION` (confidence < 0.6) → call `Conductor.author_clarification(intent)` to generate two short options; return `{reply, possibilities: [option_a, option_b]}`.
  - `CORRECTION` → similar to `OVERRIDE` but only valid in `meridian_scoping`.
  - `WHAT_IF_REQUEST` → create a Prism clone via `conductor_tools.create_prism_run`; return `{reply: "scenario run created", run_id: <prism_run_id>, possibilities: []}`.
- [x] **10.3.2** Add a `run_meridian_chat_turn` helper in `backend/forecasting/conductor.py` that takes the user's message + current state and produces a `ConductorStepResult` with a `reply` text + `possibilities` list. The first cut uses a small templated reply (no LLM call in the chat turn itself; the agentic Meridian LLM call lands in Phase 10.4 once the loop works end-to-end without it).
- [x] **10.3.3** Add `LensConversationHistory` adapter: load `outputs/{run_id}/obs_log.json`, parse `event=message` entries into `ConversationTurn(role, content, agent)` objects. For Phase 10's first cut, if no obs log exists, return an empty list and let Lens classify on `pipeline_state` alone (Lens already handles this — see `_SYSTEM` prompt line 4–6).
- [x] **10.3.4** Tests: `tests/test_api_messages.py` — using a stub Lens + stub conductor, exercise each of the 6 intent types; assert the response shape. 6 tests minimum.
- [x] **10.3.5** Verify: `uv run pytest tests/test_api_messages.py -v` — 6 passed.

### Phase 10 CB4 — `/runs/{id}/advance` endpoint (driver button)

- [ ] **10.4.1** Add `POST /runs/{id}/advance` route in `api/app.py`. Body: `{force: bool = false}`. Server: load `RunState`, pick the right `conductor.drive_run_to_<phase>` method based on `state.phase`:
  - `preflight` → `drive_run_to_meridian`
  - `meridian_scoping` → reject (user must `/messages` first); 409 with `{error: "waiting for user input"}`
  - `forge_eda` → `drive_run_to_forge` (also drives `foundry_modelling` + `report_ready` in one call, since EDA is fast — see CB5 for the long-run path)
  - `foundry_modelling` → `drive_run_to_report`
  - `report_ready` → 200 noop, returns `{state, reply: "already at report_ready"}`
- [ ] **10.4.2** Add `Conductor.drive_run_to_next(state, force=False) -> ConductorStepResult` that dispatches by phase. Same dispatch table as above. Returns the result from the underlying `drive_run_to_*` method.
- [ ] **10.4.3** Tests: `tests/test_api_advance.py` — using a stub conductor, exercise each phase transition. Verify the 409 on `meridian_scoping` without force=true. 5 tests minimum.
- [ ] **10.4.4** Verify: `uv run pytest tests/test_api_advance.py -v` — 5 passed.

### Phase 10 CB5 — Wire Conductor through Forge + Foundry + Report end-to-end

- [ ] **10.5.1** In `Conductor.drive_run_to_forge`: call `eda_toolbox.build_eda_report(canonical_table, ...)` using the data the preflight wrote. Persist `outputs/{run_id}/eda_report.json`. Transition `meridian_scoping → forge_eda` via `advance_phase(state, Phase.FORGE_EDA)`.
- [ ] **10.5.2** In `Conductor.drive_run_to_foundry`: build feature table via `feature_factory.build_feature_table`, call `forecast_harness.run_forecast_harness(request, features)`, call `ensemble.summarise_scorecards(scorecards)`, build the per-series `SeriesResult` (target met / caution / unforecastable per the `Forecastability` glossary term). Persist `outputs/{run_id}/foundry_report.json` + `forecast_harness_report.json`. Transition `forge_eda → foundry_modelling`.
- [ ] **10.5.3** In `Conductor.drive_run_to_report`: assemble the Report equivalent — overall MASE, per-series outcomes (forecastable / caution / unforecastable), open risks, override consequences, Foundry narrative. Persist `outputs/{run_id}/report.json`. Transition `foundry_modelling → report_ready`. Optionally trigger replenishment if the run's recommendations cleared the threshold.
- [ ] **10.5.4** Full-chain integration test: `tests/test_conductor_e2e.py` — upload a synthetic FMCG CSV → assert `preflight` phase, advance to `meridian_scoping` → assert reply has possibilities, send 3 fake `SCOPE_RESPONSE` messages to fill the Claim Ledger → advance → assert `forge_eda` done → advance → assert `foundry_modelling` done with scorecards → advance → assert `report_ready` with `report.json` written. 1 big test, 8 asserts.
- [ ] **10.5.5** Verify: `uv run pytest tests/test_conductor_e2e.py -v` — 1 passed; then `uv run pytest -q` (full backend suite) — all green, no regressions.

### Phase 10 CB6 — Frontend `RunConsole` page

- [ ] **10.6.1** Create `frontend/src/pages/RunConsole.tsx`. Layout: left rail with the run's `CockpitState` (current step, active agent, confidence, blockers); main panel split into two tabs: "Scope" (chat with Meridian) and "Run" (advance button + report link once ready). `RunConsole` is wired at `/runs/:runId/console` in `frontend/src/router.tsx`.
- [ ] **10.6.2** Add `useUploadCsv`, `usePostMessage`, `useAdvanceRun` hooks in `frontend/src/api/hooks.ts`. Each wraps the corresponding POST endpoint in TanStack Query mutations. The mutations expose `isPending` for the spinner, `error` for the toast, and `data` for the response (which the page writes to local state).
- [ ] **10.6.3** Add an upload form at the top of `RunConsole` (visible when no `:runId` is in the URL, i.e. when the page renders at `/`). File input → `useUploadCsv.mutate(file)` → on success, `navigate(/runs/${runId}/console)`. Domain select dropdown with FMCG as the only option for Phase 10.1.
- [ ] **10.6.4** In the Scope tab: chat history rendered as a vertical stack of message bubbles (alternating assistant / user, accent color from DESIGN.md for assistant turns). Below: a textarea + send button that calls `usePostMessage`. To the right of the textarea, the `possibilities` list from the last server reply rendered as `<StatusChip>`-style buttons; clicking a chip auto-fills the textarea with the chip's `label` and submits. Empty possibilities list = no chips shown.
- [ ] **10.6.5** In the Run tab: a single primary button "Advance to next phase" that calls `useAdvanceRun`. Disabled while `isPending` or when state is `meridian_scoping` (in that phase the user must chat, not advance). When state is `report_ready`, render a link to `/surfaces/forecast_review/:runId` + `/surfaces/replenishment_board/:runId` instead of the advance button.
- [ ] **10.6.6** Live CockpitState: poll `GET /cockpit-state/:runId` (add this endpoint in CB6.7) every 5s while a run is in flight. Render the JSON in the left rail with `<StatusChip>` for confidence / blocker tones.
- [ ] **10.6.7** Add `GET /cockpit-state/{run_id}` in `api/app.py` — returns `cockpit_state.CockpitState.from_run_state(load_run_state(run_id)).to_public_dict()` plus a `phase` field for the UI's dispatch logic.
- [ ] **10.6.8** Tests: `frontend/src/pages/RunConsole.test.tsx` — upload form submit, message send, possibility chip click, advance button click, error toast. Mock the API hooks. 8 tests minimum.
- [ ] **10.6.9** Verify: `pnpm test` — all 85 frontend tests green; `pnpm build` — clean bundle; `pnpm test:e2e` — the existing e2e suite (from Phase 9 CB12) still walks all 10 surfaces against real FastAPI.

### Phase 10 CB7 — Gloss, ticks, and live-upload smoke

- [ ] **10.7.1** Update `CONTEXT.md` to add the cockpit driver terms: `Upload Endpoint`, `Conductor`, `Chat Loop`, `Accepted Possibility`. Use the vocabulary already in the doc (`Seam`, `Adapter`, `Module`) for the architecture description; use domain terms (`Run`, `Phase`, `Meridian`) for the runtime description.
- [ ] **10.7.2** Tick the checkboxes in `docs/superpowers/plans/2026-06-05-agentic-demand-forecasting-workspace.md` (append a `### Phase 10: Cockpit Driver` section with CB1–CB7 listed and ticked). Add the Progress Summary row for Phase 10 (status: ✅ Complete when shipped).
- [ ] **10.7.3** Live-upload smoke: start the server with `uv run python -m api`, open the cockpit at `http://localhost:5173`, upload a real 12-week FMCG CSV (use the existing test CSV from `tests/test_preflight_orchestrator.py:_csv()`), chat through Meridian (pick "yes" / "looks good" / "proceed" from possibilities), click advance 2–3 times, land on the report. Capture a screenshot or a 30-line transcript for the commit message.

---

## Files likely to change

**New:**
- `backend/forecasting/conductor.py` — `Conductor` orchestrator + `ConductorStepResult` namedtuple
- `api/playbooks.py` (or inside `api/app.py`) — domain playbook loader
- `tests/test_conductor.py` — unit tests for the orchestrator
- `tests/test_conductor_e2e.py` — full-chain integration test
- `tests/test_api_uploads.py`, `tests/test_api_messages.py`, `tests/test_api_advance.py` — endpoint tests
- `frontend/src/pages/RunConsole.tsx` — the driver page
- `frontend/src/pages/RunConsole.test.tsx` — page tests

**Modified:**
- `api/app.py` — 4 new routes (`POST /uploads`, `POST /messages`, `POST /runs/{id}/advance`, `GET /cockpit-state/{run_id}`)
- `frontend/src/router.tsx` — wire `/runs/:runId/console` and replace `/` placeholder with the upload form
- `frontend/src/api/hooks.ts` — `useUploadCsv`, `usePostMessage`, `useAdvanceRun`, `useCockpitState` mutations/queries
- `backend/forecasting/contracts.py` — add `ConductorStepResult` + `Possibility` (if not colocated in conductor.py)
- `docs/superpowers/plans/2026-06-05-agentic-demand-forecasting-workspace.md` — Phase 10 section
- `CONTEXT.md` — 4 new glossary terms

**Not modified (but read carefully before each task):**
- `backend/forecasting/preflight.py` — `run_preflight` is already complete; just call it.
- `backend/forecasting/tools/conductor_tools.py` — 6 tool functions are already complete; just orchestrate them.
- `backend/forecasting/agents/lens.py` — `Lens.classify_intent` is already complete; just call it.
- `backend/forecasting/run_state.py` — `advance_phase`, `LEGAL_TRANSITIONS`, `save_run_state`, `load_run_state` are already complete; just use them.

---

## Tests + verification

After every CB:
- `uv run pytest tests/<new_test_file>.py -v` (the new tests) — green
- `uv run pytest -q` (the full backend suite) — no regressions
- `pnpm test` (after CB6) — all 85 frontend tests green

After CB7:
- Live-upload smoke against `uv run python -m api`
- `pnpm build` — clean bundle
- `pnpm test:e2e` — all 12 e2e tests green

Test counts target after Phase 10 ships:
- Backend: 848 (current) + ~25 new = ~873
- Frontend: 77 (current) + ~10 new = ~87

---

## Risks + tradeoffs

1. **Synchronous HTTP for the full pipeline** — the `/advance` endpoint will block the request thread for as long as Forge + Foundry take. For a real FMCG CSV with 50 series and 4 folds, that's plausibly 30–120s. The cockpit's TanStack Query will show a spinner. **Mitigation:** add a `?timeout=300` URL param and the server returns 504 if exceeded; the client retries with the same run_id (the conductor is idempotent re-invocation per ADR-0005 — re-running from the top skips steps whose artifacts already exist).

2. **The Meridian chat reply in 10.3.2 is templated, not LLM-driven** — this is deliberate (YAGNI: ship the loop first, plug in the LLM call second). The first cut uses canned replies + the existing Claim Ledger writes. Phase 10.x will swap in the actual `Meridian` agent. **Mitigation:** keep the response shape stable so swapping the reply generator doesn't change the cockpit.

3. **The cockpit's `usePostMessage` mutation will need retry-on-409 logic** — the user might click an "advance" chip while in `meridian_scoping`; the server returns 409. **Mitigation:** the mutation's `onError` handler toasts "waiting for your scope decision" and refocuses the chat input.

4. **Polling `/cockpit-state/{run_id}` every 5s** — fine for Phase 10's synchronous model. WebSockets / SSE are out of scope (Phase 9 explicitly excludes them: "Real-time updates (websockets, SSE) — the cockpit polls on focus + on manual refresh"). When Phase 11 adds async workers, polling becomes the wrong primitive and we revisit.

5. **The `Lens` Anthropic SDK import is at module top** — `from anthropic import Anthropic` will fail in test environments without the SDK installed (it IS installed in this repo per `requirements.txt` + `pyproject.toml`, but tests that stub Lens shouldn't actually call the SDK). **Mitigation:** the `Lens.classify_intent` function already takes an `injected_client` parameter (line 88–94); tests pass a stub. Verify this in the test file before shipping.

6. **`run_preflight` is the Phase 2 entry point, not a generic CSV parser** — it expects the canonical FMCG schema (or whatever the playbook says). If a user uploads a CSV in a non-FMCG domain, `run_preflight` will reject it with `PreflightBlockingError`. **Mitigation:** Phase 10 only ships the FMCG default; the playbook dropdown only offers FMCG. Future domains land in Phase 10.x.

---

## Open questions (decided)

- **Async workers in scope?** No (Phase 11+).
- **External scheduler integration in scope?** No (in-process `Scheduler` already exists from Phase 6 CB4; no new wiring).
- **Replenishment auto-trigger in scope?** Yes, optional — conductor writes a replenishment batch when Foundry reaches `report_ready` and the run's recommendations clear the threshold. CB5.3 includes this.
- **ADR-0005 Checkpoint / Loopback machinery in scope?** No, deferred. The first cut drives the linear lifecycle; the conductor is a flat `drive_run_to_<next>` dispatcher. Phase 10.x adds the Checkpoint loop.
- **`Meridian` agent (LLM-driven scoping) in scope?** Not in this phase. The chat reply in 10.3.2 is templated. The LLM-driven version lands in Phase 10.x once the loop works end-to-end without it.
- **Auth, multi-tenancy, deployment story?** No (Phase 9 already excludes them; Phase 10 inherits).
- **What if user uploads a CSV while another run is in flight?** Each upload gets a fresh UUID `run_id`; runs are independent. The cockpit renders a run list at `/runs` (from Phase 9 CB4); users pick which to focus on.

---

## Standing rules (binding for this phase's execution)

1. **Handoff at 40% context OR checkbox completion, whichever fires first.** Write the handoff doc to OS temp dir at every CB boundary (safety net + auditability). Chain past handoffs in the same session until Phase 10 is fully done or 40% fires. Use the `handoff` skill with the next CB as the argument.
2. **One commit per CB**, lead the commit message with the phase sub-step name (`phase 10 cb1 — ...`). Push after commit unless told otherwise.
3. **Per-CB uncommitted leftovers get bundled** into the current commit (user standing rule: phase commit + leftovers, not split into atomic commits).
4. **Never-commit paths**: `.claude/`, `.claude/settings.local.json`, `.vscode/`, `.idea/`, `__pycache__/`, `.venv/`, `.env`, `.pytest_cache/`, `backend/outputs/`, `.teach-architecture/`, `graphify-out/` (root AND `backend/forecasting/graphify-out/`), `platform-explorer.html`, generated `*.md` reports (`MONITORING_REPORT.md`, `DRIFT_REPORT.md`, `OVERRIDE_ANALYSIS.md`, `MODEL_HEALTH.md`), and the Phase 9 additions: `frontend/dist/`, `frontend/node_modules/`, `frontend/src/api/schema.ts`. Plus session-local artefacts like `.hermes/`, `.playwright-cli/`, `.playwright-mcp/`, `output/` — never commit.
5. **Project tooling**: `uv` for Python (`.python-version` pin 3.11, `uv.lock` committed). `uv run pytest -q` for the suite. `pnpm` for frontend.
6. **Test command verification**: `uv run pytest -q` must be green before every commit.
7. **Plan lives at** `.hermes/plans/2026-06-22_204900-phase10-cockpit-driver.md`. The workspace plan `docs/superpowers/plans/2026-06-05-agentic-demand-forecasting-workspace.md` is the parent and gets the Progress Summary row added at the end (CB7.2).
8. **Glossary is `CONTEXT.MD`** at the repo root. CB7.1 adds 4 new terms: `Upload Endpoint`, `Conductor`, `Chat Loop`, `Accepted Possibility`.

## Self-review

- Spec coverage: yes — upload + conductor + chat loop + advance + cockpit + tests + gloss. Each row above names the exact file paths, the exact endpoint shapes, and the exact test commands.
- Placeholder scan: no open `TBD`/`TODO` markers in the plan body.
- Scope check: ~25 backend tests + ~10 frontend tests across 7 CBs. Comparable to Phase 9 CB-by-CB scope (Phase 9 was 12 CBs at ~3 tests each + 12 e2e tests = ~48 frontend tests; this phase is 7 CBs at ~4 tests each = ~35 tests). Tight enough to execute in 1–2 sessions, loose enough that the conductor + cockpit driver both get real tests.
- Tick accuracy: the headline CB1–CB7 boxes match the sub-checkbox breakdown. Each sub-checkbox is one ticket worth of work (2–5 min of focused implementation + a test). The executor ticks the sub-checkbox, not the parent, when the work ships.

---

## Progress Summary (will be appended to the workspace plan)

| Phase | Status | Notes |
|---|---|---|
| 1–9 | ✅ Complete | (see existing plan) |
| 10: Cockpit Driver | ⬜ Not started | CB1 (uploads) → CB2 (conductor) → CB3 (messages) → CB4 (advance) → CB5 (end-to-end) → CB6 (cockpit page) → CB7 (gloss + live smoke) |
