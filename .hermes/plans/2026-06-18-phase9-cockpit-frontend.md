# Phase 9 — Data Intelligence Cockpit Frontend — Implementation Plan

> **For Hermes:** Use the `plan-execution-cadence` skill to drive this plan checkbox-by-checkbox. Per-CB cadence: TDD → uv run pytest + pnpm test green → commit → tick → handoff at 40% context OR CB completion.

**Goal:** Ship a React + Vite + Tailwind + TanStack Query frontend that consumes the existing FastAPI surface (`/surfaces`, `/surfaces/{name}/{run_id}`, `/plots`), replacing the 12 throwaway HTML mockups in `prototype/` with one wired SPA served by FastAPI.

**Architecture:** Foundation-first, then one surface per CB. `frontend/` at repo root (sibling to `api/`, `backend/`). Backend (FastAPI in `api/`) remains authoritative for data shape; frontend regenerates TS types via `openapi-typescript` whenever `api/models.py` changes. `prototype/*/DESIGN.md` is the visual spec; `prototype/*/code.html` files are visual reference only and are deleted at end of phase (their patterns are absorbed into `frontend/src/components/`). FastAPI serves the built SPA via `StaticFiles` mount at `/` (the existing mount point used by the prototype HTML).

**Tech Stack:** React 18 + Vite 5 + TypeScript 5 + Tailwind CSS 3 + TanStack Query 5 + Recharts (for the 7 plot kinds) + Vitest + React Testing Library. `pnpm` for the JS toolchain (matches `package.json` convention; lockfile committed). Backend stays on `uv` / `pytest` per Agent_A standing rules.

---

## Phase 9 — Data Intelligence Cockpit Frontend

The platform's own native layer for: (1) a typed, design-system-driven React SPA that consumes the FastAPI surface, (2) a token-level port of `prototype/data_intelligence_cockpit/DESIGN.md` into Tailwind config + base components, and (3) a one-surface-per-CB rollout that mirrors the Phase 8 backend cadence. The frontend is the surface the human planner interacts with; the FastAPI surface (Phase 8) is the engine. The boundary is deliberately kept behind small seams: (a) the `openapi-typescript`-generated types in `frontend/src/api/schema.ts` are the single source of truth for the contract, (b) TanStack Query is the single source of truth for cache/lifecycle, (c) the `<PlotFrame>` wrapper is the single source of truth for plot rendering. A future alternative (a Svelte port, a mobile-native client) can plug in behind the same FastAPI surface without changing the backend.

**What this repo ships (Phase 9 in-repo):**

- [ ] CB1 — rewrite the Phase 9 section in this plan to make the in-repo / out-of-repo split explicit, document the never-commit paths, add the frontend to the standing-rules handoff doc.
- [ ] CB2 — frontend scaffold + design tokens: `frontend/` package, Vite + React + TS + Tailwind config, Tailwind theme extended from `DESIGN.md` (colors, typography, spacing, radii, shadows), `package.json` scripts (`dev`, `build`, `preview`, `test`, `test:contract`, `generate-api`), `frontend/tsconfig.json`, `frontend/vite.config.ts` with `/api` proxy to `localhost:8000`, `frontend/tailwind.config.ts`, `frontend/postcss.config.js`. Vitest + RTL configured.
- [ ] CB3 — API client + codegen: `openapi-typescript` install + `pnpm generate-api` script that fetches `http://localhost:8000/openapi.json` and writes `frontend/src/api/schema.ts`. `frontend/src/api/client.ts` with typed `fetchSurface(name, runId)`, `listSurfaces()`, `renderPlot(request)` functions. `frontend/src/api/hooks.ts` with `useSurface`, `useSurfaces`, `usePlot` TanStack Query wrappers. Contract test (`frontend/tests/contract.test.ts`) that boots `uv run python -m api.app` in `beforeAll`, iterates over all 9 surface names + 7 plot kinds with a canned run_id, asserts each response validates against the generated Zod schemas.
- [ ] CB4 — routing + global layout: React Router v6 with `/` → Mission Control (default), `/surfaces/:name/:runId` → generic surface page, `/runs` → run selector. Top nav with surface menu (driven by `useSurfaces()`), left rail with run history, `<PageHeader>` and `<StatusChip>` base components from `frontend/src/components/`. Run selector reads `outputs/` runs from a thin backend endpoint (`GET /runs` — added to `api/app.py` as a 1-endpoint extension).
- [ ] CB5 — Mission Control surface: KPI grid (4-6 metric cards: total demand, MAPE, drift status, model health), alert list, recent activity feed. Reads `useSurface('mission_control', runId)`. Component tests for `<MetricCard>`, `<AlertList>`, `<MissionControl>`. Visual ref: `prototype/data_intelligence_cockpit/code.html` + `screen.png`.
- [ ] CB6 — Data Health surface: schema validation table, missingness heatmap, freshness indicators, join-coverage status. Reads `useSurface('data_health', runId)`. Component tests for `<DataTable>` (the workhorse for this surface), `<StatusChip>`. Visual ref: `prototype/data_health_desktop/code.html`.
- [ ] CB7 — EDA Explorer surface: filter panel (series selector, date range, granularity), plot grid (distribution + time-series + decomposition), segment map. Reads `useSurface('eda_explorer', runId)` + `usePlot()` for the 3 plot kinds. Component tests for `<FilterPanel>`, `<PlotFrame>`. Visual ref: `prototype/eda_explorer_desktop/code.html`.
- [ ] CB8 — Feature Factory surface: feature catalog table, feature importance bar chart, transform pipeline visualizer. Reads `useSurface('feature_factory', runId)`. Component tests for `<FeatureRow>`, `<ImportanceChart>`. Visual ref: `prototype/feature_factory_desktop/code.html`.
- [ ] CB9 — Model Arena surface: leaderboard table (model family × fold × metric), comparison view (overlay forecasts), promotion-recommendation banner. Reads `useSurface('model_arena', runId)`. Component tests for `<Leaderboard>`, `<ForecastOverlay>`. Visual ref: `prototype/model_arena_desktop/code.html`.
- [ ] CB10 — Forecast Review surface: forecast-vs-actual chart, residual diagnostics, segment-level drill-down. Reads `useSurface('forecast_review', runId)` + `usePlot()`. Component tests for `<ForecastChart>`, `<ResidualPanel>`. Visual ref: `prototype/forecast_review_desktop/code.html`.
- [ ] CB11 — Replenishment Board + Learning Journal: approval queue with acknowledge/override actions, learning journal timeline. Both surfaces share the approval-event data shape from `api/models.py`. Component tests for `<ApprovalQueue>`, `<JournalTimeline>`. Visual ref: `prototype/replenishment_board_desktop/code.html` + `prototype/learning_journal_desktop/code.html`.
- [ ] CB12 — MLOps Monitor + plot rendering layer + integration test: MLOps Monitor reads the four Phase 7 markdown artifacts from `outputs/{run_id}/` via a thin backend endpoint (`GET /monitor/{run_id}` — added to `api/app.py`). The `<PlotFrame>` wrapper absorbs all 7 plot kinds from CB7/CB10 into a single reusable component. End-to-end integration test (`frontend/tests/e2e.test.ts`) walks the full chain: dev server boot → surface nav → surface render → acknowledge action on Replenishment Board. Verify `pnpm test` (component + contract + e2e) green and `uv run pytest -q` (full backend suite) still green. Tick Phase 9 plan. Mark complete. Add glossary terms (`Cockpit Frontend`, `Plot Frame`, `Surface Route`).

**What this repo does NOT ship (out-of-repo workstream):**

- The deployment story (CDN config, nginx rules, the production `uvicorn` invocation behind a reverse proxy) — handled outside this repo.
- User accounts / auth (the platform has no auth layer; the cockpit trusts the network).
- Real-time updates (websockets, SSE) — the cockpit polls on focus + on manual refresh.
- A mobile-native shell — the SPA is responsive down to ~768px but phone-first is a separate workstream.

**Future external integrations:** an alternative frontend (a Svelte port, a mobile-native shell, a Jupyter widget) can consume the same FastAPI surface without backend changes. The contract is the `openapi.json` schema; whatever consumes it is the frontend's problem, not the platform's.

---

## Per-CB TDD discipline (every checkbox)

Each CB follows this exact pattern. The `plan-execution-cadence` skill enforces the per-CB commit + tick + handoff cadence; the per-CB TDD discipline is enforced here.

1. **Write the test first.** For component CBs: Vitest + RTL test that mounts the component with a mocked API client and asserts rendered output. For contract CBs: the test boots `uv run python -m api.app` and asserts the response shape. For surface CBs: the component test uses MSW (`Mock Service Worker`) to intercept `/surfaces/{name}/{run_id}` and return a canned response matching `api/models.py`.
2. **Verify RED.** `pnpm test` shows the new test failing (component not built / API client missing / surface not registered).
3. **Implement minimal code.** Build the component, hook, or page. For surface CBs, this is `frontend/src/pages/<SurfaceName>.tsx` + the corresponding route in `frontend/src/router.tsx` + the component test going GREEN.
4. **Verify GREEN.** `pnpm test` passes. `pnpm build` produces a working `dist/`. `uv run pytest -q` still green (the contract test on the backend side validates the surface schema hasn't drifted).
5. **Commit.** Lead message: `phase 9 cb<N> — <one-line summary>`. Push after commit.

**Bootstrap note (CB2):** the very first commit (`phase 9 cb2 — frontend scaffold`) must include `package.json`, `pnpm-lock.yaml`, `tsconfig.json`, `vite.config.ts`, `tailwind.config.ts`, and the Vitest config. Every later CB depends on `pnpm install` having run at least once. Do not split the scaffold across CBs.

---

## Files likely to change

**New (created across the phase):**
- `frontend/` — new top-level package directory
  - `package.json`, `pnpm-lock.yaml`, `tsconfig.json`, `tsconfig.node.json`
  - `vite.config.ts`, `tailwind.config.ts`, `postcss.config.js`, `vitest.config.ts`
  - `index.html`, `src/main.tsx`, `src/App.tsx`, `src/router.tsx`
  - `src/api/client.ts`, `src/api/hooks.ts`, `src/api/schema.ts` (generated, gitignored)
  - `src/components/MetricCard.tsx`, `src/components/DataTable.tsx`, `src/components/StatusChip.tsx`, `src/components/PageHeader.tsx`, `src/components/PlotFrame.tsx`
  - `src/pages/MissionControl.tsx`, `src/pages/DataHealth.tsx`, `src/pages/EdaExplorer.tsx`, `src/pages/FeatureFactory.tsx`, `src/pages/ModelArena.tsx`, `src/pages/ForecastReview.tsx`, `src/pages/ReplenishmentBoard.tsx`, `src/pages/LearningJournal.tsx`, `src/pages/MlopsMonitor.tsx`
  - `tests/contract.test.ts`, `tests/e2e.test.ts`, plus per-component `*.test.tsx` siblings
- `.gitignore` — add `frontend/dist/`, `frontend/node_modules/`, `frontend/src/api/schema.ts` (regenerated)
- `docs/superpowers/plans/2026-06-05-agentic-demand-forecasting-workspace.md` — add `### Phase 9: Cockpit Frontend` section, tick checkboxes as they land

**Modified (small, surgical):**
- `api/app.py` — add `GET /runs` (run selector endpoint) in CB4, add `GET /monitor/{run_id}` (MLOps Monitor endpoint) in CB12. Both are thin pass-throughs to existing registry; no new contract surface, no Pydantic schema changes.
- `.gitignore` — see above
- `CONTEXT.MD` — add glossary terms in CB12

**Deleted (at end of phase, after CB12 lands):**
- `prototype/data_intelligence_cockpit/` — design system ported to Tailwind, redundant
- `prototype/data_intelligence_cockpit_desktop/` — Mission Control, replaced by CB5
- `prototype/data_health_desktop/` — replaced by CB6
- `prototype/eda_explorer_desktop/` — replaced by CB7
- `prototype/feature_factory_desktop/` — replaced by CB8
- `prototype/model_arena_desktop/` — replaced by CB9
- `prototype/forecast_review_desktop/` — replaced by CB10
- `prototype/replenishment_board_desktop/` — replaced by CB11
- `prototype/learning_journal_desktop/` — replaced by CB11
- `prototype/mlops_monitor_desktop/` — replaced by CB12
- `prototype/canonical_builder_desktop/` — never wired (Canonical Table Builder is a backend-only surface, no UI consumer in Phase 9)
- `prototype/intelligence_platform_v2/` — exploration probe, no committed use

The deletions land in CB12 as a single commit titled `phase 9 cb12 — delete prototype/ (absorbed by frontend/)`.

---

## Tests / validation

**Per-CB:**
- `pnpm test` — Vitest component tests + contract test (after CB3) + e2e test (after CB12). Expected: monotonically increasing pass count.
- `uv run pytest -q` — full backend suite, must stay at 846 passing (no backend changes other than 2 thin endpoints in CB4/CB12).
- `pnpm build` — produces `frontend/dist/` without TypeScript errors.

**End of phase (CB12 acceptance gates):**
- [ ] `pnpm test` is green; component + contract + e2e test counts reported in commit body.
- [ ] `uv run pytest -q` is green; full suite still at 846 passing.
- [ ] `pnpm build` produces a `frontend/dist/` that FastAPI serves at `http://localhost:8000/` (verifiable by `curl http://localhost:8000/` returning the SPA index.html).
- [ ] All 9 surfaces render in a browser against a real FastAPI process + a real run_id.
- [ ] Replenishment Board's acknowledge action calls `POST /approvals/{request_id}/acknowledge` (or whatever the Phase 6 endpoint shape is — verify in CB11) and the audit log records the decision.
- [ ] `prototype/` is deleted; the `git log` shows the deletion commit.
- [ ] `docs/superpowers/plans/2026-06-05-agentic-demand-forecasting-workspace.md` has Phase 9 ticked with all 12 CB SHAs.

---

## Risks, tradeoffs, and open questions

1. **Prototype HTML deletion is irreversible.** The 11 prototype folders are the only existing visual reference. CB12's deletion commits them away. Mitigation: each surface CB (CB5–CB12) is the per-surface visual-decision moment — read the matching prototype once, extract the pattern, build from tokens. By CB12 the patterns are all in `frontend/src/components/` and `frontend/src/pages/`, so deletion is lossless.

2. **openapi-typescript drift.** The contract test (CB3) catches schema drift at test time, but the codegen step itself (`pnpm generate-api`) is a manual `pnpm` invocation. Risk: a backend change to `api/models.py` ships without a regen, and the contract test passes against stale generated types. Mitigation: add a `precommit` hook in CB12 that runs `pnpm generate-api` and fails if `frontend/src/api/schema.ts` is out of date. Until CB12, the contract test itself is the safety net (it re-bootstraps from `/openapi.json` at test time, so the contract assertion is always against current backend).

3. **Plot rendering library choice.** Recharts covers most of the 7 plot kinds (line, bar, area, scatter, composed) but the decomposition plot (trend + seasonal + residual) and the segment map are awkward in Recharts. Options for those two: (a) hand-roll with D3 + the `<PlotFrame>` wrapper, (b) swap Recharts for visx, (c) accept Recharts's limitations. CB7 and CB10 are where this surfaces. If it becomes a real problem, raise it at the CB7 boundary — don't preempt.

4. **The `intelligence_platform_v2` prototype is unexplored.** It's the only prototype folder without a matching surface in the Phase 8 backend. Possibilities: (a) it was an early cockpit concept that was superseded by `data_intelligence_cockpit/`, (b) it represents a future-state design that informs Phase 9's long-term direction. CB12 deletes it. If during CB5–CB11 you find a layout decision that the active prototype doesn't cover but `intelligence_platform_v2` does, surface it and decide whether to deviate from the active prototype. Default: don't deviate; the active prototype + DESIGN.md are the spec.

5. **The plan assumes `pnpm` is the JS toolchain.** If the repo has any existing JS tooling in `package.json` or `pnpm-lock.yaml`, this assumption is wrong. CB1 verifies: read the repo for existing JS artifacts, and switch to `npm` or `yarn` if the lockfile says so. The plan's `pnpm` references become `npm` / `yarn` references without semantic change.

6. **The plan adds 2 thin endpoints to `api/app.py` (`GET /runs` in CB4, `GET /monitor/{run_id}` in CB12).** Both are read-only pass-throughs to existing registry state. No new Pydantic schemas. No new business logic. If CB4 or CB12 surfaces a real backend gap (e.g. MLOps Monitor needs drift metrics that aren't exposed today), raise it as a new CB rather than expanding CB4/CB12 scope.

---

## Standing rules for the handoff doc

These 6 lines get carried in every handoff doc across this phase (per `plan-execution-cadence`):

1. One checkbox at a time. Handoff at 40% context OR checkbox completion. Use the `handoff` skill with the next CB as the argument.
2. Commit cadence: one commit per phase sub-step. Lead message = `phase 9 cb<N> — <one-line>`. Push after commit unless told otherwise.
3. Test commands: `pnpm test` (frontend) + `uv run pytest -q` (backend). Both green before every commit.
4. Never-commit paths: `.claude/`, `.claude/settings.local.json`, `.vscode/`, `.idea/`, `__pycache__/`, `.venv/`, `.env`, `.pytest_cache/`, `backend/outputs/`, `.teach-architecture/`, `graphify-out/` (root AND `backend/forecasting/graphify-out/`), `platform-explorer.html`, generated `*.md` reports (`MONITORING_REPORT.md`, `DRIFT_REPORT.md`, `OVERRIDE_ANALYSIS.md`, `MODEL_HEALTH.md`), `frontend/dist/`, `frontend/node_modules/`, `frontend/src/api/schema.ts` (regenerated by `pnpm generate-api`).
5. Plan path: `docs/superpowers/plans/2026-06-05-agentic-demand-forecasting-workspace.md` (Phase 9 section, this plan). Glossary: `CONTEXT.MD`.
6. UI design system: follow `prototype/*/DESIGN.md` exactly (Deep Indigo #3525cd, Electric Cyan #4cd7f6, Light Slate #f8fafc, Inter + JetBrains Mono). The 11 `prototype/*/code.html` files are visual reference only — read once per surface to extract layout patterns, never copy markup.

---

## Open question for the user (before CB1)

**Phase numbering.** This plan calls itself "Phase 9" by extending the existing plan's 1–8 sequence. Two alternatives:

- **A. Phase 9 (this plan).** Append `### Phase 9: Cockpit Frontend` to `docs/superpowers/plans/2026-06-05-agentic-demand-forecasting-workspace.md`. Tick checkboxes there. Consistent with how Phases 1–8 are tracked.
- **B. Separate plan file at `docs/superpowers/plans/2026-06-18-cockpit-frontend.md`.** The frontend gets its own plan doc, decoupled from the original 8-phase plan. Easier to find ("where's the frontend plan?") but the original plan stops at Phase 8 with no obvious continuation marker.

**Recommendation: A.** The plan's existing structure already tracks 8 phases; CB1 of this phase is the in-place rewrite that makes the in-repo / out-of-repo split explicit, mirroring the Phase 6/8 pattern. A separate plan file risks the frontend drifting from the backend's terminology, glossary, and never-commit paths. The original plan is the source of truth for the whole platform.

If A, the CB1 commit will: (1) rewrite the `### Phase 8: Data Intelligence Cockpit` "Future external integrations" paragraph to point at Phase 9, (2) append `### Phase 9: Cockpit Frontend` with the 12-CB checkbox list above, (3) update the progress-summary table to add the Phase 9 row, (4) update the Hard Rules section to add the frontend's never-commit paths.
