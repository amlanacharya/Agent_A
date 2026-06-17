# Issue 005 — DEFERRED: split `contracts.py` into phase-scoped contract files

**Type:** Deferred (parent tracker issue)
**Source:** PRD-001 §Out of Scope OOS-2
**Blocked by:** None (no work has started; this is the future parent ticket)
**Labels:** `refactor`, `deferred`, `post-phase-6`

## What to build

`contracts.py` (1113 lines) holds every Pydantic model for every phase: Pre-flight, EDA, Feature Factory, Models, Proposals, Escalation, Promotion, Approvals, Scheduling, ERP. The interface (what you must know to add a new contract) is nearly as complex as the file itself. Every module in the platform imports from it.

This issue is a **parent tracker** for the future split. When the work lands, it should:

1. Move types into phase-scoped files: `preflight_contracts.py`, `feature_contracts.py`, `model_contracts.py`, `promotion_contracts.py`, `phase6_contracts.py`, `conductor_contracts.py`.
2. Update every importer in the codebase to the new canonical path. **No backward-compat re-export facade** — the project's standing rule is no backward-compat re-exports in dev. Every importer is updated in the same CB (or in tightly-coupled CBs with the facade removed at the end).
3. Delete `contracts.py` once all importers are migrated.

The split is deferred because there is no defect today, and the blast radius is the largest in the review: every importer in the codebase touches `contracts.py`. Per the standing rule, the facade must not be left behind, which means the work has to be careful and all-at-once.

**When to start:** when the file hits ~2000 lines and the locality cost is real, not when it is at 1113 lines and the locality cost is hypothetical. Also re-evaluate if a future phase adds a major new type group (e.g. Phase 7's monitoring types, Phase 8's cockpit types).

## Acceptance criteria (for the future work, not today)

- [ ] Six new contract files exist, holding the relevant types
- [ ] `contracts.py` is removed (not left as a re-export facade — standing rule)
- [ ] Every importer in `backend/forecasting/` and `tests/` is updated to the new canonical paths
- [ ] `uv run pytest -q` is green; full suite continues to pass
- [ ] Commits land on `main` (likely one large commit, or a sequence with intermediate states)
- [ ] The PRD's OOS-2 paragraph is removed in a docs follow-up (the item is no longer "out of scope" once done)

## Blocked by

None — this is a parent issue. No work has started.

## Status

**DEFERRED.** Do not start. Re-evaluate when the file grows or a new type group arrives.
