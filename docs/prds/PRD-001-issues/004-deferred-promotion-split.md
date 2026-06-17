# Issue 004 — DEFERRED: split `promotion.py` into comparison / shadow-mode / audit modules

**Type:** Deferred (parent tracker issue)
**Source:** PRD-001 §Out of Scope OOS-1
**Blocked by:** None (no work has started; this is the future parent ticket)
**Labels:** `refactor`, `deferred`, `post-phase-6`

## What to build

`promotion.py` (702 lines) holds three distinct responsibilities: pure champion/challenger comparison (`compare_candidate_to_champion`, `check_window_leakage`), the shadow-mode runner (`run_shadow_mode` re-runs the full forecast harness on live data), and markdown audit output (`format_promotion_decision`, `write_promotion_decision` writes `PROMOTION_DECISIONS.md`). The three have different test shapes (table-driven unit tests vs harness integration test vs snapshot test), and bundling them means a maintainer cannot test comparison logic without importing the harness.

This issue is a **parent tracker** for the future split. When the work lands, it should:

1. Move pure comparison into `promotion_comparison.py` (~200 lines, imports only from `contracts.py` and `backtest.py`).
2. Move the shadow-mode runner into `shadow_mode.py` (~150 lines, calls `forecast_harness.py`).
3. Move the markdown audit into `promotion_audit.py` (~150 lines, writes to disk).
4. Make `promotion.py` a thin re-export facade for backward compat (one release), then remove the facade in a follow-up.

The split is deferred because there is no defect today. It is **not** the user's preferred path: per the project's standing rule, the re-export facade should be removed in the same work, not left as cruft. That means every importer is updated in one CB.

**When to start:** when a future phase (e.g. Phase 8 Cockpit needing shadow-mode observability) makes the split load-bearing. The parent issue's body references this PRD and the architecture review's candidate #3.

## Acceptance criteria (for the future work, not today)

- [ ] Three new modules exist: `promotion_comparison.py`, `shadow_mode.py`, `promotion_audit.py`
- [ ] `promotion.py` re-exports the public surface for exactly one release, then the facade is removed
- [ ] All existing tests pass; new test files per module have unit-test-only imports (no harness import in `test_promotion_comparison.py`)
- [ ] Commits land on `main` and the suite stays green
- [ ] The PRD's OOS-1 paragraph is removed in a docs follow-up (the item is no longer "out of scope" once done)

## Blocked by

None — this is a parent issue. No work has started.

## Status

**DEFERRED.** Do not start. Re-evaluate when a future phase needs the new locality.
