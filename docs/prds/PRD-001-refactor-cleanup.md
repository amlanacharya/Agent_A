# PRD-001 — Post-Phase 6 Refactor Cleanup

**Status:** Draft
**Date:** 2026-06-17
**Source review:** `file:///C:/Users/amlan/AppData/Local/Temp/architecture-review-20260617.html` (Phases 3–6, 5 candidates, 2 strong / 2 worth exploring / 1 speculative)
**Scope:** Fix the two "Strong" findings and the one "Worth exploring" finding whose cost is justified. Defer the two pure-locality splits (no defect today, no plan to add one).

---

## Problem Statement

After Phase 6 (Approvals, Scheduling, ERP Handoff) shipped, the codebase has 646 passing tests and the platform's feature set is complete through Phase 6. A 5-candidate architecture review (`architecture-review-20260617.html`) surfaced two structural issues that the team would like resolved before Phase 7 (Monitoring & Drift) and Phase 8 (Cockpit UI) add more code on top:

1. **`backtest.py` has a wrong-direction dependency on `model_escalation.py`.** A pure evaluation operation (backtest fold → score) imports a gate-layer function (`check_data_contract`) from the layer above it. The function is pure, but the import pulls in the module body of `model_escalation`, which in turn pulls in `code_escalation` (which holds `EscalationTracker`, a file-backed state machine). Today the module bodies are pure (no side effects at import time), so the runtime impact is latent — but a future maintainer who adds a module-level registry or cache to `model_escalation` will silently break the backtest import path. The fix is mechanical: inline the 8-line `check_data_contract` function into `backtest.py` and delete the import.

2. **`MeasureMASE` is a Protocol that is meant to be a stub-friendly seam, but it has only one adapter in production.** The Protocol lives in `config_escalation.py` (the same file as the production adapter), and there is no named stub adapter — tests pass ad-hoc callables. The seam is recognized in the docstring ("Stub-friendly MASE") but the seam has not been *promoted* to a first-class shape: a Protocol in `contracts.py` and a named `StubMeasureMASE` in a tests module. Two named adapters = real seam.

The two deferred items (splitting `promotion.py` into 3 files, splitting `contracts.py` into phase-scoped files) are pure-locality improvements with no defects. The team is choosing to land them only under pressure (e.g. when a future change needs the new locality to be safe). Documenting this decision is part of the PRD.

The "document the dual-tracker distinction" candidate from the review (code-escalation's file-backed `EscalationTracker` vs config-escalation's in-memory `Counter`) is in scope: a comment in each module explaining the intentional semantic difference (cross-run vs per-run) is essentially free and prevents future confusion. No code change.

---

## Solution

Three small, test-protected refactors, each landing as a separate commit on `main`:

1. **CB1 — Fix `backtest.py` dependency direction.** Inline the 8-line `check_data_contract` function into `backtest.py` as a private helper. Delete the `from forecasting.model_escalation import check_data_contract` line. Existing `test_backtest.py` continues to pass unchanged because the function's behaviour, signature, and the `RobustnessCheck` it returns are identical.

2. **CB2 — Promote `MeasureMASE` to a first-class seam.** Move the Protocol from `config_escalation.py` to `contracts.py`. Add a named `StubMeasureMASE` adapter in a new `tests/stubs.py` module. Update `config_escalation.py` to import the Protocol from `contracts` and export `default_measure_mase` (the production adapter). Existing `test_config_escalation.py` continues to pass; new tests in `test_stubs.py` (or appended to `test_config_escalation.py`) exercise the stub adapter directly.

3. **CB3 — Document the dual-tracker distinction.** Add a brief comment in `code_escalation.py` (above `EscalationTracker`) and in `config_escalation.py` (above the per-action `Counter`) explaining why the two are intentionally different shapes (cross-run persistence vs per-run only). No code change. Zero test impact.

**Total tests added:** ~5 (for the new stub adapter surface). **Total tests removed:** 0. **Total tests modified:** 0. **Suite goes from 646 → ~651 green.**

---

## User Stories

1. As a maintainer of `backtest.py`, I want the module to depend only on its true dependencies (forecasting models + contracts), so that adding a module-level registry to `model_escalation` does not silently break backtest imports.

2. As a maintainer of `backtest.py`, I want the data-contract validation to be co-located with the fold-evaluation code, so that the check's invariants are visible in the same file that produces the forecast that the check validates.

3. As a reader of `backtest.py`, I want to see the full backtest pipeline (fit → predict → validate contract → score) in one module, so that I can understand the operation without jumping to a higher layer.

4. As a maintainer of the config-escalation loop, I want `MeasureMASE` to be a first-class seam with two named adapters, so that the loop's stop-condition logic can be unit-tested in milliseconds without invoking the real harness.

5. As a maintainer of the config-escalation loop, I want the `MeasureMASE` Protocol to live in `contracts.py` (the canonical type home), so that any future consumer of the loop's MASE-measurement surface can import the type without depending on the implementation file.

6. As a maintainer of `config_escalation.py`, I want the production adapter (`default_measure_mase`) and the test stub (`StubMeasureMASE`) to be named, documented shapes, so that the seam is not "hypothetical" (one adapter, future-shaped) but "real" (two adapters, exercised today).

7. As a future user of the config-escalation loop, I want to be able to inject a custom MASE measurer by name (not by passing an ad-hoc lambda), so that production code can wire in alternative measurers (e.g. a measurer that runs only on a subset of folds) with a discoverable interface.

8. As a maintainer of the test suite, I want a single `tests/stubs.py` module that exports the platform's stub adapters, so that test files do not redefine the same stub pattern inline in every test module.

9. As a reader of `code_escalation.py`, I want a brief comment above `EscalationTracker` explaining why it is file-backed and survives restarts, so that I do not "fix" the apparent inconsistency with `config_escalation.py`'s in-memory `Counter` by unifying them.

10. As a reader of `config_escalation.py`, I want a brief comment above the per-action `Counter` explaining why it is in-memory and per-run-only, so that I do not "fix" the apparent inconsistency with `code_escalation.py`'s file-backed `EscalationTracker` by promoting the `Counter` to disk.

11. As a future maintainer who finds the dual-tracker distinction load-bearing (e.g. when a cockpit view needs unified observability), I want the comments to flag the option of extracting a shared `AttemptTracker` Protocol at that time, so that the unification is not premature but is also not forgotten.

12. As a developer reading the architecture review (`architecture-review-20260617.html`), I want the 5 candidates to be either resolved in the codebase or explicitly deferred in the plan, so that the review's findings do not silently sit unresolved.

13. As the project lead, I want the deferred items (the two `promotion.py` and `contracts.py` splits) to be tracked in the issue tracker, so that the next time someone proposes a change that would benefit from the new locality, the work is visible and discoverable.

14. As a future Phase 7 contributor, I want the backtest dependency direction to be correct before I add the monitoring layer that reads backtest outputs, so that monitoring does not inherit a backwards arrow.

---

## Implementation Decisions

**ID-1: Inline `check_data_contract` into `backtest.py` as a private helper, do not move it to a shared utilities module.**

The function is 8 lines, has no shared state, and is used in exactly one call site (`backtest.py:179`). The simplest fix is inlining. A shared `validation.py` module would be premature for a single use; if a second consumer appears in Phase 7+, that is the right time to extract.

**ID-2: Move `MeasureMASE` Protocol to `contracts.py`. Do not create a new `escalation_protocols.py` file.**

The Protocol is the *type* the loop uses. Types live in `contracts.py`. Creating a new file just for one Protocol is the kind of churn the deferred splits were avoided for. `contracts.py` is the canonical type home; the Protocol belongs there even if it is currently used by only one module.

**ID-3: Add `StubMeasureMASE` to a new `tests/stubs.py` module. Do not put stubs inside `test_config_escalation.py`.**

A shared stubs module is the right place for any future stub adapters. Test files that need a stub can `from tests.stubs import StubMeasureMASE`. Keeping the stub inside the test file would scatter the stub pattern as more stubs are added.

**ID-4: The `default_measure_mase` function stays in `config_escalation.py`. Do not move it to a new `harness_adapters.py`.**

The function is the production adapter for the loop. It lives with the loop, not in a separate adapters module. The split (Protocol in `contracts`, adapter in `config_escalation`) is the minimum that promotes the seam; further splitting is the kind of pure-locality churn the deferred items avoided.

**ID-5: The documentation comments in `code_escalation.py` and `config_escalation.py` are 3–5 lines each. Do not write an essay.**

The point is to answer the question a future reader will ask ("why is this shape different from the other one?"). A pointer to the dual-tracker section of the architecture review, plus one sentence per module, is enough. The architecture review document itself is the longer-form treatment.

**ID-6: The two deferred items (the `promotion.py` split and the `contracts.py` split) become GitHub issues, not plan-doc checkboxes.**

The plan doc tracks phase work, not refactor backlog. Refactor work that has not been decided-on goes in the issue tracker. If a future phase needs either split, the issue is already there to attach the work to.

**ID-7: Each of CB1, CB2, CB3 lands as a single commit on `main` with the existing commit-message convention (`feat: phase 6.x cbN - <one-line summary>`).**

Per the project's standing rule for cadence: one commit per checkbox, `uv run pytest` green before commit, push after. The CB numbers in this PRD match the commit numbers in the log.

**ID-8: Backward-compat re-exports are NOT added when the splits land.**

Per the project's standing rule: no backward-compat re-exports in dev. (This rule does not apply to CB1–CB3, since none of them move any name; it is included here so the rule is on the record for the deferred items when they do land.)

---

## Testing Decisions

**TD-1: A good test for this PRD exercises a structural property of the change, not an implementation detail.**

Examples of structural properties (good): "`backtest` does not import from `model_escalation`" (CB1), "`MeasureMASE` is importable from `contracts`" (CB2), "`StubMeasureMASE` returns the values its constructor was given" (CB2), "the two tracker files have the explanatory comment" (CB3 — can be enforced with a static check, see TD-4).

Examples of implementation details (bad): "the inlined function is byte-identical to the original" (CB1), "the Protocol's source location is `contracts.py` line 1234" (CB2 — too brittle).

**TD-2: Modules tested:** `backtest.py` (CB1), `contracts.py` + `config_escalation.py` + `tests/stubs.py` (CB2), `code_escalation.py` + `config_escalation.py` (CB3, comment presence only).

**TD-3: Prior art for the tests:** `tests/test_backtest.py` (CB1 — unchanged, the test surface already covers the fold-evaluation logic), `tests/test_config_escalation.py` (CB2 — unchanged for the loop logic, with new tests appended for the stub adapter), and `tests/test_orchestration_contracts.py` (CB2 — same pattern: parametrize over Literal kinds, assert contract shape, assert importability from `contracts`).

**TD-4: CB3's "comments are present" check is enforced by a tiny AST-style test, not by a regex on the docstring.**

A test that grep-matches the comment is fragile (whitespace changes break it). The right test imports both modules, inspects `__doc__` on the relevant class, and asserts that the explanation string is a substring. This survives reformatting and is still a structural property.

**TD-5: All existing 646 tests must continue to pass after every CB.**

The refactor is test-protected: any import change in `backtest.py` (CB1) or `config_escalation.py` (CB2) that breaks a downstream test is a contract violation, not a refactor. The full suite is the safety net.

**TD-6: No test for "the deferral" of the two skipped items.**

The fact that those items are deferred is documented in the GitHub issues (ID-6), not in the test suite. Tests assert what the code does, not what it doesn't do.

---

## Out of Scope

**OOS-1: Splitting `promotion.py` (702 lines) into `promotion_comparison.py` / `shadow_mode.py` / `promotion_audit.py`.**

The reviewer's candidate #3. The three responsibilities (pure comparison, shadow-mode runner, markdown audit) are real, and the split has locality benefits. But there is no defect today: every test passes, no maintainer is currently confused, and the test surface does not require a split to be efficient. The split is deferred. If a future phase (e.g. Phase 8 cockpit needs shadow-mode observability) makes the split load-bearing, the work is tracked as a GitHub issue.

**OOS-2: Splitting `contracts.py` (1113 lines) into phase-scoped files.**

The reviewer's candidate #1. The biggest blast radius of any item in the review: every importer in the codebase touches `contracts.py`. The split is the kind of work that pays off when the file hits ~2000 lines and the locality cost is real, not when it is at 1113 lines and the locality cost is hypothetical. Per the project's standing rule, the split (if it lands) would not leave a backward-compat re-export facade — every importer is updated to the new canonical path. That is a large CB. Deferred.

**OOS-3: Unifying the dual attempt-tracking shape (code escalation's `EscalationTracker` vs config escalation's per-action `Counter`).**

The reviewer's candidate #5. The reviewer's own caveat: the semantic difference (cross-run persistence vs per-run only) may be intentional and load-bearing. Unification without a concrete consumer of the unified shape would be a premature abstraction. Documented in the comments per CB3, not unified.

**OOS-4: New refactor seams not in the review.**

The review identified 5 candidates. If a future maintainer finds a 6th, it goes in the issue tracker and gets its own PRD. This PRD covers exactly the 5 candidates, with 3 in scope and 2 deferred.

**OOS-5: Any behaviour change.**

The PRD is refactor-only. No functional change, no new features, no API additions, no contract changes (other than the `MeasureMASE` Protocol moving from one file to another, which is a location change, not a shape change).

---

## Further Notes

**FN-1: The architecture review document (`architecture-review-20260617.html`) is a local Temp file, not committed to the repo.**

It was generated as a one-off review. The findings are summarized in this PRD; the longer-form treatment lives in the HTML. If the team wants a permanent record, the review can be committed to `docs/reviews/2026-06-17-architecture-review.html` as a follow-up — not in this PRD's scope.

**FN-2: The PRD's CB numbering matches the expected commit log:**

```
<pending>  feat: refactor cb1 - inline check_data_contract into backtest, fix dependency direction
<pending>  feat: refactor cb2 - promote MeasureMASE to contracts + add StubMeasureMASE adapter
<pending>  feat: refactor cb3 - document intentional code-vs-config escalation tracker distinction
```

Each commit should leave the suite at 646 (CB1), 646 (CB2, no test changes), and ~651 (CB2 with the new stub tests), 651 (CB3, doc-only). The exact final count depends on how many new tests CB2 adds (estimated 5: one per stub value, one for "importable from tests.stubs", one for the Protocol being importable from `contracts`, one asserting the stub is a `MeasureMASE`, and one for the default adapter still working).

**FN-3: After this PRD lands, the next planned work is Phase 7 (Monitoring & Drift).**

The backtest dependency fix (CB1) is forward-looking for Phase 7: the monitoring layer will read backtest outputs, and a backwards dependency arrow at this layer would propagate. Landing CB1 before Phase 7 keeps the layer ordering correct as more code lands on top of it.

**FN-4: The two deferred items become GitHub issues tagged `refactor` and `post-phase-6`.**

Issue titles:
- "Split `promotion.py` into comparison / shadow-mode / audit modules"
- "Split `contracts.py` into phase-scoped contract files"

Each issue body links to the architecture review's relevant section and to the OOS-1 / OOS-2 paragraphs of this PRD.
