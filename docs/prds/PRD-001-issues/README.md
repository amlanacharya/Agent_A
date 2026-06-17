# PRD-001 Issues — Index

This folder holds the 5 issues derived from PRD-001 (Post-Phase 6 Refactor Cleanup), broken into tracer-bullet vertical slices per the `to-issues` skill. Issues are independent — no work has started on any of them. Each file in this folder is a standalone issue body, ready to paste into a GitHub issue.

**Source:** `docs/prds/PRD-001-refactor-cleanup.md` (committed 2026-06-17, `93c5b56`).

**Source review:** `file:///C:/Users/amlan/AppData/Local/Temp/architecture-review-20260617.html` (Phases 3–6, 5 candidates).

## Execution order

The three in-scope issues are independent (no blockers between them). The two deferred items are parent tracker issues — do not start work on them.

| Order | File | Type | Blocked by | Source PRD ref |
|---|---|---|---|---|
| 1 | [`001-backtest-inline-check-data-contract.md`](./001-backtest-inline-check-data-contract.md) | AFK | None | PRD-001 CB1 |
| 2 | [`002-measuremase-protocol-promote.md`](./002-measuremase-protocol-promote.md) | AFK | None | PRD-001 CB2 |
| 3 | [`003-dual-tracker-doc.md`](./003-dual-tracker-doc.md) | AFK | None | PRD-001 CB3 |
| — | [`004-deferred-promotion-split.md`](./004-deferred-promotion-split.md) | Deferred | None | PRD-001 OOS-1 |
| — | [`005-deferred-contracts-split.md`](./005-deferred-contracts-split.md) | Deferred | None | PRD-001 OOS-2 |

## Suggested labels

- `refactor` — all 5
- `post-phase-6` — all 5
- `docs` — issue 003 only
- `deferred` — issues 004 and 005 only

## Suite trajectory

646 (current) → 646 (issue 001, no test changes) → ~651 (issue 002 with ~5 new stub tests) → ~652 (issue 003 with 1 new doc-assertion test). All issues must keep `uv run pytest -q` green.

## How to publish to GitHub

No `gh` CLI and no GitHub token in this environment. To publish, open the issues page for `amlanacharya/Agent_A`, click "New issue", and paste the body of each file. Suggested order matches the execution order above.
