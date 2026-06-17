# Issue 003 — Document the intentional code-vs-config escalation tracker distinction

**Type:** AFK
**Source:** PRD-001 §Solution CB3
**Blocked by:** None (independent of Issues 001 and 002)
**Labels:** `refactor`, `post-phase-6`, `docs`

## What to build

The two escalation paths implement a "3-attempt cap" with different shapes: `code_escalation.py` uses `EscalationTracker` (file-backed JSON state, cross-run persistence); `config_escalation.py` uses a per-action in-memory `Counter` (per-run only). The semantic difference is intentional — custom model families (code escalation) are permanent additions, FeatureFlag flips (config escalation) are not. A future maintainer reading both files in isolation might "fix" the apparent inconsistency by unifying the two, which would be the wrong call.

This issue adds a brief comment (3–5 lines) above each tracker explaining the intentional shape difference and pointing at the other module. No code change. A tiny test in `tests/test_escalation_tracker_docs.py` asserts both comments exist (via module `__doc__` or by parsing the source), so the comment cannot silently disappear in a future refactor.

## Acceptance criteria

- [ ] A comment block of 3–5 lines sits directly above the `EscalationTracker` class in `code_escalation.py`, explaining why it is file-backed and survives restarts, and pointing at `config_escalation.py` for the contrasting design
- [ ] A comment block of 3–5 lines sits directly above the per-action `Counter` in `config_escalation.py` (or whichever identifier holds the per-run attempt count), explaining why it is in-memory and per-run only, and pointing at `code_escalation.py` for the contrasting design
- [ ] Each comment notes the option of extracting a shared `AttemptTracker` Protocol at a future time if unified observability is needed (a single sentence is enough)
- [ ] A new test file `tests/test_escalation_tracker_docs.py` (or an addition to an existing tests file) imports both modules and asserts each comment's presence — the assertion is structural (substring on the module's source) so it survives reformatting
- [ ] `uv run pytest -q` is green; full suite remains at ~651 passing + 1 new doc-assertion test = ~652
- [ ] One commit on `main` with message `feat: refactor cb3 - document intentional code-vs-config escalation tracker distinction`
- [ ] Commit is pushed

## Blocked by

None — can start immediately. Independent of Issues 001 and 002.
