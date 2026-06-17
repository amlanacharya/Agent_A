# Issue 002 — Promote `MeasureMASE` Protocol to `contracts.py`, add `StubMeasureMASE` adapter

**Type:** AFK
**Source:** PRD-001 §Solution CB2
**Blocked by:** None (independent of Issue 001)
**Labels:** `refactor`, `post-phase-6`

## What to build

`MeasureMASE` is a Protocol in `config_escalation.py` (line ~81) that the config-escalation loop uses to measure post-application MASE. The docstring already calls it "stub-friendly", but the Protocol and its only production adapter (`default_measure_mase`) live in the same file, and there is no named stub adapter — tests pass ad-hoc callables. The seam is recognized but not promoted to a first-class shape.

This issue:

1. Moves the `MeasureMASE` Protocol declaration from `config_escalation.py` to `contracts.py` (the canonical type home).
2. Updates `config_escalation.py` to import the Protocol from `contracts` and re-export it from its own `__all__` for backward source compat (one session's worth of compat only — if any external test still imports `from forecasting.config_escalation import MeasureMASE`, it continues to work).
3. Creates a new `tests/stubs.py` module exporting a named `StubMeasureMASE` class — a deterministic measurer that returns a fixed value (or iterates a constructor-given list) regardless of the request/flags it receives. The stub satisfies the `MeasureMASE` Protocol.
4. Updates existing tests that hand-roll a stub to use `StubMeasureMASE` from the new module.
5. Adds new tests in `tests/test_stubs.py` covering: stub returns the configured value, stub iterates a list across calls, stub satisfies the `MeasureMASE` Protocol (structural check), `MeasureMASE` is importable from `forecasting.contracts`.

## Acceptance criteria

- [x] `MeasureMASE` Protocol is importable from `forecasting.contracts`
- [x] `MeasureMASE` is still importable from `forecasting.config_escalation` (re-export preserved)
- [x] `default_measure_mase` (the production adapter) remains in `config_escalation.py` and is unchanged in behaviour
- [x] `tests/stubs.py` exists and exports `StubMeasureMASE` with a documented constructor (`StubMeasureMASE(value: float)` for a constant; optional `StubMeasureMASE(values: list[float])` for iteration across calls)
- [x] `StubMeasureMASE` satisfies the `MeasureMASE` Protocol (a structural-typing test asserts this)
- [x] All existing tests that use a stub measurer are updated to import `StubMeasureMASE` from `tests/stubs.py` (no inline lambdas or anonymous classes remain in test files for this purpose)
- [x] `uv run pytest -q` is green; full suite remains at 646 passing + ~5 new stub tests = ~651
- [x] One commit on `main` with message `feat: refactor cb2 - promote MeasureMASE to contracts + add StubMeasureMASE adapter`
- [x] Commit is pushed

## Blocked by

None — can start immediately. Independent of Issue 001.
