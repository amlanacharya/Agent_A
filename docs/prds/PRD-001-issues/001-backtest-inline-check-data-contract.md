# Issue 001 — Inline `check_data_contract` into `backtest.py`, fix dependency direction

**Type:** AFK
**Source:** PRD-001 §Solution CB1
**Blocked by:** None
**Labels:** `refactor`, `post-phase-6`

## What to build

`backtest.py` currently imports `check_data_contract` from `model_escalation.py`. The dependency direction is wrong: a pure evaluation operation (backtest fold → score) should not depend on the gate layer above it, which transitively pulls in `code_escalation.py` (file-backed `EscalationTracker` state machine). The import path is `backtest → model_escalation → code_escalation → run_state`.

This issue moves the 8-line `check_data_contract` function into `backtest.py` as a private helper (e.g. `_check_data_contract`), keeps the existing `RobustnessCheck` return type from `contracts.py`, and deletes the cross-module import. After the change, `backtest.py`'s dependencies are `forecasting_models` (its data) + `contracts` (its types) — no escalation layer.

The function's behaviour, signature, and return shape are unchanged. Existing `test_backtest.py` exercises the check end-to-end and must continue to pass without modification.

## Acceptance criteria

- [x] `backtest.py` no longer imports from `model_escalation` (verify with `grep -n "from forecasting.model_escalation" backend/forecasting/backtest.py` returns nothing)
- [x] `check_data_contract` is implemented as a private helper inside `backtest.py` with the same signature `(forecast: list[float], actual: list[float] | None, horizon: int) -> RobustnessCheck` and identical behaviour
- [x] `model_escalation.py` no longer exports `check_data_contract` (or, if any other module uses it, the export is preserved there with the wrong-direction import removed from `backtest.py` only)
- [x] `uv run pytest -q` is green; full suite remains at 646 passing
- [x] One commit on `main` with message `feat: refactor cb1 - inline check_data_contract into backtest, fix dependency direction`
- [x] Commit is pushed

## Blocked by

None — can start immediately.
