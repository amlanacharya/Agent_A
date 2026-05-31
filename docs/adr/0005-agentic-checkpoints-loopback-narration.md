---
status: accepted
---

# Agents pause for human input, loop back, and narrate — the pipeline is not strictly linear

After pack confirmation the pipeline used to run silently to `report_ready` (only Meridian scoping was interactive). To make the system feel like an agent team rather than a fixed chain, we let any domain agent emit a structured **Need** (`raise_need(kind, question, options, context)`) that raises a resumable `PauseForInput` and suspends the run. The Conductor routes the need by `kind`: `USER_DECISION` → Conductor voices a quick choice and the resuming agent records the outcome; `SCOPE_AMENDMENT` → Meridian voices it, amends + re-locks the pack (`pack_version++`) and re-runs only the affected series. Agents also **narrate** their inter-tool reasoning over a new `agent_reasoning` SSE event. The Run lifecycle is therefore no longer strictly linear — `phase` can move backward via a bounded loop-back.

**Why this shape, not the obvious one:** resume is **idempotent re-invocation** (the agent re-runs from the top and skips steps whose artifacts already exist), not a suspended call stack — so a pause survives a backend restart and stays consistent with file-backed Run State (ADR-0003). A pause is **not** a Halt: `PauseForInput` is resumable and never resets budgets, whereas `GuardHalt` is terminal.

**Considered and rejected:**
- *Keep the linear pipeline, log everything for review after the fact.* Rejected — the human can't steer mid-run, which is the whole point.
- *Suspend the agent's live call stack and wake it on the answer (threads/async).* Rejected — breaks the single-process, file-backed model and can't survive a restart.
- *Full mid-tool-call interruption (lever 4).* Deferred — needs true concurrency for marginal POC benefit; users interject at turn boundaries / checkpoints instead.

**Consequences:**
- Cumulative budgets (`tokens_used_total`, `foundry_calls_total`, `loopback_count`) must persist in Run State and seed/write-back per invocation, or idempotent re-invocation would silently reset the Guard. This refines the per-run Foundry counter (review §8) into a persisted one.
- Loop-back is bounded by `loopback_count` (default 3, `.env`); exceeding it raises `GuardHalt`.
- Loop-back reuses Prism's affected-series re-run engine, parameterised by namespace (main run vs `whatif` clone). Pre-report changes mutate the run in place; post-report changes go through a cloned Prism what-if.
- Checkpoints are deliberately rare: agents take a documented default and log a Claim wherever one exists, raising a Need only when the choice materially changes the deliverable and no defensible default exists.
