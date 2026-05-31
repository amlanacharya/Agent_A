# Run State is a single Pydantic model persisted to disk as JSON

Each Run has one authoritative state file at `outputs/{run_id}/run_state.json`. The backend loads it at the start of every request and writes it after every mutation. Conductor reads and writes Run State exclusively through `get_run_state` / `update_run_state` tool calls.

Alternatives considered:

- **In-memory module-level dict** — lost on process restart, untestable in isolation, hidden coupling between requests. Rejected.
- **LangGraph** — solves state transitions natively but requires rearchitecting the Conductor/Lens design around LangGraph graph primitives. Too invasive for a POC where the agent design is already settled. Revisit in Phase 2 if state complexity grows.
- **Redis / database** — correct for multi-user production; overkill for a solo-analyst POC with no concurrency requirement.

File-backed state is simple, debuggable (you can inspect it between requests), and consistent with the existing sentinel pattern where all run artifacts live under `outputs/{run_id}/`.