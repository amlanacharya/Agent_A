# Agent P — Full Implementation Spec
**Date:** 2026-05-30
**Status:** Complete — awaiting user review
**Source:** plan_v2.md + CONTEXT.md + ADRs 0001–0004
**Depth:** Behavioural contracts + implementation-ready detail (Pydantic models, function signatures, enums)

Each layer is specced in dependency order. A layer may only depend on layers above it in this document.

---

## Layer Index

1. [Data & State](#1-data--state-layer)
2. [Pre-flight](#2-pre-flight-layer)
3. [Guard Layer](#3-guard-layer)
4. [Lens](#4-lens)
5. [Conductor](#5-conductor)
6. [Meridian](#6-meridian)
7. [Forge](#7-forge)
8. [Foundry](#8-foundry)
9. [Prism](#9-prism)
10. [Backend API](#10-backend-api)
11. [Frontend](#11-frontend)

---

## 1. Data & State Layer

### Purpose & Invariants

The data and state layer is the shared persistence foundation for every Run. It has two responsibilities that must never be conflated:

- **Series data** (`data_store`) — immutable per-request read surface for raw time-series values. Written once on upload; read by tools via `run_id` + `series_key`. Never travels through Claude's context.
- **Run control state** (`RunState`) — mutable, persisted to disk after every write, loaded fresh at the start of every HTTP request. The single source of truth for where a Run is and what it has decided.

**Key invariant:** No agent or tool holds mutable state in memory across requests. Everything that must survive an HTTP boundary lives in `run_state.json` or `data_store` (keyed by `run_id`).

---

### Series Key

Pipe-delimited concatenation of grain dimension values in the order defined by the domain playbook's `common_grains` field. The time dimension is always excluded.

**Normalisation (applied once at pre-flight, immutable thereafter):**

```
1. Split grain dimensions per playbook order (exclude time column)
2. Uppercase all values
3. Replace spaces with underscores
4. Strip any character outside [A-Z0-9_|]
5. Join with |
```

Example — grain `sku_region_week`, time col `week`:
`"sku 101", "West"` → `SKU_101|WEST`

Series keys are used as:
- Dict keys in `data_store[run_id]`
- File path components: `outputs/{run_id}/series_results/{series_key}.json`
- Identifiers in all tool inputs/outputs that reference individual series

---

### `data_store` — Behavioural Specification

`data_store` is a module-level dict in `forecasting/data_store.py`. It is the only in-memory structure permitted to hold series data.

**Pre/post conditions:**

| Operation | Precondition | Postcondition |
|---|---|---|
| `store_series(run_id, series_key, df)` | `run_id` exists; `df` has columns `[date, demand]` at minimum | `data_store[run_id][series_key]` holds the DataFrame; pre-existing key is overwritten |
| `get_series(run_id, series_key)` | `run_id` and `series_key` exist | Returns DataFrame; raises `SeriesNotFoundError` if either key missing |
| `get_series_keys(run_id)` | `run_id` exists | Returns `list[str]` of all series keys for the run |
| `delete_run(run_id)` | — | Removes `data_store[run_id]` entirely; no-op if run_id absent |

**Error cases:**
- `SeriesNotFoundError(run_id, series_key)` — raised by `get_series` when key missing. Callers must not catch silently; propagates to Guard Layer as a tool error.
- `data_store` is never serialised to disk. Process restart loses all data — acceptable for POC (single-process, single-session assumption).

**Implementation:**

```python
# forecasting/data_store.py

import pandas as pd
from typing import Dict

_store: Dict[str, Dict[str, pd.DataFrame]] = {}

class SeriesNotFoundError(Exception):
    def __init__(self, run_id: str, series_key: str):
        super().__init__(f"Series '{series_key}' not found for run '{run_id}'")
        self.run_id = run_id
        self.series_key = series_key

def store_series(run_id: str, series_key: str, df: pd.DataFrame) -> None:
    _store.setdefault(run_id, {})[series_key] = df

def get_series(run_id: str, series_key: str) -> pd.DataFrame:
    try:
        return _store[run_id][series_key]
    except KeyError:
        raise SeriesNotFoundError(run_id, series_key)

def get_series_keys(run_id: str) -> list[str]:
    return list(_store.get(run_id, {}).keys())

def delete_run(run_id: str) -> None:
    _store.pop(run_id, None)
```

---

### `RunState` — Behavioural Specification

A Pydantic v2 model. One instance per Run. Persisted to `outputs/{run_id}/run_state.json` after every mutation. Loaded from disk at the start of every HTTP request that touches a run.

**Fields:**

```python
class Phase(str, Enum):
    PREFLIGHT           = "preflight"
    MERIDIAN_SCOPING    = "meridian_scoping"
    FORGE_EDA           = "forge_eda"
    FOUNDRY_MODELLING   = "foundry_modelling"
    REPORT_READY        = "report_ready"
    HALTED              = "halted"

class RunState(BaseModel):
    run_id:               str
    phase:                Phase
    pack_confirmed:       bool          = False
    meridian_turn_count:  int           = 0
    forge_complete:       bool          = False
    foundry_complete:     bool          = False
    active_whatif_runs:   list[str]     = Field(default_factory=list)
    open_risks:           int           = 0
    override_count:       int           = 0
    halt_reason:          str | None    = None
    domain:               str
    created_at:           str           # ISO 8601 timestamp, set at creation

    model_config = ConfigDict(use_enum_values=True)
```

**Invariants:**
- `phase = HALTED` is terminal — no subsequent mutation is permitted. Any tool that calls `save_run_state` on a halted run raises `HaltedRunError`.
- `open_risks` is a count of *unacknowledged* risks only. It is incremented by `add_risk` and decremented when a risk becomes an `ACCEPTED_RISK` Claim or is resolved by Meridian evidence.
- `halt_reason` must be set before `phase` is set to `HALTED`.
- `pack_confirmed` may only transition `False → True`, never back.

**Load / save helpers:**

```python
# forecasting/run_state.py

import json
from pathlib import Path
from datetime import datetime, timezone

OUTPUTS_ROOT = Path("backend/outputs")

def run_dir(run_id: str) -> Path:
    return OUTPUTS_ROOT / run_id

def state_path(run_id: str) -> Path:
    return run_dir(run_id) / "run_state.json"

def create_run_state(run_id: str, domain: str) -> RunState:
    state = RunState(
        run_id=run_id,
        phase=Phase.PREFLIGHT,
        domain=domain,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    run_dir(run_id).mkdir(parents=True, exist_ok=True)
    save_run_state(state)
    return state

def load_run_state(run_id: str) -> RunState:
    path = state_path(run_id)
    if not path.exists():
        raise RunNotFoundError(run_id)
    return RunState.model_validate_json(path.read_text())

def save_run_state(state: RunState) -> None:
    if state.phase == Phase.HALTED and state.halt_reason is None:
        raise ValueError("halt_reason must be set before saving a HALTED RunState")
    path = state_path(state.run_id)
    path.write_text(state.model_dump_json(indent=2))

class RunNotFoundError(Exception):
    def __init__(self, run_id: str):
        super().__init__(f"Run '{run_id}' not found")
        self.run_id = run_id

class HaltedRunError(Exception):
    def __init__(self, run_id: str):
        super().__init__(f"Run '{run_id}' is halted and cannot be mutated")
```

---

### `outputs/{run_id}/` Layout

```
outputs/{run_id}/
├── run_state.json           ← RunState — written after every mutation
├── preflight.json           ← PreflightBundle — written by pre-flight layer
├── domain_context_pack.json ← written by compile_domain_context_pack; locked on confirmation
├── eda_report.json          ← written by compile_eda_report (Forge)
├── series_results/
│   └── {series_key}.json    ← one file per series, written by record_series_result (Foundry)
├── foundry_report.json      ← written by compile_foundry_report (Foundry)
├── claim_ledger.json        ← append-only; written by add_claim / resolve_claim (Meridian)
├── risk_register.json       ← append-only; written by add_risk / acknowledge_risk (Meridian)
├── obs_log.json             ← append-only structured tool call log (observability)
├── run_summary.json         ← written at run completion or halt
└── whatif/
    └── {whatif_id}/
        ├── modified_pack.json  ← cloned+modified domain_context_pack for this scenario
        └── comparison.json     ← baseline vs scenario per series
```

**File write rules:**
- `run_state.json` — written by `save_run_state` only. No other code touches it.
- `obs_log.json` — append-only. Written by the observability layer after every tool call. Never truncated.
- `domain_context_pack.json` — written once by `compile_domain_context_pack`. After `pack_confirmed = True`, treated as read-only; no tool may overwrite it. Prism clones it into `whatif/{whatif_id}/modified_pack.json`.
- All other files — written once by their respective compile/record tools. Not overwritten.

---

## 2. Pre-flight Layer

### Purpose & Invariants

The pre-flight layer is deterministic Python — no LLM, no agent. It runs once on CSV upload, before any agent starts. Its job is to profile the data, validate it, map the schema, and produce the `PreflightBundle` handed to Meridian via Conductor.

**Key invariants:**
- Pre-flight never modifies `data_store`. It reads the uploaded file, normalises and loads series into `data_store`, then writes `preflight.json`.
- Any Blocking Issue immediately halts the Run — Conductor is notified; no agent is invoked.
- Warnings are non-blocking — passed into the bundle; Meridian surfaces them in conversation.
- Series key normalisation is finalised here. Keys are immutable for the Run lifetime.
- Pre-flight writes `preflight.json` to `outputs/{run_id}/` and returns the bundle to the caller. The file is the durable record; the returned dict is used in-flight by the API handler.

---

### Tool Inventory

Each function is a pure Python callable, not an LLM tool. Called sequentially by the pre-flight orchestrator in `forecasting/preflight.py`.

```
profile_uploaded_data(df)              → DataQualityReport
map_schema(df, playbook)               → SchemaMapping
detect_frequency_and_grain(df, schema) → GrainReport
build_series_keys(df, schema, playbook)→ dict[series_key → pd.DataFrame]
compute_adi_cv2_per_series(series_map) → dict[series_key → AdiCv2Stats]
detect_zero_runs_per_series(series_map)→ dict[series_key → ZeroRunStats]
detect_spikes_per_series(series_map)   → dict[series_key → SpikeStats]
measure_promo_alignment(series_map, schema) → dict[series_key → PromoAlignmentStats]
detect_trend_strength(series_map)      → dict[series_key → TrendStats]
detect_seasonality_strength(series_map)→ dict[series_key → SeasonalityStats]
detect_structural_break_candidates(series_map) → list[BreakCandidate]
```

Then a provisional segmentation step followed by two aggregation steps:
```
assign_provisional_segments(series_map, schema, playbook) → SegmentMap
    # playbook `segment_by` grain hint → grouping; else single segment G1.
    # The map is provisional (a suggestion); the user refines it in Meridian and it
    # is frozen into the pack at confirmation.
aggregate_segment_profiles(series_map, adi_cv2, segment_map) → list[SegmentProfile]
collect_segment_exceptions(adi_cv2, zero_runs, spikes, segment_map) → list[SeriesException]
```

---

### Data Contracts

**`DataQualityReport`:**
```python
class BlockingIssue(BaseModel):
    code:    str   # e.g. "MISSING_DATE_COLUMN", "ALL_ZERO_DEMAND", "BELOW_MIN_SERIES"
    message: str

class Warning(BaseModel):
    code:    str   # e.g. "SPARSE_DATA", "SHORT_HISTORY", "SCHEMA_AMBIGUITY"
    message: str
    affected_series: list[str] = Field(default_factory=list)

class DataQualityReport(BaseModel):
    blocking_issues: list[BlockingIssue]
    warnings:        list[Warning]
    row_count:       int
    series_count:    int
```

**Blocking issue codes (halt on any):**

| Code | Condition |
|---|---|
| `MISSING_DATE_COLUMN` | No column parseable as a date |
| `MISSING_DEMAND_COLUMN` | No column mappable to demand |
| `ALL_ZERO_DEMAND` | Every demand value is zero across all series |
| `BELOW_MIN_SERIES` | Series count < playbook `min_series` |
| `UNPARSEABLE_FILE` | CSV cannot be read (encoding, delimiter, corruption) |

**Warning codes (non-blocking):**

| Code | Condition |
|---|---|
| `SPARSE_DATA` | >40% of demand values are zero for a series |
| `SHORT_HISTORY` | A series has fewer than `min_history_periods` rows (playbook default) |
| `SCHEMA_AMBIGUITY` | Multiple columns match a required schema field; first match used |
| `DUPLICATE_SERIES_KEYS` | Two rows produce the same series key after normalisation |

---

**`SchemaMapping`:**
```python
class SchemaMapping(BaseModel):
    date_col:     str
    demand_col:   str
    grain_cols:   list[str]   # ordered per playbook common_grains, time col excluded
    extra_cols:   list[str]   # present in file but not in playbook — passed through
```

**`GrainReport`:**
```python
class GrainReport(BaseModel):
    detected_frequency: Literal["daily", "weekly", "monthly", "unknown"]
    min_periods:        int
    max_periods:        int
    median_periods:     int
    gaps_detected:      bool   # any series has non-contiguous dates
```

**`AdiCv2Stats`:**
```python
class AdiCv2Stats(BaseModel):
    series_key: str
    adi:        float   # Average Demand Interval
    cv2:        float   # Coefficient of Variation squared
    # Syntetos-Boylan classification (informational at pre-flight; Forge makes the official call)
    sb_class:   Literal["SMOOTH", "ERRATIC", "INTERMITTENT", "LUMPY"]
```

ADI = mean(inter-demand intervals); CV² = (std(demand) / mean(demand))² computed on non-zero periods only.
Syntetos-Boylan thresholds: ADI < 1.32 and CV² < 0.49 → SMOOTH; ADI ≥ 1.32 and CV² < 0.49 → INTERMITTENT; ADI < 1.32 and CV² ≥ 0.49 → ERRATIC; ADI ≥ 1.32 and CV² ≥ 0.49 → LUMPY.

**`ZeroRunStats`:**
```python
class ZeroRunStats(BaseModel):
    series_key:       str
    zero_pct:         float   # fraction of periods with demand == 0
    max_zero_run:     int     # longest consecutive zero-demand run in periods
    zero_run_count:   int     # number of distinct zero runs
```

**`SpikeStats`:**
```python
class SpikeStats(BaseModel):
    series_key:    str
    spike_count:   int
    spike_dates:   list[str]   # ISO date strings of detected spike periods
    spike_method:  Literal["iqr_3x"]   # IQR × 3 above Q3
```

**`TrendStats`:**
```python
class TrendStats(BaseModel):
    series_key:      str
    trend_strength:  float   # R² of linear fit on non-zero periods, 0–1
    direction:       Literal["up", "down", "flat"]
```

**`SeasonalityStats`:**
```python
class SeasonalityStats(BaseModel):
    series_key:           str
    seasonality_strength: float      # 0–1; 0 = no seasonality
    dominant_period:      int | None # periods (e.g. 52 for annual in weekly data)
```

**`BreakCandidate`:**
```python
class BreakCandidate(BaseModel):
    date:        str        # ISO date string
    series_keys: list[str]  # series where signal detected
    method:      Literal["cusum", "pettitt"]
    p_value:     float | None
```

**`SegmentProfile`:**
```python
class SegmentProfile(BaseModel):
    segment_id:                str
    series_count:              int
    demand_class_distribution: dict[str, int]   # {"SMOOTH": 4, "ERRATIC": 1, ...}
    median_adi:                float
    median_cv2:                float
    forecastability_breakdown: dict[str, int]   # preliminary counts, not Forge's official call
    example_keys:              list[str] = []
```

**`SegmentDef` / `SegmentMap`** — the provisional segment map (suggestion refined in Meridian, frozen at pack confirmation):
```python
class SegmentDef(BaseModel):
    segment_id:  str                 # "G1", "G2", ...
    label:       str                 # e.g. "region=NORTH" or "all series"
    series_keys: list[str]
    provisional: bool = True         # True until pack confirmation

class SegmentMap(BaseModel):
    run_id:      str
    segments:    list[SegmentDef]
    provisional: bool = True
    derived_by:  str                 # "playbook:segment_by=region" | "default:single_segment"
```

**`SeriesException`:**
```python
class SeriesException(BaseModel):
    series_key:     str
    segment_id:     str
    exception_type: Literal["HIGH_ZERO_FRACTION", "SPIKE", "ZERO_RUN"]
    detail:         str   # human-readable; e.g. "zero_fraction=0.86"
```

---

### `PreflightBundle` — Full Contract

```python
class PreflightBundle(BaseModel):
    data_quality_report: DataQualityReport
    schema_mapping:      SchemaMapping
    grain_report:        GrainReport
    segment_profiles:    list[SegmentProfile]   # aggregate per segment — NOT per series
    segment_exceptions:  list[SeriesException]
    segments:            list[SegmentDef]        # provisional segment map
    domain_playbook:     dict   # raw YAML playbook dict
```

This is serialised to `outputs/{run_id}/preflight.json` and handed to Conductor, which injects it into Meridian's system prompt.

---

### Pre-flight Orchestrator

```python
# forecasting/preflight.py

def run_preflight(
    run_id: str,
    file_bytes: bytes,
    domain: str,
    playbook: dict,
) -> PreflightBundle:
    """
    Entry point called by the upload API handler.
    Returns PreflightBundle and writes preflight.json.
    Raises PreflightBlockingError if any blocking issue found.
    """
    df = _parse_csv(file_bytes)          # raises PreflightBlockingError on UNPARSEABLE_FILE

    dq_report = profile_uploaded_data(df)
    if dq_report.blocking_issues:
        raise PreflightBlockingError(dq_report.blocking_issues)

    schema     = map_schema(df, playbook)
    grain      = detect_frequency_and_grain(df, schema)
    series_map = build_series_keys(df, schema, playbook)

    # Load normalised series into data_store
    for key, series_df in series_map.items():
        store_series(run_id, key, series_df)

    adi_cv2     = compute_adi_cv2_per_series(series_map)
    zero_runs   = detect_zero_runs_per_series(series_map)
    spikes      = detect_spikes_per_series(series_map)
    promo_align = measure_promo_alignment(series_map, schema)
    trend       = detect_trend_strength(series_map)
    seasonality = detect_seasonality_strength(series_map)
    break_cands = detect_structural_break_candidates(series_map)

    seg_profiles   = aggregate_segment_profiles(series_map, adi_cv2, playbook)
    seg_exceptions = collect_segment_exceptions(adi_cv2, zero_runs, spikes)

    _append_stat_warnings(dq_report, adi_cv2, grain, playbook)

    bundle = PreflightBundle(
        data_quality_report=dq_report,
        schema_mapping=schema,
        grain_report=grain,
        segment_profiles=seg_profiles,
        segment_exceptions=seg_exceptions,
        segments=list({e.segment_id for e in seg_exceptions} |
                      {p.segment_id for p in seg_profiles}),
        domain_playbook=playbook,
    )

    path = run_dir(run_id) / "preflight.json"
    import json as _json
    path.write_text(_json.dumps({
        "bundle":           bundle.model_dump(),
        "break_candidates": [b.model_dump() for b in break_cands],
    }, indent=2))
    return bundle


class PreflightBlockingError(Exception):
    def __init__(self, issues: list[BlockingIssue]):
        super().__init__(f"Preflight blocked: {[i.code for i in issues]}")
        self.issues = issues
```

**Behavioural notes:**
- `build_series_keys` deduplicates: if two rows produce the same normalised key, it logs a `DUPLICATE_SERIES_KEYS` warning and keeps the later row (last-write-wins). This is consistent, deterministic, and auditable.
- Structural break candidates from `detect_structural_break_candidates` are **not** in the `PreflightBundle` model — they are stored separately in `preflight.json` under a top-level `break_candidates` key written alongside the bundle JSON. `preflight.json` therefore has two top-level keys: `bundle` (the `PreflightBundle` model) and `break_candidates` (list of `BreakCandidate`). Meridian receives break candidates only via `diagnose_structural_break_candidates` tool calls, keeping the injected bundle lean.
- `promo_align`, `trend`, and `seasonality` stats are stored in `preflight.json` for Meridian's diagnostic tools to access. They are not in the bundle struct — Meridian pulls them via tool calls.

---

## 3. Guard Layer

### Purpose & Invariants

The Guard Layer is shared enforcement applied to every LLM tool call across all agents. It is not an agent — purely deterministic Python in `forecasting/guard.py`. It sits between the tool dispatcher and the actual tool implementation.

**Key invariants:**
- Every tool call passes through the Guard Layer before execution. No exceptions.
- All limits are `.env`-configurable. Values below are defaults.
- A `GuardHalt` is terminal. Once raised, the Run's `phase` is set to `HALTED` and `halt_reason` is written. No further tool calls are dispatched for that `run_id`.
- The Guard Layer does not retry or recover — it raises and stops. Halt is terminal for the POC; the user adjusts `.env` limits and starts a new Run.
- Foundry is exempt from the per-agent tool call limit. It has its own cumulative counter (`max_tool_calls_foundry`).

---

### Configuration (`.env` defaults)

```
TOKEN_BUDGET_PER_RUN=80000          # total input+output tokens across all agents for a run
MAX_TOOL_CALLS_CONDUCTOR=20         # per Conductor invocation
MAX_TOOL_CALLS_MERIDIAN=20          # per Meridian invocation
MAX_TOOL_CALLS_FORGE=20             # per Forge invocation
MAX_TOOL_CALLS_PRISM=20             # per Prism invocation
MAX_TOOL_CALLS_FOUNDRY=500          # cumulative across all Foundry invocations for a run
DUPLICATE_CALL_HARD_STOP=2          # identical calls before halt (1 = halt on first duplicate)
```

---

### Duplicate Call Detection

Identity is determined by MD5 hash of `(tool_name, json.dumps(args, sort_keys=True))`. Hashes are stored in a per-run, per-agent-invocation set. A duplicate is any call whose hash appears in that set.

- On the **first** duplicate: warning logged, call is allowed. Counter incremented.
- On the **Nth** duplicate (N = `DUPLICATE_CALL_HARD_STOP`, default 2): `GuardHalt` raised with reason `"duplicate_tool_call:{tool_name}"`.

The duplicate detection set resets per agent invocation, not per run — this is intentional. A looping agent within a single invocation is the failure mode being detected, not the same tool being called across different pipeline phases.

---

### Data Contracts

```python
# forecasting/guard.py

import hashlib, json, os
from collections import defaultdict
from typing import Any

class GuardHalt(Exception):
    def __init__(self, run_id: str, reason: str):
        super().__init__(f"GuardHalt [{run_id}]: {reason}")
        self.run_id = run_id
        self.reason = reason

class GuardConfig:
    token_budget:        int = int(os.getenv("TOKEN_BUDGET_PER_RUN", 80_000))
    max_calls_conductor: int = int(os.getenv("MAX_TOOL_CALLS_CONDUCTOR", 20))
    max_calls_meridian:  int = int(os.getenv("MAX_TOOL_CALLS_MERIDIAN", 20))
    max_calls_forge:     int = int(os.getenv("MAX_TOOL_CALLS_FORGE", 20))
    max_calls_prism:     int = int(os.getenv("MAX_TOOL_CALLS_PRISM", 20))
    max_calls_foundry:   int = int(os.getenv("MAX_TOOL_CALLS_FOUNDRY", 500))
    duplicate_hard_stop: int = int(os.getenv("DUPLICATE_CALL_HARD_STOP", 2))

class AgentGuardState:
    """Per-agent-invocation state. Instantiated fresh for each agent call."""
    def __init__(self, agent: str, run_id: str):
        self.agent  = agent
        self.run_id = run_id
        self.call_count:       int           = 0
        self.duplicate_counts: dict[str, int] = defaultdict(int)
        self._seen_hashes:     set[str]       = set()

    def _hash(self, tool_name: str, args: dict) -> str:
        raw = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def check_and_record(
        self,
        tool_name: str,
        args: dict,
        tokens_used_so_far: int,
        config: GuardConfig = GuardConfig(),
    ) -> None:
        """Call before dispatching any tool. Raises GuardHalt if any limit exceeded."""
        # 1. Token budget
        if tokens_used_so_far >= config.token_budget:
            raise GuardHalt(self.run_id, f"token_budget_exceeded:{tokens_used_so_far}")

        # 2. Per-agent call limit (Foundry exempt — checked by FoundryRunGuard)
        limit = self._agent_limit(config)
        if limit is not None:
            self.call_count += 1
            if self.call_count > limit:
                raise GuardHalt(
                    self.run_id,
                    f"max_tool_calls_exceeded:{self.agent}:{self.call_count}",
                )

        # 3. Duplicate detection
        h = self._hash(tool_name, args)
        if h in self._seen_hashes:
            self.duplicate_counts[h] += 1
            if self.duplicate_counts[h] >= config.duplicate_hard_stop:
                raise GuardHalt(self.run_id, f"duplicate_tool_call:{tool_name}")
        else:
            self._seen_hashes.add(h)

    def _agent_limit(self, config: GuardConfig) -> int | None:
        return {
            "conductor": config.max_calls_conductor,
            "meridian":  config.max_calls_meridian,
            "forge":     config.max_calls_forge,
            "prism":     config.max_calls_prism,
            "foundry":   None,
        }.get(self.agent)


class FoundryRunGuard:
    """Cumulative Foundry call counter for a run. Shared across all Foundry invocations."""
    def __init__(self, run_id: str):
        self.run_id     = run_id
        self.call_count = 0

    def check_and_record(self, config: GuardConfig = GuardConfig()) -> None:
        self.call_count += 1
        if self.call_count > config.max_calls_foundry:
            raise GuardHalt(
                self.run_id,
                f"foundry_max_tool_calls_exceeded:{self.call_count}",
            )
```

---

### Tool Dispatcher Integration

```python
# forecasting/providers.py

import logging
log = logging.getLogger(__name__)

def dispatch_tool(
    tool_name: str,
    args: dict,
    guard: AgentGuardState,
    tokens_used: int,
    foundry_guard: FoundryRunGuard | None = None,
) -> Any:
    """Central dispatch. All agent tool calls route through here."""
    guard.check_and_record(tool_name, args, tokens_used)
    if foundry_guard is not None:
        foundry_guard.check_and_record()

    log.info(f"[{guard.run_id}] [{guard.agent}] TOOL_CALL {tool_name} args={args}")
    result = _tool_registry[tool_name](**args)
    log.info(f"[{guard.run_id}] [{guard.agent}] TOOL_DONE {tool_name}")
    return result
```

---

### Halt Propagation

When `GuardHalt` is raised:

1. The agent's tool loop catches it and calls `log_halt(run_id, reason)` (Conductor tool).
2. `log_halt` sets `RunState.halt_reason = reason`, sets `RunState.phase = Phase.HALTED`, persists `run_state.json`, appends to `obs_log.json`, and pushes `error` SSE event: `{ "reason": "<human message>", "halt_reason": reason }`.
3. The SSE stream closes.
4. All subsequent API requests for this `run_id` check `phase == HALTED` at load time and return `410 Gone` with the halt reason.

**Halt reason string format:** `"{category}:{detail}"` — e.g. `"token_budget_exceeded:81203"`, `"duplicate_tool_call:diagnose_spike_policy"`, `"max_tool_calls_exceeded:meridian:21"`.

---

## 4. Lens — Intent Agent

### Purpose & Invariants

Lens is a **structured-output call with no tool use**. It receives the last 6 conversation turns, the current user message, and `pipeline_state`, and returns a typed `IntentPack`. One call, one response — no tool loop.

**Key invariants:**
- Lens runs on every user message without exception. Conductor never parses raw user language.
- Lens makes no writes — it is purely a classification function.
- The model is Claude Haiku (fast, cheap). Speed matters here — it sits on the critical path of every user turn.
- `pipeline_state` and the last agent message are the strongest priors for short/ambiguous messages. "ok" after a risk warning → `SCOPE_RESPONSE`. "ok" after a proceed question → `ADVANCE_PIPELINE`. See ADR-0002.
- Confidence < 0.6 → Conductor handles disambiguation via `surface_clarification`. Lens never re-prompts the user itself.

---

### Input Contract

```python
class LensInput(BaseModel):
    conversation_history: list[ConversationTurn]  # last 6 turns max
    user_message:         str
    pipeline_state:       RunState

class ConversationTurn(BaseModel):
    role:    Literal["user", "assistant"]
    content: str
    agent:   str | None = None   # which agent produced this turn, if assistant
```

---

### Output Contract — `IntentPack`

```python
class IntentType(str, Enum):
    SCOPE_RESPONSE   = "SCOPE_RESPONSE"    # answering Meridian's question
    OVERRIDE         = "OVERRIDE"          # contradicting an agent recommendation
    ADVANCE_PIPELINE = "ADVANCE_PIPELINE"  # "ok let's model", "looks good"
    WHAT_IF_REQUEST  = "WHAT_IF_REQUEST"   # "what if promo on SKU X week 10"
    CLARIFICATION    = "CLARIFICATION"     # user asking a question
    CORRECTION       = "CORRECTION"        # fixing a prior statement
    # CORRECTION is only valid during meridian_scoping.
    # Post-pack-confirmation, any scoping change is treated as OVERRIDE;
    # Conductor offers a Scenario Run instead.

class IntentEntities(BaseModel):
    skus:     list[str] = Field(default_factory=list)
    segments: list[str] = Field(default_factory=list)
    dates:    list[str] = Field(default_factory=list)   # ISO strings
    metrics:  list[str] = Field(default_factory=list)
    scenario: str | None = None   # free-text scenario description if WHAT_IF_REQUEST

class IntentPack(BaseModel):
    intent:     IntentType
    entities:   IntentEntities
    confidence: float   # 0.0–1.0
    raw_quote:  str     # verbatim fragment of user message that drove classification
```

---

### Confidence Threshold Behaviour

| Confidence | Conductor action |
|---|---|
| ≥ 0.6 | Route normally per `intent` |
| < 0.6 | Call `surface_clarification` — offer user two options derived from closest candidate intents |

Lens does not return candidate intents explicitly. Conductor derives the two clarification options from `intent` (top pick) and `pipeline_state` context (what the likely alternative is given current phase). This keeps the Lens output minimal.

---

### Implementation

```python
# forecasting/agents/lens.py

import anthropic
from forecasting.contracts import LensInput, IntentPack

client = anthropic.Anthropic()

SYSTEM_PROMPT = """
You are Lens, an intent classifier for a demand forecasting assistant.

Your job: read the conversation and classify the user's latest message into one
intent type. Return a single JSON object matching the IntentPack schema. Nothing else.

Intent types:
- SCOPE_RESPONSE    — user is answering a question Meridian asked
- OVERRIDE          — user is contradicting a data-backed agent recommendation
- ADVANCE_PIPELINE  — user approves moving to the next pipeline phase
- WHAT_IF_REQUEST   — user is requesting a scenario / what-if analysis
- CLARIFICATION     — user is asking a question
- CORRECTION        — user is correcting a prior statement (only valid in meridian_scoping;
                      post-confirmation treat as OVERRIDE)

Weighting rules:
1. pipeline_state.phase and the last assistant message are the strongest signal for
   short ambiguous messages ("ok", "yes", "fine", "sure").
2. A short message after a risk warning → SCOPE_RESPONSE.
3. A short message after a "shall we proceed?" question → ADVANCE_PIPELINE.
4. Only classify WHAT_IF_REQUEST if the user explicitly describes a scenario change.
5. Set confidence honestly. If genuinely unsure between two intents, set confidence < 0.6.
6. raw_quote must be a verbatim excerpt (≤ 20 words) from the user message.
""".strip()

def classify_intent(inp: LensInput) -> IntentPack:
    messages = [
        {"role": t.role, "content": t.content}
        for t in inp.conversation_history
    ]
    messages.append({"role": "user", "content": inp.user_message})

    system = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Current pipeline_state:\n"
        f"  phase={inp.pipeline_state.phase}\n"
        f"  pack_confirmed={inp.pipeline_state.pack_confirmed}\n"
        f"  open_risks={inp.pipeline_state.open_risks}\n"
        f"  override_count={inp.pipeline_state.override_count}"
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system,
        messages=messages,
        temperature=0.0,
    )

    raw = response.content[0].text.strip()
    return IntentPack.model_validate_json(raw)
```

**Behavioural notes:**
- `temperature=0.0` — classification must be deterministic.
- `max_tokens=512` — the output is a small JSON object; 512 is generous headroom.
- No tool use, no streaming. Lens is synchronous and completes before Conductor runs.
- Token usage from Lens is counted toward the run's `TOKEN_BUDGET_PER_RUN`. The caller (Conductor) adds `response.usage.input_tokens + response.usage.output_tokens` to the running token total before calling `AgentGuardState.check_and_record`.

---

## 5. Conductor — Orchestration Agent

### Purpose & Invariants

Conductor is the control plane. It receives `IntentPack + RunState` and issues routing decisions. It does no data work and has no domain knowledge — pure orchestration logic.

**Key invariants:**
- Conductor never sees raw user language. It only consumes the `IntentPack` produced by Lens.
- Conductor output is never streamed to the user. Its decisions surface only as `phase_change` and `decision_update` SSE events.
- Conductor is Claude Sonnet with a tool loop. It calls exactly the tools needed for one routing decision, then stops.
- All hard rules below are enforced in the system prompt. Violation of a hard rule triggers `log_halt`.

---

### Hard Rules (system prompt enforced)

| Rule | Condition |
|---|---|
| Never call `confirm_pack_and_advance` | if `open_risks > 0` and user has not acknowledged risks |
| Never call `confirm_pack_and_advance` | if `phase ≠ meridian_scoping` or `pack_confirmed = True` or `intent ≠ ADVANCE_PIPELINE` |
| Never call `trigger_foundry` | before `forge_complete = True` |
| Never call `create_prism_run` | before `phase = report_ready` |
| Must call `surface_clarification` | if `intent.confidence < 0.6` — never guess |
| Never attempt to resume | if `phase = HALTED` — call `log_halt` and stop |
| Never route `CORRECTION` to Meridian | if `pack_confirmed = True` — treat as `OVERRIDE`, offer Scenario Run |

---

### Tool Contracts

```python
def get_run_state(run_id: str) -> dict:
    """Load and return full RunState as dict. Called at start of every Conductor invocation."""

def update_run_state(run_id: str, patch: dict) -> dict:
    """
    Apply patch to RunState fields, persist, return updated state.
    Raises HaltedRunError if phase == HALTED.
    Raises ValueError if patch attempts pack_confirmed: False (one-way transition).
    """

def advance_to_meridian(run_id: str, user_message: str) -> None:
    """
    Route message + preflight_bundle to Meridian, open SSE stream.
    Loads PreflightBundle from outputs/{run_id}/preflight.json.
    Updates phase to meridian_scoping if currently preflight.
    """

def confirm_pack_and_advance(run_id: str) -> None:
    """
    Lock domain_context_pack, set pack_confirmed=True, phase=forge_eda.
    Triggers Forge asynchronously.
    Preconditions (all must hold — raises ConditionViolationError otherwise):
      - phase == meridian_scoping
      - pack_confirmed == False
      - open_risks == 0
    Emits phase_change SSE event: { phase: "forge_eda" }
    """

def trigger_foundry(run_id: str) -> None:
    """
    Invoke Foundry agent. Called after Forge completes.
    Precondition: forge_complete == True. Raises ConditionViolationError otherwise.
    Updates phase to foundry_modelling.
    Emits phase_change SSE event: { phase: "foundry_modelling" }
    """

def create_prism_run(
    run_id: str,
    scenario_description: str,
    entities: dict,
) -> dict:
    """
    Create a child Scenario Run. Returns { whatif_id: str }.
    Precondition: phase == report_ready. Raises ConditionViolationError otherwise.
    Creates outputs/{run_id}/whatif/{whatif_id}/ directory.
    Adds whatif_id to RunState.active_whatif_runs.
    """

def get_agent_status(run_id: str, agent: str) -> Literal["done", "running", "failed", "not_started"]:
    """Check current status of a named agent for this run."""

def surface_clarification(run_id: str, message: str) -> None:
    """
    Push a Conductor-authored clarification to the user via SSE message_done event.
    message must offer exactly two options derived from the closest candidate intents.
    Never exposes internal mechanics (e.g. confidence scores, agent names).
    """

def log_halt(run_id: str, reason: str) -> None:
    """
    Terminal. Sets RunState.halt_reason = reason, phase = HALTED, persists.
    Appends to obs_log.json. Pushes error SSE event.
    After this call, no further tool calls are valid for this run_id.
    """
```

**`ConditionViolationError`:**
```python
class ConditionViolationError(Exception):
    def __init__(self, tool: str, condition: str):
        super().__init__(f"Conductor tool '{tool}' precondition failed: {condition}")
```

---

### Routing Decision Table

| Phase | Intent | Conductor action |
|---|---|---|
| `preflight` | any | `advance_to_meridian` (first message after upload) |
| `meridian_scoping` | `SCOPE_RESPONSE`, `CLARIFICATION`, `CORRECTION` | `advance_to_meridian` (route to Meridian) |
| `meridian_scoping` | `OVERRIDE` | `advance_to_meridian` (Meridian handles override) |
| `meridian_scoping` | `ADVANCE_PIPELINE` + `open_risks == 0` | `confirm_pack_and_advance` |
| `meridian_scoping` | `ADVANCE_PIPELINE` + `open_risks > 0` | `surface_clarification` (remind user of open risks) |
| `forge_eda` | any | `surface_clarification` ("Forge is running, please wait") |
| `foundry_modelling` | any | `surface_clarification` ("Foundry is running, please wait") |
| `report_ready` | `WHAT_IF_REQUEST` | `create_prism_run` |
| `report_ready` | `CORRECTION` | treat as `OVERRIDE`, `surface_clarification` offering Scenario Run |
| `report_ready` | `CLARIFICATION` | `surface_clarification` (Conductor answers from RunState context) |
| any | confidence < 0.6 | `surface_clarification` (disambiguation) |
| `HALTED` | any | return immediately |

---

### Implementation Skeleton

```python
# forecasting/agents/conductor.py

import anthropic
from forecasting.contracts import IntentPack, RunState
from forecasting.guard import AgentGuardState
from forecasting.providers import dispatch_tool

client = anthropic.Anthropic()

SYSTEM_PROMPT = """
You are Conductor, the orchestration agent for a demand forecasting pipeline.

You receive an IntentPack (from Lens) and the current RunState. Your job is to
call the correct routing tool — nothing else. You do not write domain knowledge,
interpret data, or compose user-facing messages (except via surface_clarification).

Hard rules — violation of any triggers log_halt:
1. Never call confirm_pack_and_advance if open_risks > 0.
2. Never call confirm_pack_and_advance if phase ≠ meridian_scoping or pack_confirmed = True.
3. Never call trigger_foundry before forge_complete = True.
4. Never call create_prism_run before phase = report_ready.
5. Always call surface_clarification if intent.confidence < 0.6.
6. If phase = HALTED, do nothing — return immediately.
7. If intent = CORRECTION and pack_confirmed = True, treat as OVERRIDE and offer Scenario Run.
""".strip()

def run_conductor(
    run_id: str,
    intent_pack: IntentPack,
    run_state: RunState,
    tokens_used: int,
) -> int:
    """Execute one Conductor routing decision. Returns updated tokens_used count."""
    guard = AgentGuardState(agent="conductor", run_id=run_id)

    messages = [{
        "role": "user",
        "content": (
            f"IntentPack:\n{intent_pack.model_dump_json(indent=2)}\n\n"
            f"RunState:\n{run_state.model_dump_json(indent=2)}"
        ),
    }]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=CONDUCTOR_TOOLS,
            messages=messages,
        )
        tokens_used += response.usage.input_tokens + response.usage.output_tokens

        if response.stop_reason == "end_turn":
            break

        for block in response.content:
            if block.type == "tool_use":
                result = dispatch_tool(
                    tool_name=block.name,
                    args=block.input,
                    guard=guard,
                    tokens_used=tokens_used,
                )
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": block.id,
                                 "content": str(result)}],
                })
                break   # one tool call per turn

    return tokens_used
```

**Behavioural notes:**
- Conductor makes at most one tool call per routing decision. The `break` after each tool result enforces this — it exits the content block loop and loops back to the model, which should then `end_turn`.
- `CONDUCTOR_TOOLS` is the list of tool schemas registered for Conductor. Domain tools (Meridian, Forge, Foundry) are not in this list.
- If `dispatch_tool` raises `GuardHalt`, it propagates up through `run_conductor`. The calling API handler catches it, calls `log_halt`, and closes the SSE stream.

---

## 6. Meridian — Domain Specialist

### Purpose & Invariants

Meridian conducts the scoping conversation. It operates at segment level — never interrogating individual series one by one. It presents evidence first and asks the user to interpret. It uses humble language and never overclaims causality.

**Key invariants:**
- Meridian receives the `PreflightBundle` injected into its system prompt by Conductor. It does not call pre-flight tools — those already ran.
- Meridian builds the Claim Ledger incrementally across turns via tool calls. It does not compose the full pack at the end — `compile_domain_context_pack` assembles what has been built throughout.
- `CONTRADICTED` is a transient status. It must resolve to `SUPPORTED` or `USER_OVERRIDE_ACCEPTED` before `compile_domain_context_pack` can succeed.
- Meridian makes its case once on `CONTRADICTED`. If the user insists, it accepts as `USER_OVERRIDE_ACCEPTED` and moves on. It never refuses or loops.
- Structural break candidates must be explicitly confirmed by the user. Silence does not confirm.
- Max tool calls: 20 per invocation (`.env`-configurable via `MAX_TOOL_CALLS_MERIDIAN`).

---

### Conversation Arc (Phases 0–6)

| Phase | Meridian action |
|---|---|
| 0 — Welcome | Orient the user, summarise what was found in preflight (segment count, series count, data quality warnings), ask first scoping question |
| 1 — Forecast scope | Confirm target metric, grain, horizon, forecast start date |
| 2 — Segment review | Walk through each segment: zero-demand policy, spike policy, forecastability concerns |
| 3 — Feature & break policy | Confirm promo flag usage, structural break candidates (require explicit user acknowledgement), any additional features |
| 4 — Pack review | Call `compile_domain_context_pack`, present readable summary (not raw JSON), surface validation errors and warnings |
| 5 — Risk acknowledgement | If `open_risks > 0`, walk through each open risk and ask user to acknowledge or dismiss |
| 6 — Handoff | Confirm user is happy to proceed, signal `ADVANCE_PIPELINE` readiness to Conductor |

Phase 4 is a review step — the pack has been assembled throughout Phases 1–3. `compile_domain_context_pack` validates and assembles; Meridian formats the result for the user.

---

### Claim Lifecycle

```
add_claim() → Claim with SUPPORTED / CONTRADICTED / AMBIGUOUS / UNVERIFIABLE
                  ↓ if CONTRADICTED
           Meridian presents counter-evidence once
                  ↓ user responds
           resolve_claim(claim_id, new_status)
             → SUPPORTED (user was right, data re-evaluated)
             → USER_OVERRIDE_ACCEPTED (user insisted, override logged)
                  ↓
           compile_domain_context_pack() — fails if any Claim still CONTRADICTED
```

**Claim data contract:**
```python
class VerificationStatus(str, Enum):
    SUPPORTED              = "SUPPORTED"
    CONTRADICTED           = "CONTRADICTED"
    AMBIGUOUS              = "AMBIGUOUS"
    UNVERIFIABLE           = "UNVERIFIABLE"
    USER_OVERRIDE_ACCEPTED = "USER_OVERRIDE_ACCEPTED"

class EvidenceType(str, Enum):
    STATISTICAL_TEST            = "statistical_test"
    ASSOCIATION                 = "association"
    PATTERN                     = "pattern"
    USER_CONFIRMED              = "user_confirmed"
    UNVERIFIABLE_BUSINESS_INPUT = "unverifiable_business_input"

class Claim(BaseModel):
    claim_id:               str                # uuid4
    claim:                  str                # human-readable assertion
    verification_status:    VerificationStatus
    evidence_type:          EvidenceType
    evidence_ref:           str | None         # tool call result summary backing this claim
    applies_to:             str                # segment_id, series_key, or "run"
    downstream_impact:      str                # what this claim affects downstream
    must_surface_in_report: bool = False       # True for USER_OVERRIDE_ACCEPTED
    created_at:             str
```

**Evidence type rules:**
- `statistical_test` and `association` require a preceding tool call with its result cited in `evidence_ref`. Meridian may not assert these without a tool call.
- `pattern` may be asserted conversationally — no tool call required.
- `user_confirmed` — user asserted it, data could not corroborate or deny.
- `unverifiable_business_input` — always accompanied by an open Risk added via `add_risk`.

---

### Meridian Tool Contracts

**Pack management:**
```python
def add_claim(
    run_id: str,
    claim: str,
    evidence_ref: str | None,
    verification_status: str,
    evidence_type: str,
    applies_to: str,
    downstream_impact: str,
) -> dict:
    """
    Append claim to claim_ledger.json under outputs/{run_id}/.
    Returns { claim_id: str }.
    Note: for evidence_type == unverifiable_business_input, Meridian must also call
    add_risk separately. add_claim does not call add_risk internally.
    """

def resolve_claim(
    run_id: str,
    claim_id: str,
    new_status: str,   # SUPPORTED or USER_OVERRIDE_ACCEPTED only
    user_reason: str | None = None,
) -> dict:
    """
    Update an existing CONTRADICTED claim's status.
    If new_status == USER_OVERRIDE_ACCEPTED:
      - Sets must_surface_in_report = True
      - Increments RunState.override_count
    Returns updated claim dict.
    """

def add_risk(
    run_id: str,
    risk: str,
    severity: Literal["low", "medium", "high"],
    source: str,
) -> dict:
    """
    Append to risk_register.json under outputs/{run_id}/.
    Increments RunState.open_risks.
    Returns { risk_id: str }.
    """

def acknowledge_risk(run_id: str, risk_id: str) -> None:
    """
    Mark risk as acknowledged (user accepted it).
    Adds ACCEPTED_RISK Claim to claim_ledger.
    Decrements RunState.open_risks.
    """

def compile_domain_context_pack(run_id: str) -> dict:
    """
    Assemble pack from claim_ledger.json.
    Returns {
        pack_complete: bool,
        validation_errors: list[str],
        validation_warnings: list[str],
        pack: dict | None,
    }
    Writes outputs/{run_id}/domain_context_pack.json if pack_complete == True.

    Validation errors (blocking):
      - no forecast scope (missing target, grain, or horizon)
      - no segments defined
      - any Claim still in CONTRADICTED status
      - open_risks > 0 with none acknowledged
      - required schema fields missing

    Validation warnings (non-blocking):
      - all Claims are UNVERIFIABLE or USER_OVERRIDE_ACCEPTED
      - a segment has no segment-level rules
      - a data-available feature has no policy set
      - override_count > 3
    """
```

**Diagnostic tools:**
```python
def summarise_demand_segments(run_id: str, segment_id: str | None = None) -> dict:
    """Return segment-level demand summary from preflight stats."""

def diagnose_zero_demand_policy(run_id: str, segment_id: str) -> dict:
    """
    Return zero-run stats for segment from preflight.json.
    Returns { zero_pct, max_zero_run, recommendation: str }.
    """

def diagnose_spike_policy(run_id: str, segment_id: str) -> dict:
    """
    Return spike stats for segment.
    Returns { spike_count, spike_dates, recommendation: str }.
    """

def diagnose_granularity_feasibility(run_id: str, min_series: int | None = None) -> dict:
    """Check if grain supports sufficient series for modelling."""

def diagnose_horizon_feasibility(
    run_id: str,
    horizon_periods: int,
    segment_id: str | None = None,
) -> dict:
    """
    Returns { feasible: bool, reason: str, max_recommended_horizon: int }.
    """

def diagnose_structural_break_candidates(
    run_id: str,
    date: str | None = None,
    segment_id: str | None = None,
) -> dict:
    """
    Return structural break candidates from preflight.json break_candidates field.
    Returns { candidates: list[BreakCandidate] }.
    """

def diagnose_forecastability_by_segment(run_id: str, segment_id: str) -> dict:
    """
    Return preliminary forecastability breakdown using preflight ADI/CV² stats.
    Returns { forecastable_pct, caution_pct, unforecastable_pct, basis: str }.
    """
```

---

### `domain_context_pack.json` Structure

```json
{
  "run_id": "...",
  "domain": "FMCG",
  "confirmed_at": "ISO timestamp",
  "forecast_scope": {
    "target_col":     "demand",
    "grain":          "sku_region_week",
    "horizon":        12,
    "forecast_start": "2024-10-01"
  },
  "segments": [
    {
      "segment_id":       "G1",
      "zero_policy":      "exclude",
      "spike_policy":     "cap_iqr3x",
      "feature_flags":    { "promo": true, "price": false },
      "confirmed_breaks": ["2023-06-01"]
    }
  ],
  "claim_ledger":   [ ],
  "risk_register":  [ ],
  "override_count": 1,
  "open_risks":     0
}
```

---

### Implementation Skeleton

```python
# forecasting/agents/meridian.py

import anthropic
from forecasting.guard import AgentGuardState
from forecasting.providers import dispatch_tool

client = anthropic.Anthropic()

def run_meridian(
    run_id: str,
    user_message: str,
    conversation_history: list[dict],
    preflight_bundle: dict,
    tokens_used: int,
    sse_emit: Callable,
) -> int:
    """Execute one Meridian turn. Streams tokens via sse_emit. Returns updated tokens_used."""
    guard = AgentGuardState(agent="meridian", run_id=run_id)
    system = _build_meridian_system(preflight_bundle)
    messages = conversation_history + [{"role": "user", "content": user_message}]

    while True:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            tools=MERIDIAN_TOOLS,
            messages=messages,
        ) as stream:
            full_text = ""
            for event in stream:
                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    sse_emit("token", {"content": event.delta.text})
                    full_text += event.delta.text
            response = stream.get_final_message()
            tokens_used += response.usage.input_tokens + response.usage.output_tokens

        if response.stop_reason == "end_turn":
            sse_emit("message_done", {"agent": "meridian", "full_text": full_text})
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = dispatch_tool(
                    tool_name=block.name,
                    args=block.input,
                    guard=guard,
                    tokens_used=tokens_used,
                )
                _emit_decision_event(sse_emit, block.name, result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return tokens_used

def _emit_decision_event(sse_emit: Callable, tool_name: str, result: dict) -> None:
    if tool_name == "add_claim":
        sse_emit("decision_update", result)
    elif tool_name == "add_risk":
        sse_emit("risk_update", result)
    elif tool_name == "resolve_claim" and result.get("verification_status") == "USER_OVERRIDE_ACCEPTED":
        sse_emit("override_update", result)
```

---

## 7. Forge — EDA Agent

### Purpose & Invariants

Forge receives the confirmed `domain_context_pack` and runs a single-pass EDA — no conversation, no streaming to the user. It works segment by segment, producing the `eda_report.json` that Foundry consumes.

**Key invariants:**
- Forge is triggered by Conductor after `pack_confirmed = True`. It runs fully before Foundry starts.
- No conversation — one pass of tool calls, then `compile_eda_report`. Forge does not respond to user messages.
- The `domain_context_pack` is read-only. Forge never mutates it.
- Structural breaks: Forge only runs the Chow test at **confirmed** break dates from the pack. Unconfirmed candidates are ignored.
- Walk-forward folds: minimum 2 folds post-break required. If a series cannot support 2 folds, it is flagged `caution` — not skipped, not silently degraded.
- Forge emits `forge_progress` SSE events per segment so the frontend can show progress.
- Max tool calls: 20 per invocation (`.env`-configurable).

---

### Tool Contracts

```python
def run_full_eda(run_id: str, segment_id: str) -> dict:
    """
    Run complete EDA for a segment: stationarity tests (ADF, KPSS),
    autocorrelation, trend decomposition.
    Returns { segment_id, adf_results, kpss_results, acf_summary, decomposition_summary }.
    Reads series from data_store via get_series_keys(run_id) filtered to segment.
    """

def classify_demand_profiles(run_id: str, segment_id: str) -> dict:
    """
    Run Syntetos-Boylan classification for every series in segment.
    ADI threshold: 1.32. CV² threshold: 0.49.
    Returns { segment_id, classifications: { series_key: DemandClass } }.
    This is the official Demand Class assignment — overrides preflight estimates.
    Writes classifications to outputs/{run_id}/eda_report_partial.json (appended per segment).
    """

def detect_structural_breaks(
    run_id: str,
    segment_id: str,
    confirmed_dates: list[str],
) -> dict:
    """
    Run Chow test at each confirmed_dates entry for every series in segment.
    confirmed_dates must come from domain_context_pack.segments[n].confirmed_breaks.
    Returns { segment_id, break_results: { series_key: { date: ChowResult } } }.
    ChowResult: { significant: bool, p_value: float, f_stat: float }
    """

def flag_stockouts(run_id: str, segment_id: str, threshold_weeks: int) -> dict:
    """
    Flag series with zero runs >= threshold_weeks as likely stockouts.
    Returns { segment_id, flagged_series: list[str], reason: str }.
    """

def specify_feature_config(run_id: str, segment_id: str, demand_class: str) -> dict:
    """
    Translate pack feature_flags + demand_class into per-segment feature config.
    Returns {
        segment_id,
        demand_class,
        features: { promo: bool, price: bool, fourier_terms: int, lag_windows: list[int] },
        rationale: str,
    }
    """

def design_walk_forward_folds(
    run_id: str,
    segment_id: str,
    horizon: int,
    break_dates: list[str],
) -> dict:
    """
    Design walk-forward validation folds for segment.
    Rules:
      - Minimum 2 folds post-break required
      - Pre-break data never used as fallback
      - If < 2 folds possible post-break: series flagged caution,
        reason = "insufficient_post_break_history"
    Returns {
        segment_id,
        folds: list[{ train_start, train_end, test_start, test_end }],
        caution_series: list[{ series_key, reason }],
        n_folds: int,
    }
    """

def select_evaluation_metric(run_id: str, demand_class: str) -> dict:
    """
    Returns { metric: "MASE" | "MAD_fill_rate", rationale: str }.
    MASE for SMOOTH / ERRATIC / INTERMITTENT.
    MAD + fill_rate for LUMPY.
    """

def compile_eda_report(run_id: str) -> dict:
    """
    Assemble full EDA report from per-segment partial results.
    Writes outputs/{run_id}/eda_report.json.
    Sets RunState.forge_complete = True.
    Emits forge_progress SSE event: { segment_id: "all", status: "done" }.
    Returns { run_id, segment_count, series_count, caution_count }.
    """
```

---

### `eda_report.json` Structure

```json
{
  "run_id": "...",
  "compiled_at": "ISO timestamp",
  "segments": [
    {
      "segment_id":        "G1",
      "demand_classes":    { "SKU_101|WEST": "SMOOTH", "SKU_102|EAST": "LUMPY" },
      "feature_config":    { "promo": true, "price": false, "fourier_terms": 2, "lag_windows": [1, 4, 12] },
      "evaluation_metric": "MASE",
      "folds": [
        { "train_start": "2022-01-01", "train_end": "2023-09-30",
          "test_start":  "2023-10-01", "test_end":  "2023-12-31" }
      ],
      "break_results":   { "SKU_101|WEST": { "2023-06-01": { "significant": true, "p_value": 0.02 } } },
      "caution_series":  [{ "series_key": "SKU_103|NORTH", "reason": "insufficient_post_break_history" }],
      "stockout_flags":  ["SKU_104|SOUTH"]
    }
  ]
}
```

---

### Forge Execution Order (per segment)

```
1. emit forge_progress { segment_id, status: "running" }
2. run_full_eda(segment_id)
3. classify_demand_profiles(segment_id)
4. detect_structural_breaks(segment_id, confirmed_dates)
5. flag_stockouts(segment_id, threshold_weeks)
6. for each demand_class in segment:
     specify_feature_config(segment_id, demand_class)
7. design_walk_forward_folds(segment_id, horizon, break_dates)
8. select_evaluation_metric(demand_class)
9. emit forge_progress { segment_id, status: "done" }

After all segments:
10. compile_eda_report(run_id)
    → Conductor.trigger_foundry(run_id) called by Conductor, not Forge
```

---

### Implementation Skeleton

```python
# forecasting/agents/forge.py

import anthropic, json
from forecasting.guard import AgentGuardState
from forecasting.providers import dispatch_tool

client = anthropic.Anthropic()

SYSTEM_PROMPT = """
You are Forge, the EDA agent for a demand forecasting pipeline.

You receive the confirmed domain_context_pack. Your job is to run EDA for every
segment defined in the pack, in order, then compile the eda_report.

Rules:
- Work one segment at a time, in the order they appear in the pack.
- Only run detect_structural_breaks at dates listed in confirmed_breaks. Never infer breaks.
- After all segments are complete, call compile_eda_report.
- Do not produce any user-facing text. Tool calls only.
""".strip()

def run_forge(
    run_id: str,
    domain_context_pack: dict,
    tokens_used: int,
    sse_emit: Callable,
) -> int:
    """Execute full Forge EDA pass. Returns updated tokens_used."""
    guard = AgentGuardState(agent="forge", run_id=run_id)
    messages = [{
        "role": "user",
        "content": f"domain_context_pack:\n{json.dumps(domain_context_pack, indent=2)}",
    }]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=FORGE_TOOLS,
            messages=messages,
        )
        tokens_used += response.usage.input_tokens + response.usage.output_tokens

        if response.stop_reason == "end_turn":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = dispatch_tool(
                    tool_name=block.name,
                    args=block.input,
                    guard=guard,
                    tokens_used=tokens_used,
                )
                if block.name == "compile_eda_report":
                    sse_emit("forge_progress", {"segment_id": "all", "status": "done"})
                elif block.name in ("run_full_eda", "classify_demand_profiles"):
                    sse_emit("forge_progress",
                             {"segment_id": block.input.get("segment_id"), "status": "running"})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return tokens_used
```

---

## 8. Foundry — Model Selection & Ensemble

### Purpose & Invariants

Foundry runs at series level. It receives the `eda_report` and `domain_context_pack` and executes the self-correction loop for every series across every segment. Segment policy gates the model search space at the tool layer — Foundry cannot call a model outside its demand class gate.

**Key invariants:**
- Foundry is triggered by Conductor after `forge_complete = True`.
- Demand class gates are enforced in the tool dispatcher, not in the system prompt. `train_and_evaluate` rejects a model if it is outside the gate for the series' demand class.
- Ensemble is only considered if a single model plateaus AND the delta exceeds 5% on MASE.
- MASE target follows a three-layer hierarchy: universal floor (< 1.0) → playbook default → user override Claim.
- Foundry emits `foundry_progress` SSE events so the frontend can show live counts.
- Foundry uses the cumulative `FoundryRunGuard` (max 500 tool calls per run, `.env`-configurable).

---

### Demand Class → Allowed Models

```python
DEMAND_CLASS_GATES: dict[str, list[str]] = {
    "SMOOTH":       ["XGBoost", "LightGBM", "RandomForest", "Ridge", "ARIMA", "ETS_additive"],
    "ERRATIC":      ["GradientBoosting", "ETS_multiplicative", "Holt_Winters"],
    "INTERMITTENT": ["Croston", "SBA", "ADIDA", "TSB"],
    "LUMPY":        ["SBA", "TSB", "ADIDA", "ZeroInflated"],
}
```

The tool dispatcher checks `model_name in DEMAND_CLASS_GATES[series_demand_class]` before executing `train_and_evaluate`. Violations raise `ModelGateViolationError` (not `GuardHalt` — it is a logic error, not a budget error).

---

### MASE Target Hierarchy

| Layer | Source | Value (FMCG default) |
|---|---|---|
| Universal floor | hardcoded | MASE < 1.0 |
| Playbook default | `fmcg.yaml` → `mase_target` | MASE < 0.8 |
| User override | Claim from Meridian scoping | as specified |

```python
def resolve_mase_target(
    run_id: str,
    series_key: str,
    domain_context_pack: dict,
    playbook: dict,
) -> float:
    """
    Returns the active MASE target for a series.
    Priority: user override Claim > playbook default > universal floor (1.0).
    A user override Claim is any Claim with applies_to == series_key or "run"
    and downstream_impact containing "mase_target".
    """
```

---

### Tool Contracts

```python
def get_segment_series_list(run_id: str, segment_id: str) -> dict:
    """
    Return all series keys for a segment.
    Returns { segment_id, series_keys: list[str], demand_classes: { series_key: str } }.
    Reads from eda_report.json.
    """

def train_and_evaluate(
    run_id: str,
    series_key: str,
    model_name: str,
    hyperparams: dict,
) -> dict:
    """
    Train model on series, evaluate on walk-forward folds from eda_report.
    Rejects if model_name not in DEMAND_CLASS_GATES[series_demand_class] →
      raises ModelGateViolationError.
    Returns {
        series_key, model_name,
        mase: float, mad: float | None,
        fold_scores: list[float],
        training_periods: int,
        feature_importance: dict | None,
    }
    """

def walk_forward_validate(
    run_id: str,
    series_key: str,
    model_name: str,
    n_folds: int,
) -> dict:
    """
    Re-run walk-forward validation with explicit fold count.
    Used in self-correction rounds when structural changes are made.
    Returns { series_key, model_name, fold_scores: list[float], mase: float }.
    """

def build_ensemble(
    run_id: str,
    series_key: str,
    base_models: list[str],
    strategy: Literal["simple_average", "weighted_mase"],
) -> dict:
    """
    Build ensemble of base_models for series.
    Only valid if single model plateaued AND delta > 5% MASE over best single.
    Returns { series_key, ensemble_mase: float, delta_vs_best_single: float }.
    """

def assess_target_feasibility(run_id: str, series_key: str) -> dict:
    """
    Called after all self-correction rounds are exhausted.
    Returns {
        series_key,
        achievable: bool,
        theoretical_floor_mase: float,
        active_target: float,
        recommendations: list[str],
    }
    If achievable == False → series is unforecastable.
    """

def record_series_result(run_id: str, series_key: str, result: dict) -> None:
    """
    Write per-series outcome to outputs/{run_id}/series_results/{series_key}.json.
    result must include: model_name, mase, forecastability, self_correction_rounds,
                         demand_class, fold_scores, caution_reasons (list).
    """

def compile_foundry_report(run_id: str) -> dict:
    """
    Aggregate per-series results into foundry_report.json.
    Sets RunState.foundry_complete = True, phase = report_ready.
    Emits pipeline_done SSE event: { forecastable, caution, unforecastable }.
    Returns { run_id, forecastable_count, caution_count, unforecastable_count }.
    """
```

---

### Self-Correction Loop

```
For each series_key in segment:

  Round 1 — Best single model
    train_and_evaluate(series_key, best_model_for_demand_class, default_hyperparams)
    if mase <= mase_target: record_series_result(forecastability="forecastable") → next series

  Round 2 — Structural change within same family
    try next model in demand class gate OR adjust hyperparams
    walk_forward_validate(series_key, new_model, n_folds)
    if mase <= mase_target: record_series_result(forecastability="forecastable") → next series

  Round 3 — Most complex model + ensemble
    train_and_evaluate(series_key, most_complex_model, tuned_hyperparams)
    if delta > 5% MASE over Round 2 best:
      build_ensemble(series_key, [round2_best, round3_model], strategy="weighted_mase")
    if mase <= mase_target: record_series_result(forecastability="forecastable") → next series

  All rounds failed:
    assess_target_feasibility(series_key)
    if achievable == False:
      record_series_result(forecastability="unforecastable")
    else:
      record_series_result(forecastability="caution", caution_reasons=["below_target_but_feasible"])
```

**Forecastability outcomes:**

| Value | Condition |
|---|---|
| `forecastable` | MASE meets active target |
| `caution` | Modellable but carries known risk — marginal history, high CV², `ACCEPTED_RISK` Claim, insufficient post-break folds, or below target but `assess_target_feasibility` says achievable |
| `unforecastable` | All three rounds exhausted; `assess_target_feasibility` confirms theoretical floor exceeds target |

---

### `series_results/{series_key}.json` Structure

```json
{
  "series_key":             "SKU_101|WEST",
  "segment_id":             "G1",
  "demand_class":           "SMOOTH",
  "forecastability":        "forecastable",
  "model_name":             "XGBoost",
  "mase":                   0.72,
  "mad":                    null,
  "fold_scores":            [0.68, 0.74, 0.71],
  "self_correction_rounds": 1,
  "ensemble_used":          false,
  "caution_reasons":        [],
  "feature_importance":     { "promo": 0.41, "lag_1": 0.33, "fourier_1": 0.26 },
  "training_periods":       104
}
```

---

### Implementation Skeleton

```python
# forecasting/agents/foundry.py

import anthropic, json
from forecasting.guard import AgentGuardState, FoundryRunGuard
from forecasting.providers import dispatch_tool

client = anthropic.Anthropic()

SYSTEM_PROMPT = """
You are Foundry, the model selection agent for a demand forecasting pipeline.

You receive the eda_report and domain_context_pack. Your job is to run the
self-correction loop for every series, record results, then compile the foundry_report.

Rules:
- Work one series at a time. Complete all self-correction rounds before moving on.
- Never call a model outside the demand class gate — the tool will reject it.
- Only call build_ensemble if a single model has plateaued AND delta > 5% MASE.
- After exhausting all rounds, always call assess_target_feasibility before marking unforecastable.
- After every series, call record_series_result.
- When all series are done, call compile_foundry_report.
- Do not produce any user-facing text. Tool calls only.
""".strip()

def run_foundry(
    run_id: str,
    eda_report: dict,
    domain_context_pack: dict,
    tokens_used: int,
    sse_emit: Callable,
    foundry_guard: FoundryRunGuard,
) -> int:
    """Execute full Foundry modelling pass. Returns updated tokens_used."""
    guard = AgentGuardState(agent="foundry", run_id=run_id)
    total_series = sum(len(seg["demand_classes"]) for seg in eda_report["segments"])
    done_count = 0

    messages = [{
        "role": "user",
        "content": (
            f"eda_report:\n{json.dumps(eda_report, indent=2)}\n\n"
            f"domain_context_pack:\n{json.dumps(domain_context_pack, indent=2)}"
        ),
    }]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=FOUNDRY_TOOLS,
            messages=messages,
        )
        tokens_used += response.usage.input_tokens + response.usage.output_tokens

        if response.stop_reason == "end_turn":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = dispatch_tool(
                    tool_name=block.name,
                    args=block.input,
                    guard=guard,
                    tokens_used=tokens_used,
                    foundry_guard=foundry_guard,
                )
                if block.name == "record_series_result":
                    done_count += 1
                    sse_emit("foundry_progress", {
                        "done": done_count,
                        "total": total_series,
                        "by_segment": {},
                    })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return tokens_used
```
## 9. Prism — Scenario Runner

### Purpose & Invariants

Prism is a child run. It inherits the locked `domain_context_pack` and baseline `foundry_report` from the parent run. It applies a scenario override and re-runs Forge + Foundry for affected series only — not the full pipeline.

**Key invariants:**
- Prism only runs when `phase = report_ready` on the parent run. Conductor enforces this via `create_prism_run`.
- Prism clones the `domain_context_pack` into `whatif/{whatif_id}/modified_pack.json`. The original pack is never mutated.
- If a scenario override materially changes a series' demand pattern, Prism re-runs `classify_demand_profiles_for_scenario` before `run_forge_for_scenario`. Demand Class is not frozen at baseline — see ADR-0001.
- Prism has its own SSE stream: `GET /api/v1/runs/{run_id}/whatif/{whatif_id}/stream`.
- Max tool calls: 20 per invocation (`.env`-configurable via `MAX_TOOL_CALLS_PRISM`).

---

### Scenario Override Types

```python
class ScenarioOverrideType(str, Enum):
    ADD_PROMO_EVENT     = "ADD_PROMO_EVENT"
    CHANGE_HORIZON      = "CHANGE_HORIZON"
    EXCLUDE_SERIES      = "EXCLUDE_SERIES"
    CHANGE_PRICE        = "CHANGE_PRICE"
    INJECT_STOCKOUT     = "INJECT_STOCKOUT"
    CHANGE_FEATURE_FLAG = "CHANGE_FEATURE_FLAG"

class ScenarioOverride(BaseModel):
    type:            ScenarioOverrideType
    affected_series: list[str]   # series keys
    modification:    dict         # type-specific payload

# Examples:
# ADD_PROMO_EVENT:     { "weeks": [10, 11, 12], "uplift_pct": 20 }
# CHANGE_HORIZON:      { "new_horizon": 24 }
# INJECT_STOCKOUT:     { "weeks": [5, 6, 7] }
# CHANGE_FEATURE_FLAG: { "flag": "price", "value": true }
```

**Demand Class reclassification triggers (ADR-0001):**
- `ADD_PROMO_EVENT` → may shift SMOOTH → ERRATIC
- `INJECT_STOCKOUT` → may shift any class → INTERMITTENT or LUMPY
- `CHANGE_PRICE` → may shift demand pattern
- `EXCLUDE_SERIES`, `CHANGE_HORIZON`, `CHANGE_FEATURE_FLAG` → no reclassification needed

---

### Tool Contracts

```python
def get_baseline_result(run_id: str, series_key: str | None = None) -> dict:
    """
    Return baseline foundry_report or single series result.
    If series_key given: returns outputs/{run_id}/series_results/{series_key}.json.
    Else: returns outputs/{run_id}/foundry_report.json.
    """

def parse_scenario(scenario_description: str, intent_entities: dict) -> dict:
    """
    Parse free-text scenario description into a structured ScenarioOverride.
    Returns { type, affected_series, modification }.
    Uses intent_entities (from Lens IntentPack) to resolve series keys.
    Raises ScenarioParseError if description cannot be resolved to a known override type.
    """

def apply_scenario_to_pack(run_id: str, whatif_id: str, structured_override: dict) -> dict:
    """
    Clone domain_context_pack, apply override, write to
    outputs/{run_id}/whatif/{whatif_id}/modified_pack.json.
    Returns { whatif_id, modified_pack_path, affected_series }.
    """

def classify_demand_profiles_for_scenario(
    run_id: str,
    whatif_id: str,
    affected_series: list[str],
) -> dict:
    """
    Re-run Syntetos-Boylan classification for affected_series using scenario-modified data.
    Required before run_forge_for_scenario for ADD_PROMO_EVENT, INJECT_STOCKOUT, CHANGE_PRICE.
    Returns { reclassified: { series_key: new_demand_class }, unchanged: list[str] }.
    """

def run_forge_for_scenario(run_id: str, whatif_id: str, affected_series: list[str]) -> dict:
    """
    Re-run specify_feature_config and design_walk_forward_folds for affected_series.
    Stationarity and break detection inherited from baseline eda_report.
    Returns { whatif_id, updated_feature_configs: { series_key: dict } }.
    """

def run_foundry_for_scenario(run_id: str, whatif_id: str, affected_series: list[str]) -> dict:
    """
    Re-run train_and_evaluate + walk_forward_validate for affected_series.
    Uses same model family as baseline unless demand class shifted or
    metric degrades > 10% vs baseline (triggers model family reconsideration).
    Returns { whatif_id, scenario_results: { series_key: SeriesResult } }.
    """

def compile_comparison(run_id: str, whatif_id: str) -> dict:
    """
    Build side-by-side comparison: baseline vs scenario per affected series.
    Writes outputs/{run_id}/whatif/{whatif_id}/comparison.json.
    Returns { whatif_id, comparison: { series_key: { baseline, scenario, delta_mase } } }.
    """
```

---

### Prism Execution Order

```
1. parse_scenario(scenario_description, intent_entities)
2. apply_scenario_to_pack(run_id, whatif_id, structured_override)
3. if override_type in (ADD_PROMO_EVENT, INJECT_STOCKOUT, CHANGE_PRICE):
     classify_demand_profiles_for_scenario(run_id, whatif_id, affected_series)
4. run_forge_for_scenario(run_id, whatif_id, affected_series)
5. run_foundry_for_scenario(run_id, whatif_id, affected_series)
6. compile_comparison(run_id, whatif_id)
   → comparison.json written; SSE message_done emitted on whatif stream
```

---

### `comparison.json` Structure

```json
{
  "whatif_id":   "wif_abc123",
  "run_id":      "run_xyz",
  "scenario":    "20% promo on SKU_101 weeks 10–12",
  "compiled_at": "ISO timestamp",
  "comparison": {
    "SKU_101|WEST": {
      "baseline": {
        "model_name": "XGBoost", "mase": 0.72,
        "forecastability": "forecastable", "demand_class": "SMOOTH"
      },
      "scenario": {
        "model_name": "GradientBoosting", "mase": 0.81,
        "forecastability": "caution", "demand_class": "ERRATIC"
      },
      "delta_mase": 0.09,
      "demand_class_shifted": true
    }
  }
}
```

---

### Implementation Skeleton

```python
# forecasting/agents/prism.py

import anthropic, json
from forecasting.guard import AgentGuardState
from forecasting.providers import dispatch_tool

client = anthropic.Anthropic()

SYSTEM_PROMPT = """
You are Prism, the scenario runner for a demand forecasting pipeline.

You receive a scenario description, the parent run_id, and a whatif_id.
Your job: parse the scenario, apply it to the pack, re-run Forge and Foundry
for affected series only, then compile the comparison.

Rules:
- Always call parse_scenario first. Do not infer the override type from the description.
- Only call classify_demand_profiles_for_scenario for ADD_PROMO_EVENT, INJECT_STOCKOUT,
  or CHANGE_PRICE overrides.
- Stationarity and break detection are inherited from baseline — do not re-run them.
- After compile_comparison, stop.
""".strip()

def run_prism(
    run_id: str,
    whatif_id: str,
    scenario_description: str,
    intent_entities: dict,
    tokens_used: int,
    sse_emit: Callable,
) -> int:
    """Execute Prism scenario run. Returns updated tokens_used."""
    guard = AgentGuardState(agent="prism", run_id=run_id)
    messages = [{
        "role": "user",
        "content": (
            f"run_id: {run_id}\nwhatif_id: {whatif_id}\n"
            f"scenario_description: {scenario_description}\n"
            f"intent_entities: {json.dumps(intent_entities)}"
        ),
    }]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=PRISM_TOOLS,
            messages=messages,
        )
        tokens_used += response.usage.input_tokens + response.usage.output_tokens

        if response.stop_reason == "end_turn":
            sse_emit("message_done", {"agent": "prism", "whatif_id": whatif_id})
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = dispatch_tool(
                    tool_name=block.name,
                    args=block.input,
                    guard=guard,
                    tokens_used=tokens_used,
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return tokens_used
```

---

## 10. Backend API

### Purpose & Invariants

FastAPI application in `backend/app.py`. Serves the built React bundle at `/` in production. All agent orchestration is triggered from API handlers — handlers are the only entry point into the agent layer.

**Key invariants:**
- Sync handlers for the POC (no `async def`). FastAPI runs them in a thread pool.
- No auth — solo analyst POC.
- Every handler that touches a run loads `RunState` from disk at the start and checks `phase != HALTED` before proceeding. Halted runs return `410 Gone`.
- SSE streams are long-lived responses. Each stream is one `EventSourceResponse` per request.
- `GuardHalt` propagating out of any agent call is caught by the handler, which calls `log_halt` and closes the stream.

---

### Request / Response Models

```python
# forecasting/api/models.py

class CreateRunRequest(BaseModel):
    domain: str

class CreateRunResponse(BaseModel):
    run_id: str

class UploadResponse(BaseModel):
    run_id:          str
    series_count:    int
    segment_count:   int
    segments:        list[str]
    blocking_issues: list[BlockingIssue]
    warnings:        list[Warning]

class MessageRequest(BaseModel):
    content: str

class RunSummaryResponse(BaseModel):
    run_id:             str
    domain:             str
    phase:              str
    claim_count:        int
    risk_count:         int
    override_count:     int
    token_usage:        dict         # { input: int, output: int }
    halt_reason:        str | None
    final_pack_created: bool

class RunListItem(BaseModel):
    run_id:       str
    domain:       str
    phase:        str
    series_count: int
    created_at:   str

class WhatIfRequest(BaseModel):
    scenario_description: str

class WhatIfResponse(BaseModel):
    whatif_id: str

class DecisionsResponse(BaseModel):
    claims:    list[dict]
    risks:     list[dict]
    overrides: list[dict]   # filtered view: verification_status == USER_OVERRIDE_ACCEPTED
```

---

### Router Contracts

**`routers/runs.py`**
```python
POST /api/v1/runs/create
    body:    CreateRunRequest
    action:  generate run_id (uuid4), call create_run_state(run_id, domain), load playbook
    returns: CreateRunResponse
    errors:  400 if domain playbook not found

POST /api/v1/runs/{run_id}/upload
    body:    multipart — file (CSV), domain (str)
    action:  call run_preflight(run_id, file_bytes, domain, playbook)
             on PreflightBlockingError: update RunState.halt_reason, return 422
    returns: UploadResponse
    errors:  422 with blocking_issues if PreflightBlockingError; 410 if HALTED

GET /api/v1/runs
    returns: list[RunListItem] — reads all run_state.json under outputs/

GET /api/v1/runs/{run_id}
    returns: RunSummaryResponse
    errors:  404 if run not found

GET /api/v1/runs/{run_id}/decisions
    returns: DecisionsResponse — reads claim_ledger.json and risk_register.json
    errors:  404 if run not found

GET /api/v1/runs/{run_id}/report
    returns: foundry_report.json contents
    errors:  404 if run not found; 409 if phase != report_ready

GET /api/v1/runs/{run_id}/artifacts/{name}
    name:    one of [domain_context_pack, eda_report, foundry_report,
                     obs_log, run_state, run_summary, preflight]
    returns: file download (application/json)
    errors:  404 if file not found
```

**`routers/message.py`**
```python
POST /api/v1/runs/{run_id}/message
    body:    MessageRequest
    action:
      1. load RunState — 410 if HALTED
      2. call classify_intent(LensInput(...))
      3. call run_conductor(run_id, intent_pack, run_state, tokens_used)
         GuardHalt → log_halt, push error SSE, return 200
    returns: 202 Accepted — { "status": "processing" }
    errors:  410 if HALTED; 404 if run not found
    note:    Response body is minimal — real response comes via SSE stream
```

**`routers/stream.py`**
```python
GET /api/v1/runs/{run_id}/stream
    returns: text/event-stream (EventSourceResponse)
    action:  open SSE stream; push queued events; keep alive until message_done or error
    errors:  404 if run not found; 410 pushes final error event then closes

GET /api/v1/runs/{run_id}/whatif/{whatif_id}/stream
    returns: text/event-stream scoped to scenario run
    errors:  404 if whatif_id not found
```

**`routers/whatif.py`**
```python
POST /api/v1/runs/{run_id}/whatif
    body:    WhatIfRequest
    action:  calls Conductor.create_prism_run, then run_prism(...)
    returns: WhatIfResponse — { whatif_id }
    errors:  409 if phase != report_ready; 410 if HALTED

GET /api/v1/runs/{run_id}/whatif/{whatif_id}/compare
    returns: comparison.json contents
    errors:  404 if not found; 409 if scenario run not complete
```

---

### SSE Event Contract (complete)

All events: `event: {type}\ndata: {json}\n\n`

| Event | Payload | Consumer |
|---|---|---|
| `token` | `{ content: str }` | Append to streaming bubble |
| `message_done` | `{ agent: str, full_text: str }` | Commit to history, clear stream buffer |
| `decision_update` | `{ claim_id, claim, verification_status, evidence_type, applies_to }` | Push to DecisionPanel |
| `risk_update` | `{ risk_id, risk, severity, source }` | Push to RiskList |
| `override_update` | `{ claim_id, decision, consequence, severity }` | Push to OverrideList |
| `phase_change` | `{ phase: str }` | Update PhaseBar, trigger view switch |
| `forge_progress` | `{ segment_id: str, status: "pending"\|"running"\|"done" }` | Update ForgeProgress |
| `foundry_progress` | `{ done: int, total: int, by_segment: dict }` | Update FoundryProgress |
| `pipeline_done` | `{ forecastable: int, caution: int, unforecastable: int }` | Show ReportSummary |
| `error` | `{ reason: str, halt_reason: str }` | Show error state, close stream |

Conductor's tool call trace is never emitted. Only `phase_change` and `decision_update` surface Conductor decisions.

---

### SSE Queue Implementation

```python
# forecasting/api/sse.py

import queue, json
from typing import Generator

_queues: dict[str, queue.Queue] = {}

def get_or_create_queue(run_id: str) -> queue.Queue:
    if run_id not in _queues:
        _queues[run_id] = queue.Queue()
    return _queues[run_id]

def emit(run_id: str, event_type: str, payload: dict) -> None:
    get_or_create_queue(run_id).put({"event": event_type, "data": payload})

def stream_events(run_id: str) -> Generator[str, None, None]:
    q = get_or_create_queue(run_id)
    while True:
        try:
            item = q.get(timeout=30)
            yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
            if item["event"] in ("message_done", "error", "pipeline_done"):
                break
        except queue.Empty:
            yield ": keepalive\n\n"
```

---

### `app.py` Entry Point

```python
# backend/app.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from forecasting.api.routers import runs, message, stream, whatif

app = FastAPI(title="Agent P")
app.include_router(runs.router,    prefix="/api/v1")
app.include_router(message.router, prefix="/api/v1")
app.include_router(stream.router,  prefix="/api/v1")
app.include_router(whatif.router,  prefix="/api/v1")

app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
```

---

## 11. Frontend

### Purpose & Invariants

React 18 + Vite + TypeScript + Tailwind CSS + shadcn/ui. State managed by Zustand (4 stores). SSE consumed via native `EventSource`. HTTP via `fetch`.

**Key invariants:**
- The frontend never holds series data — it only holds conversation history, phase state, decisions, and streaming buffer.
- SSE is the only channel for agent output. The `POST /message` response body is ignored for content — `202 Accepted` just means the request was queued.
- All store mutations that originate from SSE events must be idempotent — reconnection replays events.
- The DecisionPanel is read-only. It displays claims, risks, and overrides but offers no editing surface.
- Three fixed layout zones: left sidebar (200px), centre flex, right panel (320px collapsible).

---

### Zustand Stores

```typescript
// src/stores/runStore.ts
interface RunStore {
  run_id:      string | null
  phase:       Phase | null
  domain:      string | null
  history:     ConversationTurn[]
  setRun:      (run_id: string, domain: string) => void
  setPhase:    (phase: Phase) => void
  pushMessage: (turn: ConversationTurn) => void
  reset:       () => void
}

// src/stores/streamStore.ts
interface StreamStore {
  partial:     string       // current streaming token buffer
  isStreaming: boolean
  append:      (token: string) => void
  commit:      () => void   // clears partial, sets isStreaming=false
  setStreaming: (v: boolean) => void
}

// src/stores/decisionStore.ts
interface DecisionStore {
  claims:       Claim[]
  risks:        Risk[]
  overrides:    Override[]  // filtered view — same objects, different list
  pushClaim:    (claim: Claim) => void
  pushRisk:     (risk: Risk) => void
  pushOverride: (override: Override) => void
  reset:        () => void
}

// src/stores/prismStore.ts
interface PrismStore {
  whatif_id:         string | null
  scenario:          string | null
  isRunning:         boolean
  comparison_result: ComparisonResult | null
  setWhatif:    (whatif_id: string, scenario: string) => void
  setRunning:   (v: boolean) => void
  setComparison:(result: ComparisonResult) => void
  reset:        () => void
}
```

---

### TypeScript Types

```typescript
type Phase =
  | "preflight" | "meridian_scoping" | "forge_eda"
  | "foundry_modelling" | "report_ready" | "halted"

interface ConversationTurn {
  role:    "user" | "assistant"
  content: string
  agent?:  string
}

interface Claim {
  claim_id:               string
  claim:                  string
  verification_status:    VerificationStatus
  evidence_type:          EvidenceType
  applies_to:             string
  downstream_impact:      string
  must_surface_in_report: boolean
}

type VerificationStatus =
  | "SUPPORTED" | "CONTRADICTED" | "AMBIGUOUS"
  | "UNVERIFIABLE" | "USER_OVERRIDE_ACCEPTED"

type EvidenceType =
  | "statistical_test" | "association" | "pattern"
  | "user_confirmed" | "unverifiable_business_input"

interface Risk {
  risk_id:  string
  risk:     string
  severity: "low" | "medium" | "high"
  source:   string
}

interface Override {
  claim_id:    string
  decision:    string
  consequence: string
  severity:    "low" | "medium" | "high"
}

interface ComparisonResult {
  whatif_id: string
  scenario:  string
  comparison: Record<string, {
    baseline:             SeriesResult
    scenario:             SeriesResult
    delta_mase:           number
    demand_class_shifted: boolean
  }>
}

interface SeriesResult {
  model_name:      string
  mase:            number
  forecastability: "forecastable" | "caution" | "unforecastable"
  demand_class:    string
}
```

---

### SSE Hook

```typescript
// src/hooks/useSSE.ts

import { useEffect, useRef } from "react"
import { useRunStore }      from "../stores/runStore"
import { useStreamStore }   from "../stores/streamStore"
import { useDecisionStore } from "../stores/decisionStore"

export function useSSE(run_id: string | null) {
  const esRef       = useRef<EventSource | null>(null)
  const setPhase    = useRunStore(s => s.setPhase)
  const pushMessage = useRunStore(s => s.pushMessage)
  const append      = useStreamStore(s => s.append)
  const commit      = useStreamStore(s => s.commit)
  const setStreaming = useStreamStore(s => s.setStreaming)
  const pushClaim   = useDecisionStore(s => s.pushClaim)
  const pushRisk    = useDecisionStore(s => s.pushRisk)
  const pushOverride = useDecisionStore(s => s.pushOverride)

  useEffect(() => {
    if (!run_id) return
    const es = new EventSource(`/api/v1/runs/${run_id}/stream`)
    esRef.current = es

    es.addEventListener("token",         e => { setStreaming(true); append(JSON.parse(e.data).content) })
    es.addEventListener("message_done",  e => { const { agent, full_text } = JSON.parse(e.data); pushMessage({ role: "assistant", content: full_text, agent }); commit() })
    es.addEventListener("phase_change",  e => setPhase(JSON.parse(e.data).phase))
    es.addEventListener("decision_update", e => pushClaim(JSON.parse(e.data)))
    es.addEventListener("risk_update",     e => pushRisk(JSON.parse(e.data)))
    es.addEventListener("override_update", e => pushOverride(JSON.parse(e.data)))
    es.addEventListener("error",           () => { setPhase("halted"); es.close() })

    return () => { es.close(); esRef.current = null }
  }, [run_id])
}
```

---

### Component Tree & Responsibilities

```
App
├── RunsSidebar (200px fixed)
│   └── RunCard × N          — click to switch active run; shows phase chip
│
├── MainPanel (flex centre)
│   ├── PhaseBar             — phase indicator strip; highlights current phase
│   ├── ConversationView     — visible during meridian_scoping
│   │   ├── MessageBubble × N
│   │   └── StreamingBubble  — live from streamStore.partial
│   ├── PipelineProgress     — visible during forge_eda + foundry_modelling
│   │   ├── ForgeProgress    — per-segment status from forge_progress events
│   │   └── FoundryProgress  — done/total counter + per-segment breakdown
│   ├── ReportSummary        — visible when phase = report_ready
│   │   ├── TechnicalView    — MASE, fold ranges, self-correction, model reasoning
│   │   ├── BusinessView     — plain language, MAPE translation, forecastability counts
│   │   └── PrismButton      — hidden until report_ready
│   └── InputBar             — disabled during forge_eda + foundry_modelling
│       └── MessageInput
│
├── DecisionPanel (320px collapsible right)
│   ├── ClaimCard × N        — verification_status chip + evidence type
│   ├── RiskList             — severity badge + risk text
│   └── OverrideList         — consequence text; must_surface_in_report flagged
│
└── PrismDrawer (slide-over)
    ├── ScenarioInput        — free-text; sends POST /whatif
    ├── PrismProgress        — SSE from whatif stream
    └── ComparisonTable      — baseline vs scenario; delta_mase; demand_class_shifted highlighted
```

---

### View Switching Logic

```typescript
const phase = useRunStore(s => s.phase)

const view = ((): "upload" | "conversation" | "pipeline_progress" | "report" | "error" => {
  if (!phase || phase === "preflight")    return "upload"
  if (phase === "meridian_scoping")       return "conversation"
  if (phase === "forge_eda" ||
      phase === "foundry_modelling")      return "pipeline_progress"
  if (phase === "report_ready")           return "report"
  if (phase === "halted")                 return "error"
  return "upload"
})()
```

`InputBar` is disabled (`pointer-events-none`, greyed) during `forge_eda` and `foundry_modelling`. Re-enables on `report_ready`.

---

### API Client

```typescript
// src/api/client.ts

const BASE = "/api/v1"

export const api = {
  createRun: (domain: string) =>
    fetch(`${BASE}/runs/create`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ domain }) }).then(r => r.json()),

  uploadFile: (run_id: string, file: File, domain: string) => {
    const fd = new FormData(); fd.append("file", file); fd.append("domain", domain)
    return fetch(`${BASE}/runs/${run_id}/upload`, { method: "POST", body: fd }).then(r => r.json())
  },

  sendMessage: (run_id: string, content: string) =>
    fetch(`${BASE}/runs/${run_id}/message`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content }) }),

  createWhatIf: (run_id: string, scenario_description: string) =>
    fetch(`${BASE}/runs/${run_id}/whatif`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scenario_description }) }).then(r => r.json()),

  getComparison: (run_id: string, whatif_id: string) =>
    fetch(`${BASE}/runs/${run_id}/whatif/${whatif_id}/compare`).then(r => r.json()),
}
```

---
---

## Agentic Interaction Layer (Project B)

Layered onto the contracts above; TDD tasks live in `docs/plans/plan.md` Phase G (Tasks 32–42) and the rationale in ADR-0005.

**New contracts** (`contracts.py`): `NeedKind = Literal["USER_DECISION","SCOPE_AMENDMENT"]`; `AgentNeed {agent, kind, question, options[], context}`; `AwaitingInput {need, raised_at}`; `PauseForInput(Exception)` carrying an `AgentNeed` (resumable — NOT a `GuardHalt`).

**RunState additions**: `pack_version`, `awaiting_input: AwaitingInput | None`, `loopback_count`, `tokens_used_total`, `foundry_calls_total`.

**Tools**: agent-side `raise_need(agent, kind, question, options, context)`; Conductor `resolve_need(run_id, answer)` and `reroute(run_id, target_phase, reason)`; `rerun_affected_series(run_id, namespace, series_keys, run_forge, run_foundry)` (loop-back reuses Prism's engine with `namespace=run_id`).

**SSE**: `agent_reasoning {agent, text}`, `agent_needs_input {agent, kind, question, options, context}`.

**Invariants**:
- Pause is non-terminal and resets nothing; only `GuardHalt` ends a run.
- Resume = idempotent re-invocation: agents skip steps whose artifacts already exist and apply the answer at the decision point.
- Exactly one outstanding need at a time; agents batch homogeneous decisions.
- Cumulative budgets persist in RunState and are seeded/written-back per invocation (or resume defeats the Guard).
- Loop-back is a scoped pack amendment that re-locks (`pack_version++`) and re-runs only the affected slice; bounded by `loopback_count` (default 3, `.env`); pre-report mutates in place, post-report uses a cloned Prism what-if.
- Two checkpoints only: Forge strong-unconfirmed-break (`SCOPE_AMENDMENT`) and Foundry target-shortfall batch (`USER_DECISION`); everything else takes a documented default + logs a Claim + narrates.
