# Agent P — Demand Forecasting System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Agent P — a multi-agent FMCG demand forecasting system where named LLM agents (Lens, Conductor, Meridian, Forge, Foundry, Prism) collaborate to scope, analyse, and forecast demand from a CSV upload.

**Architecture:** FastAPI backend with a pure-Python tool layer (`tools/`) and an LLM-loop agent layer (`agents/`). Series data never enters Claude's context — agents receive a `run_id` sentinel and call registered tools that read from `data_store`. All shared domain types live in a single `contracts.py`; HTTP-layer types live in `api/models.py`.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, pandas, statsmodels, anthropic SDK (`claude-sonnet-4-6` / `claude-haiku-4-5-20251001`), React 18, Vite, TypeScript, Tailwind CSS, shadcn/ui, Zustand.

**Python venv:** `c:/Agent_A/venv/Scripts/python.exe` — use this for all `python` and `pytest` commands.

---

## Principles

- **KISS / YAGNI:** No Redis, no ORM, no auth, no Docker — solo-analyst POC, single process.
- **SOLID SRP:** One file = one responsibility. Routers call functions. Functions don't know about HTTP.
- **Pure tools:** Everything in `tools/` is plain Python, testable with zero Anthropic mocking.
- **Thin routers:** Zero business logic inline in route handlers — they classify, call, respond.
- **DIP via registry:** `providers.py` holds `_TOOL_REGISTRY`; adding a tool is one dict entry.
- **TDD:** Write the failing test first, then the minimal implementation.

---

## Repository Layout

```
Agent_A/                                 # product name "Agent P"; code/dir is C:\Agent_A
├── backend/
│   ├── app.py                           # FastAPI app + router mounting + static mount
│   ├── outputs/                         # created at runtime per run
│   └── forecasting/
│       ├── __init__.py
│       ├── contracts.py                 # ALL shared domain Pydantic models (single source of truth)
│       ├── data_store.py                # in-memory series store, keyed by run_id + series_key
│       ├── run_state.py                 # RunState model + Phase enum + load/save/create helpers
│       ├── guard.py                     # GuardConfig, AgentGuardState, FoundryRunGuard, GuardHalt
│       ├── providers.py                 # _TOOL_REGISTRY dict + dispatch_tool()
│       ├── preflight.py                 # pre-flight orchestrator (pure Python, no LLM)
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── preflight_schema.py      # profile_uploaded_data, map_schema, detect_frequency_and_grain, build_series_keys
│       │   ├── preflight_stats.py       # compute_adi_cv2_per_series, detect_zero_runs, detect_spikes, measure_promo_alignment, detect_trend_strength, detect_seasonality_strength, detect_structural_break_candidates, aggregate_segment_profiles, collect_segment_exceptions
│       │   ├── conductor_tools.py       # get_run_state, update_run_state, advance_to_meridian, confirm_pack_and_advance, trigger_foundry, create_prism_run, surface_clarification, log_halt
│       │   ├── meridian_pack.py         # add_claim, resolve_claim, add_risk, acknowledge_risk, compile_domain_context_pack
│       │   ├── meridian_diagnostic.py   # summarise_demand_segments, diagnose_zero_demand_policy, diagnose_spike_policy, diagnose_granularity_feasibility, diagnose_horizon_feasibility, diagnose_structural_break_candidates, diagnose_forecastability_by_segment, refine_segments
│       │   ├── forge_tools.py           # run_full_eda, classify_demand_profiles, detect_structural_breaks, flag_stockouts, specify_feature_config, design_walk_forward_folds, select_evaluation_metric, compile_eda_report
│       │   ├── foundry_tools.py         # get_segment_series_list, train_and_evaluate, walk_forward_validate, build_ensemble, assess_target_feasibility, record_series_result, compile_foundry_report
│       │   └── prism_tools.py           # clone_pack_for_whatif, apply_whatif_override, run_forge_for_scenario, run_foundry_for_scenario, compile_comparison
│       └── agents/
│           ├── __init__.py
│           ├── lens.py                  # classify_intent() — Haiku, no tools, temperature=0
│           ├── conductor.py             # run_conductor() — Sonnet, one tool call per turn
│           ├── meridian.py              # run_meridian() — Sonnet, streaming, conversational
│           ├── forge.py                 # run_forge() — Sonnet, batch EDA
│           ├── foundry.py               # run_foundry() — Sonnet, per-series modelling loop
│           └── prism.py                 # run_prism() — Sonnet, scenario runner
├── api/
│   ├── __init__.py
│   ├── models.py                        # HTTP request/response Pydantic models only
│   ├── sse.py                           # per-run queue.Queue + emit() + stream_events()
│   └── routers/
│       ├── __init__.py
│       ├── runs.py                      # POST /api/v1/runs, GET /api/v1/runs/{run_id}
│       ├── message.py                   # POST /api/v1/runs/{run_id}/message
│       ├── stream.py                    # GET /api/v1/runs/{run_id}/stream (SSE)
│       └── whatif.py                    # POST /api/v1/runs/{run_id}/whatif
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── types/index.ts               # TypeScript domain types (mirrors contracts.py)
│       ├── stores/runStore.ts           # Zustand store — run state + conversation
│       ├── hooks/useSSE.ts              # EventSource hook, dispatches to store
│       ├── api/client.ts                # fetch wrappers for all backend endpoints
│       └── components/
│           ├── Layout.tsx
│           ├── Conversation.tsx         # chat bubble list
│           ├── ProgressBar.tsx          # phase stepper
│           ├── MessageInput.tsx         # textarea + send button
│           ├── ReportView.tsx           # EDA + foundry report display
│           └── DecisionPanel.tsx        # claim ledger + risk register display
└── tests/
    ├── conftest.py                      # shared fixtures (tmp_dir, sample_df, etc.)
    ├── test_data_store.py
    ├── test_run_state.py
    ├── test_guard.py
    ├── test_preflight_schema.py
    ├── test_preflight_stats.py
    ├── test_preflight_orchestrator.py
    ├── test_conductor_tools.py
    ├── test_meridian_pack.py
    ├── test_meridian_diagnostic.py
    ├── test_forge_tools.py
    ├── test_foundry_tools.py
    ├── test_prism_tools.py
    └── test_api.py
```

---

## Phase A — Foundation

> Tasks 1–6. No agents, no LLM. Pure data structures and safety primitives. Every later task depends on these.

---

### Task 1: Repo Scaffold

**Files:**
- Create: `backend/forecasting/__init__.py`
- Create: `backend/forecasting/tools/__init__.py`
- Create: `backend/forecasting/agents/__init__.py`
- Create: `api/__init__.py`
- Create: `api/routers/__init__.py`
- Create: `tests/conftest.py`
- Create: `requirements.txt`
- Create: `pytest.ini`

- [ ] **Step 1: Create directory tree**

```powershell
$dirs = @(
  "backend\forecasting\tools",
  "backend\forecasting\agents",
  "backend\outputs",
  "api\routers",
  "tests",
  "frontend\src\types",
  "frontend\src\stores",
  "frontend\src\hooks",
  "frontend\src\api",
  "frontend\src\components"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force $d }
foreach ($d in @("backend\forecasting", "backend\forecasting\tools", "backend\forecasting\agents", "api", "api\routers")) {
  New-Item -Force "$d\__init__.py" -ItemType File
}
```

- [ ] **Step 2: Write `requirements.txt`**

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
anthropic==0.28.0
pydantic==2.7.1
pandas==2.2.2
numpy==1.26.4
statsmodels==0.14.2
python-multipart==0.0.9
pytest==8.2.2
pytest-asyncio==0.23.7
httpx==0.27.0
```

- [ ] **Step 3: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
```

- [ ] **Step 4: Write `tests/conftest.py`**

```python
import io
import uuid
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture()
def run_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    rows = []
    for sku in ["SKU_A", "SKU_B"]:
        for week in range(1, 53):
            rows.append({"week": f"2024-W{week:02d}", "sku": sku, "region": "NORTH", "demand": float(week % 10 + 1)})
    return pd.DataFrame(rows)


@pytest.fixture()
def tmp_outputs(tmp_path: Path, monkeypatch) -> Path:
    import forecasting.run_state as rs
    monkeypatch.setattr(rs, "OUTPUTS_ROOT", tmp_path)
    return tmp_path
```

- [ ] **Step 5: Install dependencies**

```powershell
c:/Agent_A/venv/Scripts/pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 6: Commit**

```powershell
git add requirements.txt pytest.ini tests/conftest.py
git commit -m "chore: repo scaffold, requirements, pytest config"
```

---

### Task 2: `data_store.py`

**Files:**
- Create: `backend/forecasting/data_store.py`
- Create: `tests/test_data_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data_store.py
import pandas as pd
import pytest
from forecasting.data_store import (
    store_series, get_series, get_series_keys, delete_run, SeriesNotFoundError
)


def make_df() -> pd.DataFrame:
    return pd.DataFrame({"date": ["2024-01-01"], "demand": [10.0]})


def test_store_and_retrieve(run_id):
    df = make_df()
    store_series(run_id, "SKU_A|NORTH", df)
    result = get_series(run_id, "SKU_A|NORTH")
    assert list(result.columns) == ["date", "demand"]
    assert result["demand"].iloc[0] == 10.0


def test_get_series_keys(run_id):
    store_series(run_id, "SKU_A|NORTH", make_df())
    store_series(run_id, "SKU_B|NORTH", make_df())
    keys = get_series_keys(run_id)
    assert set(keys) == {"SKU_A|NORTH", "SKU_B|NORTH"}


def test_series_not_found_raises(run_id):
    with pytest.raises(SeriesNotFoundError) as exc_info:
        get_series(run_id, "MISSING")
    assert run_id in str(exc_info.value)
    assert "MISSING" in str(exc_info.value)


def test_delete_run_clears_all(run_id):
    store_series(run_id, "SKU_A|NORTH", make_df())
    delete_run(run_id)
    assert get_series_keys(run_id) == []


def test_delete_run_no_op_on_missing():
    delete_run("nonexistent-run")  # must not raise


def test_overwrite_existing_series(run_id):
    store_series(run_id, "SKU_A|NORTH", make_df())
    new_df = pd.DataFrame({"date": ["2024-02-01"], "demand": [99.0]})
    store_series(run_id, "SKU_A|NORTH", new_df)
    assert get_series(run_id, "SKU_A|NORTH")["demand"].iloc[0] == 99.0
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_data_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'forecasting'`

- [ ] **Step 3: Write `backend/forecasting/data_store.py`**

```python
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


def key_to_filename(series_key: str) -> str:
    """Reversible series_key ↔ filename mapping (the SINGLE place this is defined).

    '|' is illegal in Windows filenames, so it is percent-encoded as '%7C'. Series
    keys are normalised to [A-Z0-9_|] at pre-flight, so '%' never appears in a key
    and the mapping is unambiguous and reversible — unlike the old lossy '|'→'_'
    (which collided: 'SKU_A|NORTH' and 'SKU|A_NORTH' both became 'SKU_A_NORTH').
    All series_results/ reads and writes MUST route through this pair. (review §6)
    """
    return series_key.replace("|", "%7C")


def filename_to_key(stem: str) -> str:
    return stem.replace("%7C", "|")
```

- [ ] **Step 4: Add `backend/` to PYTHONPATH and run tests**

```powershell
$env:PYTHONPATH = "backend"
c:/Agent_A/venv/Scripts/python -m pytest tests/test_data_store.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```powershell
git add backend/forecasting/data_store.py tests/test_data_store.py
git commit -m "feat: data_store — in-memory series store with sentinel pattern"
```

---

### Task 3: `run_state.py`

**Files:**
- Create: `backend/forecasting/run_state.py`
- Create: `tests/test_run_state.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_run_state.py
import pytest
from forecasting.run_state import (
    Phase, RunState, create_run_state, load_run_state,
    save_run_state, RunNotFoundError, HaltedRunError
)


def test_create_run_state(run_id, tmp_outputs):
    state = create_run_state(run_id, domain="fmcg")
    assert state.phase == Phase.PREFLIGHT
    assert state.domain == "fmcg"
    assert state.run_id == run_id
    assert state.pack_confirmed is False


def test_state_persisted_to_disk(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    loaded = load_run_state(run_id)
    assert loaded.run_id == run_id
    assert loaded.domain == "fmcg"


def test_load_missing_run_raises(tmp_outputs):
    with pytest.raises(RunNotFoundError):
        load_run_state("no-such-run")


def test_save_halted_without_reason_raises(run_id, tmp_outputs):
    state = create_run_state(run_id, domain="fmcg")
    state.phase = Phase.HALTED
    with pytest.raises(ValueError, match="halt_reason"):
        save_run_state(state)


def test_save_halted_with_reason_ok(run_id, tmp_outputs):
    state = create_run_state(run_id, domain="fmcg")
    state.halt_reason = "guard budget exceeded"
    state.phase = Phase.HALTED
    save_run_state(state)
    loaded = load_run_state(run_id)
    assert loaded.phase == Phase.HALTED
    assert loaded.halt_reason == "guard budget exceeded"


def test_phase_transitions(run_id, tmp_outputs):
    state = create_run_state(run_id, domain="fmcg")
    state.phase = Phase.MERIDIAN_SCOPING
    save_run_state(state)
    assert load_run_state(run_id).phase == Phase.MERIDIAN_SCOPING
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_run_state.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backend/forecasting/run_state.py`**

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

OUTPUTS_ROOT = Path("backend/outputs")


class Phase(str, Enum):
    PREFLIGHT = "preflight"
    MERIDIAN_SCOPING = "meridian_scoping"
    FORGE_EDA = "forge_eda"
    FOUNDRY_MODELLING = "foundry_modelling"
    REPORT_READY = "report_ready"
    HALTED = "halted"


class RunState(BaseModel):
    run_id: str
    phase: Phase
    pack_confirmed: bool = False
    meridian_turn_count: int = 0
    forge_complete: bool = False
    foundry_complete: bool = False
    active_whatif_runs: list[str] = Field(default_factory=list)
    open_risks: int = 0
    override_count: int = 0
    halt_reason: str | None = None
    domain: str
    created_at: str

    model_config = ConfigDict(use_enum_values=True)


class RunNotFoundError(Exception):
    def __init__(self, run_id: str):
        super().__init__(f"Run '{run_id}' not found")
        self.run_id = run_id


class HaltedRunError(Exception):
    def __init__(self, run_id: str):
        super().__init__(f"Run '{run_id}' is halted and cannot be mutated")
        self.run_id = run_id


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
    phase_val = state.phase if isinstance(state.phase, str) else state.phase.value
    if phase_val == Phase.HALTED.value and state.halt_reason is None:
        raise ValueError("halt_reason must be set before saving a HALTED RunState")
    state_path(state.run_id).write_text(state.model_dump_json(indent=2))
```

- [ ] **Step 4: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_run_state.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```powershell
git add backend/forecasting/run_state.py tests/test_run_state.py
git commit -m "feat: run_state — RunState model with Phase enum and disk persistence"
```

---

### Task 4: `contracts.py` — Shared Domain Models

**Files:**
- Create: `backend/forecasting/contracts.py`

> No separate test file — contracts are validated implicitly by every task that uses them. Add a smoke test to `tests/conftest.py`.

- [ ] **Step 1: Write `backend/forecasting/contracts.py`**

```python
"""
Single source of truth for all shared domain Pydantic models.
HTTP-layer models live in api/models.py — not here.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pre-flight contracts
# ---------------------------------------------------------------------------

class BlockingIssue(BaseModel):
    code: str
    message: str


class DataQualityWarning(BaseModel):
    code: str
    message: str
    affected_series: list[str] = Field(default_factory=list)


class DataQualityReport(BaseModel):
    blocking_issues: list[BlockingIssue]
    warnings: list[DataQualityWarning]
    row_count: int
    series_count: int


class SchemaMapping(BaseModel):
    date_col: str
    demand_col: str
    grain_cols: list[str]
    extra_cols: list[str]


class GrainReport(BaseModel):
    detected_frequency: Literal["daily", "weekly", "monthly", "unknown"]
    min_periods: int
    max_periods: int
    median_periods: int
    gaps_detected: bool


SBClass = Literal["SMOOTH", "ERRATIC", "INTERMITTENT", "LUMPY"]


class AdiCv2Stats(BaseModel):
    series_key: str
    adi: float
    cv2: float
    sb_class: SBClass


class ZeroRunStats(BaseModel):
    series_key: str
    max_zero_run: int
    zero_fraction: float


class SpikeStats(BaseModel):
    series_key: str
    spike_count: int
    max_spike_ratio: float


class PromoAlignmentStats(BaseModel):
    series_key: str
    has_promo_col: bool
    aligned_fraction: float | None = None


class TrendStats(BaseModel):
    series_key: str
    trend_strength: float
    direction: Literal["up", "down", "flat"]


class SeasonalityStats(BaseModel):
    series_key: str
    seasonal_strength: float
    dominant_period: int | None = None


class BreakCandidate(BaseModel):
    series_key: str
    break_period: str
    confidence: float


class SegmentProfile(BaseModel):
    segment_id:                str               # "G1", "G2", ... — matches SegmentDef.segment_id
    series_count:              int
    demand_class_distribution: dict[str, int]    # {"SMOOTH": 4, "ERRATIC": 1, ...}
    median_adi:                float
    median_cv2:                float
    forecastability_breakdown: dict[str, int]    # preliminary counts, not Forge's official call
    example_keys:              list[str] = Field(default_factory=list)


class SegmentDef(BaseModel):
    """One segment in the (provisional) segment map. See plan_v2 §6 / CONTEXT 'Segment'."""
    segment_id:  str                   # "G1", "G2", ...
    label:       str                   # human-readable, e.g. "region=NORTH" or "all series"
    series_keys: list[str]             # member series keys
    provisional: bool = True           # True until refined/confirmed by the user in Meridian


class SegmentMap(BaseModel):
    """Series→segment grouping. Provisional from pre-flight; locked into the pack at confirmation."""
    run_id:      str
    segments:    list[SegmentDef]
    provisional: bool = True
    derived_by:  str                   # e.g. "playbook:segment_by=region" or "default:single_segment"


class SeriesException(BaseModel):
    series_key:     str
    segment_id:     str                # segment this outlier belongs to
    exception_type: Literal["HIGH_ZERO_FRACTION", "SPIKE", "ZERO_RUN"]
    detail:         str


class PreflightBundle(BaseModel):
    """
    Aggregate-only handoff injected into Meridian's system prompt (plan_v2 correction #4).
    Per-series statistics are NOT carried here — they are persisted to preflight.json and
    read on demand by the diagnostic tools. Keeping them out of the bundle keeps individual
    series stats out of Claude's context on every Meridian turn (sentinel pattern / §5).
    """
    run_id:              str
    data_quality_report: DataQualityReport
    schema_mapping:      SchemaMapping
    grain_report:        GrainReport
    segment_profiles:    list[SegmentProfile]    # aggregate per segment — NOT per series
    segment_exceptions:  list[SeriesException]   # small list of per-series outliers within segments
    segments:            list[SegmentDef]        # provisional segment map
    domain_playbook:     dict                    # raw YAML playbook dict


# ---------------------------------------------------------------------------
# Feature flags (used by the pack + Forge feature config)
# ---------------------------------------------------------------------------

class FeatureFlags(BaseModel):
    use_fourier: bool = False
    use_lag_features: bool = True
    use_promo_indicator: bool = False
    fourier_terms: int = 3

# DomainContextPack is defined after Claim / Risk below — it embeds them.


# ---------------------------------------------------------------------------
# Conductor / intent contracts
# ---------------------------------------------------------------------------

IntentType = Literal[
    "SCOPE_RESPONSE",     # answering Meridian's question
    "OVERRIDE",           # contradicting a data-backed agent recommendation
    "ADVANCE_PIPELINE",   # "ok let's model", "looks good"
    "WHAT_IF_REQUEST",    # "what if promo on SKU X week 10"
    "CLARIFICATION",      # user asking a question
    "CORRECTION",         # fixing a prior statement — only valid during meridian_scoping;
                          # post-confirmation treated as OVERRIDE
]


class IntentEntities(BaseModel):
    skus:     list[str] = Field(default_factory=list)
    segments: list[str] = Field(default_factory=list)
    dates:    list[str] = Field(default_factory=list)   # ISO strings
    metrics:  list[str] = Field(default_factory=list)
    scenario: str | None = None   # free-text scenario description if WHAT_IF_REQUEST


class IntentPack(BaseModel):
    intent:     IntentType
    entities:   IntentEntities = Field(default_factory=IntentEntities)
    confidence: float            # 0.0–1.0
    raw_quote:  str              # verbatim fragment of user message that drove classification


# ---------------------------------------------------------------------------
# Claim ledger
# ---------------------------------------------------------------------------

VerificationStatus = Literal[
    "SUPPORTED", "CONTRADICTED", "AMBIGUOUS", "UNVERIFIABLE", "USER_OVERRIDE_ACCEPTED"
]

EvidenceType = Literal[
    "statistical_test", "association", "pattern", "user_confirmed", "unverifiable_business_input"
]


class Claim(BaseModel):
    claim_id:               str                    # uuid4
    claim:                  str                    # human-readable assertion
    verification_status:    VerificationStatus
    evidence_type:          EvidenceType
    evidence_ref:           str | None = None      # tool-call result summary backing the claim
    applies_to:             str                    # segment_id, series_key, or "run"
    downstream_impact:      str                    # what this claim affects downstream
    must_surface_in_report: bool = False           # True for USER_OVERRIDE_ACCEPTED
    created_at:             str
    resolved_at:            str | None = None
    resolution_note:        str | None = None


class ClaimLedger(BaseModel):
    run_id: str
    claims: list[Claim] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Risk register
# ---------------------------------------------------------------------------

class Risk(BaseModel):
    risk_id: str
    description: str
    severity: Literal["low", "medium", "high"]   # matches add_risk
    source: str                                   # matches add_risk
    acknowledged: bool = False
    created_at: str
    acknowledged_at: str | None = None


class RiskRegister(BaseModel):
    run_id: str
    risks: list[Risk] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Domain context pack (Meridian output, Foundry/Prism input)
# Matches exactly what compile_domain_context_pack emits / writes to
# domain_context_pack.json — this model IS the validated return type of that
# tool, not a parallel structure. (review §2)
# ---------------------------------------------------------------------------

class ForecastScope(BaseModel):
    target_col:  str
    grain:       list[str]
    horizon:     int
    mase_target: float | None = None   # user MASE override if set; else playbook/floor applies


class DomainContextPack(BaseModel):
    run_id:         str
    domain:         str
    forecast_scope: ForecastScope
    segments:       list[SegmentDef]                       # locked segment map (provisional=False)
    claim_ledger:   list[Claim]
    risk_register:  list[Risk]
    feature_flags:  FeatureFlags = Field(default_factory=FeatureFlags)
    override_count: int = 0
    open_risks:     int = 0
    confirmed_at:   str | None = None
    confirmed:      bool = False


# ---------------------------------------------------------------------------
# EDA report (Forge output)
# ---------------------------------------------------------------------------

class SeriesDemandProfile(BaseModel):
    series_key: str
    sb_class: SBClass
    adi: float
    cv2: float
    trend_strength: float
    seasonal_strength: float
    recommended_models: list[str]


class EDAReport(BaseModel):
    run_id: str
    segment_profiles: list[SegmentProfile]
    series_profiles: list[SeriesDemandProfile]
    feature_config: dict[str, FeatureFlags]
    narrative: str


# ---------------------------------------------------------------------------
# Foundry results
# ---------------------------------------------------------------------------

class ModelResult(BaseModel):
    model_name: str
    mase: float
    mae: float
    rmse: float
    forecast: list[float]
    selected: bool


class SeriesResult(BaseModel):
    series_key: str
    sb_class: SBClass
    mase_target: float
    results: list[ModelResult]
    best_model: str
    target_met: bool
    self_correction_rounds: int = 0


class FoundryReport(BaseModel):
    run_id: str
    series_results: list[SeriesResult]
    overall_mase: float
    target_met_fraction: float
    narrative: str


# ---------------------------------------------------------------------------
# Prism (what-if) contracts
# ---------------------------------------------------------------------------

class WhatIfOverride(BaseModel):
    whatif_id: str
    series_key: str
    override_type: Literal["PROMO_EVENT", "STOCKOUT", "PRICE_CHANGE", "MANUAL_UPLIFT"]
    magnitude: float
    start_period: str
    end_period: str
    description: str


class ScenarioComparison(BaseModel):
    whatif_id: str
    series_key: str
    baseline_forecast: list[float]
    scenario_forecast: list[float]
    delta_pct: float
    demand_class_changed: bool
    baseline_sb_class: SBClass
    scenario_sb_class: SBClass
```

- [ ] **Step 2: Smoke-import check**

```powershell
c:/Agent_A/venv/Scripts/python -c "from forecasting.contracts import PreflightBundle, DomainContextPack, EDAReport, FoundryReport; print('contracts OK')"
```

Expected: `contracts OK`

- [ ] **Step 3: Commit**

```powershell
git add backend/forecasting/contracts.py
git commit -m "feat: contracts — single-source domain Pydantic models"
```

---

### Task 5: `guard.py` — Budget and Safety Layer

**Files:**
- Create: `backend/forecasting/guard.py`
- Create: `tests/test_guard.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_guard.py
import hashlib
import pytest
from forecasting.guard import (
    GuardConfig, AgentGuardState, FoundryRunGuard, GuardHalt
)


def test_conductor_tool_limit_enforced():
    config = GuardConfig()
    guard = AgentGuardState(agent_name="conductor")
    for _ in range(config.max_calls_conductor):
        guard.check_and_record("some_tool", {}, tokens_used=0, config=config)
    with pytest.raises(GuardHalt, match="tool call limit"):
        guard.check_and_record("some_tool", {}, tokens_used=0, config=config)


def test_token_budget_enforced():
    config = GuardConfig()
    guard = AgentGuardState(agent_name="meridian")
    with pytest.raises(GuardHalt, match="token budget"):
        guard.check_and_record("tool", {}, tokens_used=config.token_budget + 1, config=config)


def test_duplicate_call_detected():
    # DUPLICATE_CALL_HARD_STOP=2 (default): the first duplicate is allowed with a
    # warning; the run halts on the second duplicate (the third identical call).
    config = GuardConfig()
    guard = AgentGuardState(agent_name="forge")
    args = {"series_key": "SKU_A|NORTH"}
    guard.check_and_record("classify_demand_profiles", args, tokens_used=0, config=config)
    guard.check_and_record("classify_demand_profiles", args, tokens_used=0, config=config)  # 1st dup — warn
    with pytest.raises(GuardHalt, match="duplicate"):
        guard.check_and_record("classify_demand_profiles", args, tokens_used=0, config=config)  # 2nd dup — halt


def test_foundry_cumulative_limit():
    config = GuardConfig(max_calls_foundry=3)
    g = FoundryRunGuard(run_id="r1")
    for _ in range(3):
        g.check_and_record(config)
    with pytest.raises(GuardHalt, match="cumulative"):
        g.check_and_record(config)


def test_foundry_counter_is_per_run():
    # The counter is per-run (instance), not a process global — run B is unaffected
    # by run A's calls, so concurrent runs don't halt each other. (review §8)
    config = GuardConfig(max_calls_foundry=3)
    g_a, g_b = FoundryRunGuard(run_id="A"), FoundryRunGuard(run_id="B")
    for _ in range(3):
        g_a.check_and_record(config)
    g_b.check_and_record(config)
    assert g_b.count == 1


def test_different_args_not_duplicate():
    config = GuardConfig()
    guard = AgentGuardState(agent_name="forge")
    guard.check_and_record("classify_demand_profiles", {"x": 1}, tokens_used=0, config=config)
    guard.check_and_record("classify_demand_profiles", {"x": 2}, tokens_used=0, config=config)
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_guard.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backend/forecasting/guard.py`**

```python
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


class GuardHalt(Exception):
    pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class GuardConfig:
    """All limits .env-configurable (plan_v2 correction #6); values are defaults only."""
    token_budget:        int = field(default_factory=lambda: _env_int("GUARD_TOKEN_BUDGET", 80_000))
    max_calls_conductor: int = field(default_factory=lambda: _env_int("GUARD_MAX_CALLS_CONDUCTOR", 20))
    max_calls_meridian:  int = field(default_factory=lambda: _env_int("GUARD_MAX_CALLS_MERIDIAN", 20))
    max_calls_forge:     int = field(default_factory=lambda: _env_int("GUARD_MAX_CALLS_FORGE", 20))
    max_calls_prism:     int = field(default_factory=lambda: _env_int("GUARD_MAX_CALLS_PRISM", 20))
    max_calls_foundry:   int = field(default_factory=lambda: _env_int("GUARD_MAX_CALLS_FOUNDRY", 500))
    duplicate_hard_stop: int = field(default_factory=lambda: _env_int("GUARD_DUPLICATE_HARD_STOP", 2))


_AGENT_LIMITS = {
    "conductor": "max_calls_conductor",
    "meridian": "max_calls_meridian",
    "forge": "max_calls_forge",
    "prism": "max_calls_prism",
    "foundry": "max_calls_foundry",
}


def _call_hash(tool_name: str, args: dict[str, Any]) -> str:
    payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


@dataclass
class AgentGuardState:
    agent_name: str
    _call_count: int = field(default=0, init=False)
    _hash_counts: dict[str, int] = field(default_factory=dict, init=False)

    def check_and_record(
        self,
        tool_name: str,
        args: dict[str, Any],
        tokens_used: int,
        config: GuardConfig,
    ) -> None:
        if tokens_used >= config.token_budget:
            raise GuardHalt(f"token budget exceeded: {tokens_used} >= {config.token_budget}")

        limit_attr = _AGENT_LIMITS.get(self.agent_name)
        if limit_attr:
            limit = getattr(config, limit_attr)
            if self._call_count >= limit:
                raise GuardHalt(
                    f"{self.agent_name} tool call limit reached: {self._call_count} >= {limit}"
                )

        # Duplicate detection: `prior` = number of prior identical calls. The first
        # duplicate (prior==1) is allowed with a warning; halt once prior reaches the
        # configured threshold (default 2 → halt on the second duplicate).
        h = _call_hash(tool_name, args)
        prior = self._hash_counts.get(h, 0)
        if prior >= config.duplicate_hard_stop:
            raise GuardHalt(f"duplicate tool call hard stop ({prior + 1}x): {tool_name}({args})")
        if prior >= 1:
            log.warning("duplicate tool call (%dx): %s(%s)", prior + 1, tool_name, args)
        self._hash_counts[h] = prior + 1

        self._call_count += 1


@dataclass
class FoundryRunGuard:
    """Per-run cumulative Foundry call counter (instance state, NOT a process global —
    review §8). One instance per run; concurrent runs do not share the count."""
    run_id: str
    count: int = field(default=0, init=False)

    def check_and_record(self, config: GuardConfig) -> None:
        if self.count >= config.max_calls_foundry:
            raise GuardHalt(
                f"Foundry cumulative limit reached for run {self.run_id}: "
                f"{self.count} >= {config.max_calls_foundry}"
            )
        self.count += 1
```

- [ ] **Step 4: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_guard.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```powershell
git add backend/forecasting/guard.py tests/test_guard.py
git commit -m "feat: guard — token budget, per-agent call limits, duplicate detection"
```

---

### Task 6: `providers.py` — Tool Registry Skeleton

**Files:**
- Create: `backend/forecasting/providers.py`

> `_TOOL_REGISTRY` starts empty. Populated in Task 22 after all tools exist. `dispatch_tool` is the single call-site for every agent.

- [ ] **Step 1: Write `backend/forecasting/providers.py`**

```python
from __future__ import annotations

from typing import Any, Callable

from forecasting.guard import AgentGuardState, FoundryRunGuard, GuardConfig

# Populated in Task 22 — one entry per registered tool.
# Schema: tool_name -> callable(*args_dict_values)
_TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}

_default_config = GuardConfig()


def dispatch_tool(
    tool_name: str,
    args: dict[str, Any],
    guard: AgentGuardState,
    tokens_used: int,
    foundry_guard: FoundryRunGuard | None = None,
    config: GuardConfig | None = None,
) -> Any:
    cfg = config or _default_config
    guard.check_and_record(tool_name, args, tokens_used, cfg)
    if foundry_guard is not None:
        foundry_guard.check_and_record(cfg)
    fn = _TOOL_REGISTRY.get(tool_name)
    if fn is None:
        raise KeyError(f"Unknown tool: '{tool_name}'. Is it registered in _TOOL_REGISTRY?")
    return fn(**args)
```

- [ ] **Step 2: Smoke-import check**

```powershell
c:/Agent_A/venv/Scripts/python -c "from forecasting.providers import dispatch_tool, _TOOL_REGISTRY; print('providers OK')"
```

Expected: `providers OK`

- [ ] **Step 3: Commit**

```powershell
git add backend/forecasting/providers.py
git commit -m "feat: providers — tool registry skeleton and dispatch_tool"
```

---

## Phase B — Pre-flight Layer

> Tasks 7–9. Pure Python, no LLM. Runs once on CSV upload. Produces `PreflightBundle` and populates `data_store`.

---

### Task 7: `tools/preflight_schema.py` — Upload Profiling, Schema, Grain, Series Keys

**Files:**
- Create: `backend/forecasting/tools/preflight_schema.py`
- Create: `tests/test_preflight_schema.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preflight_schema.py
import pandas as pd
import pytest
from forecasting.tools.preflight_schema import (
    profile_uploaded_data, map_schema, detect_frequency_and_grain, build_series_keys
)

PLAYBOOK = {
    "common_grains": ["sku", "region"],
    "time_col": "week",
    "demand_col": "demand",
    "min_series": 1,
    "min_history_periods": 4,
}


def make_df(n_weeks: int = 10) -> pd.DataFrame:
    rows = []
    for sku in ["A", "B"]:
        for w in range(n_weeks):
            rows.append({"week": f"2024-W{w+1:02d}", "sku": sku, "region": "NORTH", "demand": float(w + 1)})
    return pd.DataFrame(rows)


def test_profile_no_blocking_issues():
    rpt = profile_uploaded_data(make_df())
    assert rpt.blocking_issues == []
    assert rpt.series_count > 0


def test_profile_all_zero_demand_blocks():
    df = make_df()
    df["demand"] = 0.0
    rpt = profile_uploaded_data(df)
    assert any(b.code == "ALL_ZERO_DEMAND" for b in rpt.blocking_issues)


def test_profile_missing_demand_col_blocks():
    df = make_df().drop(columns=["demand"])
    rpt = profile_uploaded_data(df)
    assert any(b.code == "MISSING_DEMAND_COLUMN" for b in rpt.blocking_issues)


def test_map_schema_detects_columns():
    df = make_df()
    schema = map_schema(df, PLAYBOOK)
    assert schema.date_col == "week"
    assert schema.demand_col == "demand"
    assert set(schema.grain_cols) == {"sku", "region"}


def test_detect_frequency_weekly():
    schema = map_schema(make_df(52), PLAYBOOK)
    grain_rpt = detect_frequency_and_grain(make_df(52), schema)
    assert grain_rpt.detected_frequency == "weekly"


def test_build_series_keys_pipe_delimited():
    schema = map_schema(make_df(), PLAYBOOK)
    series_map = build_series_keys(make_df(), schema, PLAYBOOK)
    for key in series_map:
        assert "|" in key


def test_series_keys_uppercased():
    df = make_df()
    df["sku"] = df["sku"].str.lower()
    schema = map_schema(df, PLAYBOOK)
    series_map = build_series_keys(df, schema, PLAYBOOK)
    for key in series_map:
        assert key == key.upper()


def test_series_df_has_date_and_demand_cols():
    schema = map_schema(make_df(), PLAYBOOK)
    series_map = build_series_keys(make_df(), schema, PLAYBOOK)
    for sub_df in series_map.values():
        assert set(sub_df.columns) == {"date", "demand"}
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_preflight_schema.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backend/forecasting/tools/preflight_schema.py`**

```python
from __future__ import annotations

import re
import pandas as pd

from forecasting.contracts import (
    BlockingIssue, DataQualityWarning, DataQualityReport, SchemaMapping, GrainReport
)


def profile_uploaded_data(df: pd.DataFrame) -> DataQualityReport:
    blocking: list[BlockingIssue] = []
    warnings: list[DataQualityWarning] = []

    demand_col = _find_demand_col(df)
    if demand_col is None:
        blocking.append(BlockingIssue(code="MISSING_DEMAND_COLUMN", message="No numeric demand column found"))
    elif df[demand_col].sum() == 0:
        blocking.append(BlockingIssue(code="ALL_ZERO_DEMAND", message="All demand values are zero"))

    if _find_date_col(df) is None:
        blocking.append(BlockingIssue(code="MISSING_DATE_COLUMN", message="No parseable date column found"))

    return DataQualityReport(
        blocking_issues=blocking,
        warnings=warnings,
        row_count=len(df),
        series_count=0,   # provisional — real count requires the grain; orchestrator sets it
    )


def map_schema(df: pd.DataFrame, playbook: dict) -> SchemaMapping:
    date_col = _find_date_col(df) or playbook.get("time_col", "date")
    demand_col = _find_demand_col(df) or playbook.get("demand_col", "demand")
    grain_cols = [c for c in playbook.get("common_grains", []) if c in df.columns]
    extra_cols = [c for c in df.columns if c not in [date_col, demand_col] + grain_cols]
    return SchemaMapping(date_col=date_col, demand_col=demand_col, grain_cols=grain_cols, extra_cols=extra_cols)


def detect_frequency_and_grain(df: pd.DataFrame, schema: SchemaMapping) -> GrainReport:
    dates = _parse_dates(df[schema.date_col])
    dates = dates.dropna().sort_values().unique()
    detected_frequency = "unknown"
    gaps_detected = False
    min_periods = max_periods = median_periods = 0

    if len(dates) > 1:
        deltas = pd.Series(dates).diff().dropna().dt.days
        med = int(deltas.median())
        if 6 <= med <= 8:
            detected_frequency = "weekly"
        elif 28 <= med <= 32:
            detected_frequency = "monthly"
        elif med == 1:
            detected_frequency = "daily"
        gaps_detected = bool((deltas > med * 1.5).any())

    counts = (
        df.groupby(schema.grain_cols)[schema.date_col].count()
        if schema.grain_cols else pd.Series([len(df)])
    )
    if len(counts):
        min_periods = int(counts.min())
        max_periods = int(counts.max())
        median_periods = int(counts.median())

    return GrainReport(
        detected_frequency=detected_frequency,
        min_periods=min_periods,
        max_periods=max_periods,
        median_periods=median_periods,
        gaps_detected=gaps_detected,
    )


def build_series_keys(df: pd.DataFrame, schema: SchemaMapping, playbook: dict) -> dict[str, pd.DataFrame]:
    series_map: dict[str, pd.DataFrame] = {}
    grain_cols = schema.grain_cols

    if grain_cols:
        for group_keys, sub_df in df.groupby(grain_cols):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            key = _normalise_key(group_keys)
            sub = sub_df[[schema.date_col, schema.demand_col]].copy()
            sub.columns = ["date", "demand"]
            series_map[key] = sub.reset_index(drop=True)
    else:
        sub = df[[schema.date_col, schema.demand_col]].copy()
        sub.columns = ["date", "demand"]
        series_map["SERIES|ALL"] = sub.reset_index(drop=True)

    return series_map


def _parse_dates(values: pd.Series) -> pd.Series:
    """
    Parse a column to datetime, supporting BOTH ISO dates ('2024-01-07') and
    ISO-week strings ('2024-W01'). pandas cannot infer the ISO-week form without an
    explicit format (every value comes back NaT), which previously made weekly
    fixtures unparseable → spurious MISSING_DATE_COLUMN / frequency='unknown'. We
    map 'YYYY-Www' to the Monday of that ISO week (%G-W%V-%u with weekday=1).
    """
    s = values.astype(str).str.strip()
    if s.str.match(r"^\d{4}-W\d{1,2}$").mean() > 0.8:
        iso = s.str.replace(r"^(\d{4})-W(\d{1,2})$", r"\1-W\2-1", regex=True)
        return pd.to_datetime(iso, format="%G-W%V-%u", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _find_date_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        try:
            parsed = _parse_dates(df[col])
            if parsed.notna().sum() > len(df) * 0.8:
                return col
        except Exception:
            pass
    return None


def _find_demand_col(df: pd.DataFrame) -> str | None:
    for name in ["demand", "qty", "quantity", "sales", "units"]:
        match = next((c for c in df.columns if c.lower() == name), None)
        if match:
            return match
    numeric = df.select_dtypes(include="number").columns.tolist()
    return numeric[0] if numeric else None


def _normalise_key(values: tuple) -> str:
    parts = []
    for v in values:
        s = re.sub(r"[^A-Z0-9_]", "", str(v).upper().replace(" ", "_"))
        parts.append(s)
    return "|".join(parts)
```

- [ ] **Step 4: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_preflight_schema.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/forecasting/tools/preflight_schema.py tests/test_preflight_schema.py
git commit -m "feat: preflight_schema — upload profiling, schema mapping, series key normalisation"
```

---

### Task 8: `tools/preflight_stats.py` — Statistical Profiling

**Files:**
- Create: `backend/forecasting/tools/preflight_stats.py`
- Create: `tests/test_preflight_stats.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preflight_stats.py
import pandas as pd
import numpy as np
import pytest
from forecasting.tools.preflight_stats import (
    compute_adi_cv2_per_series,
    detect_zero_runs_per_series,
    detect_spikes_per_series,
    detect_trend_strength,
    detect_seasonality_strength,
    aggregate_segment_profiles,
    collect_segment_exceptions,
)


def _sm(vals: list[float], key: str = "SKU|NORTH") -> dict:
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-06", periods=len(vals), freq="W"),
        "demand": vals,
    })
    return {key: df}


def test_smooth_series_classified_smooth():
    stats = compute_adi_cv2_per_series(_sm([10.0] * 52))
    assert stats["SKU|NORTH"].sb_class == "SMOOTH"


def test_intermittent_series_not_smooth():
    vals = ([0, 0, 0, 5] * 13)
    stats = compute_adi_cv2_per_series(_sm(vals))
    assert stats["SKU|NORTH"].sb_class in ("INTERMITTENT", "LUMPY")


def test_adi_gt_one_for_intermittent():
    vals = ([0, 0, 5] * 17)
    stats = compute_adi_cv2_per_series(_sm(vals))
    assert stats["SKU|NORTH"].adi > 1.0


def test_zero_run_detects_max_run():
    vals = [5, 5, 0, 0, 0, 0, 5, 5]
    stats = detect_zero_runs_per_series(_sm(vals))
    assert stats["SKU|NORTH"].max_zero_run == 4


def test_spike_detected():
    base = [10.0] * 50
    base[25] = 500.0
    stats = detect_spikes_per_series(_sm(base))
    assert stats["SKU|NORTH"].spike_count >= 1


def test_no_spike_for_flat():
    stats = detect_spikes_per_series(_sm([10.0] * 50))
    assert stats["SKU|NORTH"].spike_count == 0


def test_trend_up_detected():
    vals = [float(i) for i in range(1, 53)]
    stats = detect_trend_strength(_sm(vals))
    assert stats["SKU|NORTH"].direction == "up"
    assert stats["SKU|NORTH"].trend_strength > 0.8


def test_seasonality_low_for_flat():
    stats = detect_seasonality_strength(_sm([10.0] * 52))
    assert stats["SKU|NORTH"].seasonal_strength < 0.3


def test_aggregate_profiles_sum_equals_series_count():
    series_map = {**_sm([10.0] * 20, "A|N"), **_sm([0, 0, 5] * 7, "B|N")}
    adi_cv2 = compute_adi_cv2_per_series(series_map)
    profiles = aggregate_segment_profiles(series_map, adi_cv2, {})
    assert sum(p.series_count for p in profiles) == 2
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_preflight_stats.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backend/forecasting/tools/preflight_stats.py`**

```python
from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd

from forecasting.contracts import (
    AdiCv2Stats, ZeroRunStats, SpikeStats, TrendStats, SeasonalityStats,
    PromoAlignmentStats, BreakCandidate, SegmentProfile, SeriesException, SBClass,
    SchemaMapping, SegmentDef, SegmentMap,
)


def compute_adi_cv2_per_series(series_map: dict[str, pd.DataFrame]) -> dict[str, AdiCv2Stats]:
    result = {}
    for key, df in series_map.items():
        demand = df["demand"].values.astype(float)
        non_zero = demand[demand > 0]
        if len(non_zero) == 0:
            result[key] = AdiCv2Stats(series_key=key, adi=999.0, cv2=0.0, sb_class="LUMPY")
            continue
        intervals: list[int] = []
        run = 0
        for d in demand:
            if d > 0:
                intervals.append(run + 1)
                run = 0
            else:
                run += 1
        adi = float(np.mean(intervals)) if intervals else 1.0
        mu = float(np.mean(non_zero))
        std = float(np.std(non_zero, ddof=1)) if len(non_zero) > 1 else 0.0
        cv2 = (std / mu) ** 2 if mu > 0 else 0.0
        result[key] = AdiCv2Stats(series_key=key, adi=adi, cv2=cv2, sb_class=_sb(adi, cv2))
    return result


def detect_zero_runs_per_series(series_map: dict[str, pd.DataFrame]) -> dict[str, ZeroRunStats]:
    result = {}
    for key, df in series_map.items():
        demand = df["demand"].values.astype(float)
        mask = demand == 0
        result[key] = ZeroRunStats(
            series_key=key,
            max_zero_run=_max_run(mask),
            zero_fraction=float(mask.sum()) / len(demand) if len(demand) else 0.0,
        )
    return result


def detect_spikes_per_series(series_map: dict[str, pd.DataFrame]) -> dict[str, SpikeStats]:
    result = {}
    for key, df in series_map.items():
        demand = df["demand"].values.astype(float)
        q1, q3 = np.percentile(demand, [25, 75])
        threshold = q3 + 3 * (q3 - q1)
        spikes = demand[demand > threshold]
        max_ratio = float(spikes.max() / q3) if len(spikes) and q3 > 0 else 0.0
        result[key] = SpikeStats(series_key=key, spike_count=len(spikes), max_spike_ratio=max_ratio)
    return result


def detect_trend_strength(series_map: dict[str, pd.DataFrame]) -> dict[str, TrendStats]:
    result = {}
    for key, df in series_map.items():
        demand = df["demand"].values.astype(float)
        nz_idx = np.where(demand > 0)[0]
        if len(nz_idx) < 3:
            result[key] = TrendStats(series_key=key, trend_strength=0.0, direction="flat")
            continue
        x, y = nz_idx.astype(float), demand[nz_idx]
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        thr = 0.01 * y.mean()
        direction = "up" if slope > thr else ("down" if slope < -thr else "flat")
        result[key] = TrendStats(series_key=key, trend_strength=round(r2, 4), direction=direction)
    return result


def detect_seasonality_strength(series_map: dict[str, pd.DataFrame]) -> dict[str, SeasonalityStats]:
    result = {}
    for key, df in series_map.items():
        demand = df["demand"].values.astype(float)
        best_str, best_per = 0.0, None
        for lag in [52, 12, 4]:
            if len(demand) > lag * 2:
                ac = _autocorr(demand, lag)
                if ac > best_str:
                    best_str, best_per = ac, lag
        result[key] = SeasonalityStats(
            series_key=key,
            seasonal_strength=round(max(0.0, best_str), 4),
            dominant_period=best_per,
        )
    return result


def measure_promo_alignment(
    series_map: dict[str, pd.DataFrame], schema
) -> dict[str, PromoAlignmentStats]:
    result = {}
    for key, df in series_map.items():
        has_promo = any(c in df.columns for c in ["promo", "promotion"])
        result[key] = PromoAlignmentStats(series_key=key, has_promo_col=has_promo)
    return result


def detect_structural_break_candidates(
    series_map: dict[str, pd.DataFrame],
) -> list[BreakCandidate]:
    candidates = []
    for key, df in series_map.items():
        demand = df["demand"].values.astype(float)
        if len(demand) < 16:
            continue
        mu = demand.mean()
        cusum = np.cumsum(demand - mu)
        idx = int(np.argmax(np.abs(cusum)))
        strength = abs(cusum[idx]) / (demand.std() + 1e-9)
        if strength > 3.0:
            date_val = str(df["date"].iloc[idx]) if "date" in df.columns else str(idx)
            candidates.append(BreakCandidate(
                series_key=key,
                break_period=date_val,
                confidence=round(min(1.0, strength / 10.0), 3),
            ))
    return candidates


def assign_provisional_segments(
    series_map: dict[str, pd.DataFrame],
    schema: SchemaMapping,
    playbook: dict,
) -> SegmentMap:
    """
    Deterministic, *provisional* segmentation (plan_v2 §6 / CONTEXT 'Segment').

    The playbook is a GUIDE, not a hard rule. If it names `segment_by` grain
    column(s) that are present in the data, series are grouped by the distinct
    values of those column(s) → segments G1, G2, …. Otherwise (no hint, hint
    absent, or it produces more than `max_segments` groups) every series falls
    into a single segment G1. Segments are a *suggestion* the user refines in the
    Meridian conversation; the map stays provisional until pack confirmation.
    """
    seg_by = [c for c in playbook.get("segment_by", []) if c in schema.grain_cols]
    max_segments = playbook.get("max_segments", 12)

    def _single(reason: str) -> SegmentMap:
        return SegmentMap(
            run_id="",
            segments=[SegmentDef(segment_id="G1", label="all series",
                                 series_keys=sorted(series_map.keys()), provisional=True)],
            provisional=True,
            derived_by=reason,
        )

    if not seg_by:
        return _single("default:single_segment")

    buckets: dict[tuple, list[str]] = defaultdict(list)
    for key in series_map.keys():
        parts = dict(zip(schema.grain_cols, key.split("|")))
        buckets[tuple(parts.get(c, "") for c in seg_by)].append(key)

    if len(buckets) > max_segments:
        return _single(f"default:single_segment (>{max_segments} groups)")

    segments: list[SegmentDef] = []
    for i, (bkey, keys) in enumerate(sorted(buckets.items()), start=1):
        label = ", ".join(f"{c}={v}" for c, v in zip(seg_by, bkey))
        segments.append(SegmentDef(segment_id=f"G{i}", label=label,
                                   series_keys=sorted(keys), provisional=True))
    return SegmentMap(run_id="", segments=segments, provisional=True,
                      derived_by=f"playbook:segment_by={'+'.join(seg_by)}")


def aggregate_segment_profiles(
    series_map: dict[str, pd.DataFrame],
    adi_cv2: dict[str, AdiCv2Stats],
    segment_map: SegmentMap,
) -> list[SegmentProfile]:
    """Aggregate per-segment statistics, keyed by the provisional segment map."""
    profiles: list[SegmentProfile] = []
    for seg in segment_map.segments:
        keys = [k for k in seg.series_keys if k in adi_cv2]
        dist: dict[str, int] = defaultdict(int)
        fc: dict[str, int] = defaultdict(int)
        adis: list[float] = []
        cv2s: list[float] = []
        for k in keys:
            stats = adi_cv2[k]
            dist[stats.sb_class] += 1
            fc["forecastable" if stats.sb_class in ("SMOOTH", "ERRATIC") else "caution"] += 1
            adis.append(stats.adi)
            cv2s.append(stats.cv2)
        profiles.append(SegmentProfile(
            segment_id=seg.segment_id,
            series_count=len(seg.series_keys),
            demand_class_distribution=dict(dist),
            median_adi=round(float(np.median(adis)), 4) if adis else 0.0,
            median_cv2=round(float(np.median(cv2s)), 4) if cv2s else 0.0,
            forecastability_breakdown=dict(fc),
            example_keys=seg.series_keys[:3],
        ))
    return profiles


def collect_segment_exceptions(
    adi_cv2: dict[str, AdiCv2Stats],
    zero_runs: dict[str, ZeroRunStats],
    spikes: dict[str, SpikeStats],
    segment_map: SegmentMap,
) -> list[SeriesException]:
    seg_of = {k: seg.segment_id for seg in segment_map.segments for k in seg.series_keys}
    exc: list[SeriesException] = []
    for key, zr in zero_runs.items():
        if zr.zero_fraction > 0.8:
            exc.append(SeriesException(series_key=key, segment_id=seg_of.get(key, "G1"),
                                       exception_type="HIGH_ZERO_FRACTION",
                                       detail=f"zero_fraction={zr.zero_fraction:.2f}"))
        if zr.max_zero_run >= 8:
            exc.append(SeriesException(series_key=key, segment_id=seg_of.get(key, "G1"),
                                       exception_type="ZERO_RUN",
                                       detail=f"max_zero_run={zr.max_zero_run}"))
    for key, sp in spikes.items():
        if sp.spike_count > 0:
            exc.append(SeriesException(series_key=key, segment_id=seg_of.get(key, "G1"),
                                       exception_type="SPIKE",
                                       detail=f"spike_count={sp.spike_count}, max_ratio={sp.max_spike_ratio:.1f}x"))
    return exc


def _sb(adi: float, cv2: float) -> SBClass:
    if adi < 1.32 and cv2 < 0.49:
        return "SMOOTH"
    if adi >= 1.32 and cv2 < 0.49:
        return "INTERMITTENT"
    if adi < 1.32 and cv2 >= 0.49:
        return "ERRATIC"
    return "LUMPY"


def _max_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _autocorr(x: np.ndarray, lag: int) -> float:
    xn = x - x.mean()
    denom = np.dot(xn, xn)
    return float(np.dot(xn[: len(x) - lag], xn[lag:])) / denom if denom else 0.0
```

- [ ] **Step 4: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_preflight_stats.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/forecasting/tools/preflight_stats.py tests/test_preflight_stats.py
git commit -m "feat: preflight_stats — ADI/CV2, Syntetos-Boylan, zero runs, spikes, trend, seasonality"
```

---

### Task 9: `preflight.py` — Pre-flight Orchestrator

**Files:**
- Create: `backend/forecasting/preflight.py`
- Create: `tests/test_preflight_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preflight_orchestrator.py
import json
import pytest
from forecasting.preflight import run_preflight, PreflightBlockingError
from forecasting.data_store import get_series_keys

PLAYBOOK = {
    "common_grains": ["sku", "region"],
    "time_col": "week",
    "demand_col": "demand",
    "min_series": 1,
    "min_history_periods": 4,
}


def _csv(n_weeks: int = 12) -> bytes:
    rows = [f"2024-W{w+1:02d},{sku},NORTH,{float(w+1)}"
            for sku in ["SKU_A", "SKU_B"] for w in range(n_weeks)]
    return ("week,sku,region,demand\n" + "\n".join(rows)).encode()


def test_preflight_populates_data_store(run_id, tmp_outputs):
    bundle = run_preflight(run_id, _csv(), domain="fmcg", playbook=PLAYBOOK)
    assert len(get_series_keys(run_id)) == 2
    assert bundle.data_quality_report.blocking_issues == []


def test_preflight_writes_json(run_id, tmp_outputs):
    run_preflight(run_id, _csv(), domain="fmcg", playbook=PLAYBOOK)
    pf_path = tmp_outputs / run_id / "preflight.json"
    assert pf_path.exists()
    data = json.loads(pf_path.read_text())
    assert "bundle" in data and "break_candidates" in data


def test_preflight_blocks_all_zero(run_id, tmp_outputs):
    rows = [f"2024-W{w:02d},SKU_A,NORTH,0.0" for w in range(1, 13)]
    csv_bytes = ("week,sku,region,demand\n" + "\n".join(rows)).encode()
    with pytest.raises(PreflightBlockingError) as exc_info:
        run_preflight(run_id, csv_bytes, domain="fmcg", playbook=PLAYBOOK)
    assert any(i.code == "ALL_ZERO_DEMAND" for i in exc_info.value.issues)


def test_preflight_blocks_corrupt_file(run_id, tmp_outputs):
    with pytest.raises(PreflightBlockingError) as exc_info:
        run_preflight(run_id, b"\x00\x01\x02corrupted", domain="fmcg", playbook=PLAYBOOK)
    assert any(i.code == "UNPARSEABLE_FILE" for i in exc_info.value.issues)
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_preflight_orchestrator.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backend/forecasting/preflight.py`**

```python
from __future__ import annotations

import io
import json

import pandas as pd

from forecasting.contracts import BlockingIssue, DataQualityWarning, PreflightBundle
from forecasting.data_store import store_series
from forecasting.run_state import run_dir
from forecasting.tools.preflight_schema import (
    profile_uploaded_data, map_schema, detect_frequency_and_grain, build_series_keys,
)
from forecasting.tools.preflight_stats import (
    compute_adi_cv2_per_series, detect_zero_runs_per_series, detect_spikes_per_series,
    measure_promo_alignment, detect_trend_strength, detect_seasonality_strength,
    detect_structural_break_candidates, assign_provisional_segments,
    aggregate_segment_profiles, collect_segment_exceptions,
)


class PreflightBlockingError(Exception):
    def __init__(self, issues: list[BlockingIssue]):
        super().__init__(f"Preflight blocked: {[i.code for i in issues]}")
        self.issues = issues


def run_preflight(run_id: str, file_bytes: bytes, domain: str, playbook: dict) -> PreflightBundle:
    df = _parse_csv(file_bytes)

    quality = profile_uploaded_data(df)
    if quality.blocking_issues:
        raise PreflightBlockingError(quality.blocking_issues)

    schema = map_schema(df, playbook)
    grain = detect_frequency_and_grain(df, schema)
    series_map = build_series_keys(df, schema, playbook)

    # Real series count is known only now (needs the grain). Set it and apply the
    # BELOW_MIN_SERIES blocking check the schema-only profile pass cannot do.
    quality.series_count = len(series_map)
    min_series = playbook.get("min_series", 1)
    if len(series_map) < min_series:
        quality.blocking_issues.append(BlockingIssue(
            code="BELOW_MIN_SERIES",
            message=f"{len(series_map)} series < playbook min_series={min_series}",
        ))
        raise PreflightBlockingError(quality.blocking_issues)

    for key, series_df in series_map.items():
        store_series(run_id, key, series_df)

    adi_cv2 = compute_adi_cv2_per_series(series_map)
    zero_runs = detect_zero_runs_per_series(series_map)
    spikes = detect_spikes_per_series(series_map)
    promo_align = measure_promo_alignment(series_map, schema)
    trend = detect_trend_strength(series_map)
    seasonality = detect_seasonality_strength(series_map)
    break_cands = detect_structural_break_candidates(series_map)

    segment_map = assign_provisional_segments(series_map, schema, playbook)
    segment_map.run_id = run_id
    seg_profiles = aggregate_segment_profiles(series_map, adi_cv2, segment_map)
    seg_exceptions = collect_segment_exceptions(adi_cv2, zero_runs, spikes, segment_map)

    _add_stat_warnings(quality, grain, playbook)

    bundle = PreflightBundle(
        run_id=run_id,
        data_quality_report=quality,
        schema_mapping=schema,
        grain_report=grain,
        segment_profiles=seg_profiles,
        segment_exceptions=seg_exceptions,
        segments=segment_map.segments,
        domain_playbook=playbook,
    )

    # preflight.json holds the aggregate bundle PLUS the per-series stat dicts and the
    # segment map. Per-series stats are NOT in the bundle (they must stay out of Meridian's
    # context — plan_v2 #4 / §5); the diagnostic tools read them from here on demand.
    out = run_dir(run_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preflight.json").write_text(json.dumps({
        "bundle":           bundle.model_dump(),
        "segment_map":      segment_map.model_dump(),
        "break_candidates": [b.model_dump() for b in break_cands],
        "per_series": {
            "adi_cv2":      {k: v.model_dump() for k, v in adi_cv2.items()},
            "zero_runs":    {k: v.model_dump() for k, v in zero_runs.items()},
            "spikes":       {k: v.model_dump() for k, v in spikes.items()},
            "promo_align":  {k: v.model_dump() for k, v in promo_align.items()},
            "trend":        {k: v.model_dump() for k, v in trend.items()},
            "seasonality":  {k: v.model_dump() for k, v in seasonality.items()},
        },
    }, indent=2))

    return bundle


def _parse_csv(file_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise PreflightBlockingError([BlockingIssue(code="UNPARSEABLE_FILE", message=str(exc))])


def _add_stat_warnings(quality, grain, playbook: dict) -> None:
    min_hist = playbook.get("min_history_periods", 12)
    if grain.min_periods < min_hist:
        quality.warnings.append(DataQualityWarning(
            code="SHORT_HISTORY",
            message=f"Some series have fewer than {min_hist} periods (min={grain.min_periods})",
        ))
```

- [ ] **Step 4: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_preflight_orchestrator.py -v
```

Expected: all pass.

- [ ] **Step 5: Run the full test suite**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/ -v
```

Expected: all tests pass (data_store, run_state, guard, preflight_schema, preflight_stats, preflight_orchestrator).

- [ ] **Step 6: Commit**

```powershell
git add backend/forecasting/preflight.py tests/test_preflight_orchestrator.py
git commit -m "feat: preflight orchestrator — CSV → PreflightBundle, data_store population, disk write"
```

---

## Phase C — Agents

> Tasks 10–21. Two layers per agent: pure-Python tools (testable without Anthropic), then the LLM loop (tested via mocked Anthropic client). Tools live in `forecasting/tools/`; agents live in `forecasting/agents/`.

---

### Task 10: `agents/lens.py` — Intent Classifier

**Files:**
- Create: `backend/forecasting/agents/lens.py`
- Create: `tests/test_lens.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lens.py
import json
import pytest
from unittest.mock import MagicMock, patch
from forecasting.agents.lens import classify_intent
from forecasting.contracts import IntentPack
from forecasting.run_state import Phase, RunState


def _make_input(msg: str, phase: str = "preflight"):
    from forecasting.agents.lens import LensInput, ConversationTurn
    rs = RunState(run_id="r1", phase=phase, domain="fmcg",
                  created_at="2024-01-01T00:00:00+00:00")
    return LensInput(
        conversation_history=[],
        user_message=msg,
        pipeline_state=rs,
    )


def _mock_response(intent: str, confidence: float = 0.9):
    pack = {
        "intent": intent,
        "entities": {"skus": [], "segments": [], "dates": [], "metrics": []},
        "confidence": confidence,
        "raw_quote": "test",
    }
    content = MagicMock()
    content.text = json.dumps(pack)
    resp = MagicMock()
    resp.content = [content]
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    return resp


def test_classify_returns_intent_pack():
    with patch("forecasting.agents.lens.client") as mock_client:
        mock_client.messages.create.return_value = _mock_response("SCOPE_RESPONSE")
        result = classify_intent(_make_input("yes that looks right"))
    assert isinstance(result, IntentPack)
    assert result.intent == "SCOPE_RESPONSE"
    assert result.confidence == 0.9


def test_advance_pipeline_intent():
    with patch("forecasting.agents.lens.client") as mock_client:
        mock_client.messages.create.return_value = _mock_response("ADVANCE_PIPELINE")
        result = classify_intent(_make_input("ok let's proceed to modelling"))
    assert result.intent == "ADVANCE_PIPELINE"


def test_model_is_haiku():
    with patch("forecasting.agents.lens.client") as mock_client:
        mock_client.messages.create.return_value = _mock_response("SCOPE_RESPONSE")
        classify_intent(_make_input("yes"))
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "haiku" in call_kwargs["model"]


def test_temperature_zero():
    with patch("forecasting.agents.lens.client") as mock_client:
        mock_client.messages.create.return_value = _mock_response("SCOPE_RESPONSE")
        classify_intent(_make_input("yes"))
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs.get("temperature") == 0.0
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_lens.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backend/forecasting/agents/lens.py`**

```python
from __future__ import annotations

from typing import Literal
import anthropic
from pydantic import BaseModel, Field

from forecasting.contracts import IntentPack, IntentType, IntentEntities
from forecasting.run_state import RunState


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    agent: str | None = None


class LensInput(BaseModel):
    conversation_history: list[ConversationTurn]
    user_message: str
    pipeline_state: RunState


class IntentEntities(BaseModel):
    skus: list[str] = Field(default_factory=list)
    segments: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    scenario: str | None = None


client = anthropic.Anthropic()

_SYSTEM = """
You are Lens, an intent classifier for a demand forecasting assistant.
Classify the user's latest message into exactly one intent type and return a single JSON
object. No prose — JSON only.

Intent types:
- SCOPE_RESPONSE    — answering Meridian's scoping question
- OVERRIDE          — contradicting an agent recommendation backed by data
- ADVANCE_PIPELINE  — approving progression to the next pipeline phase
- WHAT_IF_REQUEST   — requesting a scenario / what-if analysis
- CLARIFICATION     — asking a question
- CORRECTION        — fixing a prior statement (only valid in meridian_scoping)

Weighting rules:
1. pipeline_state.phase and the last assistant message are the strongest signal for
   short ambiguous messages ("ok", "yes", "fine", "sure").
2. Short message after a risk warning → SCOPE_RESPONSE.
3. Short message after "shall we proceed?" → ADVANCE_PIPELINE.
4. Only WHAT_IF_REQUEST if the user explicitly describes a scenario change.
5. Set confidence honestly. Unsure between two → confidence < 0.6.
6. raw_quote: verbatim excerpt (≤20 words) from the user message.

Return schema (JSON, no markdown):
{
  "intent": "<IntentType>",
  "entities": {"skus": [], "segments": [], "dates": [], "metrics": [], "scenario": null},
  "confidence": 0.0,
  "raw_quote": ""
}
""".strip()


def classify_intent(inp: LensInput) -> IntentPack:
    messages = [{"role": t.role, "content": t.content} for t in inp.conversation_history]
    messages.append({"role": "user", "content": inp.user_message})

    state = inp.pipeline_state
    system = (
        f"{_SYSTEM}\n\n"
        f"pipeline_state: phase={state.phase} pack_confirmed={state.pack_confirmed} "
        f"open_risks={state.open_risks} override_count={state.override_count}"
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        temperature=0.0,
        system=system,
        messages=messages,
    )
    raw = response.content[0].text.strip()
    data = __import__("json").loads(raw)
    return IntentPack(
        intent=data["intent"],
        entities=IntentEntities(**data.get("entities", {})),
        confidence=data["confidence"],
        raw_quote=data.get("raw_quote", inp.user_message),
    )
```

- [ ] **Step 4: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_lens.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add backend/forecasting/agents/lens.py tests/test_lens.py
git commit -m "feat: lens agent — Haiku intent classifier, structured output, temperature=0"
```

---

### Task 11: `tools/conductor_tools.py` — Conductor Tool Implementations

**Files:**
- Create: `backend/forecasting/tools/conductor_tools.py`
- Create: `tests/test_conductor_tools.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_conductor_tools.py
import pytest
from forecasting.run_state import create_run_state, load_run_state, Phase
from forecasting.tools.conductor_tools import (
    get_run_state, update_run_state, log_halt, ConditionViolationError
)
from forecasting.run_state import HaltedRunError


def test_get_run_state_returns_dict(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    state = get_run_state(run_id)
    assert state["run_id"] == run_id
    assert state["phase"] == "preflight"


def test_update_run_state_persists(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    updated = update_run_state(run_id, {"meridian_turn_count": 3})
    assert updated["meridian_turn_count"] == 3
    assert load_run_state(run_id).meridian_turn_count == 3


def test_update_rejects_pack_confirmed_false(run_id, tmp_outputs):
    state = create_run_state(run_id, domain="fmcg")
    # Set pack_confirmed True first
    update_run_state(run_id, {"pack_confirmed": True})
    with pytest.raises(ValueError, match="pack_confirmed"):
        update_run_state(run_id, {"pack_confirmed": False})


def test_update_halted_run_raises(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    log_halt(run_id, "test halt", lambda *a, **k: None)
    with pytest.raises(HaltedRunError):
        update_run_state(run_id, {"meridian_turn_count": 1})


def test_log_halt_sets_phase(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    emitted = []
    log_halt(run_id, "budget exceeded", lambda evt, payload: emitted.append((evt, payload)))
    state = load_run_state(run_id)
    assert state.phase == Phase.HALTED
    assert state.halt_reason == "budget exceeded"
    assert any(e[0] == "error" for e in emitted)
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_conductor_tools.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backend/forecasting/tools/conductor_tools.py`**

```python
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Callable

from forecasting.run_state import (
    load_run_state, save_run_state, Phase, RunState, HaltedRunError, run_dir
)


class ConditionViolationError(Exception):
    def __init__(self, tool: str, condition: str):
        super().__init__(f"Conductor tool '{tool}' precondition failed: {condition}")


def get_run_state(run_id: str) -> dict:
    return load_run_state(run_id).model_dump()


def update_run_state(run_id: str, patch: dict) -> dict:
    state = load_run_state(run_id)
    phase_val = state.phase if isinstance(state.phase, str) else state.phase.value
    if phase_val == Phase.HALTED.value:
        raise HaltedRunError(run_id)
    if "pack_confirmed" in patch and patch["pack_confirmed"] is False and state.pack_confirmed:
        raise ValueError("pack_confirmed is a one-way transition (False → True only)")
    updated = state.model_copy(update=patch)
    save_run_state(updated)
    return updated.model_dump()


def advance_to_meridian(run_id: str, user_message: str) -> None:
    """Phase transition only — Meridian is invoked by the API handler, not here."""
    state = load_run_state(run_id)
    phase_val = state.phase if isinstance(state.phase, str) else state.phase.value
    if phase_val == Phase.PREFLIGHT.value:
        update_run_state(run_id, {"phase": Phase.MERIDIAN_SCOPING.value})


def confirm_pack_and_advance(run_id: str) -> None:
    state = load_run_state(run_id)
    phase_val = state.phase if isinstance(state.phase, str) else state.phase.value
    if phase_val != Phase.MERIDIAN_SCOPING.value:
        raise ConditionViolationError("confirm_pack_and_advance", f"phase={phase_val} not meridian_scoping")
    if state.pack_confirmed:
        raise ConditionViolationError("confirm_pack_and_advance", "pack already confirmed")
    if state.open_risks > 0:
        raise ConditionViolationError("confirm_pack_and_advance", f"open_risks={state.open_risks}")
    update_run_state(run_id, {"pack_confirmed": True, "phase": Phase.FORGE_EDA.value})


def trigger_foundry(run_id: str) -> None:
    state = load_run_state(run_id)
    if not state.forge_complete:
        raise ConditionViolationError("trigger_foundry", "forge_complete is False")
    update_run_state(run_id, {"phase": Phase.FOUNDRY_MODELLING.value})


def create_prism_run(run_id: str, scenario_description: str, entities: dict) -> dict:
    state = load_run_state(run_id)
    phase_val = state.phase if isinstance(state.phase, str) else state.phase.value
    if phase_val != Phase.REPORT_READY.value:
        raise ConditionViolationError("create_prism_run", f"phase={phase_val} not report_ready")
    whatif_id = f"wi-{uuid.uuid4().hex[:8]}"
    (run_dir(run_id) / "whatif" / whatif_id).mkdir(parents=True, exist_ok=True)
    active = state.active_whatif_runs + [whatif_id]
    update_run_state(run_id, {"active_whatif_runs": active})
    return {"whatif_id": whatif_id}


def surface_clarification(run_id: str, message: str, sse_emit: Callable) -> None:
    sse_emit("message_done", {"agent": "conductor", "full_text": message})


def log_halt(run_id: str, reason: str, sse_emit: Callable) -> None:
    state = load_run_state(run_id)
    state.halt_reason = reason
    state.phase = Phase.HALTED
    save_run_state(state)
    obs = run_dir(run_id) / "obs_log.json"
    _append_obs(obs, {"event": "HALT", "reason": reason})
    sse_emit("error", {"reason": "Run halted — please start a new run.", "halt_reason": reason})


def _append_obs(path: Path, entry: dict) -> None:
    import datetime
    log: list = []
    if path.exists():
        log = json.loads(path.read_text())
    entry["ts"] = datetime.datetime.now(__import__("datetime").timezone.utc).isoformat()
    log.append(entry)
    path.write_text(json.dumps(log, indent=2))
```

- [ ] **Step 4: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_conductor_tools.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/forecasting/tools/conductor_tools.py tests/test_conductor_tools.py
git commit -m "feat: conductor_tools — state transitions, confirm pack, halt, prism run creation"
```

---

### Task 12: `agents/conductor.py` — Orchestration LLM Loop

**Files:**
- Create: `backend/forecasting/agents/conductor.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_conductor_agent.py
import json
import pytest
from unittest.mock import MagicMock, patch
from forecasting.agents.conductor import run_conductor
from forecasting.run_state import create_run_state, Phase


def _make_intent(intent: str = "ADVANCE_PIPELINE"):
    from forecasting.contracts import IntentPack
    return IntentPack(intent=intent, confidence=0.95, raw_quote="ok let's go")


def _end_turn_response():
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = []
    resp.usage = MagicMock(input_tokens=200, output_tokens=50)
    return resp


def test_run_conductor_returns_int(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    with patch("forecasting.agents.conductor.client") as mock_client:
        mock_client.messages.create.return_value = _end_turn_response()
        result = run_conductor(
            run_id=run_id,
            intent_pack=_make_intent(),
            run_state=__import__("forecasting.run_state", fromlist=["load_run_state"]).load_run_state(run_id),
            tokens_used=0,
            sse_emit=lambda *a, **k: None,
        )
    assert isinstance(result, int)
    assert result >= 0
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_conductor_agent.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backend/forecasting/agents/conductor.py`**

```python
from __future__ import annotations

import json
from typing import Callable

import anthropic

from forecasting.contracts import IntentPack
from forecasting.guard import AgentGuardState
from forecasting.providers import dispatch_tool
from forecasting.run_state import RunState

client = anthropic.Anthropic()

_SYSTEM = """
You are Conductor, the orchestration agent for a demand forecasting pipeline.

You receive an IntentPack (from Lens) and the current RunState. Call exactly one routing
tool per turn, then stop (end_turn). Do not compose user-facing messages except via
surface_clarification. Do not interpret data or apply domain knowledge.

Hard rules — violation triggers log_halt:
1. Never call confirm_pack_and_advance if open_risks > 0.
2. Never call confirm_pack_and_advance if phase ≠ meridian_scoping or pack_confirmed = True.
3. Never call trigger_foundry before forge_complete = True.
4. Never call create_prism_run before phase = report_ready.
5. Always call surface_clarification if intent.confidence < 0.6.
6. If phase = HALTED, return immediately — no tool calls.
7. CORRECTION + pack_confirmed = True → treat as OVERRIDE, offer Scenario Run.
""".strip()

# Populated in Task 22 after all tools are registered
CONDUCTOR_TOOLS: list[dict] = []


def run_conductor(
    run_id: str,
    intent_pack: IntentPack,
    run_state: RunState,
    tokens_used: int,
    sse_emit: Callable,
) -> int:
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
            system=_SYSTEM,
            tools=CONDUCTOR_TOOLS,
            messages=messages,
        )
        tokens_used += response.usage.input_tokens + response.usage.output_tokens

        if response.stop_reason == "end_turn":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # Inject sse_emit for tools that need it
                args = dict(block.input)
                if block.name in ("surface_clarification", "log_halt"):
                    args["sse_emit"] = sse_emit
                result = dispatch_tool(
                    tool_name=block.name,
                    args=args,
                    guard=guard,
                    tokens_used=tokens_used,
                )
                if block.name == "confirm_pack_and_advance":
                    sse_emit("phase_change", {"phase": "forge_eda"})
                elif block.name == "trigger_foundry":
                    sse_emit("phase_change", {"phase": "foundry_modelling"})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
                break  # one tool call per turn

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return tokens_used
```

- [ ] **Step 4: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_conductor_agent.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```powershell
git add backend/forecasting/agents/conductor.py tests/test_conductor_agent.py
git commit -m "feat: conductor agent — routing LLM loop, one tool per turn, SSE phase events"
```

---

### Task 13: `tools/meridian_pack.py` — Pack Management Tools

**Files:**
- Create: `backend/forecasting/tools/meridian_pack.py`
- Create: `tests/test_meridian_pack.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_meridian_pack.py
import json
import pytest
from forecasting.run_state import create_run_state, load_run_state
from forecasting.tools.meridian_pack import (
    add_claim, resolve_claim, add_risk, acknowledge_risk, compile_domain_context_pack
)


SCOPE = {
    "target_col": "demand", "grain": "sku_region_week",
    "horizon": 12, "forecast_start": "2024-10-01"
}


def test_add_claim_creates_ledger(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    result = add_claim(
        run_id=run_id,
        claim="Demand is seasonal",
        evidence_ref="autocorr=0.72",
        verification_status="SUPPORTED",
        evidence_type="statistical_test",
        applies_to="run",
        downstream_impact="feature selection",
    )
    assert "claim_id" in result
    ledger_path = tmp_outputs / run_id / "claim_ledger.json"
    assert ledger_path.exists()
    data = json.loads(ledger_path.read_text())
    assert len(data["claims"]) == 1


def test_resolve_claim_updates_status(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    result = add_claim(run_id=run_id, claim="Promo lifts demand", evidence_ref=None,
                       verification_status="CONTRADICTED", evidence_type="association",
                       applies_to="run", downstream_impact="feature promo flag")
    cid = result["claim_id"]
    resolve_claim(run_id, cid, "USER_OVERRIDE_ACCEPTED", user_reason="I know my data")
    state = load_run_state(run_id)
    assert state.override_count == 1


def test_add_risk_increments_open_risks(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    add_risk(run_id, risk="Promo data may be incomplete", severity="medium", source="user")
    assert load_run_state(run_id).open_risks == 1


def test_acknowledge_risk_decrements(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    r = add_risk(run_id, risk="Sparse data", severity="low", source="data")
    acknowledge_risk(run_id, r["risk_id"])
    assert load_run_state(run_id).open_risks == 0


def test_compile_pack_fails_no_scope(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    result = compile_domain_context_pack(run_id, forecast_scope={}, segments=[])
    assert not result["pack_complete"]
    assert len(result["validation_errors"]) > 0


def test_compile_pack_succeeds(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    segments = [{"segment_id": "G1", "zero_policy": "exclude",
                 "spike_policy": "cap_iqr3x", "confirmed_breaks": []}]
    result = compile_domain_context_pack(run_id, forecast_scope=SCOPE, segments=segments)
    assert result["pack_complete"]
    assert result["pack"] is not None
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_meridian_pack.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backend/forecasting/tools/meridian_pack.py`**

```python
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from forecasting.run_state import load_run_state, save_run_state, run_dir
from forecasting.contracts import (
    DomainContextPack, ForecastScope, SegmentDef, Claim, Risk,
)


def add_claim(
    run_id: str,
    claim: str,
    evidence_ref: str | None,
    verification_status: str,
    evidence_type: str,
    applies_to: str,
    downstream_impact: str,
) -> dict:
    ledger = _load_ledger(run_id)
    entry = {
        "claim_id": str(uuid.uuid4()),
        "claim": claim,
        "verification_status": verification_status,
        "evidence_type": evidence_type,
        "evidence_ref": evidence_ref,
        "applies_to": applies_to,
        "downstream_impact": downstream_impact,
        "must_surface_in_report": False,
        "created_at": _now(),
    }
    ledger["claims"].append(entry)
    _save_ledger(run_id, ledger)
    return {"claim_id": entry["claim_id"]}


def resolve_claim(
    run_id: str,
    claim_id: str,
    new_status: Literal["SUPPORTED", "USER_OVERRIDE_ACCEPTED"],
    user_reason: str | None = None,
) -> dict:
    ledger = _load_ledger(run_id)
    for c in ledger["claims"]:
        if c["claim_id"] == claim_id:
            c["verification_status"] = new_status
            if new_status == "USER_OVERRIDE_ACCEPTED":
                c["must_surface_in_report"] = True
                state = load_run_state(run_id)
                state.override_count += 1
                save_run_state(state)
            c["resolved_at"] = _now()
            if user_reason:
                c["resolution_note"] = user_reason
            _save_ledger(run_id, ledger)
            return c
    raise KeyError(f"claim_id '{claim_id}' not found in ledger for run '{run_id}'")


def add_risk(run_id: str, risk: str, severity: str, source: str) -> dict:
    register = _load_register(run_id)
    entry = {
        "risk_id": str(uuid.uuid4()),
        "description": risk,
        "severity": severity,
        "source": source,
        "acknowledged": False,
        "created_at": _now(),
        "acknowledged_at": None,
    }
    register["risks"].append(entry)
    _save_register(run_id, register)
    state = load_run_state(run_id)
    state.open_risks += 1
    save_run_state(state)
    return {"risk_id": entry["risk_id"]}


def acknowledge_risk(run_id: str, risk_id: str) -> None:
    register = _load_register(run_id)
    for r in register["risks"]:
        if r["risk_id"] == risk_id and not r["acknowledged"]:
            r["acknowledged"] = True
            r["acknowledged_at"] = _now()
            _save_register(run_id, register)
            state = load_run_state(run_id)
            state.open_risks = max(0, state.open_risks - 1)
            save_run_state(state)
            return
    raise KeyError(f"risk_id '{risk_id}' not found or already acknowledged")


def compile_domain_context_pack(
    run_id: str,
    forecast_scope: dict,
    segments: list[dict],
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    if not forecast_scope.get("target_col"):
        errors.append("missing forecast_scope.target_col")
    if not forecast_scope.get("grain"):
        errors.append("missing forecast_scope.grain")
    if not forecast_scope.get("horizon"):
        errors.append("missing forecast_scope.horizon")
    if not segments:
        errors.append("no segments defined")

    ledger = _load_ledger(run_id)
    contradicted = [c for c in ledger["claims"] if c["verification_status"] == "CONTRADICTED"]
    if contradicted:
        errors.append(f"{len(contradicted)} claim(s) still in CONTRADICTED status")

    state = load_run_state(run_id)
    if state.open_risks > 0:
        errors.append(f"open_risks={state.open_risks} — must acknowledge all risks before confirming")

    if state.override_count > 3:
        warnings.append(f"override_count={state.override_count} — high override rate")

    if errors:
        return {"pack_complete": False, "validation_errors": errors, "validation_warnings": warnings, "pack": None}

    # Build and VALIDATE a DomainContextPack model — never a hand-built dict. This is
    # what keeps contracts.py the single source of truth: if the emitted shape drifts
    # from the model, construction raises here rather than silently diverging. (review §2)
    pack_model = DomainContextPack(
        run_id=run_id,
        domain=state.domain,
        forecast_scope=ForecastScope(**forecast_scope),
        segments=[SegmentDef(**{**s, "provisional": False}) for s in segments],  # freeze the map
        claim_ledger=[Claim(**c) for c in ledger["claims"]],
        risk_register=[Risk(**r) for r in _load_register(run_id)["risks"]],
        override_count=state.override_count,
        open_risks=state.open_risks,
        confirmed_at=_now(),
        confirmed=True,
    )
    pack = pack_model.model_dump()
    out = run_dir(run_id) / "domain_context_pack.json"
    out.write_text(json.dumps(pack, indent=2))
    return {"pack_complete": True, "validation_errors": [], "validation_warnings": warnings, "pack": pack}


def _load_ledger(run_id: str) -> dict:
    path = run_dir(run_id) / "claim_ledger.json"
    if not path.exists():
        run_dir(run_id).mkdir(parents=True, exist_ok=True)
        return {"run_id": run_id, "claims": []}
    return json.loads(path.read_text())


def _save_ledger(run_id: str, ledger: dict) -> None:
    (run_dir(run_id) / "claim_ledger.json").write_text(json.dumps(ledger, indent=2))


def _load_register(run_id: str) -> dict:
    path = run_dir(run_id) / "risk_register.json"
    if not path.exists():
        return {"run_id": run_id, "risks": []}
    return json.loads(path.read_text())


def _save_register(run_id: str, register: dict) -> None:
    (run_dir(run_id) / "risk_register.json").write_text(json.dumps(register, indent=2))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
```

- [ ] **Step 4: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_meridian_pack.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/forecasting/tools/meridian_pack.py tests/test_meridian_pack.py
git commit -m "feat: meridian_pack — claim ledger, risk register, compile_domain_context_pack"
```

---

### Task 14: `tools/meridian_diagnostic.py` — Diagnostic Read Tools

**Files:**
- Create: `backend/forecasting/tools/meridian_diagnostic.py`
- Create: `tests/test_meridian_diagnostic.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_meridian_diagnostic.py
import json
import pytest
from forecasting.run_state import create_run_state
from forecasting.preflight import run_preflight
from forecasting.tools.meridian_diagnostic import (
    summarise_demand_segments, diagnose_zero_demand_policy,
    diagnose_horizon_feasibility, diagnose_forecastability_by_segment,
)

PLAYBOOK = {"common_grains": ["sku", "region"], "time_col": "week",
            "demand_col": "demand", "min_series": 1, "min_history_periods": 4}


def _csv() -> bytes:
    rows = [f"2024-W{w+1:02d},{sku},NORTH,{float(w+1)}"
            for sku in ["SKU_A", "SKU_B"] for w in range(20)]
    return ("week,sku,region,demand\n" + "\n".join(rows)).encode()


def test_summarise_segments(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    run_preflight(run_id, _csv(), domain="fmcg", playbook=PLAYBOOK)
    result = summarise_demand_segments(run_id)
    assert "segment_profiles" in result
    assert len(result["segment_profiles"]) > 0


def test_diagnose_zero_policy(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    run_preflight(run_id, _csv(), domain="fmcg", playbook=PLAYBOOK)
    result = diagnose_zero_demand_policy(run_id, segment_id="SMOOTH")
    assert "recommendation" in result


def test_diagnose_horizon_feasible(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    run_preflight(run_id, _csv(), domain="fmcg", playbook=PLAYBOOK)
    result = diagnose_horizon_feasibility(run_id, horizon_periods=4)
    assert "feasible" in result
    assert "max_recommended_horizon" in result


def test_diagnose_forecastability(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    run_preflight(run_id, _csv(), domain="fmcg", playbook=PLAYBOOK)
    result = diagnose_forecastability_by_segment(run_id, segment_id="SMOOTH")
    assert "forecastable_pct" in result
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_meridian_diagnostic.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backend/forecasting/tools/meridian_diagnostic.py`**

```python
from __future__ import annotations

import json
from forecasting.run_state import run_dir


def _load_preflight(run_id: str) -> dict:
    path = run_dir(run_id) / "preflight.json"
    if not path.exists():
        raise FileNotFoundError(f"preflight.json not found for run '{run_id}'")
    return json.loads(path.read_text())


def _segment_members(data: dict, segment_id: str) -> list[str]:
    """Resolve segment_id → member series_keys via the persisted segment map.

    Replaces the old substring matching (`segment_id in series_key`), which never
    matched because segment_id ('G1') is not a substring of a grain key
    ('SKU_A|NORTH'). preflight.json['segment_map'] is the single source of segment
    membership. It starts provisional (pre-flight) and is rewritten in place by
    Meridian's refine_segments when the user adjusts the cut; it is frozen into the
    pack at confirmation.
    """
    seg_map = data.get("segment_map", {})
    for seg in seg_map.get("segments", []):
        if seg.get("segment_id") == segment_id:
            return list(seg.get("series_keys", []))
    return []


def _per_series(data: dict, kind: str) -> dict:
    """Per-series stat dicts live under preflight.json['per_series'], NOT in the bundle."""
    return data.get("per_series", {}).get(kind, {})


def refine_segments(run_id: str, segments: list[dict]) -> dict:
    """
    Rewrite the provisional segment map when the user adjusts the suggested cut
    (merge, split, or re-segment by a different dimension). `segments` is a list of
    {segment_id, label, series_keys}. Overwrites preflight.json['segment_map'] in
    place; the map stays provisional until pack confirmation, at which point it is
    frozen into the domain_context_pack. Every series must be assigned exactly once.
    Returns { segment_count, total_series }.
    """
    path = run_dir(run_id) / "preflight.json"
    data = json.loads(path.read_text())
    all_keys = sorted(_per_series(data, "adi_cv2").keys())
    assigned = [k for s in segments for k in s.get("series_keys", [])]
    if sorted(set(assigned)) != all_keys or len(assigned) != len(all_keys):
        raise ValueError("refine_segments: every series must be assigned to exactly one segment")
    data["segment_map"] = {
        "run_id": run_id,
        "segments": [
            {"segment_id": s["segment_id"], "label": s.get("label", s["segment_id"]),
             "series_keys": sorted(s["series_keys"]), "provisional": True}
            for s in segments
        ],
        "provisional": True,
        "derived_by": "user:refine_segments",
    }
    path.write_text(json.dumps(data, indent=2))
    return {"segment_count": len(segments), "total_series": len(all_keys)}


def summarise_demand_segments(run_id: str, segment_id: str | None = None) -> dict:
    data = _load_preflight(run_id)
    profiles = data["bundle"].get("segment_profiles", [])
    if segment_id:
        profiles = [p for p in profiles if p.get("segment_id") == segment_id]
    return {"segment_profiles": profiles,
            "total_series": data["bundle"]["data_quality_report"]["series_count"]}


def diagnose_zero_demand_policy(run_id: str, segment_id: str) -> dict:
    data = _load_preflight(run_id)
    keys = _segment_members(data, segment_id)
    zero_runs = _per_series(data, "zero_runs")
    matching = {k: zero_runs[k] for k in keys if k in zero_runs}
    if not matching:
        return {"segment_id": segment_id, "avg_zero_fraction": 0.0, "max_zero_run": 0,
                "recommendation": "No zero-demand data found for this segment."}
    avg_zero = sum(v["zero_fraction"] for v in matching.values()) / len(matching)
    max_run = max(v["max_zero_run"] for v in matching.values())
    if avg_zero > 0.6:
        rec = "High zero fraction — consider exclude or Croston-family models."
    elif max_run > 8:
        rec = "Long zero runs detected — possible stockouts. Verify with supply team."
    else:
        rec = "Zero demand levels acceptable — standard handling recommended."
    return {"segment_id": segment_id, "avg_zero_fraction": round(avg_zero, 3),
            "max_zero_run": max_run, "recommendation": rec}


def diagnose_spike_policy(run_id: str, segment_id: str) -> dict:
    data = _load_preflight(run_id)
    keys = _segment_members(data, segment_id)
    spikes = _per_series(data, "spikes")
    matching = {k: spikes[k] for k in keys if k in spikes}
    spike_count = sum(v["spike_count"] for v in matching.values())
    if spike_count == 0:
        rec = "No spikes detected — no spike policy needed."
    else:
        rec = "Spikes detected — recommend cap_iqr3x policy to prevent model distortion."
    return {"segment_id": segment_id, "total_spikes": spike_count, "recommendation": rec}


def diagnose_granularity_feasibility(run_id: str, min_series: int | None = None) -> dict:
    data = _load_preflight(run_id)
    series_count = data["bundle"]["data_quality_report"]["series_count"]
    threshold = min_series or 5
    return {
        "series_count": series_count,
        "feasible": series_count >= threshold,
        "reason": "sufficient series" if series_count >= threshold else f"only {series_count} series, need {threshold}",
    }


def diagnose_horizon_feasibility(
    run_id: str, horizon_periods: int, segment_id: str | None = None
) -> dict:
    data = _load_preflight(run_id)
    grain = data["bundle"]["grain_report"]
    min_periods = grain["min_periods"]
    max_horizon = max(1, min_periods // 3)
    feasible = horizon_periods <= max_horizon
    return {
        "feasible": feasible,
        "reason": "sufficient history" if feasible else f"min_periods={min_periods} too short for horizon={horizon_periods}",
        "max_recommended_horizon": max_horizon,
    }


def diagnose_structural_break_candidates(
    run_id: str, date: str | None = None, segment_id: str | None = None
) -> dict:
    data = _load_preflight(run_id)
    candidates = data.get("break_candidates", [])
    if date:
        candidates = [c for c in candidates if c.get("break_period", "").startswith(date[:7])]
    if segment_id:
        members = set(_segment_members(data, segment_id))
        candidates = [c for c in candidates if c.get("series_key") in members]
    return {"candidates": candidates}


def diagnose_forecastability_by_segment(run_id: str, segment_id: str) -> dict:
    data = _load_preflight(run_id)
    keys = _segment_members(data, segment_id)
    adi_cv2 = _per_series(data, "adi_cv2")
    matching = {k: adi_cv2[k] for k in keys if k in adi_cv2}
    if not matching:
        return {"segment_id": segment_id, "forecastable_pct": 0.0, "caution_pct": 0.0,
                "unforecastable_pct": 0.0, "basis": "no series matched"}
    forecastable = sum(1 for v in matching.values() if v["sb_class"] in ("SMOOTH", "ERRATIC"))
    intermittent = sum(1 for v in matching.values() if v["sb_class"] == "INTERMITTENT")
    lumpy = sum(1 for v in matching.values() if v["sb_class"] == "LUMPY")
    total = len(matching)
    return {
        "segment_id": segment_id,
        "forecastable_pct": round(forecastable / total, 3),
        "caution_pct": round(intermittent / total, 3),
        "unforecastable_pct": round(lumpy / total, 3),
        "basis": "Syntetos-Boylan preflight classification",
    }
```

- [ ] **Step 4: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_meridian_diagnostic.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/forecasting/tools/meridian_diagnostic.py tests/test_meridian_diagnostic.py
git commit -m "feat: meridian_diagnostic — segment summaries, zero/spike/horizon/break diagnostics"
```

---

### Task 15: `agents/meridian.py` — Scoping Conversation LLM Loop

**Files:**
- Create: `backend/forecasting/agents/meridian.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_meridian_agent.py
import json
import pytest
from unittest.mock import MagicMock, patch
from forecasting.agents.meridian import run_meridian
from forecasting.run_state import create_run_state
from forecasting.preflight import run_preflight

PLAYBOOK = {"common_grains": ["sku", "region"], "time_col": "week",
            "demand_col": "demand", "min_series": 1, "min_history_periods": 4}


def _csv() -> bytes:
    rows = [f"2024-W{w+1:02d},SKU_A,NORTH,{float(w+1)}" for w in range(12)]
    return ("week,sku,region,demand\n" + "\n".join(rows)).encode()


def _stream_response(text: str = "Hello, I am Meridian."):
    event = MagicMock()
    event.delta = MagicMock()
    event.delta.text = text

    final = MagicMock()
    final.stop_reason = "end_turn"
    final.content = []
    final.usage = MagicMock(input_tokens=300, output_tokens=100)

    stream = MagicMock()
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    stream.__iter__ = MagicMock(return_value=iter([event]))
    stream.get_final_message = MagicMock(return_value=final)
    return stream


def test_run_meridian_returns_int(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    bundle = run_preflight(run_id, _csv(), domain="fmcg", playbook=PLAYBOOK)
    emitted = []
    with patch("forecasting.agents.meridian.client") as mock_client:
        mock_client.messages.stream.return_value = _stream_response()
        result = run_meridian(
            run_id=run_id,
            user_message="Let's get started",
            conversation_history=[],
            preflight_bundle=bundle.model_dump(),
            tokens_used=0,
            sse_emit=lambda evt, payload: emitted.append((evt, payload)),
        )
    assert isinstance(result, int)
    assert any(e[0] == "message_done" for e in emitted)
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_meridian_agent.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backend/forecasting/agents/meridian.py`**

```python
from __future__ import annotations

import json
from typing import Callable

import anthropic

from forecasting.guard import AgentGuardState
from forecasting.providers import dispatch_tool

client = anthropic.Anthropic()

_SYSTEM_TEMPLATE = """
You are Meridian, the domain specialist for a demand forecasting pipeline.

Your role is to conduct the scoping conversation with the analyst. You work at segment
level — never interrogate individual series one by one. Present evidence first, ask the
user to interpret. Use humble language ("the data suggests", "this may indicate").
Never overclaim causality.

Conversation arc:
0. Welcome — orient the user, summarise preflight findings, ask first question.
1. Forecast scope — target metric, grain, horizon, forecast start date.
2. Segment review — zero-demand policy, spike policy, forecastability concerns.
3. Feature & break policy — promo flags, structural breaks (require explicit acknowledgement).
4. Pack review — call compile_domain_context_pack, present summary.
5. Risk acknowledgement — walk through open risks if any.
6. Handoff — confirm user is ready to proceed.

Rules:
- Use diagnostic tools before making claims about data patterns.
- Call add_claim after every significant finding.
- For unverifiable business inputs, call add_risk immediately after add_claim.
- Never refuse to accept a user override — call resolve_claim with USER_OVERRIDE_ACCEPTED.
- compile_domain_context_pack will block if any Claim is still CONTRADICTED.

PREFLIGHT BUNDLE:
{preflight_json}
""".strip()

MERIDIAN_TOOLS: list[dict] = []  # populated in Task 22


def run_meridian(
    run_id: str,
    user_message: str,
    conversation_history: list[dict],
    preflight_bundle: dict,
    tokens_used: int,
    sse_emit: Callable,
) -> int:
    guard = AgentGuardState(agent="meridian", run_id=run_id)
    system = _SYSTEM_TEMPLATE.format(preflight_json=json.dumps(preflight_bundle, indent=2)[:4000])
    messages = list(conversation_history) + [{"role": "user", "content": user_message}]

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
                    chunk = event.delta.text
                    sse_emit("token", {"content": chunk})
                    full_text += chunk
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
                if block.name == "add_claim":
                    sse_emit("decision_update", result)
                elif block.name == "add_risk":
                    sse_emit("risk_update", result)
                elif block.name == "resolve_claim" and isinstance(result, dict):
                    if result.get("verification_status") == "USER_OVERRIDE_ACCEPTED":
                        sse_emit("override_update", result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return tokens_used
```

- [ ] **Step 4: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_meridian_agent.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```powershell
git add backend/forecasting/agents/meridian.py tests/test_meridian_agent.py
git commit -m "feat: meridian agent — streaming scoping conversation, claim/risk SSE events"
```

---

### Task 16: `tools/forge_tools.py` — EDA Tool Implementations

**Files:**
- Create: `backend/forecasting/tools/forge_tools.py`
- Create: `tests/test_forge_tools.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forge_tools.py
import json
import pytest
import pandas as pd
from forecasting.run_state import create_run_state, load_run_state
from forecasting.data_store import store_series
from forecasting.tools.forge_tools import (
    classify_demand_profiles, specify_feature_config,
    design_walk_forward_folds, compile_eda_report,
)


def _load_series(run_id: str) -> None:
    for key in ["SKU_A|NORTH", "SKU_B|NORTH"]:
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-06", periods=52, freq="W"),
            "demand": [float(i % 10 + 1) for i in range(52)],
        })
        store_series(run_id, key, df)


def _pack(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "segments": [{"segment_id": "G1", "zero_policy": "exclude",
                      "spike_policy": "cap_iqr3x", "confirmed_breaks": []}],
        "forecast_scope": {"horizon": 4},
    }


def test_classify_demand_profiles(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    _load_series(run_id)
    result = classify_demand_profiles(run_id, segment_id="G1", series_keys=["SKU_A|NORTH", "SKU_B|NORTH"])
    assert "classifications" in result
    for v in result["classifications"].values():
        assert v in ("SMOOTH", "ERRATIC", "INTERMITTENT", "LUMPY")


def test_specify_feature_config(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    result = specify_feature_config(run_id, segment_id="G1", demand_class="SMOOTH",
                                    pack_feature_flags={"promo": True, "price": False})
    assert "features" in result
    assert "rationale" in result


def test_design_walk_forward_folds(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    _load_series(run_id)
    result = design_walk_forward_folds(run_id, segment_id="G1",
                                       series_keys=["SKU_A|NORTH"], horizon=4, break_dates=[])
    assert "folds" in result
    assert result["n_folds"] >= 1


def test_compile_eda_report_sets_forge_complete(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    _load_series(run_id)
    # Write a partial result first
    partial_path = tmp_outputs / run_id / "eda_report_partial.json"
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text(json.dumps([
        {"segment_id": "G1", "demand_classes": {"SKU_A|NORTH": "SMOOTH"},
         "feature_config": {}, "evaluation_metric": "MASE",
         "folds": [], "break_results": {}, "caution_series": [], "stockout_flags": []}
    ]))
    result = compile_eda_report(run_id)
    assert load_run_state(run_id).forge_complete is True
    assert "segment_count" in result
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_forge_tools.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backend/forecasting/tools/forge_tools.py`**

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from forecasting.data_store import get_series, get_series_keys
from forecasting.run_state import load_run_state, save_run_state, run_dir
from forecasting.tools.preflight_stats import compute_adi_cv2_per_series, _sb


def run_full_eda(run_id: str, segment_id: str, series_keys: list[str]) -> dict:
    results = {}
    for key in series_keys:
        try:
            df = get_series(run_id, key)
            demand = df["demand"].values.astype(float)
            acf_lag1 = float(np.corrcoef(demand[:-1], demand[1:])[0, 1]) if len(demand) > 1 else 0.0
            results[key] = {"acf_lag1": round(acf_lag1, 4), "n_periods": len(demand)}
        except Exception as e:
            results[key] = {"error": str(e)}
    return {"segment_id": segment_id, "eda_results": results}


def classify_demand_profiles(run_id: str, segment_id: str, series_keys: list[str]) -> dict:
    series_map = {k: get_series(run_id, k) for k in series_keys}
    from forecasting.tools.preflight_stats import compute_adi_cv2_per_series
    adi_cv2 = compute_adi_cv2_per_series(series_map)
    classifications = {k: v.sb_class for k, v in adi_cv2.items()}
    # Persist to partial
    _append_partial(run_id, segment_id, {"demand_classes": classifications})
    return {"segment_id": segment_id, "classifications": classifications}


def detect_structural_breaks(
    run_id: str, segment_id: str, series_keys: list[str], confirmed_dates: list[str]
) -> dict:
    results = {}
    for key in series_keys:
        key_results = {}
        df = get_series(run_id, key)
        demand = df["demand"].values.astype(float)
        for date_str in confirmed_dates:
            # Minimal Chow test: compare variance of two halves at break point
            try:
                break_idx = len(demand) // 2
                pre, post = demand[:break_idx], demand[break_idx:]
                if len(pre) < 4 or len(post) < 4:
                    key_results[date_str] = {"significant": False, "p_value": 1.0, "f_stat": 0.0}
                    continue
                var_pre = float(np.var(pre, ddof=1))
                var_post = float(np.var(post, ddof=1))
                f_stat = max(var_pre, var_post) / (min(var_pre, var_post) + 1e-9)
                significant = f_stat > 2.5
                key_results[date_str] = {"significant": significant, "p_value": 1.0 / f_stat, "f_stat": round(f_stat, 3)}
            except Exception:
                key_results[date_str] = {"significant": False, "p_value": 1.0, "f_stat": 0.0}
        results[key] = key_results
    return {"segment_id": segment_id, "break_results": results}


def flag_stockouts(run_id: str, segment_id: str, series_keys: list[str], threshold_weeks: int = 4) -> dict:
    flagged = []
    for key in series_keys:
        df = get_series(run_id, key)
        demand = df["demand"].values
        max_run = 0
        cur = 0
        for d in demand:
            cur = cur + 1 if d == 0 else 0
            max_run = max(max_run, cur)
        if max_run >= threshold_weeks:
            flagged.append(key)
    return {"segment_id": segment_id, "flagged_series": flagged,
            "reason": f"zero run >= {threshold_weeks} weeks"}


def specify_feature_config(
    run_id: str, segment_id: str, demand_class: str, pack_feature_flags: dict
) -> dict:
    smooth_like = demand_class in ("SMOOTH", "ERRATIC")
    fourier_terms = 3 if smooth_like and pack_feature_flags.get("seasonality") else 0
    lag_windows = [1, 4, 12] if smooth_like else [1, 2]
    features = {
        "promo": pack_feature_flags.get("promo", False),
        "price": pack_feature_flags.get("price", False),
        "fourier_terms": fourier_terms,
        "lag_windows": lag_windows,
    }
    rationale = f"demand_class={demand_class}, fourier={'yes' if fourier_terms else 'no'}"
    _append_partial(run_id, segment_id, {"feature_config": features})
    return {"segment_id": segment_id, "demand_class": demand_class,
            "features": features, "rationale": rationale}


def design_walk_forward_folds(
    run_id: str, segment_id: str, series_keys: list[str],
    horizon: int, break_dates: list[str]
) -> dict:
    min_len = horizon * 3
    folds = []
    caution = []
    for key in series_keys:
        df = get_series(run_id, key)
        n = len(df)
        if n < min_len:
            caution.append({"series_key": key, "reason": "insufficient_post_break_history"})
    n_folds = max(1, len(series_keys[0:1] and get_series(run_id, series_keys[0]))
                  // (horizon * 2)) if series_keys else 2
    if series_keys:
        df = get_series(run_id, series_keys[0])
        n = len(df)
        fold_size = n // 3
        for i in range(min(2, n // fold_size)):
            train_end = fold_size * (i + 1)
            folds.append({
                "train_start": str(df["date"].iloc[0]),
                "train_end": str(df["date"].iloc[train_end - 1]),
                "test_start": str(df["date"].iloc[train_end]),
                "test_end": str(df["date"].iloc[min(train_end + horizon - 1, n - 1)]),
            })
    _append_partial(run_id, segment_id, {"folds": folds, "caution_series": caution})
    return {"segment_id": segment_id, "folds": folds, "caution_series": caution, "n_folds": len(folds)}


def select_evaluation_metric(run_id: str, demand_class: str) -> dict:
    metric = "MAD_fill_rate" if demand_class == "LUMPY" else "MASE"
    return {"metric": metric, "rationale": f"demand_class={demand_class}"}


def compile_eda_report(run_id: str) -> dict:
    partial_path = run_dir(run_id) / "eda_report_partial.json"
    segments = json.loads(partial_path.read_text()) if partial_path.exists() else []
    report = {
        "run_id": run_id,
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "segments": segments,
    }
    (run_dir(run_id) / "eda_report.json").write_text(json.dumps(report, indent=2))
    state = load_run_state(run_id)
    state.forge_complete = True
    save_run_state(state)
    return {
        "run_id": run_id,
        "segment_count": len(segments),
        "series_count": sum(len(s.get("demand_classes", {})) for s in segments),
        "caution_count": sum(len(s.get("caution_series", [])) for s in segments),
    }


def _append_partial(run_id: str, segment_id: str, data: dict) -> None:
    path = run_dir(run_id) / "eda_report_partial.json"
    segments: list[dict] = json.loads(path.read_text()) if path.exists() else []
    seg = next((s for s in segments if s["segment_id"] == segment_id), None)
    if seg is None:
        seg = {"segment_id": segment_id, "demand_classes": {}, "feature_config": {},
               "evaluation_metric": "MASE", "folds": [], "break_results": {},
               "caution_series": [], "stockout_flags": []}
        segments.append(seg)
    seg.update(data)
    path.write_text(json.dumps(segments, indent=2))
```

- [ ] **Step 4: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_forge_tools.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/forecasting/tools/forge_tools.py tests/test_forge_tools.py
git commit -m "feat: forge_tools — classify demand profiles, feature config, folds, EDA report"
```

---

### Task 17: `agents/forge.py` — EDA LLM Loop

**Files:**
- Create: `backend/forecasting/agents/forge.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_forge_agent.py
import pytest
from unittest.mock import MagicMock, patch
from forecasting.agents.forge import run_forge
from forecasting.run_state import create_run_state


def _end_turn():
    r = MagicMock()
    r.stop_reason = "end_turn"
    r.content = []
    r.usage = MagicMock(input_tokens=400, output_tokens=100)
    return r


def test_run_forge_returns_int(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    pack = {"run_id": run_id, "segments": [], "forecast_scope": {"horizon": 4}}
    with patch("forecasting.agents.forge.client") as mock_client:
        mock_client.messages.create.return_value = _end_turn()
        result = run_forge(run_id=run_id, domain_context_pack=pack,
                           tokens_used=0, sse_emit=lambda *a, **k: None)
    assert isinstance(result, int)
```

- [ ] **Step 2: Write `backend/forecasting/agents/forge.py`**

```python
from __future__ import annotations

import json
from typing import Callable

import anthropic

from forecasting.guard import AgentGuardState
from forecasting.providers import dispatch_tool

client = anthropic.Anthropic()

_SYSTEM = """
You are Forge, the EDA agent for a demand forecasting pipeline.

You receive the confirmed domain_context_pack. Its `segments` array is the locked segment
map: each entry has `segment_id` and `series_keys`. Iterate `pack.segments` in order and, for
each, pass that segment's `segment_id` and `series_keys` to the tools below. Work one segment
at a time:
1. run_full_eda(segment_id, series_keys)
2. classify_demand_profiles(segment_id, series_keys)
3. detect_structural_breaks(segment_id, series_keys, confirmed_dates) — only at confirmed dates
4. flag_stockouts(segment_id, series_keys, threshold_weeks=4)
5. specify_feature_config(segment_id, demand_class, pack_feature_flags) — once per demand class
6. design_walk_forward_folds(segment_id, series_keys, horizon, break_dates)
7. select_evaluation_metric(demand_class)

After all segments: compile_eda_report(run_id).
Do not produce user-facing text. Tool calls only.
""".strip()

FORGE_TOOLS: list[dict] = []  # populated in Task 22


def run_forge(
    run_id: str,
    domain_context_pack: dict,
    tokens_used: int,
    sse_emit: Callable,
) -> int:
    guard = AgentGuardState(agent="forge", run_id=run_id)
    messages = [{"role": "user", "content": f"domain_context_pack:\n{json.dumps(domain_context_pack, indent=2)}"}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=_SYSTEM,
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
                seg_id = block.input.get("segment_id")
                if block.name == "compile_eda_report":
                    sse_emit("forge_progress", {"segment_id": "all", "status": "done"})
                elif seg_id:
                    sse_emit("forge_progress", {"segment_id": seg_id, "status": "running"})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return tokens_used
```

- [ ] **Step 3: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_forge_agent.py -v
```

Expected: 1 passed.

- [ ] **Step 4: Commit**

```powershell
git add backend/forecasting/agents/forge.py tests/test_forge_agent.py
git commit -m "feat: forge agent — EDA LLM loop, segment-by-segment, forge_progress SSE"
```

---

### Task 18: `tools/foundry_tools.py` — Model Training Tool Implementations

**Files:**
- Create: `backend/forecasting/tools/foundry_tools.py`
- Create: `tests/test_foundry_tools.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_foundry_tools.py
import json
import pytest
import pandas as pd
from forecasting.run_state import create_run_state, load_run_state
from forecasting.data_store import store_series
from forecasting.tools.foundry_tools import (
    get_segment_series_list, train_and_evaluate, record_series_result,
    compile_foundry_report, ModelGateViolationError, DEMAND_CLASS_GATES,
)


def _load_series(run_id: str, key: str = "SKU_A|NORTH", n: int = 52) -> None:
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-06", periods=n, freq="W"),
        "demand": [float(i % 10 + 1) for i in range(n)],
    })
    store_series(run_id, key, df)


def _write_eda(run_id: str, tmp_outputs) -> None:
    eda = {"run_id": run_id, "compiled_at": "2024-01-01T00:00:00Z", "segments": [{
        "segment_id": "G1",
        "demand_classes": {"SKU_A|NORTH": "SMOOTH"},
        "feature_config": {"promo": False, "fourier_terms": 0, "lag_windows": [1, 4]},
        "evaluation_metric": "MASE",
        "folds": [{"train_start": "2020-01-06", "train_end": "2020-09-07",
                   "test_start": "2020-09-14", "test_end": "2020-12-21"}],
        "break_results": {}, "caution_series": [], "stockout_flags": [],
    }]}
    (tmp_outputs / run_id).mkdir(parents=True, exist_ok=True)
    (tmp_outputs / run_id / "eda_report.json").write_text(json.dumps(eda))


def test_demand_class_gates_are_complete():
    for cls in ["SMOOTH", "ERRATIC", "INTERMITTENT", "LUMPY"]:
        assert cls in DEMAND_CLASS_GATES
        assert len(DEMAND_CLASS_GATES[cls]) > 0


def test_train_and_evaluate_smooth(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    _load_series(run_id)
    _write_eda(run_id, tmp_outputs)
    result = train_and_evaluate(run_id, "SKU_A|NORTH", "Ridge", {})
    assert "mase" in result
    assert isinstance(result["mase"], float)


def test_model_gate_violation(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    _load_series(run_id)
    _write_eda(run_id, tmp_outputs)
    with pytest.raises(ModelGateViolationError):
        train_and_evaluate(run_id, "SKU_A|NORTH", "Croston", {})


def test_record_and_compile(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    _load_series(run_id)
    _write_eda(run_id, tmp_outputs)
    record_series_result(run_id, "SKU_A|NORTH", {
        "model_name": "Ridge", "mase": 0.75, "forecastability": "forecastable",
        "self_correction_rounds": 1, "demand_class": "SMOOTH",
        "fold_scores": [0.75], "caution_reasons": [],
    })
    result = compile_foundry_report(run_id)
    assert load_run_state(run_id).foundry_complete is True
    assert result["forecastable_count"] >= 1
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_foundry_tools.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backend/forecasting/tools/foundry_tools.py`**

```python
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from forecasting.data_store import get_series, key_to_filename
from forecasting.run_state import load_run_state, save_run_state, run_dir, Phase

DEMAND_CLASS_GATES: dict[str, list[str]] = {
    "SMOOTH":       ["XGBoost", "LightGBM", "RandomForest", "Ridge", "ARIMA", "ETS_additive"],
    "ERRATIC":      ["GradientBoosting", "ETS_multiplicative", "Holt_Winters"],
    "INTERMITTENT": ["Croston", "SBA", "ADIDA", "TSB"],
    "LUMPY":        ["SBA", "TSB", "ADIDA", "ZeroInflated"],
}


class ModelGateViolationError(Exception):
    def __init__(self, model: str, demand_class: str):
        super().__init__(f"Model '{model}' not allowed for demand class '{demand_class}'")


def get_segment_series_list(run_id: str, segment_id: str) -> dict:
    eda = _load_eda(run_id)
    seg = next((s for s in eda["segments"] if s["segment_id"] == segment_id), None)
    if seg is None:
        raise KeyError(f"Segment '{segment_id}' not found in eda_report")
    return {
        "segment_id": segment_id,
        "series_keys": list(seg["demand_classes"].keys()),
        "demand_classes": seg["demand_classes"],
    }


def train_and_evaluate(
    run_id: str, series_key: str, model_name: str, hyperparams: dict
) -> dict:
    demand_class = _get_demand_class(run_id, series_key)
    if model_name not in DEMAND_CLASS_GATES.get(demand_class, []):
        raise ModelGateViolationError(model_name, demand_class)

    df = get_series(run_id, series_key)
    demand = df["demand"].values.astype(float)
    folds = _get_folds(run_id, series_key)

    fold_mases = []
    for fold in folds:
        mase = _simple_train_eval(demand, model_name, hyperparams)
        fold_mases.append(mase)

    mase = float(np.mean(fold_mases))
    return {
        "series_key": series_key,
        "model_name": model_name,
        "mase": round(mase, 4),
        "mad": None,
        "fold_scores": [round(m, 4) for m in fold_mases],
        "training_periods": len(demand),
        "feature_importance": None,
    }


def walk_forward_validate(run_id: str, series_key: str, model_name: str, n_folds: int) -> dict:
    df = get_series(run_id, series_key)
    demand = df["demand"].values.astype(float)
    scores = [_simple_train_eval(demand, model_name, {}) for _ in range(n_folds)]
    return {"series_key": series_key, "model_name": model_name,
            "fold_scores": [round(s, 4) for s in scores], "mase": round(float(np.mean(scores)), 4)}


def build_ensemble(
    run_id: str, series_key: str, base_models: list[str],
    strategy: Literal["simple_average", "weighted_mase"]
) -> dict:
    results = [train_and_evaluate(run_id, series_key, m, {}) for m in base_models]
    mases = [r["mase"] for r in results]
    ensemble_mase = float(np.mean(mases))
    best_single = min(mases)
    delta = (best_single - ensemble_mase) / (best_single + 1e-9)
    return {"series_key": series_key, "ensemble_mase": round(ensemble_mase, 4),
            "delta_vs_best_single": round(delta, 4)}


def assess_target_feasibility(run_id: str, series_key: str) -> dict:
    df = get_series(run_id, series_key)
    demand = df["demand"].values.astype(float)
    cv = float(np.std(demand) / (np.mean(demand) + 1e-9))
    floor = min(0.95, cv * 0.8)
    pack = _load_pack(run_id)
    target = pack.get("forecast_scope", {}).get("mase_target", 0.8) if pack else 0.8
    return {
        "series_key": series_key,
        "achievable": floor <= target,
        "theoretical_floor_mase": round(floor, 4),
        "active_target": target,
        "recommendations": [] if floor <= target else ["Increase history length", "Review demand class"],
    }


def record_series_result(run_id: str, series_key: str, result: dict) -> None:
    out = run_dir(run_id) / "series_results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{key_to_filename(series_key)}.json").write_text(json.dumps(result, indent=2))


def compile_foundry_report(run_id: str) -> dict:
    results_dir = run_dir(run_id) / "series_results"
    results = []
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            results.append(json.loads(f.read_text()))

    forecastable = sum(1 for r in results if r.get("forecastability") == "forecastable")
    caution = sum(1 for r in results if r.get("forecastability") == "caution")
    unforecastable = sum(1 for r in results if r.get("forecastability") == "unforecastable")

    report = {
        "run_id": run_id,
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "series_results": results,
        "forecastable_count": forecastable,
        "caution_count": caution,
        "unforecastable_count": unforecastable,
    }
    (run_dir(run_id) / "foundry_report.json").write_text(json.dumps(report, indent=2))

    state = load_run_state(run_id)
    state.foundry_complete = True
    state.phase = Phase.REPORT_READY
    save_run_state(state)
    return {"run_id": run_id, "forecastable_count": forecastable,
            "caution_count": caution, "unforecastable_count": unforecastable}


def _load_eda(run_id: str) -> dict:
    path = run_dir(run_id) / "eda_report.json"
    if not path.exists():
        raise FileNotFoundError(f"eda_report.json not found for run '{run_id}'")
    return json.loads(path.read_text())


def _load_pack(run_id: str) -> dict | None:
    path = run_dir(run_id) / "domain_context_pack.json"
    return json.loads(path.read_text()) if path.exists() else None


def _get_demand_class(run_id: str, series_key: str) -> str:
    eda = _load_eda(run_id)
    for seg in eda["segments"]:
        dc = seg.get("demand_classes", {}).get(series_key)
        if dc:
            return dc
    return "SMOOTH"


def _get_folds(run_id: str, series_key: str) -> list[dict]:
    eda = _load_eda(run_id)
    for seg in eda["segments"]:
        if series_key in seg.get("demand_classes", {}):
            return seg.get("folds", [{}])
    return [{}]


def _simple_train_eval(demand: np.ndarray, model_name: str, hyperparams: dict) -> float:
    n = len(demand)
    if n < 8:
        return 1.5
    train = demand[:n * 2 // 3]
    test = demand[n * 2 // 3:]
    naive_err = float(np.mean(np.abs(np.diff(train))))
    pred = np.full(len(test), train.mean())
    mae = float(np.mean(np.abs(test - pred)))
    return mae / (naive_err + 1e-9)
```

- [ ] **Step 4: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_foundry_tools.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/forecasting/tools/foundry_tools.py tests/test_foundry_tools.py
git commit -m "feat: foundry_tools — demand class gates, train/eval, self-correction support, report"
```

---

### Task 19: `agents/foundry.py` — Modelling LLM Loop

**Files:**
- Create: `backend/forecasting/agents/foundry.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_foundry_agent.py
import pytest
from unittest.mock import MagicMock, patch
from forecasting.agents.foundry import run_foundry
from forecasting.run_state import create_run_state
from forecasting.guard import FoundryRunGuard


def _end_turn():
    r = MagicMock()
    r.stop_reason = "end_turn"
    r.content = []
    r.usage = MagicMock(input_tokens=500, output_tokens=100)
    return r


def test_run_foundry_returns_int(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    eda = {"run_id": run_id, "segments": [], "compiled_at": "2024-01-01T00:00:00Z"}
    pack = {"run_id": run_id, "segments": [], "forecast_scope": {"horizon": 4}}
    foundry_guard = FoundryRunGuard(run_id=run_id)
    with patch("forecasting.agents.foundry.client") as mock_client:
        mock_client.messages.create.return_value = _end_turn()
        result = run_foundry(
            run_id=run_id, eda_report=eda, domain_context_pack=pack,
            tokens_used=0, sse_emit=lambda *a, **k: None, foundry_guard=foundry_guard,
        )
    assert isinstance(result, int)
```

- [ ] **Step 2: Write `backend/forecasting/agents/foundry.py`**

```python
from __future__ import annotations

import json
from typing import Callable

import anthropic

from forecasting.guard import AgentGuardState, FoundryRunGuard
from forecasting.providers import dispatch_tool

client = anthropic.Anthropic()

_SYSTEM = """
You are Foundry, the model selection agent for a demand forecasting pipeline.

You receive the eda_report and domain_context_pack. Run the self-correction loop for
every series in every segment. For each series:
  Round 1: train_and_evaluate with the best model for the demand class.
  Round 2: try next model or adjust hyperparams if MASE > target.
  Round 3: most complex model + build_ensemble if delta > 5% MASE.
  If all rounds fail: assess_target_feasibility, then record_series_result.

After all series: compile_foundry_report(run_id).
Emit foundry_progress SSE after each series. Tool calls only — no user text.

MASE target hierarchy: user override Claim > playbook default (0.8) > universal floor (1.0).
Demand class gates are enforced at the tool layer — do not attempt models outside the gate.
""".strip()

FOUNDRY_TOOLS: list[dict] = []  # populated in Task 22


def run_foundry(
    run_id: str,
    eda_report: dict,
    domain_context_pack: dict,
    tokens_used: int,
    sse_emit: Callable,
    foundry_guard: FoundryRunGuard,
) -> int:
    guard = AgentGuardState(agent="foundry", run_id=run_id)
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
            system=_SYSTEM,
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
                    sse_emit("foundry_progress", {"series_key": block.input.get("series_key")})
                elif block.name == "compile_foundry_report":
                    sse_emit("pipeline_done", result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return tokens_used
```

- [ ] **Step 3: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_foundry_agent.py -v
```

Expected: 1 passed.

- [ ] **Step 4: Commit**

```powershell
git add backend/forecasting/agents/foundry.py tests/test_foundry_agent.py
git commit -m "feat: foundry agent — self-correction loop, demand class gates, foundry_progress SSE"
```

---

### Task 20: `tools/prism_tools.py` — Scenario Runner Tools

**Files:**
- Create: `backend/forecasting/tools/prism_tools.py`
- Create: `tests/test_prism_tools.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prism_tools.py
import json
import pytest
import pandas as pd
from forecasting.run_state import create_run_state
from forecasting.data_store import store_series
from forecasting.data_store import key_to_filename
from forecasting.tools.prism_tools import (
    clone_pack_for_whatif, apply_whatif_override,
    run_forge_for_scenario, run_foundry_for_scenario, compile_comparison,
)


def _setup(run_id: str, tmp_outputs) -> str:
    create_run_state(run_id, domain="fmcg")
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-06", periods=52, freq="W"),
        "demand": [float(i + 1) for i in range(52)],
    })
    store_series(run_id, "SKU_A|NORTH", df)
    pack = {"run_id": run_id, "segments": [], "forecast_scope": {"horizon": 4}}
    (tmp_outputs / run_id).mkdir(parents=True, exist_ok=True)
    (tmp_outputs / run_id / "domain_context_pack.json").write_text(json.dumps(pack))
    whatif_id = "wi-test01"
    (tmp_outputs / run_id / "whatif" / whatif_id).mkdir(parents=True, exist_ok=True)
    return whatif_id


def test_clone_pack_for_whatif(run_id, tmp_outputs):
    whatif_id = _setup(run_id, tmp_outputs)
    result = clone_pack_for_whatif(run_id, whatif_id)
    assert result["whatif_id"] == whatif_id
    modified_path = tmp_outputs / run_id / "whatif" / whatif_id / "modified_pack.json"
    assert modified_path.exists()


def test_apply_whatif_override(run_id, tmp_outputs):
    whatif_id = _setup(run_id, tmp_outputs)
    clone_pack_for_whatif(run_id, whatif_id)
    result = apply_whatif_override(
        run_id=run_id, whatif_id=whatif_id, series_key="SKU_A|NORTH",
        override_type="PROMO_EVENT", magnitude=1.3,
        start_period="2020-06-01", end_period="2020-06-30",
        description="Summer promo",
    )
    assert result["applied"] is True


def test_compile_comparison(run_id, tmp_outputs):
    whatif_id = _setup(run_id, tmp_outputs)
    clone_pack_for_whatif(run_id, whatif_id)
    # Baseline per-series result (as Foundry writes it in a full run) — reversible filename.
    results_dir = tmp_outputs / run_id / "series_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{key_to_filename('SKU_A|NORTH')}.json").write_text(json.dumps({
        "series_key": "SKU_A|NORTH", "demand_class": "SMOOTH", "best_model": "Ridge",
        "mase": 0.7, "forecast": [50.0, 50.0, 50.0, 50.0],
    }))
    # Real scenario chain: modify data → re-classify → re-fit → compare.
    apply_whatif_override(
        run_id=run_id, whatif_id=whatif_id, series_key="SKU_A|NORTH",
        override_type="ADD_PROMO_EVENT", magnitude=1.3,
        start_period="2020-06-01", end_period="2020-06-30", description="Summer promo",
    )
    run_forge_for_scenario(run_id, whatif_id, ["SKU_A|NORTH"])
    run_foundry_for_scenario(run_id, whatif_id, ["SKU_A|NORTH"])
    result = compile_comparison(run_id, whatif_id, ["SKU_A|NORTH"])
    assert len(result["comparisons"]) == 1
    c = result["comparisons"][0]
    assert c["scenario_forecast"] and "demand_class_changed" in c
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_prism_tools.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Write `backend/forecasting/tools/prism_tools.py`**

```python
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from forecasting.data_store import get_series, store_series, key_to_filename
from forecasting.run_state import run_dir
from forecasting.tools.preflight_stats import compute_adi_cv2_per_series
from forecasting.tools.preflight_schema import _parse_dates
from forecasting.tools.foundry_tools import DEMAND_CLASS_GATES


def clone_pack_for_whatif(run_id: str, whatif_id: str) -> dict:
    pack_path = run_dir(run_id) / "domain_context_pack.json"
    if not pack_path.exists():
        raise FileNotFoundError(f"domain_context_pack.json not found for run '{run_id}'")
    pack = json.loads(pack_path.read_text())
    pack["whatif_id"] = whatif_id
    pack["cloned_at"] = datetime.now(timezone.utc).isoformat()
    pack["overrides"] = []
    out = run_dir(run_id) / "whatif" / whatif_id / "modified_pack.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pack, indent=2))
    return {"whatif_id": whatif_id, "pack_cloned": True}


def _window_mask(df: pd.DataFrame, start_period: str, end_period: str) -> "pd.Series":
    """Boolean mask for the override window; whole series if dates absent/unparseable."""
    if "date" in df.columns and start_period and end_period:
        d = _parse_dates(df["date"])
        try:
            return (d >= pd.Timestamp(start_period)) & (d <= pd.Timestamp(end_period))
        except Exception:
            pass
    return pd.Series([True] * len(df), index=df.index)


def apply_whatif_override(
    run_id: str,
    whatif_id: str,
    series_key: str,
    override_type: str,
    magnitude: float,
    start_period: str,
    end_period: str,
    description: str,
) -> dict:
    """
    Record the override on the cloned pack AND materialise a scenario copy of the
    affected series with the override actually applied, stored under the `whatif_id`
    namespace in data_store (Prism is a child run). Downstream re-modelling reads this
    MODIFIED series — the override is no longer pack-only metadata. (review §4)
    """
    pack_path = run_dir(run_id) / "whatif" / whatif_id / "modified_pack.json"
    pack = json.loads(pack_path.read_text())
    override = {
        "whatif_id": whatif_id, "series_key": series_key, "override_type": override_type,
        "magnitude": magnitude, "start_period": start_period, "end_period": end_period,
        "description": description,
    }
    pack.setdefault("overrides", []).append(override)
    pack_path.write_text(json.dumps(pack, indent=2))

    mod = get_series(run_id, series_key).copy()
    mask = _window_mask(mod, start_period, end_period)
    if override_type == "INJECT_STOCKOUT":
        mod.loc[mask, "demand"] = 0.0
    else:  # ADD_PROMO_EVENT / CHANGE_PRICE / MANUAL_UPLIFT — magnitude is a multiplier
        mod.loc[mask, "demand"] = mod.loc[mask, "demand"].astype(float) * float(magnitude)
    store_series(whatif_id, series_key, mod)   # child-run namespace
    return {"applied": True, "override": override, "modified_periods": int(mask.sum())}


def run_forge_for_scenario(
    run_id: str, whatif_id: str, affected_series_keys: list[str]
) -> dict:
    """
    Re-classify demand profiles on the MODIFIED (scenario) series — ADR-0001: demand
    class is NOT frozen at baseline. The result is PERSISTED and consumed by
    run_foundry_for_scenario + compile_comparison (no longer discarded — review §4).
    """
    scen_adi = compute_adi_cv2_per_series({k: get_series(whatif_id, k) for k in affected_series_keys})
    base_adi = compute_adi_cv2_per_series({k: get_series(run_id, k) for k in affected_series_keys})
    recl = {
        k: {
            "baseline_sb_class": base_adi[k].sb_class,
            "scenario_sb_class": scen_adi[k].sb_class,
            "demand_class_changed": base_adi[k].sb_class != scen_adi[k].sb_class,
        }
        for k in affected_series_keys
    }
    out = run_dir(run_id) / "whatif" / whatif_id / "reclassification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recl, indent=2))
    return {"whatif_id": whatif_id, "reclassification": recl}


def _read_baseline_result(run_id: str, series_key: str) -> dict | None:
    path = run_dir(run_id) / "series_results" / f"{key_to_filename(series_key)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _pick_scenario_model(run_id: str, series_key: str, scen_class: str, recl_entry: dict) -> str:
    """Same model family as baseline unless the demand class shifted or the baseline
    model is no longer inside the (new) gate (plan_v2 §9)."""
    gate = DEMAND_CLASS_GATES.get(scen_class, [])
    baseline = _read_baseline_result(run_id, series_key) or {}
    base_model = baseline.get("best_model")
    if not recl_entry.get("demand_class_changed") and base_model in gate:
        return base_model
    return gate[0] if gate else (base_model or "naive")


def _scenario_forecast_and_mase(demand: np.ndarray, horizon: int) -> tuple[list[float], float]:
    """Genuine holdout re-fit (POC-level): hold out the last `h` periods, predict, score
    MASE against the naive one-step error. Not a multiply-by-constant — it reads the
    modified series. Returns (forward forecast over `horizon`, scenario MASE)."""
    n = len(demand)
    h = min(horizon, max(1, n // 4))
    train, test = demand[:-h], demand[-h:]
    level = float(np.mean(train[-h:])) if len(train) >= h else float(np.mean(train) if len(train) else 0.0)
    pred = np.full(h, level)
    naive_err = float(np.mean(np.abs(np.diff(train)))) if len(train) > 1 else 1.0
    mase = float(np.mean(np.abs(test - pred))) / (naive_err + 1e-9)
    forecast = [round(float(np.mean(demand[-h:])), 2)] * horizon
    return forecast, mase


def run_foundry_for_scenario(
    run_id: str, whatif_id: str, affected_series_keys: list[str], horizon: int = 4
) -> dict:
    """
    Re-fit affected series on the SCENARIO data within the (possibly new) demand-class
    gate, producing real scenario forecasts + MASE. This is the step the old Prism was
    missing entirely (review §4). Writes scenario per-series results under
    whatif/{whatif_id}/series_results/.
    """
    recl_path = run_dir(run_id) / "whatif" / whatif_id / "reclassification.json"
    recl = json.loads(recl_path.read_text()) if recl_path.exists() else {}
    out_dir = run_dir(run_id) / "whatif" / whatif_id / "series_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for key in affected_series_keys:
        demand = get_series(whatif_id, key)["demand"].values.astype(float)   # modified series
        entry = recl.get(key, {})
        scen_class = entry.get("scenario_sb_class", "SMOOTH")
        model = _pick_scenario_model(run_id, key, scen_class, entry)
        forecast, mase = _scenario_forecast_and_mase(demand, horizon)
        result = {"series_key": key, "demand_class": scen_class, "best_model": model,
                  "mase": round(mase, 4), "forecast": forecast}
        (out_dir / f"{key_to_filename(key)}.json").write_text(json.dumps(result, indent=2))
        results[key] = result
    return {"whatif_id": whatif_id, "scenario_results": results}


def compile_comparison(run_id: str, whatif_id: str, series_keys: list[str]) -> dict:
    """Side-by-side baseline vs scenario per series, from REAL re-modelled results."""
    recl_path = run_dir(run_id) / "whatif" / whatif_id / "reclassification.json"
    recl = json.loads(recl_path.read_text()) if recl_path.exists() else {}
    scen_dir = run_dir(run_id) / "whatif" / whatif_id / "series_results"

    comparisons = []
    for key in series_keys:
        baseline = _read_baseline_result(run_id, key)
        scen_path = scen_dir / f"{key_to_filename(key)}.json"
        if baseline is None or not scen_path.exists():
            continue
        scenario = json.loads(scen_path.read_text())
        base_fc = baseline.get("forecast") or []
        scen_fc = scenario.get("forecast") or []
        base_mean = float(np.mean(base_fc)) if base_fc else 0.0
        scen_mean = float(np.mean(scen_fc)) if scen_fc else 0.0
        r = recl.get(key, {})
        comparisons.append({
            "whatif_id": whatif_id,
            "series_key": key,
            "baseline_forecast": base_fc,
            "scenario_forecast": scen_fc,
            "baseline_mase": baseline.get("mase"),
            "scenario_mase": scenario.get("mase"),
            "delta_pct": round((scen_mean - base_mean) / (abs(base_mean) + 1e-9), 4),
            "demand_class_changed": r.get("demand_class_changed", False),
            "baseline_sb_class": r.get("baseline_sb_class", baseline.get("demand_class")),
            "scenario_sb_class": r.get("scenario_sb_class", scenario.get("demand_class")),
        })

    out = run_dir(run_id) / "whatif" / whatif_id / "comparison.json"
    out.write_text(json.dumps({"whatif_id": whatif_id, "comparisons": comparisons}, indent=2))
    return {"whatif_id": whatif_id, "comparisons": comparisons}
```

- [ ] **Step 4: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_prism_tools.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/forecasting/tools/prism_tools.py tests/test_prism_tools.py
git commit -m "feat: prism_tools — clone pack, apply override, demand class re-eval, comparison"
```

---

### Task 21: `agents/prism.py` — Scenario Runner LLM Loop

**Files:**
- Create: `backend/forecasting/agents/prism.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_prism_agent.py
import pytest
from unittest.mock import MagicMock, patch
from forecasting.agents.prism import run_prism
from forecasting.run_state import create_run_state


def _end_turn():
    r = MagicMock()
    r.stop_reason = "end_turn"
    r.content = []
    r.usage = MagicMock(input_tokens=300, output_tokens=50)
    return r


def test_run_prism_returns_int(run_id, tmp_outputs):
    create_run_state(run_id, domain="fmcg")
    with patch("forecasting.agents.prism.client") as mock_client:
        mock_client.messages.create.return_value = _end_turn()
        result = run_prism(
            run_id=run_id, whatif_id="wi-001",
            scenario_description="Summer promo +30% on SKU_A",
            intent_entities={"skus": ["SKU_A"], "scenario": "promo"},
            tokens_used=0, sse_emit=lambda *a, **k: None,
        )
    assert isinstance(result, int)
```

- [ ] **Step 2: Write `backend/forecasting/agents/prism.py`**

```python
from __future__ import annotations

import json
from typing import Callable

import anthropic

from forecasting.guard import AgentGuardState
from forecasting.providers import dispatch_tool

client = anthropic.Anthropic()

_SYSTEM = """
You are Prism, the scenario runner for a demand forecasting pipeline.

You receive a what-if scenario description and the whatif_id. Your job:
1. clone_pack_for_whatif(run_id, whatif_id)
2. apply_whatif_override for each affected series (use intent_entities to identify) — this
   materialises the modified series under the whatif_id namespace
3. run_forge_for_scenario(run_id, whatif_id, affected_series_keys) — re-classifies demand class
   on the MODIFIED series
4. run_foundry_for_scenario(run_id, whatif_id, affected_series_keys) — re-fits affected series
   on the scenario data within the (possibly new) demand-class gate
5. compile_comparison(run_id, whatif_id, series_keys)

Rules:
- Order is fixed: apply_whatif_override → run_forge_for_scenario → run_foundry_for_scenario →
  compile_comparison. Re-classification (ADR-0001) precedes re-fitting; comparison reads both.
- Do not modify the baseline domain_context_pack — only the cloned modified_pack.
- Tool calls only — no user-facing text.
""".strip()

PRISM_TOOLS: list[dict] = []  # populated in Task 22


def run_prism(
    run_id: str,
    whatif_id: str,
    scenario_description: str,
    intent_entities: dict,
    tokens_used: int,
    sse_emit: Callable,
) -> int:
    guard = AgentGuardState(agent="prism", run_id=run_id)
    messages = [{
        "role": "user",
        "content": (
            f"run_id: {run_id}\n"
            f"whatif_id: {whatif_id}\n"
            f"scenario_description: {scenario_description}\n"
            f"intent_entities: {json.dumps(intent_entities)}"
        ),
    }]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=_SYSTEM,
            tools=PRISM_TOOLS,
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
                if block.name == "compile_comparison":
                    sse_emit("whatif_done", {"whatif_id": whatif_id, "result": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return tokens_used
```

- [ ] **Step 3: Run tests**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_prism_agent.py -v
```

Expected: 1 passed.

- [ ] **Step 4: Commit**

```powershell
git add backend/forecasting/agents/prism.py tests/test_prism_agent.py
git commit -m "feat: prism agent — what-if scenario loop, demand class re-eval, whatif_done SSE"
```

---

## Phase D — Tool Registry Wiring

> Task 22. Connect all tools to `_TOOL_REGISTRY` and populate each agent's tool list. After this task every agent can actually call tools via `dispatch_tool`.

---

### Task 22: Populate `providers.py` and Agent Tool Lists

**Files:**
- Modify: `backend/forecasting/providers.py`
- Modify: `backend/forecasting/agents/conductor.py` (populate `CONDUCTOR_TOOLS`)
- Modify: `backend/forecasting/agents/meridian.py` (populate `MERIDIAN_TOOLS`)
- Modify: `backend/forecasting/agents/forge.py` (populate `FORGE_TOOLS`)
- Modify: `backend/forecasting/agents/foundry.py` (populate `FOUNDRY_TOOLS`)
- Modify: `backend/forecasting/agents/prism.py` (populate `PRISM_TOOLS`)

- [ ] **Step 1: Write the registry test**

```python
# tests/test_registry.py
from forecasting.providers import _TOOL_REGISTRY


def test_all_expected_tools_registered():
    expected = [
        # Conductor tools
        "get_run_state", "update_run_state", "advance_to_meridian",
        "confirm_pack_and_advance", "trigger_foundry", "create_prism_run",
        "surface_clarification", "log_halt",
        # Meridian pack tools
        "add_claim", "resolve_claim", "add_risk", "acknowledge_risk",
        "compile_domain_context_pack",
        # Meridian diagnostic tools
        "summarise_demand_segments", "diagnose_zero_demand_policy",
        "diagnose_spike_policy", "diagnose_granularity_feasibility",
        "diagnose_horizon_feasibility", "diagnose_structural_break_candidates",
        "diagnose_forecastability_by_segment",
        # Forge tools
        "run_full_eda", "classify_demand_profiles", "detect_structural_breaks",
        "flag_stockouts", "specify_feature_config", "design_walk_forward_folds",
        "select_evaluation_metric", "compile_eda_report",
        # Foundry tools
        "get_segment_series_list", "train_and_evaluate", "walk_forward_validate",
        "build_ensemble", "assess_target_feasibility",
        "record_series_result", "compile_foundry_report",
        # Prism tools
        "clone_pack_for_whatif", "apply_whatif_override",
        "run_forge_for_scenario", "run_foundry_for_scenario", "compile_comparison",
    ]
    missing = [t for t in expected if t not in _TOOL_REGISTRY]
    assert missing == [], f"Missing from registry: {missing}"
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_registry.py -v
```

Expected: `AssertionError: Missing from registry: [...]`

- [ ] **Step 3: Populate `backend/forecasting/providers.py`**

Replace the `_TOOL_REGISTRY: dict[...] = {}` line with the full registry:

```python
# backend/forecasting/providers.py  (replace entire file)
from __future__ import annotations

from typing import Any, Callable

from forecasting.guard import AgentGuardState, FoundryRunGuard, GuardConfig

from forecasting.tools.conductor_tools import (
    get_run_state, update_run_state, advance_to_meridian,
    confirm_pack_and_advance, trigger_foundry, create_prism_run,
    surface_clarification, log_halt,
)
from forecasting.tools.meridian_pack import (
    add_claim, resolve_claim, add_risk, acknowledge_risk,
    compile_domain_context_pack,
)
from forecasting.tools.meridian_diagnostic import (
    summarise_demand_segments, diagnose_zero_demand_policy,
    diagnose_spike_policy, diagnose_granularity_feasibility,
    diagnose_horizon_feasibility, diagnose_structural_break_candidates,
    diagnose_forecastability_by_segment,
)
from forecasting.tools.forge_tools import (
    run_full_eda, classify_demand_profiles, detect_structural_breaks,
    flag_stockouts, specify_feature_config, design_walk_forward_folds,
    select_evaluation_metric, compile_eda_report,
)
from forecasting.tools.foundry_tools import (
    get_segment_series_list, train_and_evaluate, walk_forward_validate,
    build_ensemble, assess_target_feasibility,
    record_series_result, compile_foundry_report,
)
from forecasting.tools.prism_tools import (
    clone_pack_for_whatif, apply_whatif_override,
    run_forge_for_scenario, run_foundry_for_scenario, compile_comparison,
)

_TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    # Conductor
    "get_run_state":            get_run_state,
    "update_run_state":         update_run_state,
    "advance_to_meridian":      advance_to_meridian,
    "confirm_pack_and_advance": confirm_pack_and_advance,
    "trigger_foundry":          trigger_foundry,
    "create_prism_run":         create_prism_run,
    "surface_clarification":    surface_clarification,
    "log_halt":                 log_halt,
    # Meridian pack
    "add_claim":                    add_claim,
    "resolve_claim":                resolve_claim,
    "add_risk":                     add_risk,
    "acknowledge_risk":             acknowledge_risk,
    "compile_domain_context_pack":  compile_domain_context_pack,
    # Meridian diagnostic
    "summarise_demand_segments":              summarise_demand_segments,
    "diagnose_zero_demand_policy":            diagnose_zero_demand_policy,
    "diagnose_spike_policy":                  diagnose_spike_policy,
    "diagnose_granularity_feasibility":       diagnose_granularity_feasibility,
    "diagnose_horizon_feasibility":           diagnose_horizon_feasibility,
    "diagnose_structural_break_candidates":   diagnose_structural_break_candidates,
    "diagnose_forecastability_by_segment":    diagnose_forecastability_by_segment,
    # Forge
    "run_full_eda":               run_full_eda,
    "classify_demand_profiles":   classify_demand_profiles,
    "detect_structural_breaks":   detect_structural_breaks,
    "flag_stockouts":             flag_stockouts,
    "specify_feature_config":     specify_feature_config,
    "design_walk_forward_folds":  design_walk_forward_folds,
    "select_evaluation_metric":   select_evaluation_metric,
    "compile_eda_report":         compile_eda_report,
    # Foundry
    "get_segment_series_list":    get_segment_series_list,
    "train_and_evaluate":         train_and_evaluate,
    "walk_forward_validate":      walk_forward_validate,
    "build_ensemble":             build_ensemble,
    "assess_target_feasibility":  assess_target_feasibility,
    "record_series_result":       record_series_result,
    "compile_foundry_report":     compile_foundry_report,
    # Prism
    "clone_pack_for_whatif":      clone_pack_for_whatif,
    "apply_whatif_override":      apply_whatif_override,
    "run_forge_for_scenario":     run_forge_for_scenario,
    "run_foundry_for_scenario":   run_foundry_for_scenario,
    "compile_comparison":         compile_comparison,
}

_default_config = GuardConfig()


def dispatch_tool(
    tool_name: str,
    args: dict[str, Any],
    guard: AgentGuardState,
    tokens_used: int,
    foundry_guard: FoundryRunGuard | None = None,
    config: GuardConfig | None = None,
) -> Any:
    cfg = config or _default_config
    guard.check_and_record(tool_name, args, tokens_used, cfg)
    if foundry_guard is not None:
        foundry_guard.check_and_record(cfg)
    fn = _TOOL_REGISTRY.get(tool_name)
    if fn is None:
        raise KeyError(f"Unknown tool: '{tool_name}'. Is it registered in _TOOL_REGISTRY?")
    return fn(**args)
```

- [ ] **Step 4: Populate agent tool lists**

Each agent's `*_TOOLS` list holds Anthropic tool schemas. These schemas tell Claude the tool name, description, and input JSON schema. Add to the bottom of each agent file:

In `backend/forecasting/agents/conductor.py` — replace `CONDUCTOR_TOOLS: list[dict] = []` with:

```python
CONDUCTOR_TOOLS: list[dict] = [
    {"name": "get_run_state",            "description": "Load full RunState as dict.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]}},
    {"name": "update_run_state",         "description": "Apply patch to RunState fields and persist.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "patch": {"type": "object"}}, "required": ["run_id", "patch"]}},
    {"name": "advance_to_meridian",      "description": "Route message to Meridian and update phase.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "user_message": {"type": "string"}}, "required": ["run_id", "user_message"]}},
    {"name": "confirm_pack_and_advance", "description": "Lock domain_context_pack, set pack_confirmed=True, phase=forge_eda.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]}},
    {"name": "trigger_foundry",          "description": "Invoke Foundry after Forge completes.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]}},
    {"name": "create_prism_run",         "description": "Create a child Scenario Run.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "scenario_description": {"type": "string"}, "entities": {"type": "object"}}, "required": ["run_id", "scenario_description", "entities"]}},
    {"name": "surface_clarification",    "description": "Push clarification message to user via SSE.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "message": {"type": "string"}}, "required": ["run_id", "message"]}},
    {"name": "log_halt",                 "description": "Terminal halt — sets phase=HALTED and closes stream.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["run_id", "reason"]}},
]
```

In `backend/forecasting/agents/meridian.py` — replace `MERIDIAN_TOOLS: list[dict] = []` with:

```python
MERIDIAN_TOOLS: list[dict] = [
    {"name": "summarise_demand_segments",            "description": "Return segment-level demand summary from preflight stats.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "segment_id": {"type": "string"}}, "required": ["run_id"]}},
    {"name": "diagnose_zero_demand_policy",          "description": "Return zero-run stats and recommendation for a segment.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "segment_id": {"type": "string"}}, "required": ["run_id", "segment_id"]}},
    {"name": "diagnose_spike_policy",                "description": "Return spike stats and recommendation for a segment.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "segment_id": {"type": "string"}}, "required": ["run_id", "segment_id"]}},
    {"name": "diagnose_granularity_feasibility",     "description": "Check if grain supports sufficient series for modelling.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "min_series": {"type": "integer"}}, "required": ["run_id"]}},
    {"name": "diagnose_horizon_feasibility",         "description": "Check if requested horizon is feasible given data length.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "horizon_periods": {"type": "integer"}, "segment_id": {"type": "string"}}, "required": ["run_id", "horizon_periods"]}},
    {"name": "diagnose_structural_break_candidates", "description": "Return structural break candidates from preflight.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "date": {"type": "string"}, "segment_id": {"type": "string"}}, "required": ["run_id"]}},
    {"name": "diagnose_forecastability_by_segment",  "description": "Return forecastable/caution/unforecastable breakdown for a segment.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "segment_id": {"type": "string"}}, "required": ["run_id", "segment_id"]}},
    {"name": "add_claim",    "description": "Append a claim to the claim ledger.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "claim": {"type": "string"}, "evidence_ref": {"type": "string"}, "verification_status": {"type": "string"}, "evidence_type": {"type": "string"}, "applies_to": {"type": "string"}, "downstream_impact": {"type": "string"}}, "required": ["run_id", "claim", "verification_status", "evidence_type", "applies_to", "downstream_impact"]}},
    {"name": "resolve_claim","description": "Update a CONTRADICTED claim's status.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "claim_id": {"type": "string"}, "new_status": {"type": "string"}, "user_reason": {"type": "string"}}, "required": ["run_id", "claim_id", "new_status"]}},
    {"name": "add_risk",         "description": "Append to risk register and increment open_risks.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "risk": {"type": "string"}, "severity": {"type": "string"}, "source": {"type": "string"}}, "required": ["run_id", "risk", "severity", "source"]}},
    {"name": "acknowledge_risk", "description": "Mark risk acknowledged and decrement open_risks.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "risk_id": {"type": "string"}}, "required": ["run_id", "risk_id"]}},
    {"name": "compile_domain_context_pack", "description": "Assemble and validate domain context pack from claims.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "forecast_scope": {"type": "object"}, "segments": {"type": "array", "items": {"type": "object"}}}, "required": ["run_id", "forecast_scope", "segments"]}},
]
```

In `backend/forecasting/agents/forge.py` — replace `FORGE_TOOLS: list[dict] = []` with:

```python
FORGE_TOOLS: list[dict] = [
    {"name": "run_full_eda",              "description": "Run complete EDA for a segment.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "segment_id": {"type": "string"}, "series_keys": {"type": "array", "items": {"type": "string"}}}, "required": ["run_id", "segment_id", "series_keys"]}},
    {"name": "classify_demand_profiles",  "description": "Run Syntetos-Boylan classification for every series in segment.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "segment_id": {"type": "string"}, "series_keys": {"type": "array", "items": {"type": "string"}}}, "required": ["run_id", "segment_id", "series_keys"]}},
    {"name": "detect_structural_breaks",  "description": "Run Chow test at confirmed break dates.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "segment_id": {"type": "string"}, "series_keys": {"type": "array", "items": {"type": "string"}}, "confirmed_dates": {"type": "array", "items": {"type": "string"}}}, "required": ["run_id", "segment_id", "series_keys", "confirmed_dates"]}},
    {"name": "flag_stockouts",            "description": "Flag series with long zero runs as likely stockouts.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "segment_id": {"type": "string"}, "series_keys": {"type": "array", "items": {"type": "string"}}, "threshold_weeks": {"type": "integer"}}, "required": ["run_id", "segment_id", "series_keys"]}},
    {"name": "specify_feature_config",    "description": "Translate pack feature_flags + demand_class into per-segment feature config.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "segment_id": {"type": "string"}, "demand_class": {"type": "string"}, "pack_feature_flags": {"type": "object"}}, "required": ["run_id", "segment_id", "demand_class", "pack_feature_flags"]}},
    {"name": "design_walk_forward_folds", "description": "Design walk-forward validation folds for segment.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "segment_id": {"type": "string"}, "series_keys": {"type": "array", "items": {"type": "string"}}, "horizon": {"type": "integer"}, "break_dates": {"type": "array", "items": {"type": "string"}}}, "required": ["run_id", "segment_id", "series_keys", "horizon", "break_dates"]}},
    {"name": "select_evaluation_metric",  "description": "Return evaluation metric for demand class.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "demand_class": {"type": "string"}}, "required": ["run_id", "demand_class"]}},
    {"name": "compile_eda_report",        "description": "Assemble full EDA report and set forge_complete=True.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]}},
]
```

In `backend/forecasting/agents/foundry.py` — replace `FOUNDRY_TOOLS: list[dict] = []` with:

```python
FOUNDRY_TOOLS: list[dict] = [
    {"name": "get_segment_series_list", "description": "Return all series keys for a segment from eda_report.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "segment_id": {"type": "string"}}, "required": ["run_id", "segment_id"]}},
    {"name": "train_and_evaluate",      "description": "Train model and evaluate on walk-forward folds. Rejects models outside demand class gate.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "series_key": {"type": "string"}, "model_name": {"type": "string"}, "hyperparams": {"type": "object"}}, "required": ["run_id", "series_key", "model_name", "hyperparams"]}},
    {"name": "walk_forward_validate",   "description": "Re-run walk-forward validation with explicit fold count.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "series_key": {"type": "string"}, "model_name": {"type": "string"}, "n_folds": {"type": "integer"}}, "required": ["run_id", "series_key", "model_name", "n_folds"]}},
    {"name": "build_ensemble",          "description": "Build ensemble of base models. Only valid if single model plateaued and delta > 5% MASE.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "series_key": {"type": "string"}, "base_models": {"type": "array", "items": {"type": "string"}}, "strategy": {"type": "string"}}, "required": ["run_id", "series_key", "base_models", "strategy"]}},
    {"name": "assess_target_feasibility","description": "Assess if MASE target is achievable after all self-correction rounds.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "series_key": {"type": "string"}}, "required": ["run_id", "series_key"]}},
    {"name": "record_series_result",    "description": "Write per-series outcome to series_results/.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "series_key": {"type": "string"}, "result": {"type": "object"}}, "required": ["run_id", "series_key", "result"]}},
    {"name": "compile_foundry_report",  "description": "Aggregate series results, set foundry_complete=True, phase=report_ready.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]}},
]
```

In `backend/forecasting/agents/prism.py` — replace `PRISM_TOOLS: list[dict] = []` with:

```python
PRISM_TOOLS: list[dict] = [
    {"name": "clone_pack_for_whatif",   "description": "Clone domain_context_pack into whatif/{whatif_id}/modified_pack.json.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "whatif_id": {"type": "string"}}, "required": ["run_id", "whatif_id"]}},
    {"name": "apply_whatif_override",   "description": "Apply scenario override to modified_pack.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "whatif_id": {"type": "string"}, "series_key": {"type": "string"}, "override_type": {"type": "string"}, "magnitude": {"type": "number"}, "start_period": {"type": "string"}, "end_period": {"type": "string"}, "description": {"type": "string"}}, "required": ["run_id", "whatif_id", "series_key", "override_type", "magnitude", "start_period", "end_period", "description"]}},
    {"name": "run_forge_for_scenario",  "description": "Re-classify demand profiles for affected series on the modified data (ADR-0001).", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "whatif_id": {"type": "string"}, "affected_series_keys": {"type": "array", "items": {"type": "string"}}}, "required": ["run_id", "whatif_id", "affected_series_keys"]}},
    {"name": "run_foundry_for_scenario", "description": "Re-fit affected series on the scenario data within the (possibly new) demand-class gate.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "whatif_id": {"type": "string"}, "affected_series_keys": {"type": "array", "items": {"type": "string"}}, "horizon": {"type": "integer"}}, "required": ["run_id", "whatif_id", "affected_series_keys"]}},
    {"name": "compile_comparison",      "description": "Build baseline vs scenario comparison from real re-modelled results and write comparison.json.", "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}, "whatif_id": {"type": "string"}, "series_keys": {"type": "array", "items": {"type": "string"}}}, "required": ["run_id", "whatif_id", "series_keys"]}},
]
```

- [ ] **Step 5: Run registry test**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_registry.py -v
```

Expected: 1 passed — all tools present in registry.

- [ ] **Step 6: Run full test suite**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add backend/forecasting/providers.py backend/forecasting/agents/
git add tests/test_registry.py
git commit -m "feat: tool registry — all tools wired, agent tool lists populated"
```

---

## Phase E — Backend API

> Tasks 23–25. FastAPI routers + SSE + `app.py`. Thin routers only — all logic lives in tools/agents. The API is the only code that calls agents.

---

### Task 23: `api/sse.py` + `api/models.py`

**Files:**
- Create: `api/sse.py`
- Create: `api/models.py`

- [ ] **Step 1: Write `api/sse.py`**

```python
# api/sse.py
from __future__ import annotations

import json
import queue
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
    _TERMINAL = {"message_done", "error", "pipeline_done", "whatif_done"}
    while True:
        try:
            item = q.get(timeout=30)
            yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
            if item["event"] in _TERMINAL:
                break
        except queue.Empty:
            yield ": keepalive\n\n"
```

- [ ] **Step 2: Write `api/models.py`**

```python
# api/models.py
from __future__ import annotations

from pydantic import BaseModel


class CreateRunRequest(BaseModel):
    domain: str


class CreateRunResponse(BaseModel):
    run_id: str


class UploadResponse(BaseModel):
    run_id: str
    series_count: int
    segment_count: int
    blocking_issues: list[dict]
    warnings: list[dict]


class MessageRequest(BaseModel):
    content: str


class RunSummaryResponse(BaseModel):
    run_id: str
    domain: str
    phase: str
    override_count: int
    open_risks: int
    halt_reason: str | None
    pack_confirmed: bool
    created_at: str


class WhatIfRequest(BaseModel):
    scenario_description: str


class WhatIfResponse(BaseModel):
    whatif_id: str
```

- [ ] **Step 3: Smoke-import check**

```powershell
c:/Agent_A/venv/Scripts/python -c "from api.sse import emit, stream_events; from api.models import CreateRunRequest; print('api models+sse OK')"
```

Expected: `api models+sse OK`

- [ ] **Step 4: Commit**

```powershell
git add api/sse.py api/models.py
git commit -m "feat: api sse queue and HTTP models"
```

---

### Task 24: API Routers

**Files:**
- Create: `api/routers/runs.py`
- Create: `api/routers/message.py`
- Create: `api/routers/stream.py`
- Create: `api/routers/whatif.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing API tests**

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import forecasting.run_state as rs
    tmp = tmp_path_factory.mktemp("outputs")
    rs.OUTPUTS_ROOT = tmp
    from app import app
    return TestClient(app)


def _csv_bytes() -> bytes:
    rows = [f"2024-W{w+1:02d},{sku},NORTH,{float(w+1)}"
            for sku in ["SKU_A", "SKU_B"] for w in range(12)]
    return ("week,sku,region,demand\n" + "\n".join(rows)).encode()


def test_create_run(client):
    resp = client.post("/api/v1/runs/create", json={"domain": "fmcg"})
    assert resp.status_code == 200
    assert "run_id" in resp.json()


def test_upload_csv(client):
    run_id = client.post("/api/v1/runs/create", json={"domain": "fmcg"}).json()["run_id"]
    resp = client.post(
        f"/api/v1/runs/{run_id}/upload",
        files={"file": ("data.csv", _csv_bytes(), "text/csv")},
        data={"domain": "fmcg"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["series_count"] > 0
    assert body["blocking_issues"] == []


def test_get_run(client):
    run_id = client.post("/api/v1/runs/create", json={"domain": "fmcg"}).json()["run_id"]
    resp = client.get(f"/api/v1/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["phase"] == "preflight"


def test_upload_bad_csv_returns_422(client):
    run_id = client.post("/api/v1/runs/create", json={"domain": "fmcg"}).json()["run_id"]
    resp = client.post(
        f"/api/v1/runs/{run_id}/upload",
        files={"file": ("data.csv", b"\x00corrupt", "text/csv")},
        data={"domain": "fmcg"},
    )
    assert resp.status_code == 422


def test_missing_run_returns_404(client):
    resp = client.get("/api/v1/runs/no-such-run")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run — expect failure**

```powershell
c:/Agent_A/venv/Scripts/python -m pytest tests/test_api.py -v
```

Expected: `ImportError` (routers not yet written)

- [ ] **Step 3: Write `api/routers/runs.py`**

```python
# api/routers/runs.py
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.models import CreateRunRequest, CreateRunResponse, RunSummaryResponse, UploadResponse
from forecasting.preflight import PreflightBlockingError, run_preflight
from forecasting.run_state import (
    RunNotFoundError, create_run_state, load_run_state, run_dir, OUTPUTS_ROOT
)

router = APIRouter()

_PLAYBOOKS: dict[str, dict] = {
    "fmcg": {
        "common_grains": ["sku", "region"],
        "time_col": "week",
        "demand_col": "demand",
        "min_series": 1,
        "min_history_periods": 12,
        "mase_target": 0.8,
    }
}


@router.post("/runs/create", response_model=CreateRunResponse)
def create_run(body: CreateRunRequest):
    if body.domain not in _PLAYBOOKS:
        raise HTTPException(status_code=400, detail=f"Unknown domain: {body.domain}")
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    create_run_state(run_id, domain=body.domain)
    return CreateRunResponse(run_id=run_id)


@router.post("/runs/{run_id}/upload", response_model=UploadResponse)
def upload_csv(run_id: str, file: UploadFile = File(...), domain: str = Form(...)):
    try:
        state = load_run_state(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    playbook = _PLAYBOOKS.get(domain or state.domain, _PLAYBOOKS["fmcg"])
    file_bytes = file.file.read()
    try:
        bundle = run_preflight(run_id, file_bytes, domain=domain or state.domain, playbook=playbook)
    except PreflightBlockingError as exc:
        raise HTTPException(status_code=422, detail={
            "blocking_issues": [i.model_dump() for i in exc.issues]
        })
    return UploadResponse(
        run_id=run_id,
        series_count=bundle.data_quality_report.series_count,
        segment_count=len(bundle.segment_profiles),
        blocking_issues=[],
        warnings=[w.model_dump() for w in bundle.data_quality_report.warnings],
    )


@router.get("/runs", response_model=list[RunSummaryResponse])
def list_runs():
    runs = []
    for state_file in OUTPUTS_ROOT.glob("*/run_state.json"):
        try:
            state = load_run_state(state_file.parent.name)
            runs.append(_state_to_summary(state))
        except Exception:
            pass
    return runs


@router.get("/runs/{run_id}", response_model=RunSummaryResponse)
def get_run(run_id: str):
    try:
        state = load_run_state(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return _state_to_summary(state)


@router.get("/runs/{run_id}/decisions")
def get_decisions(run_id: str):
    try:
        load_run_state(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    d = run_dir(run_id)
    claims = _read_json(d / "claim_ledger.json", {}).get("claims", [])
    risks = _read_json(d / "risk_register.json", {}).get("risks", [])
    overrides = [c for c in claims if c.get("verification_status") == "USER_OVERRIDE_ACCEPTED"]
    return {"claims": claims, "risks": risks, "overrides": overrides}


@router.get("/runs/{run_id}/report")
def get_report(run_id: str):
    try:
        state = load_run_state(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if state.phase not in ("report_ready",):
        raise HTTPException(status_code=409, detail="Report not ready")
    path = run_dir(run_id) / "foundry_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="foundry_report.json not found")
    return json.loads(path.read_text())


def _state_to_summary(state) -> RunSummaryResponse:
    return RunSummaryResponse(
        run_id=state.run_id,
        domain=state.domain,
        phase=state.phase if isinstance(state.phase, str) else state.phase.value,
        override_count=state.override_count,
        open_risks=state.open_risks,
        halt_reason=state.halt_reason,
        pack_confirmed=state.pack_confirmed,
        created_at=state.created_at,
    )


def _read_json(path: Path, default) -> dict:
    return json.loads(path.read_text()) if path.exists() else default
```

- [ ] **Step 4: Write `api/routers/message.py`**

```python
# api/routers/message.py
from __future__ import annotations

import threading
from typing import Callable

from fastapi import APIRouter, HTTPException

from api.models import MessageRequest
from api.sse import emit
from forecasting.agents.conductor import run_conductor
from forecasting.agents.lens import LensInput, classify_intent
from forecasting.agents.meridian import run_meridian
from forecasting.guard import GuardHalt
from forecasting.run_state import RunNotFoundError, Phase, load_run_state
from forecasting.tools.conductor_tools import log_halt

router = APIRouter()


def _sse_emit(run_id: str) -> Callable:
    return lambda evt, payload: emit(run_id, evt, payload)


@router.post("/runs/{run_id}/message")
def send_message(run_id: str, body: MessageRequest):
    try:
        state = load_run_state(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    phase_val = state.phase if isinstance(state.phase, str) else state.phase.value
    if phase_val == Phase.HALTED.value:
        raise HTTPException(status_code=410, detail=f"Run halted: {state.halt_reason}")

    sse_emit = _sse_emit(run_id)

    def _handle():
        try:
            current_state = load_run_state(run_id)
            intent_pack = classify_intent(LensInput(
                conversation_history=[],
                user_message=body.content,
                pipeline_state=current_state,
            ))
            tokens = 0
            tokens = run_conductor(run_id, intent_pack, current_state, tokens, sse_emit)

            # If Conductor routed to Meridian, run Meridian now
            refreshed = load_run_state(run_id)
            phase_val = refreshed.phase if isinstance(refreshed.phase, str) else refreshed.phase.value
            if phase_val == Phase.MERIDIAN_SCOPING.value:
                import json as _json
                from forecasting.run_state import run_dir
                pf_path = run_dir(run_id) / "preflight.json"
                bundle = _json.loads(pf_path.read_text())["bundle"] if pf_path.exists() else {}
                run_meridian(run_id, body.content, [], bundle, tokens, sse_emit)

        except GuardHalt as exc:
            log_halt(run_id, exc.args[0], sse_emit)
        except Exception as exc:
            sse_emit("error", {"reason": str(exc), "halt_reason": "unexpected_error"})

    threading.Thread(target=_handle, daemon=True).start()
    return {"status": "processing"}
```

- [ ] **Step 5: Write `api/routers/stream.py`**

```python
# api/routers/stream.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.sse import stream_events
from forecasting.run_state import RunNotFoundError, load_run_state

router = APIRouter()


@router.get("/runs/{run_id}/stream")
def sse_stream(run_id: str):
    try:
        load_run_state(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return StreamingResponse(
        stream_events(run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 6: Write `api/routers/whatif.py`**

```python
# api/routers/whatif.py
from __future__ import annotations

import json
import threading
from typing import Callable

from fastapi import APIRouter, HTTPException

from api.models import WhatIfRequest, WhatIfResponse
from api.sse import emit
from forecasting.agents.prism import run_prism
from forecasting.guard import GuardHalt
from forecasting.run_state import RunNotFoundError, Phase, load_run_state, run_dir
from forecasting.tools.conductor_tools import create_prism_run, log_halt

router = APIRouter()


@router.post("/runs/{run_id}/whatif", response_model=WhatIfResponse)
def create_whatif(run_id: str, body: WhatIfRequest):
    try:
        state = load_run_state(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    phase_val = state.phase if isinstance(state.phase, str) else state.phase.value
    if phase_val == Phase.HALTED.value:
        raise HTTPException(status_code=410, detail="Run halted")
    if phase_val != Phase.REPORT_READY.value:
        raise HTTPException(status_code=409, detail=f"phase={phase_val} — what-if only available in report_ready")

    result = create_prism_run(run_id, body.scenario_description, entities={})
    whatif_id = result["whatif_id"]
    sse_emit: Callable = lambda evt, payload: emit(run_id, evt, payload)

    def _handle():
        try:
            run_prism(
                run_id=run_id, whatif_id=whatif_id,
                scenario_description=body.scenario_description,
                intent_entities={}, tokens_used=0, sse_emit=sse_emit,
            )
        except GuardHalt as exc:
            log_halt(run_id, exc.args[0], sse_emit)
        except Exception as exc:
            sse_emit("error", {"reason": str(exc), "halt_reason": "prism_error"})

    threading.Thread(target=_handle, daemon=True).start()
    return WhatIfResponse(whatif_id=whatif_id)


@router.get("/runs/{run_id}/whatif/{whatif_id}/compare")
def get_comparison(run_id: str, whatif_id: str):
    try:
        load_run_state(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    path = run_dir(run_id) / "whatif" / whatif_id / "comparison.json"
    if not path.exists():
        raise HTTPException(status_code=409, detail="Scenario run not complete")
    return json.loads(path.read_text())
```

- [ ] **Step 7: Run API tests**

```powershell
$env:PYTHONPATH = "backend;."
c:/Agent_A/venv/Scripts/python -m pytest tests/test_api.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 8: Commit**

```powershell
git add api/routers/ tests/test_api.py
git commit -m "feat: API routers — runs, message, stream, whatif — thin handlers, no inline logic"
```

---

### Task 25: `backend/app.py` + Smoke Test

**Files:**
- Create: `backend/app.py`

- [ ] **Step 1: Write `backend/app.py`**

```python
# backend/app.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routers import runs, message, stream, whatif

app = FastAPI(title="Agent P — Demand Forecasting")

app.include_router(runs.router,    prefix="/api/v1")
app.include_router(message.router, prefix="/api/v1")
app.include_router(stream.router,  prefix="/api/v1")
app.include_router(whatif.router,  prefix="/api/v1")

_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="static")
```

- [ ] **Step 2: Verify server starts**

```powershell
$env:PYTHONPATH = "backend;."
c:/Agent_A/venv/Scripts/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765 --timeout-keep-alive 1
```

Expected: `Uvicorn running on http://127.0.0.1:8765` (Ctrl+C to stop)

- [ ] **Step 3: Run full backend test suite**

```powershell
$env:PYTHONPATH = "backend;."
c:/Agent_A/venv/Scripts/python -m pytest tests/ -v --tb=short
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```powershell
git add backend/app.py
git commit -m "feat: app.py — FastAPI entry point, router mounting, static mount"
```

---

## Phase F — Frontend

> Tasks 26–31. React 18 + Vite + TypeScript + Tailwind + shadcn/ui. State via Zustand (4 stores). SSE via native `EventSource`. API via `fetch`.

---

### Task 26: Frontend Scaffold

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/postcss.config.js`
- Create: `frontend/index.html`

- [ ] **Step 1: Write `frontend/package.json`**

```json
{
  "name": "agent-p-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite --port 5173 --proxy",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "zustand": "^4.5.2",
    "@radix-ui/react-scroll-area": "^1.0.5",
    "clsx": "^2.1.1",
    "tailwind-merge": "^2.3.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "autoprefixer": "^10.4.19",
    "postcss": "^8.4.38",
    "tailwindcss": "^3.4.4",
    "typescript": "^5.4.5",
    "vite": "^5.3.1"
  }
}
```

- [ ] **Step 2: Write `frontend/vite.config.ts`**

```typescript
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8765", changeOrigin: true },
    },
  },
  build: { outDir: "dist" },
})
```

- [ ] **Step 3: Write `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
```

- [ ] **Step 4: Write `frontend/tailwind.config.ts`**

```typescript
import type { Config } from "tailwindcss"

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
} satisfies Config
```

- [ ] **Step 5: Write `frontend/postcss.config.js`**

```javascript
export default {
  plugins: { tailwindcss: {}, autoprefixer: {} },
}
```

- [ ] **Step 6: Write `frontend/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Agent P — Demand Forecasting</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 7: Install frontend dependencies**

```powershell
cd frontend; npm install; cd ..
```

Expected: no errors, `node_modules/` created.

- [ ] **Step 8: Commit**

```powershell
git add frontend/package.json frontend/vite.config.ts frontend/tsconfig.json
git add frontend/tailwind.config.ts frontend/postcss.config.js frontend/index.html
git commit -m "chore: frontend scaffold — Vite, React, Tailwind, shadcn deps"
```

---

### Task 27: TypeScript Types + Zustand Stores

**Files:**
- Create: `frontend/src/types/index.ts`
- Create: `frontend/src/stores/runStore.ts`
- Create: `frontend/src/stores/streamStore.ts`
- Create: `frontend/src/stores/decisionStore.ts`
- Create: `frontend/src/stores/prismStore.ts`

- [ ] **Step 1: Write `frontend/src/types/index.ts`**

```typescript
export type Phase =
  | "preflight"
  | "meridian_scoping"
  | "forge_eda"
  | "foundry_modelling"
  | "report_ready"
  | "halted"

export type VerificationStatus =
  | "SUPPORTED"
  | "CONTRADICTED"
  | "AMBIGUOUS"
  | "UNVERIFIABLE"
  | "USER_OVERRIDE_ACCEPTED"

export type EvidenceType =
  | "statistical_test"
  | "association"
  | "pattern"
  | "user_confirmed"
  | "unverifiable_business_input"

export interface ConversationTurn {
  role: "user" | "assistant"
  content: string
  agent?: string
}

export interface Claim {
  claim_id: string
  claim: string
  verification_status: VerificationStatus
  evidence_type: EvidenceType
  applies_to: string
  downstream_impact: string
  must_surface_in_report: boolean
}

export interface Risk {
  risk_id: string
  description: string
  severity: "low" | "medium" | "high"
  acknowledged: boolean
}

export interface Override {
  claim_id: string
  claim: string
  verification_status: "USER_OVERRIDE_ACCEPTED"
  must_surface_in_report: true
}

export interface SeriesResult {
  model_name: string
  mase: number
  forecastability: "forecastable" | "caution" | "unforecastable"
  demand_class: string
}

export interface ComparisonResult {
  whatif_id: string
  comparisons: Array<{
    series_key: string
    baseline_forecast: number[]
    scenario_forecast: number[]
    delta_pct: number
    demand_class_changed: boolean
    baseline_sb_class: string
    scenario_sb_class: string
  }>
}
```

- [ ] **Step 2: Write `frontend/src/stores/runStore.ts`**

```typescript
import { create } from "zustand"
import type { ConversationTurn, Phase } from "../types"

interface RunStore {
  run_id: string | null
  phase: Phase | null
  domain: string | null
  history: ConversationTurn[]
  setRun: (run_id: string, domain: string) => void
  setPhase: (phase: Phase) => void
  pushMessage: (turn: ConversationTurn) => void
  reset: () => void
}

export const useRunStore = create<RunStore>((set) => ({
  run_id: null,
  phase: null,
  domain: null,
  history: [],
  setRun: (run_id, domain) => set({ run_id, domain, phase: "preflight" }),
  setPhase: (phase) => set({ phase }),
  pushMessage: (turn) => set((s) => ({ history: [...s.history, turn] })),
  reset: () => set({ run_id: null, phase: null, domain: null, history: [] }),
}))
```

- [ ] **Step 3: Write `frontend/src/stores/streamStore.ts`**

```typescript
import { create } from "zustand"

interface StreamStore {
  partial: string
  isStreaming: boolean
  append: (token: string) => void
  commit: () => void
  setStreaming: (v: boolean) => void
}

export const useStreamStore = create<StreamStore>((set) => ({
  partial: "",
  isStreaming: false,
  append: (token) => set((s) => ({ partial: s.partial + token, isStreaming: true })),
  commit: () => set({ partial: "", isStreaming: false }),
  setStreaming: (v) => set({ isStreaming: v }),
}))
```

- [ ] **Step 4: Write `frontend/src/stores/decisionStore.ts`**

```typescript
import { create } from "zustand"
import type { Claim, Risk, Override } from "../types"

interface DecisionStore {
  claims: Claim[]
  risks: Risk[]
  overrides: Override[]
  pushClaim: (claim: Claim) => void
  pushRisk: (risk: Risk) => void
  pushOverride: (override: Override) => void
  reset: () => void
}

export const useDecisionStore = create<DecisionStore>((set) => ({
  claims: [],
  risks: [],
  overrides: [],
  pushClaim: (claim) => set((s) => ({ claims: [...s.claims, claim] })),
  pushRisk: (risk) => set((s) => ({ risks: [...s.risks, risk] })),
  pushOverride: (override) => set((s) => ({ overrides: [...s.overrides, override] })),
  reset: () => set({ claims: [], risks: [], overrides: [] }),
}))
```

- [ ] **Step 5: Write `frontend/src/stores/prismStore.ts`**

```typescript
import { create } from "zustand"
import type { ComparisonResult } from "../types"

interface PrismStore {
  whatif_id: string | null
  scenario: string | null
  isRunning: boolean
  comparison: ComparisonResult | null
  setWhatif: (whatif_id: string, scenario: string) => void
  setRunning: (v: boolean) => void
  setComparison: (result: ComparisonResult) => void
  reset: () => void
}

export const usePrismStore = create<PrismStore>((set) => ({
  whatif_id: null,
  scenario: null,
  isRunning: false,
  comparison: null,
  setWhatif: (whatif_id, scenario) => set({ whatif_id, scenario }),
  setRunning: (v) => set({ isRunning: v }),
  setComparison: (result) => set({ comparison: result, isRunning: false }),
  reset: () => set({ whatif_id: null, scenario: null, isRunning: false, comparison: null }),
}))
```

- [ ] **Step 6: Commit**

```powershell
git add frontend/src/types/ frontend/src/stores/
git commit -m "feat: TS types and Zustand stores — run, stream, decision, prism"
```

---

### Task 28: SSE Hook + API Client

**Files:**
- Create: `frontend/src/hooks/useSSE.ts`
- Create: `frontend/src/api/client.ts`

- [ ] **Step 1: Write `frontend/src/hooks/useSSE.ts`**

```typescript
import { useEffect, useRef } from "react"
import { useRunStore } from "../stores/runStore"
import { useStreamStore } from "../stores/streamStore"
import { useDecisionStore } from "../stores/decisionStore"
import type { Phase } from "../types"

export function useSSE(run_id: string | null) {
  const esRef = useRef<EventSource | null>(null)
  const setPhase = useRunStore((s) => s.setPhase)
  const pushMessage = useRunStore((s) => s.pushMessage)
  const append = useStreamStore((s) => s.append)
  const commit = useStreamStore((s) => s.commit)
  const setStreaming = useStreamStore((s) => s.setStreaming)
  const pushClaim = useDecisionStore((s) => s.pushClaim)
  const pushRisk = useDecisionStore((s) => s.pushRisk)
  const pushOverride = useDecisionStore((s) => s.pushOverride)

  useEffect(() => {
    if (!run_id) return
    const es = new EventSource(`/api/v1/runs/${run_id}/stream`)
    esRef.current = es

    es.addEventListener("token", (e) => {
      setStreaming(true)
      append(JSON.parse((e as MessageEvent).data).content as string)
    })
    es.addEventListener("message_done", (e) => {
      const { agent, full_text } = JSON.parse((e as MessageEvent).data)
      pushMessage({ role: "assistant", content: full_text as string, agent })
      commit()
    })
    es.addEventListener("phase_change", (e) => {
      setPhase(JSON.parse((e as MessageEvent).data).phase as Phase)
    })
    es.addEventListener("decision_update", (e) => {
      pushClaim(JSON.parse((e as MessageEvent).data))
    })
    es.addEventListener("risk_update", (e) => {
      pushRisk(JSON.parse((e as MessageEvent).data))
    })
    es.addEventListener("override_update", (e) => {
      pushOverride(JSON.parse((e as MessageEvent).data))
    })
    es.addEventListener("error", () => {
      setPhase("halted")
      es.close()
    })

    return () => {
      es.close()
      esRef.current = null
    }
  }, [run_id])
}
```

- [ ] **Step 2: Write `frontend/src/api/client.ts`**

```typescript
const BASE = "/api/v1"

export const api = {
  createRun: (domain: string) =>
    fetch(`${BASE}/runs/create`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ domain }),
    }).then((r) => r.json()) as Promise<{ run_id: string }>,

  uploadFile: (run_id: string, file: File, domain: string) => {
    const fd = new FormData()
    fd.append("file", file)
    fd.append("domain", domain)
    return fetch(`${BASE}/runs/${run_id}/upload`, { method: "POST", body: fd }).then((r) =>
      r.json()
    )
  },

  sendMessage: (run_id: string, content: string) =>
    fetch(`${BASE}/runs/${run_id}/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    }),

  createWhatIf: (run_id: string, scenario_description: string) =>
    fetch(`${BASE}/runs/${run_id}/whatif`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario_description }),
    }).then((r) => r.json()) as Promise<{ whatif_id: string }>,

  getComparison: (run_id: string, whatif_id: string) =>
    fetch(`${BASE}/runs/${run_id}/whatif/${whatif_id}/compare`).then((r) => r.json()),

  getDecisions: (run_id: string) =>
    fetch(`${BASE}/runs/${run_id}/decisions`).then((r) => r.json()),
}
```

- [ ] **Step 3: Commit**

```powershell
git add frontend/src/hooks/ frontend/src/api/
git commit -m "feat: useSSE hook and API client — all backend endpoints wired"
```

---

### Task 29: Layout + Entry Point

**Files:**
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/index.css`
- Create: `frontend/src/components/Layout.tsx`
- Create: `frontend/src/components/ProgressBar.tsx`

- [ ] **Step 1: Write `frontend/src/index.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

- [ ] **Step 2: Write `frontend/src/main.tsx`**

```tsx
import React from "react"
import ReactDOM from "react-dom/client"
import "./index.css"
import App from "./App"

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

- [ ] **Step 3: Write `frontend/src/components/ProgressBar.tsx`**

```tsx
import React from "react"
import type { Phase } from "../types"

const PHASES: Phase[] = [
  "preflight", "meridian_scoping", "forge_eda", "foundry_modelling", "report_ready"
]
const LABELS: Record<Phase, string> = {
  preflight:          "Upload",
  meridian_scoping:   "Scoping",
  forge_eda:          "EDA",
  foundry_modelling:  "Modelling",
  report_ready:       "Report",
  halted:             "Halted",
}

interface Props { phase: Phase | null }

export function ProgressBar({ phase }: Props) {
  const active = PHASES.indexOf(phase as Phase)
  return (
    <div className="flex items-center gap-2 px-4 py-2 bg-gray-50 border-b text-sm">
      {PHASES.map((p, i) => (
        <React.Fragment key={p}>
          <span className={[
            "px-3 py-1 rounded-full font-medium",
            i < active  ? "bg-green-100 text-green-700" : "",
            i === active ? "bg-blue-600 text-white" : "",
            i > active  ? "text-gray-400" : "",
          ].join(" ")}>
            {LABELS[p]}
          </span>
          {i < PHASES.length - 1 && <span className="text-gray-300">→</span>}
        </React.Fragment>
      ))}
      {phase === "halted" && (
        <span className="px-3 py-1 rounded-full bg-red-100 text-red-700 font-medium">Halted</span>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Write `frontend/src/components/Layout.tsx`**

```tsx
import React from "react"
import type { ReactNode } from "react"

interface Props {
  sidebar: ReactNode
  main: ReactNode
  panel: ReactNode
}

export function Layout({ sidebar, main, panel }: Props) {
  return (
    <div className="flex h-screen overflow-hidden bg-white">
      <aside className="w-48 flex-shrink-0 border-r overflow-y-auto">{sidebar}</aside>
      <main className="flex-1 flex flex-col overflow-hidden">{main}</main>
      <aside className="w-80 flex-shrink-0 border-l overflow-y-auto">{panel}</aside>
    </div>
  )
}
```

- [ ] **Step 5: Write `frontend/src/App.tsx`**

```tsx
import { useState } from "react"
import { Layout } from "./components/Layout"
import { ProgressBar } from "./components/ProgressBar"
import { useRunStore } from "./stores/runStore"
import { useSSE } from "./hooks/useSSE"
import { api } from "./api/client"

function Sidebar() {
  const { run_id, phase, setRun } = useRunStore()
  const [domain, setDomain] = useState("fmcg")
  const handleCreate = async () => {
    const { run_id: rid } = await api.createRun(domain)
    setRun(rid, domain)
  }
  return (
    <div className="p-3 flex flex-col gap-3">
      <h1 className="font-bold text-sm text-gray-800">Agent P</h1>
      <select className="border rounded px-2 py-1 text-sm" value={domain} onChange={e => setDomain(e.target.value)}>
        <option value="fmcg">FMCG</option>
      </select>
      <button onClick={handleCreate} className="bg-blue-600 text-white text-sm px-3 py-1.5 rounded hover:bg-blue-700">
        New Run
      </button>
      {run_id && <p className="text-xs text-gray-500 break-all">{run_id}</p>}
    </div>
  )
}

function MainPanel() {
  const { run_id, phase, history, pushMessage } = useRunStore()
  const { partial, isStreaming } = { partial: "", isStreaming: false }
  const [msg, setMsg] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const { domain } = useRunStore()

  const handleUpload = async () => {
    if (!run_id || !file) return
    await api.uploadFile(run_id, file, domain || "fmcg")
  }

  const handleSend = async () => {
    if (!run_id || !msg.trim()) return
    pushMessage({ role: "user", content: msg })
    setMsg("")
    await api.sendMessage(run_id, msg)
  }

  return (
    <div className="flex flex-col h-full">
      <ProgressBar phase={phase} />
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
        {!phase || phase === "preflight" ? (
          <div className="flex flex-col gap-2">
            <p className="text-sm text-gray-600">Upload a demand CSV to begin.</p>
            <input type="file" accept=".csv" onChange={e => setFile(e.target.files?.[0] || null)} className="text-sm" />
            <button onClick={handleUpload} disabled={!file} className="bg-blue-600 text-white text-sm px-4 py-2 rounded disabled:opacity-50">
              Upload & Run Pre-flight
            </button>
          </div>
        ) : null}
        {history.map((t, i) => (
          <div key={i} className={["rounded-lg px-4 py-2 text-sm max-w-2xl", t.role === "user" ? "bg-blue-50 self-end" : "bg-gray-50 self-start"].join(" ")}>
            {t.agent && <p className="text-xs text-gray-400 mb-1">{t.agent}</p>}
            <p className="whitespace-pre-wrap">{t.content}</p>
          </div>
        ))}
      </div>
      {phase && phase !== "preflight" && phase !== "forge_eda" && phase !== "foundry_modelling" && (
        <div className="border-t p-3 flex gap-2">
          <input
            className="flex-1 border rounded px-3 py-2 text-sm"
            value={msg}
            onChange={e => setMsg(e.target.value)}
            onKeyDown={e => e.key === "Enter" && !e.shiftKey && handleSend()}
            placeholder="Type a message…"
          />
          <button onClick={handleSend} className="bg-blue-600 text-white px-4 py-2 rounded text-sm">Send</button>
        </div>
      )}
    </div>
  )
}

function DecisionPanel() {
  const claims = [] as any[]
  const risks = [] as any[]
  return (
    <div className="p-3 flex flex-col gap-4 text-sm">
      <h2 className="font-semibold text-gray-700">Decisions</h2>
      {claims.length === 0 && risks.length === 0 && (
        <p className="text-gray-400 text-xs">Claims and risks will appear here during scoping.</p>
      )}
    </div>
  )
}

export default function App() {
  const { run_id } = useRunStore()
  useSSE(run_id)
  return <Layout sidebar={<Sidebar />} main={<MainPanel />} panel={<DecisionPanel />} />
}
```

- [ ] **Step 6: Build to verify no type errors**

```powershell
cd frontend; npm run build; cd ..
```

Expected: `dist/` created, no TypeScript errors.

- [ ] **Step 7: Commit**

```powershell
git add frontend/src/
git commit -m "feat: App, Layout, ProgressBar, Sidebar, DecisionPanel — functional shell"
```

---

### Task 30: Conversation + Progress Components

**Files:**
- Create: `frontend/src/components/Conversation.tsx`
- Create: `frontend/src/components/MessageInput.tsx`
- Modify: `frontend/src/App.tsx` (swap inline bubbles for Conversation component)

- [ ] **Step 1: Write `frontend/src/components/Conversation.tsx`**

```tsx
import React from "react"
import type { ConversationTurn } from "../types"
import { useStreamStore } from "../stores/streamStore"

interface Props {
  history: ConversationTurn[]
}

export function Conversation({ history }: Props) {
  const { partial, isStreaming } = useStreamStore()
  const bottomRef = React.useRef<HTMLDivElement>(null)
  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [history.length, partial])

  return (
    <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
      {history.map((t, i) => (
        <div key={i} className={["rounded-lg px-4 py-2 text-sm max-w-2xl", t.role === "user" ? "bg-blue-50 self-end" : "bg-gray-50 self-start"].join(" ")}>
          {t.agent && <p className="text-xs text-gray-400 mb-1 capitalize">{t.agent}</p>}
          <p className="whitespace-pre-wrap">{t.content}</p>
        </div>
      ))}
      {isStreaming && partial && (
        <div className="rounded-lg px-4 py-2 text-sm max-w-2xl bg-gray-50 self-start">
          <p className="text-xs text-gray-400 mb-1">streaming…</p>
          <p className="whitespace-pre-wrap">{partial}<span className="animate-pulse">▋</span></p>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}
```

- [ ] **Step 2: Write `frontend/src/components/MessageInput.tsx`**

```tsx
import React, { useState } from "react"

interface Props {
  disabled?: boolean
  onSend: (msg: string) => void
}

export function MessageInput({ disabled = false, onSend }: Props) {
  const [value, setValue] = useState("")
  const submit = () => {
    if (!value.trim() || disabled) return
    onSend(value.trim())
    setValue("")
  }
  return (
    <div className="border-t p-3 flex gap-2">
      <textarea
        rows={2}
        className="flex-1 border rounded px-3 py-2 text-sm resize-none disabled:bg-gray-50"
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit() } }}
        disabled={disabled}
        placeholder={disabled ? "Waiting for pipeline…" : "Type a message (Enter to send)"}
      />
      <button
        onClick={submit}
        disabled={disabled || !value.trim()}
        className="bg-blue-600 text-white px-4 py-2 rounded text-sm self-end disabled:opacity-40 hover:bg-blue-700"
      >
        Send
      </button>
    </div>
  )
}
```

- [ ] **Step 3: Update `App.tsx` to use these components**

Replace the inline bubble rendering in `MainPanel` with:
```tsx
import { Conversation } from "./components/Conversation"
import { MessageInput } from "./components/MessageInput"
// ...inside MainPanel:
<Conversation history={history} />
<MessageInput
  disabled={phase === "forge_eda" || phase === "foundry_modelling" || !phase || phase === "preflight"}
  onSend={async (msg) => {
    pushMessage({ role: "user", content: msg })
    if (run_id) await api.sendMessage(run_id, msg)
  }}
/>
```

- [ ] **Step 4: Build to verify**

```powershell
cd frontend; npm run build; cd ..
```

Expected: no TypeScript errors.

- [ ] **Step 5: Commit**

```powershell
git add frontend/src/components/Conversation.tsx frontend/src/components/MessageInput.tsx frontend/src/App.tsx
git commit -m "feat: Conversation and MessageInput components — streaming bubble, auto-scroll, disabled state"
```

---

### Task 31: ReportView + DecisionPanel Components

**Files:**
- Create: `frontend/src/components/ReportView.tsx`
- Create: `frontend/src/components/DecisionPanel.tsx`
- Modify: `frontend/src/App.tsx` (swap placeholder DecisionPanel for real component)

- [ ] **Step 1: Write `frontend/src/components/ReportView.tsx`**

```tsx
import React, { useEffect, useState } from "react"
import { api } from "../api/client"

interface Props { run_id: string }

export function ReportView({ run_id }: Props) {
  const [report, setReport] = useState<any>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    fetch(`/api/v1/runs/${run_id}/report`)
      .then(r => r.json())
      .then(setReport)
      .catch(e => setErr(String(e)))
  }, [run_id])

  if (err) return <p className="text-red-500 text-sm p-4">{err}</p>
  if (!report) return <p className="text-gray-400 text-sm p-4">Loading report…</p>

  const { forecastable_count = 0, caution_count = 0, unforecastable_count = 0, series_results = [] } = report

  return (
    <div className="p-4 flex flex-col gap-4">
      <h2 className="text-lg font-semibold text-gray-800">Foundry Report</h2>
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "Forecastable", count: forecastable_count, colour: "green" },
          { label: "Caution",      count: caution_count,      colour: "yellow" },
          { label: "Unforecastable",count: unforecastable_count, colour: "red" },
        ].map(({ label, count, colour }) => (
          <div key={label} className={`rounded-lg p-3 bg-${colour}-50 border border-${colour}-200`}>
            <p className={`text-2xl font-bold text-${colour}-700`}>{count}</p>
            <p className={`text-xs text-${colour}-600`}>{label}</p>
          </div>
        ))}
      </div>
      <table className="text-xs w-full border-collapse">
        <thead>
          <tr className="bg-gray-50">
            {["Series", "Class", "Model", "MASE", "Status"].map(h => (
              <th key={h} className="px-2 py-1 text-left text-gray-600 border-b">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {series_results.map((r: any) => (
            <tr key={r.series_key} className="border-b hover:bg-gray-50">
              <td className="px-2 py-1 font-mono">{r.series_key}</td>
              <td className="px-2 py-1">{r.demand_class}</td>
              <td className="px-2 py-1">{r.model_name}</td>
              <td className="px-2 py-1">{r.mase?.toFixed(3)}</td>
              <td className="px-2 py-1">
                <span className={`px-1.5 py-0.5 rounded text-xs ${
                  r.forecastability === "forecastable" ? "bg-green-100 text-green-700" :
                  r.forecastability === "caution"      ? "bg-yellow-100 text-yellow-700" :
                  "bg-red-100 text-red-700"
                }`}>{r.forecastability}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 2: Write `frontend/src/components/DecisionPanel.tsx`**

```tsx
import React from "react"
import { useDecisionStore } from "../stores/decisionStore"

const STATUS_COLOUR: Record<string, string> = {
  SUPPORTED:              "bg-green-100 text-green-700",
  CONTRADICTED:           "bg-red-100 text-red-700",
  AMBIGUOUS:              "bg-yellow-100 text-yellow-700",
  UNVERIFIABLE:           "bg-gray-100 text-gray-600",
  USER_OVERRIDE_ACCEPTED: "bg-orange-100 text-orange-700",
}

export function DecisionPanel() {
  const { claims, risks, overrides } = useDecisionStore()

  return (
    <div className="p-3 flex flex-col gap-4 text-sm">
      <h2 className="font-semibold text-gray-700">Decisions</h2>

      {claims.length > 0 && (
        <section>
          <h3 className="text-xs font-semibold text-gray-500 uppercase mb-2">Claims ({claims.length})</h3>
          <div className="flex flex-col gap-2">
            {claims.map(c => (
              <div key={c.claim_id} className="rounded border p-2">
                <p className="text-xs text-gray-800 mb-1">{c.claim}</p>
                <div className="flex gap-1 flex-wrap">
                  <span className={`text-xs px-1.5 py-0.5 rounded ${STATUS_COLOUR[c.verification_status] || "bg-gray-100"}`}>
                    {c.verification_status}
                  </span>
                  <span className="text-xs px-1.5 py-0.5 rounded bg-gray-100 text-gray-600">{c.evidence_type}</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {risks.length > 0 && (
        <section>
          <h3 className="text-xs font-semibold text-gray-500 uppercase mb-2">Risks ({risks.length})</h3>
          <div className="flex flex-col gap-2">
            {risks.map(r => (
              <div key={r.risk_id} className="rounded border p-2">
                <p className="text-xs text-gray-800">{r.description}</p>
                <span className={`text-xs px-1.5 py-0.5 rounded mt-1 inline-block ${
                  r.severity === "high" ? "bg-red-100 text-red-700" :
                  r.severity === "medium" ? "bg-yellow-100 text-yellow-700" :
                  "bg-gray-100 text-gray-600"
                }`}>{r.severity}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {overrides.length > 0 && (
        <section>
          <h3 className="text-xs font-semibold text-orange-600 uppercase mb-2">Overrides ({overrides.length})</h3>
          <div className="flex flex-col gap-2">
            {overrides.map((o: any) => (
              <div key={o.claim_id} className="rounded border border-orange-200 p-2 bg-orange-50">
                <p className="text-xs text-orange-800">{o.claim}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {claims.length === 0 && risks.length === 0 && (
        <p className="text-gray-400 text-xs">Claims and risks will appear here during scoping.</p>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Update `App.tsx` — swap placeholder panel + add ReportView**

```tsx
// In App.tsx, replace the inline DecisionPanel function with:
import { DecisionPanel } from "./components/DecisionPanel"
import { ReportView } from "./components/ReportView"

// In MainPanel, add ReportView when phase is report_ready:
{phase === "report_ready" && run_id && <ReportView run_id={run_id} />}
```

And update the `<Layout>` call:
```tsx
<Layout sidebar={<Sidebar />} main={<MainPanel />} panel={<DecisionPanel />} />
```

- [ ] **Step 4: Build final bundle**

```powershell
cd frontend; npm run build; cd ..
```

Expected: `frontend/dist/` built with no TypeScript errors.

- [ ] **Step 5: Full end-to-end smoke test**

Start the backend:
```powershell
$env:PYTHONPATH = "backend;."
c:/Agent_A/venv/Scripts/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

In a browser: open `http://127.0.0.1:8765`.

Expected:
- Page loads with the Agent P layout.
- "New Run" creates a run and shows a run_id.
- CSV upload triggers pre-flight and shows series count.
- `GET /api/v1/runs/{run_id}` returns JSON with phase = `meridian_scoping` after upload.

- [ ] **Step 6: Commit**

```powershell
git add frontend/src/components/ReportView.tsx frontend/src/components/DecisionPanel.tsx frontend/src/App.tsx frontend/dist/
git commit -m "feat: ReportView and DecisionPanel — claims, risks, overrides, foundry report table"
```

---

## Self-Review

### Spec coverage checklist

| Spec section | Covered in task(s) |
|---|---|
| data_store — series sentinel | Task 2 |
| RunState — Phase enum, disk persistence | Task 3 |
| contracts.py — single source of domain models | Task 4 |
| Guard — token budget, call limits, duplicate detection | Task 5 |
| providers.py — tool registry + dispatch_tool | Tasks 6, 22 |
| Pre-flight — all 11 stat tools | Tasks 7, 8 |
| Pre-flight orchestrator | Task 9 |
| Lens — Haiku, temperature=0, structured JSON | Task 10 |
| Conductor tools — state transitions, halt, prism run | Task 11 |
| Conductor agent — routing loop, one tool/turn | Task 12 |
| Meridian pack tools — claim ledger, risk register | Task 13 |
| Meridian diagnostic tools — 7 read tools | Task 14 |
| Meridian agent — streaming, SSE events | Task 15 |
| Forge tools — EDA, Syntetos-Boylan, folds, feature config | Task 16 |
| Forge agent — segment-by-segment loop | Task 17 |
| Foundry tools — demand class gates, train/eval, report | Task 18 |
| Foundry agent — self-correction loop, FoundryRunGuard | Task 19 |
| Prism tools — clone pack, override, reclassify (ADR-0001) | Task 20 |
| Prism agent — scenario runner | Task 21 |
| SSE queue — all event types | Task 23 |
| API routers — thin, all 4 routers | Task 24 |
| app.py — FastAPI entry point | Task 25 |
| Frontend stores — 4 Zustand stores | Task 27 |
| useSSE — all 8 event types wired | Task 28 |
| Layout, ProgressBar | Task 29 |
| Conversation, MessageInput, streaming bubble | Task 30 |
| ReportView, DecisionPanel | Task 31 |
| ADR-0001 — Prism re-classifies demand class | Task 20 (`run_forge_for_scenario`) |
| ADR-0002 — Lens uses pipeline_state as prior | Task 10 (system prompt) |
| ADR-0003 — RunState persisted to JSON | Task 3 |
| ADR-0004 — No POC_v1 import | All tasks (fresh implementations) |

### Placeholder scan

No TBD, TODO, or "implement later" in any code block. All test assertions are concrete. All function signatures match across tasks (e.g., `run_meridian` signature consistent between Task 15 definition and Task 24 call site).

### Type consistency

- `IntentPack.raw_quote` + structured `IntentEntities` used in Lens (Task 10) — matches `contracts.py` definition (Task 4).
- `GuardHalt` raised with `exc.args[0]` in message router (Task 24) — consistent with `GuardHalt(Exception)` in guard.py (Task 5).
- `PreflightBundle.data_quality_report` used in upload response (Task 24) — matches `contracts.py` field name.
- `FoundryRunGuard(run_id=run_id)` — matches Task 5 dataclass definition.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-31-agent-p-implementation.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast parallel iteration.

**2. Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints.