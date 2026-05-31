# Design Review — Agent P planning docs

**Date:** 2026-05-31
**Reviewer:** Claude (Opus 4.8)
**Scope:** `CONTEXT.MD`, `plan_v2.md`, `docs/plans/plan.md`, `docs/specs/spec.md`, `docs/adr/scenario{1,2,3}.md`

---

## 0. Orientation — what these files actually are

- `CONTEXT.MD` — domain glossary. Authoritative on vocabulary (Segment, Series, Claim, etc.).
- `plan_v2.md` — high-level architecture spec (the "idea").
- `docs/plans/plan.md` — **contains the behavioural Full Implementation Spec** (Pydantic contracts, function signatures, invariants). 2,578 lines.
- `docs/specs/spec.md` — **contains the task-by-task TDD Implementation Plan** (Tasks 1–31, the thing an agent executes). 6,103 lines.

> **Nit:** the contents of `plans/plan.md` and `specs/spec.md` are swapped relative to their folder names. The "plan" folder holds the spec; the "specs" folder holds the plan. Worth renaming or moving so the next reader isn't misled.

The central problem below is that **the implementation plan (`specs/spec.md`) has drifted from the behavioural spec (`plans/plan.md`), and the drift lives in `contracts.py` — the artifact declared as the single source of truth — and in the tools that read/write it.** Everything else is downstream of that.

---

## 1. The "Segment" abstraction has collapsed into "Demand Class" — and its wiring is broken  ⚠️ FOUNDATIONAL

`CONTEXT.MD` defines **Segment** as a *named grouping inferred from the data schema, shaped by the playbook* (e.g. `G1`, `G2`); a Series belongs to exactly one Segment; Forge and Foundry operate segment-by-segment. The behavioural spec's `SegmentProfile` carries `segment_id`, `demand_class_distribution`, `median_adi`, etc.

The implementation plan does something else entirely:

- `contracts.py` `SegmentProfile` has **no `segment_id`** — its fields are `sb_class`, `series_count`, `fraction`, `example_keys` (`specs/spec.md` Task 4).
- `aggregate_segment_profiles` **buckets series by `sb_class`** (SMOOTH/ERRATIC/INTERMITTENT/LUMPY), not by any schema-derived segment.
- `PreflightBundle` has **no `segments` field** at all.
- Meridian's diagnostic tools treat `segment_id` *as a demand-class string*: `summarise_demand_segments` filters `p.get("sb_class") == segment_id`, and the Task 14 tests call `diagnose_zero_demand_policy(run_id, segment_id="SMOOTH")`.

So "segment" now means "demand class." Nothing in pre-flight produces a real `segment_id` like `G1`. It appears only hardcoded in a Foundry **test fixture** (`_write_eda` → `segment_id: "G1"`) and as free-form dicts the Meridian LLM must invent for `compile_domain_context_pack(..., segments=[...])`. There is no deterministic segmentation step anywhere, even though CONTEXT.MD says pre-flight is the primary source of segments.

**And the wiring is outright broken:** `diagnose_zero_demand_policy` / `diagnose_spike_policy` do
`{k: v for k, v in zero_runs.items() if segment_id.upper() in k.upper()}`
where `k` is a series key like `SKU_A|NORTH`. The passed `segment_id` is a demand-class name (`"SMOOTH"`), which is **never** a substring of a grain key. → `matching` is always empty → the tool always returns the "No zero-demand data found for this segment" fallback. These diagnostics cannot function as written.

**Impact:** every downstream tool that takes `segment_id` (Forge `detect_structural_breaks`, `design_walk_forward_folds`; Foundry `get_segment_series_list`; all Meridian diagnostics) has no coherent, deterministically-produced input. This is the highest-priority issue — resolve the Segment model before building Phase C.

**Recommendation:** decide one thing and make it consistent end-to-end: either (a) implement real schema/playbook-derived segments in pre-flight (with `segment_id`, a series→segment map, and `demand_class_distribution` per segment, matching the behavioural spec), or (b) formally redefine "Segment = Demand Class" in CONTEXT.MD and fix the diagnostic tools to map series→class via `adi_cv2_by_series` rather than substring-matching the class name against grain keys.

---

## 2. `contracts.py` is not the single source of truth it claims to be  ⚠️ ROOT CAUSE

The plan declares `contracts.py` the "single source of truth for all shared domain models." In practice the models there match **neither** the behavioural spec **nor** what the tools actually read and write. Most of the type mismatches in §3 are symptoms of this. Concretely:

- **`DomainContextPack`** (contracts.py): `run_id, domain, horizon_weeks, mase_target, series_keys, feature_flags, user_notes, confirmed`. But `compile_domain_context_pack` (Task 13) writes a *raw dict* with a completely different shape: `forecast_scope, segments[], claim_ledger, risk_register, override_count, open_risks`. The Pydantic `DomainContextPack` model is **never instantiated** by the tool that produces the pack. There is no `horizon_weeks`/`mase_target`/`series_keys`/`feature_flags` in the written pack, and no `forecast_scope`/`segments` in the model. The two are unrelated structures sharing a name.
- **`Claim`** (contracts.py): `claim_id, text, status, source, created_at, resolved_at, resolution_note`. But `add_claim` (Task 13) writes `claim, verification_status, evidence_type, evidence_ref, applies_to, downstream_impact, must_surface_in_report, created_at`. Different field names (`text` vs `claim`, `status` vs `verification_status`) and the entire evidence model (`evidence_type`, `applies_to`, `downstream_impact`) is absent from the contract. `Claim`/`ClaimLedger` models are dead code.
- **`IntentPack`** (contracts.py): `intent, confidence, entities: dict[str,str], raw_message`. The behavioural spec and `lens.py`'s own docstring use structured `IntentEntities` (skus/segments/dates/metrics/scenario) and `raw_quote`. `lens.py` even *defines* `IntentEntities` but never uses it — it returns `entities` as a loose dict and `raw_message`.

**Recommendation:** before any Phase C work, reconcile `contracts.py` with (a) the behavioural spec and (b) the dicts the Task 13/16/18/20 tools actually emit. Make the compile/record tools *return validated Pydantic models*, not hand-built dicts, so drift can't silently reappear. This single fix collapses most of §3.

---

## 3. `IntentType` enum omits two intents the rest of the system requires  ⚠️ CONCRETE BUG

`contracts.py` defines:
```python
IntentType = Literal["NEW_RUN","SCOPE_RESPONSE","ADVANCE_PIPELINE","WHATIF_REQUEST","CLARIFICATION","UNKNOWN"]
```
But CONTEXT.MD, plan_v2, the behavioural spec, **and `lens.py`'s own system prompt** use a different six: `SCOPE_RESPONSE, OVERRIDE, ADVANCE_PIPELINE, WHAT_IF_REQUEST, CLARIFICATION, CORRECTION`.

Mismatches:
- **`OVERRIDE` and `CORRECTION` are missing** from the Literal, yet Conductor's hard rules and routing table (Task 11/12) branch on both, and `lens.py` is instructed to emit them. A Lens response of `OVERRIDE`/`CORRECTION` will **fail Pydantic validation** against `IntentPack.intent`.
- **`WHATIF_REQUEST`** (contracts) vs **`WHAT_IF_REQUEST`** (lens prompt + spec + Conductor routing). Spelling mismatch → the what-if path never matches.
- `NEW_RUN`/`UNKNOWN` are introduced in contracts but appear nowhere in the behavioural design.

**Recommendation:** make `IntentType` exactly the six canonical values and delete `NEW_RUN`/`UNKNOWN` (or justify them in an ADR). Add a Lens unit test that round-trips every intent value through `IntentPack` validation.

---

## 4. Prism doesn't run scenarios — it fabricates the comparison  ⚠️ FEATURE GAP

The whole point of Prism (CONTEXT.MD, plan_v2 §9, ADR scenario1) is to re-run Forge + Foundry on affected series with an override applied, re-classifying Demand Class. The Task 20 implementation does none of the modelling:

- `apply_whatif_override` appends override **metadata to the cloned pack only** — it never modifies the series in `data_store`. So even a real model re-run would see unchanged data.
- `run_forge_for_scenario` *does* re-classify demand (honouring ADR scenario1's intent — good), **but the result is discarded**: `compile_comparison` hardcodes `demand_class_changed: False` and `scenario_sb_class = baseline_sb_class`. The re-classification output is never consumed.
- There is **no `run_foundry_for_scenario`** — no model is re-trained for the scenario.
- `compile_comparison` computes the scenario forecast as `baseline_mean * magnitude`, repeated 4× (a flat line). The "comparison" is `baseline_mean` vs `baseline_mean × constant`.

Net: Prism returns a deterministic multiply-by-constant, not a forecast. This contradicts the documented behaviour even though it doesn't contradict ADR scenario1 (the re-classification call exists; its output is just thrown away).

**Recommendation:** either implement the real re-run loop (mutate series data per override → re-classify → re-fit within the gate → compare), or explicitly scope Prism down to "indicative mean-scaling preview" in plan_v2/CONTEXT and remove the misleading `baseline_sb_class`/`scenario_sb_class`/`demand_class_changed` fields that imply real re-modelling.

---

## 5. `PreflightBundle` violates plan_v2 correction #4 (context bloat)

plan_v2 correction #4 and the behavioural spec are explicit: the bundle handed to Meridian is **aggregate per segment** (`segment_profiles`) plus a *small* `segment_exceptions` list; individual per-series stats stay in `data_store`/`preflight.json` and Meridian pulls them via diagnostic tools. The behavioural spec even states promo/trend/seasonality are deliberately *not* in the bundle.

The implementation's `PreflightBundle` (Task 4) instead carries **full per-series dicts**: `adi_cv2_by_series`, `zero_runs_by_series`, `spikes_by_series`, `promo_alignment_by_series`, `trend_by_series`, `seasonality_by_series`. Since the bundle is injected into Meridian's system prompt, every series' stats land in Claude's context on every Meridian turn — exactly what correction #4 was written to prevent. (This is a *correction-#4 / scale* problem, not a strict sentinel-pattern violation — the sentinel rule is about raw series *values*, which do stay in `data_store`.)

**Recommendation:** drop the per-series dicts from the injected bundle; expose them only through the diagnostic tools (which already load `preflight.json`). Keep `segment_profiles` + a capped `segment_exceptions`.

---

## 6. Windows filename hazard — `|` in series keys used as path component

Series keys are pipe-delimited (`SKU_A|NORTH`). The behavioural spec and plan_v2 §18 say the on-disk path is `series_results/{series_key}.json` — i.e. literally `SKU_A|NORTH.json`. **`|` is illegal in Windows filenames, and the target platform is win32.** As written, that path would fail on the actual dev machine.

The implementation quietly works around it: `record_series_result` writes `series_key.replace("|","_")` and Prism's `compile_comparison` reads `key.replace("|","_")`. Write and read are consistent with *each other*, so it functions — but:
- it **silently diverges** from the documented `{series_key}.json` convention (any reader following the spec misses the files);
- `|`→`_` is **lossy and collision-prone**: `SKU_A|NORTH` and `SKU|A_NORTH` both map to `SKU_A_NORTH.json`.

**Recommendation:** define one reversible key↔filename mapping in `data_store`/`run_state` (e.g. percent-encode `|`, or hash), document it in CONTEXT.MD §Series Key and plan_v2 §18, and route all reads/writes through it. Don't leave `replace("|","_")` scattered at call sites.

---

## 7. Pre-flight test data and date parsing — confirmed test failures  ⚠️ CONCRETE BUG

The fixtures (`conftest.py`, `test_preflight_schema.py`, `test_preflight_orchestrator.py`, `test_meridian_diagnostic.py`) all generate dates as ISO-week strings: `"2024-W01"`, `"2024-W02"`, …

**Verified empirically** (`pd.to_datetime(['2024-W01','2024-W02'], errors='coerce')` → **0 of 2 non-null**; every value is `NaT`). pandas cannot parse `%Y-W%W` without an explicit `format`. Consequences in the plan's own code:
- `_find_date_col` needs >80% parseable → returns `None` for the `week` column → `profile_uploaded_data` emits **`MISSING_DATE_COLUMN`**, a *blocking issue*. So `test_profile_no_blocking_issues`, `test_preflight_populates_data_store`, and the Meridian-diagnostic tests (which call `run_preflight` on the same data) all **fail**, not just `test_detect_frequency_weekly`.
- `detect_frequency_and_grain` parses the same column → all `NaT` → frequency stays `"unknown"`, so `test_detect_frequency_weekly` asserting `"weekly"` fails too.

**Recommendation:** change fixtures to real dates (`pd.date_range(..., freq="W")` or `"2024-01-06"`-style strings), or have `_find_date_col`/`map_schema` accept an explicit ISO-week format. Either way, run the Phase B suite before building on it — as written it is red.

**Related nit:** `profile_uploaded_data` computes `series_count = max(1, len(df)//10)` — a fabricated `rows//10` heuristic, not the real series count — and never checks `BELOW_MIN_SERIES`/`UNPARSEABLE_FILE` (those blocking codes are specced but only `UNPARSEABLE_FILE` is raised, by the orchestrator). The reported `series_count` flows into `DataQualityReport` and is simply wrong.

---

## 8. Foundry cumulative counter is process-global, not per-run  ⚠️ CONCRETE BUG

The behavioural spec's `FoundryRunGuard` is per-run (`__init__(self, run_id)`, instance counter). The implementation (Task 5) uses a **module-level global**:
```python
_foundry_cumulative_calls: int = 0
```
and `FoundryRunGuard.check_and_record` mutates that global. Problems:
- **Never resets between runs** — run B inherits run A's count; eventually every run halts immediately.
- **Shared across concurrent runs** — directly undercuts the multi-run sidebar feature (`GET /api/v1/runs`).
- **Order-dependent tests** — `test_foundry_cumulative_limit` only passes in isolation.

**Recommendation:** make the counter per-run (keyed by `run_id`, or an instance field as the behavioural spec shows) and reset/instantiate per run.

---

## 9. Minor nits (low priority, but real)

- **Guard duplicate detection:** implementation halts on the **1st** duplicate (`if h in self._seen_hashes: raise`), but the behavioural spec says `DUPLICATE_CALL_HARD_STOP=2` (first duplicate allowed with a warning, halt on the 2nd) and makes it `.env`-configurable. The implementation also **drops `duplicate_hard_stop` from `GuardConfig` entirely**.
- **`GuardConfig` not `.env`-wired:** plan_v2 correction #6 and the behavioural spec require all guard limits to be `.env`-configurable; the implementation's `GuardConfig` is a plain dataclass with hardcoded defaults (no `os.getenv`).
- **Token budget boundary:** spec uses `>=`, implementation uses `>` (off-by-one at exactly the budget). Cosmetic but pick one.
- **Conductor tool naming:** behavioural spec + implementation Task 11 use `get_run_state`/`update_run_state`; CONTEXT.MD §Run State and the `specs/spec.md` repo-layout block say `get_pipeline_state`/`update_pipeline_state`. Align the names.
- **`infer_datetime_format=True`** in `preflight_schema.py`: deprecated. Harmless no-op + `FutureWarning` on the pinned pandas 2.2.2; **removed/erroring on newer pandas**. Just delete the kwarg (and pass an explicit `format=` per §7).
- **Stale repo-layout block** (`specs/spec.md` lines 43–50): lists only 3 Forge tools and 4 Foundry tools, but Tasks 16/18 correctly spec the full surface (`detect_structural_breaks`, `design_walk_forward_folds`, `walk_forward_validate`, `build_ensemble`, `assess_target_feasibility`, …). The summary is out of date — no missing functionality, just a misleading map.
- **Project naming drift:** "Agent P" (docs) vs working dir `C:\Agent_A` vs repo `agent-p/` vs plan_v2 §15 `c:\Agent_P`. Pick one.

---

## What's genuinely strong

- The behavioural spec (`plans/plan.md`) is unusually thorough: pre/post-condition tables, explicit invariants, halt-propagation flow, and ADRs that record *why*. The ADRs (esp. scenario1 demand-class re-classification, scenario3 file-backed state) are well-reasoned.
- The TDD task structure (failing test → minimal impl → commit) is exactly right for this kind of build, and dependency ordering (Phase A foundation → B pre-flight → C agents → D wiring → E API → F frontend) is sound.
- The Guard/sentinel/RunState-on-disk design is a good fit for a single-process POC; the ADRs correctly reject Redis/LangGraph as premature.
- Demand-class model gates, the three-layer MASE hierarchy, and walk-forward "min 2 folds post-break, no pre-break fallback" are well-specified in the behavioural spec.

---

## Priority order for fixes

1. **§1 Segment model** — decide and unify; nothing downstream is coherent until this is settled.
2. **§2 `contracts.py` reconciliation** — make it real and have tools emit validated models; collapses most of §3.
3. **§3 `IntentType`** — add `OVERRIDE`/`CORRECTION`, fix `WHAT_IF_REQUEST` spelling.
4. **§7 pre-flight date fixtures** — the Phase B suite is currently red.
5. **§4 Prism** — implement the real re-run or rescope honestly.
6. **§8 Foundry global counter**, **§5 bundle bloat**, **§6 Windows filenames**.
7. **§9 nits.**
