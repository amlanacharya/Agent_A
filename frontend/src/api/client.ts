/**
 * Typed API client for the Phase 8 FastAPI cockpit surface.
 *
 * `openapi-typescript` (run via `pnpm generate-api`) reads the FastAPI
 * server's `/openapi.json` and writes `src/api/schema.ts` with the
 * typed operations + components. This module wraps those types in
 * a thin fetch helper so the React components don't need to know
 * the URL shape.
 *
 * In the browser, the base URL is `/api` (Vite proxies /api/* to
 * http://localhost:8000 in dev; in production FastAPI serves the
 * SPA at `/` and the same `/api` path works behind the same
 * origin). In Node-side tests, the base URL can be overridden to
 * an absolute origin via `setApiBase()` — Node's `fetch` requires
 * absolute URLs.
 */

import type { operations } from "./schema";

// The 9 closed surface kinds + 7 closed plot kinds are typed as
// union strings here. They mirror the Python ``SurfaceName`` and
// ``PlotKind`` ``Literal`` types in ``api/models.py``; the
// openapi-typescript codegen does NOT promote these to standalone
// schemas (it inlines them into the operation types), so we
// redefine them as TS string literal unions. If a new surface or
// plot kind is added to api/models.py, add it here too — the
// ``fetchSurface`` + ``renderPlot`` signatures will fail to compile
// in the surface CB if the union is wrong.
export type SurfaceName =
  | "mission_control"
  | "data_health"
  | "canonical_table_builder"
  | "eda_explorer"
  | "feature_factory"
  | "model_arena"
  | "forecast_review"
  | "replenishment_board"
  | "mlops_monitor"
  | "learning_journal";

export type PlotKind =
  | "demand_curve"
  | "sparsity"
  | "anomalies"
  | "forecast_band"
  | "backtest"
  | "feature_importance"
  | "drift_chart";

// SurfaceSnapshot and PlotResponse are open-ended dicts on the
// FastAPI side (the per-surface ``state`` is `dict[str, object]`).
// The TS types reflect that — the surface CB adds bespoke fields
// via intersection if needed, but the contract test only checks
// the structural shape.
export interface SurfaceSnapshot {
  run_id: string;
  surface: SurfaceName;
  state: Record<string, unknown>;
}

export interface PlotResponse {
  kind: PlotKind;
  content_type: string;
  bytes_b64: string;
  width: number;
  height: number;
}

export interface CockpitPlotRequest {
  run_id: string;
  kind: PlotKind;
  params?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Phase 10 cockpit driver — upload + chat + advance response shapes.
// ---------------------------------------------------------------------------

export interface UploadResponse {
  run_id: string;
  domain: string;
  preflight: Record<string, unknown>;
  state: Record<string, unknown>;
}

export type PossibilityKind = "ACCEPT" | "OVERRIDE" | "CLARIFY";

export interface Possibility {
  kind: PossibilityKind;
  label: string;
  payload: Record<string, unknown>;
}

export interface MessageRequest {
  run_id: string;
  user_message: string;
}

export interface MessageResponse {
  intent: string;
  run_id: string;
  reply: string;
  possibilities: Possibility[];
  advanced_to?: string;
  state?: Record<string, unknown>;
  prism_run_id?: string;
}

export interface AdvanceRequest {
  force?: boolean;
}

export interface AdvanceResponse {
  run_id: string;
  advanced_to: string;
  reply: string;
  possibilities: Possibility[];
  state?: Record<string, unknown>;
}

export interface CockpitStateResponse {
  run_id: string;
  current_step: string;
  active_agent: string;
  phase: string;
  tool_result?: string | null;
  code_escalation_status?: string | null;
  code_attempt?: number | null;
  verifier_gate?: string | null;
  approval_needed?: boolean;
  confidence?: string;
  blockers?: string[];
}

// Vite's bundler replaces `import.meta.env` at build time. In Node-side
// tests there is no import.meta.env; default to `/api` (browser shape).
let BASE: string =
  typeof import.meta !== "undefined" &&
  (import.meta as unknown as { env?: { VITE_API_BASE?: string } }).env
    ?.VITE_API_BASE
    ? (import.meta as unknown as { env: { VITE_API_BASE: string } }).env
        .VITE_API_BASE
    : "/api";

/** Override the base URL (used by the contract test in Node). */
export function setApiBase(url: string): void {
  BASE = url;
}

/** Read the current base URL (handy for tests + debugging). */
export function getApiBase(): string {
  return BASE;
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "content-type": "application/json", ...init?.headers },
    ...init,
  });
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new ApiError(response.status, response.statusText, body);
  }
  return (await response.json()) as T;
}

/**
 * Multipart form-data fetch — bypasses the JSON content-type header
 * because the browser sets the boundary itself when the body is a
 * FormData. The server reads ``file`` + ``domain`` from the form.
 */
async function multipartFetch<T>(path: string, form: FormData): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new ApiError(response.status, response.statusText, body);
  }
  return (await response.json()) as T;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly statusText: string,
    public readonly body: string,
  ) {
    super(`API ${status} ${statusText}: ${body.slice(0, 200)}`);
    this.name = "ApiError";
  }
}

/** GET /surfaces — list the registered surface names (for the UI menu). */
export function listSurfaces(): Promise<{ surfaces: SurfaceName[] }> {
  return jsonFetch("/surfaces");
}

/** GET /surfaces/{surface_name}/{run_id} — render the named surface. */
export function fetchSurface(
  surfaceName: SurfaceName,
  runId: string,
): Promise<SurfaceSnapshot> {
  return jsonFetch(`/surfaces/${encodeURIComponent(surfaceName)}/${encodeURIComponent(runId)}`);
}

/** POST /plots — render a plot via the engine. */
export function renderPlot(request: CockpitPlotRequest): Promise<PlotResponse> {
  return jsonFetch("/plots", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

// ---------------------------------------------------------------------------
// Phase 10 cockpit driver client functions.
// ---------------------------------------------------------------------------

/**
 * POST /uploads — multipart CSV upload, kicks off the preflight
 * pipeline synchronously and returns the bundle + run_id.
 */
export function uploadCsv(file: File, domain: string): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("domain", domain);
  return multipartFetch("/uploads", form);
}

/** POST /messages — chat-loop dispatch (Lens classifies, conductor replies). */
export function postMessage(request: MessageRequest): Promise<MessageResponse> {
  return jsonFetch("/messages", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

/** POST /runs/{run_id}/advance — driver-button advance (force bypasses chat gate). */
export function advanceRun(
  runId: string,
  request: AdvanceRequest = {},
): Promise<AdvanceResponse> {
  return jsonFetch(`/runs/${encodeURIComponent(runId)}/advance`, {
    method: "POST",
    body: JSON.stringify(request),
  });
}

/** GET /cockpit-state/{run_id} — live state for polling the left rail. */
export function fetchCockpitState(runId: string): Promise<CockpitStateResponse> {
  return jsonFetch(`/cockpit-state/${encodeURIComponent(runId)}`);
}
