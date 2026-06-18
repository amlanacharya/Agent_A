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

import type { components, operations } from "./schema";

export type SurfaceName = components["schemas"]["SurfaceName"];
export type PlotKind = components["schemas"]["PlotKind"];
export type SurfaceSnapshot = components["schemas"]["SurfaceSnapshot"];
export type PlotResponse = components["schemas"]["PlotResponse"];
export type CockpitPlotRequest = components["schemas"]["CockpitPlotRequest"];

// Vite's bundler replaces `import.meta.env` at build time. In Node-side
// tests there is no import.meta.env; default to `/api` (browser shape).
let BASE: string =
  typeof import.meta !== "undefined" &&
  (import.meta as { env?: { VITE_API_BASE?: string } }).env?.VITE_API_BASE
    ? (import.meta as { env: { VITE_API_BASE: string } }).env.VITE_API_BASE
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
export function listSurfaces(): Promise<operations["list_surfaces_surfaces_get"]["responses"]["200"]["content"]["application/json"]> {
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
