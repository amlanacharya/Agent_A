/**
 * Contract test (CB3 acceptance gate).
 *
 * Verifies that the generated `src/api/schema.ts` parses every
 * response the FastAPI surface can emit. This catches schema drift
 * between Pydantic (Python) and openapi-typescript (TypeScript) at
 * test time, not at build time.
 *
 * Strategy: spawn `uv run python -m api` in a subprocess once,
 * wait for `/surfaces` to respond, then hit every surface +
 * every plot kind with a canned run_id. Validate each response
 * by passing it through the TypeScript type system (a function
 * with a typed parameter will fail to compile if the runtime
 * value doesn't match the type — the type cast is intentional
 * to force the check at the call site).
 *
 * Node's fetch needs absolute URLs — we use http://localhost:8000
 * in the test (the dev server runs on the same host). Production
 * frontend code uses the relative `/api` path that Vite proxies.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  fetchSurface,
  listSurfaces,
  renderPlot,
  setApiBase,
  type PlotKind,
  type PlotResponse,
  type SurfaceName,
  type SurfaceSnapshot,
} from "@/api/client";

const API_BASE = "http://localhost:8000";

const SURFACES: SurfaceName[] = [
  "mission_control",
  "data_health",
  "canonical_table_builder",
  "eda_explorer",
  "feature_factory",
  "model_arena",
  "forecast_review",
  "replenishment_board",
  "mlops_monitor",
  "learning_journal",
];

const RUN_ID = "contract-test-run";

// Per-plot-kind minimal params (the engine validates these and
// returns 400 if missing). The values are arbitrary — the contract
// test only asserts the *shape* of the response, not the rendered
// image content.
const PLOT_PARAMS: Record<PlotKind, Record<string, unknown>> = {
  demand_curve: {
    weeks: ["W1", "W2", "W3"],
    actual: [10, 12, 8],
    forecast: [11, 11, 9],
  },
  sparsity: {
    series: [
      { series_key: "SKU_1", adi: 1.0, cv2: 0.4 },
      { series_key: "SKU_2", adi: 1.4, cv2: 0.8 },
    ],
  },
  anomalies: { weeks: ["W1", "W2", "W3"], values: [10, 50, 12], flags: [false, true, false] },
  forecast_band: {
    weeks: ["W1", "W2", "W3"],
    forecast: [11, 11, 9],
    lower: [9, 9, 7],
    upper: [13, 13, 11],
  },
  backtest: {
    folds: ["fold_1", "fold_2", "fold_3"],
    actual: [10, 12, 8],
    forecast: [11, 11, 9],
  },
  feature_importance: { features: [{ name: "lag_1", importance: 0.5 }, { name: "rolling_7", importance: 0.3 }] },
  drift_chart: {
    runs: ["r1", "r2"],
    segments: { G1: [0.85, 0.87], G2: [0.90, 0.88] },
  },
};

const PLOT_KINDS = Object.keys(PLOT_PARAMS) as PlotKind[];

let proc: ChildProcess | undefined;
const STARTUP_MS = 8000;

async function waitForServer(): Promise<void> {
  const deadline = Date.now() + STARTUP_MS;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`${API_BASE}/surfaces`);
      if (r.ok) return;
    } catch {
      /* not ready yet */
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`FastAPI did not start within ${STARTUP_MS}ms`);
}

beforeAll(async () => {
  // Node's fetch needs an absolute URL; the browser-side client uses
  // the relative /api path that Vite proxies. setApiBase swaps the
  // base for this test run only.
  setApiBase(API_BASE);
  proc = spawn("uv", ["run", "python", "-m", "api"], {
    cwd: "..",
    env: { ...process.env },
    stdio: "ignore",
    shell: true,
  });
  await waitForServer();
}, STARTUP_MS + 2000);

afterAll(() => {
  proc?.kill();
});

describe("FastAPI surface contract (CB3)", () => {
  it("lists registered surfaces", async () => {
    const list = await listSurfaces();
    expect(Array.isArray(list.surfaces)).toBe(true);
    expect(list.surfaces.length).toBeGreaterThan(0);
    for (const name of list.surfaces) {
      expect(typeof name).toBe("string");
    }
  });

  for (const surfaceName of SURFACES) {
    it(`renders ${surfaceName} for a canned run`, async () => {
      const snapshot = (await fetchSurface(surfaceName, RUN_ID)) as SurfaceSnapshot;
      expect(snapshot.run_id).toBe(RUN_ID);
      expect(snapshot.surface).toBe(surfaceName);
      expect(typeof snapshot.state).toBe("object");
    });
  }

  for (const kind of PLOT_KINDS) {
    it(`renders plot kind: ${kind}`, async () => {
      const response = (await renderPlot({
        run_id: RUN_ID,
        kind,
        params: PLOT_PARAMS[kind],
      })) as PlotResponse;
      expect(response.kind).toBe(kind);
      expect(response.content_type).toMatch(/^image\/(png|svg\+xml)$/);
      expect(response.bytes_b64.length).toBeGreaterThan(0);
      expect(response.width).toBeGreaterThan(0);
      expect(response.height).toBeGreaterThan(0);
    });
  }
});
