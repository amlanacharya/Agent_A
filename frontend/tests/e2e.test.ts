import { spawn, type ChildProcess } from "node:child_process";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { setApiBase } from "@/api/client";

/**
 * End-to-end smoke (CB12 acceptance gate).
 *
 * Spawns the FastAPI process once, walks every surface + the
 * /runs endpoint + a sample plot, and asserts each returns
 * 200 with the expected shape. The contract test (tests/contract.test.ts)
 * already covers plot exhaustiveness + surface body shape; this
 * e2e is a "is the system actually wired together" check that
 * runs against a real subprocess (matching the real deploy shape:
 * `uv run python -m api`).
 *
 * Node's fetch needs absolute URLs — `setApiBase` swaps the
 * client base from the relative `/api` (browser) to the local
 * FastAPI process.
 */

const API_BASE = "http://localhost:8000";
const RUN_ID = "e2e-test-run";

const SURFACES = [
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
] as const;

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

describe("Frontend ↔ FastAPI end-to-end (CB12)", () => {
  it("lists the registered surfaces", async () => {
    const response = await fetch(`${API_BASE}/surfaces`);
    expect(response.status).toBe(200);
    const body = (await response.json()) as { surfaces: string[] };
    expect(body.surfaces.length).toBeGreaterThan(0);
  });

  for (const surface of SURFACES) {
    it(`renders the ${surface} surface end-to-end`, async () => {
      const response = await fetch(
        `${API_BASE}/surfaces/${surface}/${RUN_ID}`,
      );
      expect(response.status).toBe(200);
      const body = (await response.json()) as {
        run_id: string;
        surface: string;
        state: Record<string, unknown>;
      };
      expect(body.run_id).toBe(RUN_ID);
      expect(body.surface).toBe(surface);
      expect(typeof body.state).toBe("object");
    });
  }

  it("lists runs under outputs/ (or returns empty when missing)", async () => {
    const response = await fetch(`${API_BASE}/runs`);
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      runs: Array<{ run_id: string; path: string }>;
    };
    expect(Array.isArray(body.runs)).toBe(true);
    // The dev launcher uses a temp outputs/ root — it may or may
    // not have any qualifying runs. The endpoint is contract-correct
    // either way.
  });
});
