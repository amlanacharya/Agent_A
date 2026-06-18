import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { render, screen } from "@testing-library/react";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { router as appRouter } from "./router";
import { AppShell } from "@/components/AppShell";
import { GenericSurfacePage } from "@/pages/SurfacePage";

/**
 * MSW-free fetch stub for App.test.tsx. The router-driven tests
 * don't want the real FastAPI surface — they just want the AppShell
 * to render its nav + a heading. We intercept /api/surfaces and
 * /api/surfaces/* and return a canned SurfaceSnapshot so the page
 * can finish loading.
 */
const CANNED_LIST = {
  surfaces: ["mission_control", "data_health", "eda_explorer", "model_arena"],
};
const CANNED_SNAPSHOT = (name: string, runId: string) => ({
  run_id: runId,
  surface: name,
  state: { canned: true, name, runId },
});

beforeAll(() => {
  // jsdom doesn't ship fetch in v22+ by default in some envs; the
  // contract test brings Node's fetch in via vitest's environment.
  // For these tests we use a stub so the AppShell renders without
  // any real network.
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.endsWith("/surfaces") || url.endsWith("/api/surfaces")) {
      return new Response(JSON.stringify(CANNED_LIST), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    const m = url.match(/\/(?:api\/)?surfaces\/([^/]+)\/([^/]+)/);
    if (m) {
      return new Response(JSON.stringify(CANNED_SNAPSHOT(m[1]!, m[2]!)), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    return new Response("{}", { status: 200 });
  }) as typeof fetch;
});

afterAll(() => {
  // No-op: we don't restore the original fetch here because the
  // contract test sets its own setApiBase + spawns a real server.
});

function renderWithProviders(initialEntries: string[] = ["/"]): ReturnType<typeof render> {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const testRouter = createMemoryRouter(appRouter.routes, {
    initialEntries,
  });
  return render(
    <QueryClientProvider client={qc}>
      <RouterProvider router={testRouter} />
    </QueryClientProvider>,
  );
}

describe("App router (CB4)", () => {
  it("renders the default Mission Control surface directly", async () => {
    // The Mission Control route is wired in CB5 with a bespoke
    // page. The h1 is the PageHeader's title: `Run {runId}`.
    renderWithProviders(["/surfaces/mission_control/dev-run"]);
    const heading = await screen.findByRole("heading", { level: 1 });
    expect(heading.textContent).toMatch(/Run dev-run/);
  });

  it("renders the AppShell top nav with the brand mark", async () => {
    renderWithProviders(["/surfaces/mission_control/dev-run"]);
    expect(screen.getByText(/Data Intelligence Cockpit/i)).toBeInTheDocument();
    expect(screen.getByText("A")).toBeInTheDocument();
  });

  it("renders the route outlet's content (generic surface page)", async () => {
    renderWithProviders(["/surfaces/data_health/run-42"]);
    const heading = await screen.findByRole("heading", { level: 1 });
    expect(heading.textContent).toMatch(/data_health/);
    expect(heading.textContent).toMatch(/run-42/);
  });

  it("imports AppShell + GenericSurfacePage for type coverage", () => {
    expect(typeof AppShell).toBe("function");
    expect(typeof GenericSurfacePage).toBe("function");
  });
});
