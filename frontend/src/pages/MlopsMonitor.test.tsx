import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MlopsMonitor } from "./MlopsMonitor";

function cannedState(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    "MONITORING_REPORT.md": "# Monitoring\n\nAll metrics within bounds.",
    "DRIFT_REPORT.md": "# Drift\n\nMean shift 0.05 across SKU_1.",
    "OVERRIDE_ANALYSIS.md": null,
    "MODEL_HEALTH.md": "# Health\n\nChampion: naive.",
    ...overrides,
  };
}

function renderMlopsMonitor(state: Record<string, unknown>): ReturnType<typeof render> {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.endsWith("/surfaces") || url.endsWith("/api/surfaces")) {
      return new Response(JSON.stringify({ surfaces: ["mlops_monitor"] }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    const m = url.match(/\/(?:api\/)?surfaces\/([^/]+)\/([^/]+)/);
    if (m) {
      return new Response(
        JSON.stringify({ run_id: m[2], surface: m[1], state }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }
    return new Response("{}", { status: 200 });
  }) as unknown as typeof fetch;
  return render(
    <QueryClientProvider client={qc}>
      <MlopsMonitor runId="dev-run" />
    </QueryClientProvider>,
  );
}

describe("<MlopsMonitor> (CB12)", () => {
  it("renders the 4 artifact KPI cards (Monitoring, Drift, Overrides, Health)", async () => {
    renderMlopsMonitor(cannedState());
    // Wait for the artifacts-present KPI to render (4/4).
    await waitFor(() => {
      expect(screen.getAllByText(/3 \/ 4/).length).toBeGreaterThan(0);
    });
    expect(screen.getAllByText(/Monitoring/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Drift/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Overrides/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Health/i).length).toBeGreaterThan(0);
  });

  it("renders the artifact tabs with content", async () => {
    renderMlopsMonitor(cannedState());
    await waitFor(() => {
      expect(screen.getByText(/All metrics within bounds\./)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: "Monitoring" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Drift" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Overrides" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Health" })).toBeInTheDocument();
  });
});
