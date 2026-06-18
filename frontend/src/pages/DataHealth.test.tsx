import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { DataHealth } from "./DataHealth";

function cannedState(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    series_count: 2,
    segment_count: 1,
    segment_profiles: [
      {
        segment_id: "G1",
        series_count: 2,
        median_adi: 1.2,
        median_cv2: 0.5,
        demand_class_distribution: { SMOOTH: 1, ERRATIC: 1 },
        forecastability_breakdown: { forecastable: 2 },
      },
    ],
    demand_class_breakdown: { SMOOTH: 1, ERRATIC: 1 },
    narrative: "Two-series run.",
    ...overrides,
  };
}

function renderDataHealth(state: Record<string, unknown>): ReturnType<typeof render> {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.endsWith("/surfaces") || url.endsWith("/api/surfaces")) {
      return new Response(JSON.stringify({ surfaces: ["data_health"] }), {
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
      <DataHealth runId="dev-run" />
    </QueryClientProvider>,
  );
}

describe("<DataHealth> (CB6)", () => {
  it("renders the 3 KPI cards (series, segments, demand classes)", async () => {
    renderDataHealth(cannedState());
    await waitFor(() =>
      expect(screen.getByRole("heading", { level: 1 }).textContent).toMatch(/Run dev-run/),
    );
    expect(screen.getAllByText(/^Series$/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/^Segments$/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Demand classes/i).length).toBeGreaterThan(0);
  });

  it("renders the segment profiles table with the right columns", async () => {
    renderDataHealth(cannedState());
    await waitFor(() =>
      expect(screen.getByText(/Segment profiles/i)).toBeInTheDocument(),
    );
    expect(screen.getByText("G1")).toBeInTheDocument();
    // The "SMOOTH: 1, ERRATIC: 1" string appears in both the
    // Demand-classes MetricCard caption and the table cell;
    // assert at least one match.
    expect(screen.getAllByText("SMOOTH: 1, ERRATIC: 1").length).toBeGreaterThan(0);
  });

  it("renders the narrative when present", async () => {
    renderDataHealth(cannedState({ narrative: "A custom narrative." }));
    await waitFor(() =>
      expect(screen.getByText(/A custom narrative\./)).toBeInTheDocument(),
    );
  });

  it("shows the empty-state when there are no segment profiles", async () => {
    renderDataHealth(cannedState({ segment_profiles: [], segment_count: 0 }));
    await waitFor(() =>
      expect(screen.getByText(/No segment profiles for this run\./)).toBeInTheDocument(),
    );
  });
});
