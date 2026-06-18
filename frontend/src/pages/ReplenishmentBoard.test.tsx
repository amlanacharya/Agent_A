import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReplenishmentBoard } from "./ReplenishmentBoard";

function cannedState(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    recommendation_count: 1,
    recommendations: [
      {
        series_key: "SKU_1",
        lead_time_days: 7,
        reorder_point: 85,
        current_inventory: 20,
        order_quantity: 80,
        approval_tier: "medium",
      },
    ],
    total_order_quantity: 80,
    approval_tier_breakdown: { medium: 1 },
    ...overrides,
  };
}

function renderReplenishment(state: Record<string, unknown>): ReturnType<typeof render> {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.endsWith("/surfaces") || url.endsWith("/api/surfaces")) {
      return new Response(JSON.stringify({ surfaces: ["replenishment_board"] }), {
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
      <ReplenishmentBoard runId="dev-run" />
    </QueryClientProvider>,
  );
}

describe("<ReplenishmentBoard> (CB11)", () => {
  it("renders the 4 KPI cards (count, total qty, high tier, medium/low)", async () => {
    renderReplenishment(cannedState());
    // Wait for the SKU_1 row to render (depends on fetch).
    await waitFor(() => {
      expect(screen.getByText("SKU_1")).toBeInTheDocument();
    });
    expect(screen.getAllByText(/Recommendations/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Total order qty/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/High tier/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Medium \/ low/i).length).toBeGreaterThan(0);
  });

  it("renders the recommendations table with the tier chip", async () => {
    renderReplenishment(cannedState());
    await waitFor(() => {
      expect(screen.getByText("SKU_1")).toBeInTheDocument();
    });
    expect(screen.getAllByText("medium").length).toBeGreaterThan(0);
  });
});
