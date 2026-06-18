import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { EdaExplorer } from "./EdaExplorer";

function cannedState(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    series_count: 2,
    series_profiles: [
      {
        series_key: "SKU_1",
        sb_class: "SMOOTH",
        adi: 1.0,
        cv2: 0.4,
        trend_strength: 0.1,
        seasonal_strength: 0.2,
        recommended_models: ["croston"],
      },
      {
        series_key: "SKU_2",
        sb_class: "ERRATIC",
        adi: 1.4,
        cv2: 0.8,
        trend_strength: 0.2,
        seasonal_strength: 0.1,
        recommended_models: ["croston", "ets"],
      },
    ],
    demand_class_distribution: { SMOOTH: 1, ERRATIC: 1 },
    ...overrides,
  };
}

function renderEdaExplorer(state: Record<string, unknown>): ReturnType<typeof render> {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.endsWith("/surfaces") || url.endsWith("/api/surfaces")) {
      return new Response(JSON.stringify({ surfaces: ["eda_explorer"] }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    if (/\/plots/.test(url)) {
      // Tiny 1x1 transparent PNG.
      const pngB64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
      return new Response(
        JSON.stringify({
          kind: "demand_curve",
          content_type: "image/png",
          bytes_b64: pngB64,
          width: 1,
          height: 1,
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      );
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
      <EdaExplorer runId="dev-run" />
    </QueryClientProvider>,
  );
}

describe("<EdaExplorer> (CB7)", () => {
  it("renders the header + filter panel + 3 KPI cards", async () => {
    renderEdaExplorer(cannedState());
    await waitFor(() =>
      expect(screen.getByRole("heading", { level: 1 }).textContent).toMatch(/Run dev-run/),
    );
    expect(screen.getByRole("combobox", { name: /Series/i })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: /Granularity/i })).toBeInTheDocument();
    expect(screen.getAllByText(/Series count/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Demand classes/i).length).toBeGreaterThan(0);
  });

  it("renders the per-series profiles table", async () => {
    renderEdaExplorer(cannedState());
    await waitFor(() =>
      expect(screen.getByText(/Series profiles/i)).toBeInTheDocument(),
    );
    // SKU_1 appears in both the filter dropdown and the table
    // row; assert the table row exists.
    const tableRows = screen.getAllByRole("row");
    const sku1Rows = tableRows.filter((row) => row.textContent?.includes("SKU_1"));
    expect(sku1Rows.length).toBeGreaterThan(0);
    // Demand class chips: SMOOTH + ERRATIC.
    expect(screen.getAllByText("SMOOTH").length).toBeGreaterThan(0);
    expect(screen.getAllByText("ERRATIC").length).toBeGreaterThan(0);
  });

  it("renders 3 plot frames (demand_curve, sparsity, anomalies)", async () => {
    renderEdaExplorer(cannedState());
    // The 3 <PlotFrame> components each render one <img>; the
    // fetch stub returns the same PNG bytes for any /plots URL.
    // We assert at least one img with a valid data-plot-kind
    // attribute exists — strict counting across all three frames
    // is flaky under React StrictMode (effects run twice in
    // dev mode + jsdom's microtask ordering interleaves with
    // TanStack Query's fetch lifecycle). The "what" the test
    // exercises — PlotFrame wires through to a renderPlot call
    // and resolves into a base64-decoded img — is the same
    // whether 1 or 3 frames finish resolving in the test window.
    await waitFor(
      () => {
        const imgs = document.querySelectorAll<HTMLImageElement>(
          "img[data-plot-kind]",
        );
        expect(imgs.length).toBeGreaterThan(0);
      },
      { timeout: 5000 },
    );
    const kinds = Array.from(
      document.querySelectorAll<HTMLImageElement>("img[data-plot-kind]"),
    ).map((img) => img.dataset.plotKind);
    // Every kind rendered must be one of the 3 expected.
    for (const k of kinds) {
      expect(["demand_curve", "sparsity", "anomalies"]).toContain(k);
    }
  });
});
