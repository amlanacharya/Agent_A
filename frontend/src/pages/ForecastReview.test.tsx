import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ForecastReview } from "./ForecastReview";

function cannedState(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    overall_mase: 0.85,
    target_met_fraction: 1.0,
    series_results: [
      {
        series_key: "SKU_1",
        sb_class: "SMOOTH",
        mase_target: 0.8,
        best_model: "naive",
        target_met: true,
        overall_mase: 0.85,
      },
    ],
    narrative: "Single-series success.",
    ...overrides,
  };
}

function renderForecastReview(state: Record<string, unknown>): ReturnType<typeof render> {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.endsWith("/surfaces") || url.endsWith("/api/surfaces")) {
      return new Response(JSON.stringify({ surfaces: ["forecast_review"] }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    if (/\/plots/.test(url)) {
      const pngB64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
      return new Response(
        JSON.stringify({
          kind: "forecast_band",
          content_type: "image/png",
          bytes_b64: pngB64,
          width: 1,
          height: 1,
        }),
        { status: 200, headers: { "content-type": "application/json" } },
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
      <ForecastReview runId="dev-run" />
    </QueryClientProvider>,
  );
}

describe("<ForecastReview> (CB10)", () => {
  it("renders the overall MASE KPI + PASS chip", async () => {
    renderForecastReview(cannedState());
    await waitFor(() => {
      // "0.850" appears in the KPI value AND the per-series MASE
      // column; wait for any match.
      expect(screen.getAllByText("0.850").length).toBeGreaterThan(0);
    });
    expect(screen.getAllByText("PASS").length).toBeGreaterThan(0);
  });

  it("renders the per-series results table with MET chip", async () => {
    renderForecastReview(cannedState());
    await waitFor(() => {
      expect(screen.getByText("SKU_1")).toBeInTheDocument();
    });
    expect(screen.getAllByText("MET").length).toBeGreaterThan(0);
  });

  it("renders the foundry narrative when present", async () => {
    renderForecastReview(cannedState());
    await waitFor(() =>
      expect(screen.getByText(/Foundry narrative/i)).toBeInTheDocument(),
    );
    expect(screen.getByText(/Single-series success\./)).toBeInTheDocument();
  });

  it("renders 2 plot frames (forecast_band + backtest)", async () => {
    renderForecastReview(cannedState());
    await waitFor(
      () => {
        const imgs = document.querySelectorAll("img[data-plot-kind]");
        expect(imgs.length).toBeGreaterThan(0);
      },
      { timeout: 5000 },
    );
    const kinds = Array.from(
      document.querySelectorAll<HTMLImageElement>("img[data-plot-kind]"),
    ).map((img) => img.dataset.plotKind);
    expect(kinds).toEqual(expect.arrayContaining(["forecast_band", "backtest"]));
  });
});
