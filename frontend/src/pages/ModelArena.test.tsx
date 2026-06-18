import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ModelArena } from "./ModelArena";

function cannedState(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    scorecard_count: 1,
    scorecards: [
      {
        model_family: "naive",
        series_key: "SKU_1",
        mase: 0.85,
        mae: 1.0,
        rmse: 1.0,
        bias: 0.05,
        fold_cutoff: "2026-01-01",
      },
    ],
    ensemble_weights: { G1: { naive: 1.0 } },
    frequently_promoted: ["naive"],
    never_surfaced: ["lstm"],
    ...overrides,
  };
}

function renderModelArena(state: Record<string, unknown>): ReturnType<typeof render> {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.endsWith("/surfaces") || url.endsWith("/api/surfaces")) {
      return new Response(JSON.stringify({ surfaces: ["model_arena"] }), {
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
      <ModelArena runId="dev-run" />
    </QueryClientProvider>,
  );
}

describe("<ModelArena> (CB9)", () => {
  it("renders the leaderboard with the scorecard row", async () => {
    renderModelArena(cannedState());
    await waitFor(() =>
      expect(screen.queryByText(/No scorecards for this run\./)).not.toBeInTheDocument(),
    );
    expect(screen.getAllByText("naive").length).toBeGreaterThan(0);
    expect(screen.getAllByText("SKU_1").length).toBeGreaterThan(0);
  });

  it("renders the best-MASE KPI + PASS chip when MASE < 1", async () => {
    renderModelArena(cannedState());
    await waitFor(() =>
      expect(screen.getByText(/Best MASE/i)).toBeInTheDocument(),
    );
    // Scorecards must arrive; if "No scorecards for this run" is
    // visible, the state fetch returned empty (a real bug).
    await waitFor(() => {
      expect(screen.queryByText(/No scorecards for this run\./)).not.toBeInTheDocument();
    });
    expect(screen.getAllByText("0.850").length).toBeGreaterThan(0);
    expect(screen.getAllByText("PASS").length).toBeGreaterThan(0);
  });

  it("renders the forecast-band plot", async () => {
    renderModelArena(cannedState());
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
    expect(kinds).toContain("forecast_band");
  });

  it("shows the never-surfaced chip when present", async () => {
    renderModelArena(cannedState({ never_surfaced: ["lstm", "prophet"] }));
    await waitFor(() =>
      expect(screen.getByText(/Never surfaced/i)).toBeInTheDocument(),
    );
    expect(screen.getAllByText("lstm").length).toBeGreaterThan(0);
  });
});
