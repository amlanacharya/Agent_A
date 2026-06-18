import { describe, it, expect, beforeAll, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MissionControl } from "./MissionControl";

/**
 * The MissionControl page reads from useSurface("mission_control", runId).
 * Stub the global fetch so jsdom doesn't try to hit the real backend.
 */
function cannedState(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    run_id: "dev-run",
    current_step: "foundry_modelling",
    active_agent: "foundry",
    tool_result: "Claim: residual mean is 0.34",
    code_escalation_status: null,
    code_attempt: null,
    verifier_gate: null,
    approval_needed: false,
    confidence: "high",
    blockers: [],
    ...overrides,
  };
}

function renderMissionControl(state: Record<string, unknown>): ReturnType<typeof render> {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.endsWith("/surfaces") || url.endsWith("/api/surfaces")) {
      return new Response(JSON.stringify({ surfaces: ["mission_control"] }), {
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
  });
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  return render(
    <QueryClientProvider client={qc}>
      <MissionControl runId="dev-run" />
    </QueryClientProvider>,
  );
}

describe("<MissionControl> (CB5)", () => {
  it("renders the KPI grid from the surface state", async () => {
    renderMissionControl(cannedState());
    // Wait for the surface query to resolve and the KPI values to
    // render. The PageHeader title (h1) renders immediately so
    // it's a stable anchor.
    await waitFor(() =>
      expect(screen.getByRole("heading", { level: 1 }).textContent).toMatch(/Run dev-run/),
    );
    // The 4 metric card labels.
    const labels = ["Current step", "Active agent", "Confidence", "Blockers"];
    for (const label of labels) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
    // Wait for the current_step value to render (depends on fetch).
    await waitFor(() => {
      const matches = screen.getAllByText(/foundry_modelling/);
      expect(matches.length).toBeGreaterThan(0);
    });
    // And the active agent value.
    await waitFor(() => {
      const matches = screen.getAllByText(/^foundry$/);
      expect(matches.length).toBeGreaterThan(0);
    });
  });

  it("renders the approval banner when approval_needed is true", async () => {
    renderMissionControl(
      cannedState({ approval_needed: true, verifier_gate: "model_promote" }),
    );
    await waitFor(() =>
      expect(screen.getByText(/Approval Needed/i)).toBeInTheDocument(),
    );
    expect(screen.getByText(/model_promote/i)).toBeInTheDocument();
  });

  it("hides the approval banner when approval_needed is false", async () => {
    renderMissionControl(cannedState({ approval_needed: false }));
    await waitFor(() =>
      expect(screen.getByRole("heading", { level: 1 }).textContent).toMatch(/Run dev-run/),
    );
    expect(screen.queryByText(/Approval Needed/i)).not.toBeInTheDocument();
  });

  it("renders the activity feed with tool_result and blockers", async () => {
    renderMissionControl(
      cannedState({
        tool_result: "Decomposed residual: 0.34 mean, 1.2 std",
        blockers: ["XGBoost failed on SKU_3"],
      }),
    );
    await waitFor(() =>
      expect(screen.getByText(/Recent activity/i)).toBeInTheDocument(),
    );
    // The activity feed's body text appears after the fetch resolves.
    await waitFor(() => {
      expect(screen.getByText(/Decomposed residual/i)).toBeInTheDocument();
      expect(screen.getByText(/XGBoost failed on SKU_3/i)).toBeInTheDocument();
    });
  });

  it("renders the confidence chip with the success tone when high", async () => {
    renderMissionControl(cannedState({ confidence: "high" }));
    await waitFor(() =>
      expect(screen.getAllByText(/^HIGH$/).length).toBeGreaterThan(0),
    );
    // The confidence chip is rendered as a StatusChip with the
    // "high" label; we look it up via its lowercase form since the
    // chip uses text-label-caps uppercase rendering.
    const chip = screen.getAllByText(/^High$/i).find((el) =>
      el.className.includes("text-success-teal"),
    );
    expect(chip).toBeDefined();
  });
});
