import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LearningJournal } from "./LearningJournal";

function cannedState(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    active_cards: 2,
    retired_cards: 1,
    LEARNINGS: "# LEARNINGS\n\n## Active\n\n- card-a\n- card-b\n\n## Retired\n\n- card-c\n",
    DECISIONS: "# Decisions\n",
    ASSUMPTIONS: null,
    RUNBOOK: "# Runbook\n",
    MODEL_REGISTRY: "# Registry\n",
    PROMOTION_DECISIONS: "# Promotions\n",
    ...overrides,
  };
}

function backendArtifactState(): Record<string, unknown> {
  return {
    active_cards: 1,
    retired_cards: 0,
    "LEARNINGS.md": "# LEARNINGS\n\n## Active\n\n- dev-card\n\n## Retired\n\n",
    "DECISIONS.md": "# DECISIONS.md\n",
    "ASSUMPTIONS.md": "# ASSUMPTIONS.md\n",
    "RUNBOOK.md": "# RUNBOOK.md\n",
    "MODEL_REGISTRY.md": "# MODEL_REGISTRY.md\n",
    "PROMOTION_DECISIONS.md": "# PROMOTION_DECISIONS.md\n",
  };
}

function renderLearningJournal(state: Record<string, unknown>): ReturnType<typeof render> {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.endsWith("/surfaces") || url.endsWith("/api/surfaces")) {
      return new Response(JSON.stringify({ surfaces: ["learning_journal"] }), {
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
      <LearningJournal runId="dev-run" />
    </QueryClientProvider>,
  );
}

describe("<LearningJournal> (CB11)", () => {
  it("renders the 3 KPI cards (active, retired, artifacts)", async () => {
    renderLearningJournal(cannedState());
    // Wait for the active-card count to render.
    await waitFor(() => {
      expect(screen.getAllByText(/Active cards/i).length).toBeGreaterThan(0);
    });
    expect(screen.getAllByText(/Retired cards/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Artifacts/i).length).toBeGreaterThan(0);
  });

  it("renders the 6 artifact tabs", async () => {
    renderLearningJournal(cannedState());
    await waitFor(() =>
      expect(screen.getByRole("heading", { level: 1 }).textContent).toMatch(/Run dev-run/),
    );
    const tabLabels = ["Learnings", "Decisions", "Assumptions", "Runbook", "Model registry", "Promotions"];
    for (const label of tabLabels) {
      expect(screen.getByRole("button", { name: label })).toBeInTheDocument();
    }
  });

  it("renders the active LEARNINGS tab content by default", async () => {
    renderLearningJournal(cannedState());
    await waitFor(() =>
      expect(screen.getByText(/card-a/)).toBeInTheDocument(),
    );
  });

  it("renders markdown artifacts returned with .md backend keys", async () => {
    renderLearningJournal(backendArtifactState());
    await waitFor(() =>
      expect(screen.getByText(/dev-card/)).toBeInTheDocument(),
    );
    expect(screen.getByText("6 / 6")).toBeInTheDocument();
  });
});
