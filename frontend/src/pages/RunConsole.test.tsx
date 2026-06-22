import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  RouterProvider,
  createMemoryRouter,
} from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AppShell } from "@/components/AppShell";
import { RunConsole } from "./RunConsole";

/**
 * Tests for the RunConsole page — Phase 10 CB6.
 *
 * Stubs global ``fetch`` to return canned responses for the
 * three endpoints the page exercises: ``POST /uploads``,
 * ``POST /messages``, ``POST /runs/{id}/advance``, and
 * ``GET /cockpit-state/{id}``. Wraps the page with
 * ``QueryClientProvider`` + ``createMemoryRouter`` so the
 * TanStack Query hooks + react-router's ``useNavigate`` /
 * ``useParams`` resolve.
 *
 * The test surface mirrors what a real user does: open
 * ``/``, upload a CSV, send a chat message, click a
 * possibility chip, advance the run, land on the report
 * links at report_ready.
 */

interface FetchLogEntry {
  url: string;
  method: string;
  body: unknown;
}

function makeFetchStub(handlers: {
  cockpitState?: () => unknown;
  upload?: (body: unknown) => unknown;
  message?: (body: unknown) => unknown;
  advance?: (body: unknown) => unknown;
}) {
  const log: FetchLogEntry[] = [];
  const fetchMock = vi.fn(
    async (
      input: RequestInfo | URL,
      init?: RequestInit,
    ): Promise<Response> => {
      const url = typeof input === "string" ? input : input.toString();
      const method = init?.method ?? "GET";
      let body: unknown = null;
      if (init?.body) {
        const text =
          typeof init.body === "string"
            ? init.body
            : init.body instanceof FormData
              ? "[FormData]"
              : null;
        if (text) {
          try {
            body = JSON.parse(text);
          } catch {
            body = text;
          }
        } else {
          body = "[binary]";
        }
      }
      log.push({ url, method, body });

      if (url.includes("/cockpit-state/")) {
        return new Response(JSON.stringify(handlers.cockpitState?.() ?? {}), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/surfaces") || url.endsWith("/api/surfaces")) {
        return new Response(
          JSON.stringify({ surfaces: ["mission_control"] }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      if (url.endsWith("/uploads") && method === "POST") {
        return new Response(
          JSON.stringify(
            handlers.upload?.(body) ?? {
              run_id: "upload-test-123",
              domain: "fmcg",
              preflight: {},
              state: { current_step: "preflight" },
            },
          ),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      if (url.endsWith("/messages") && method === "POST") {
        return new Response(
          JSON.stringify(
            handlers.message?.(body) ?? {
              intent: "SCOPE_RESPONSE",
              run_id: "run-x",
              reply: "Noted.",
              possibilities: [],
            },
          ),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      if (url.match(/\/runs\/[^/]+\/advance/) && method === "POST") {
        return new Response(
          JSON.stringify(
            handlers.advance?.(body) ?? {
              run_id: "run-x",
              advanced_to: "foundry_modelling",
              reply: "advanced",
              possibilities: [],
            },
          ),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      // The AppShell's useSurfaces hook fires /api/surfaces on
      // mount; the GenericSurfacePage fallback would also fire
      // /api/surfaces/{name}/{runId}. Return canned data so
      // the AppShell nav renders without console errors.
      const surfaceMatch = url.match(/\/(?:api\/)?surfaces\/([^/]+)\/([^/]+)/);
      if (surfaceMatch) {
        return new Response(
          JSON.stringify({
            run_id: surfaceMatch[2],
            surface: surfaceMatch[1],
            state: {},
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      return new Response("{}", { status: 200 });
    },
  );
  return { log, fetchMock };
}

function makeWrapper(
  fetchMock: ReturnType<typeof makeFetchStub>["fetchMock"],
  initialEntry: string,
): React.ComponentType {
  return function Wrapper() {
    const qc = new QueryClient({
      defaultOptions: { queries: { retry: false, refetchInterval: false } },
    });
    // Disable refetchInterval in tests so the cockpit-state
    // poll doesn't fire repeatedly and noise up the log.
    const router = createMemoryRouter(
      [
        {
          path: "/",
          element: <AppShell />,
          children: [
            { index: true, element: <RunConsole /> },
            { path: "runs/:runId/console", element: <RunConsole /> },
            {
              path: "surfaces/:name/:runId",
              element: <div data-testid="generic-surface">surface</div>,
            },
          ],
        },
      ],
      { initialEntries: [initialEntry] },
    );
    // ``RouterProvider`` doesn't accept children — the router
    // itself renders the matched route.
    return (
      <QueryClientProvider client={qc}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    );
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("RunConsole", () => {
  let originalFetch: typeof fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("renders the upload form at /", async () => {
    const { fetchMock } = makeFetchStub({});
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const Wrapper = makeWrapper(fetchMock, "/");

    render(<Wrapper />);

    expect(
      await screen.findByRole("heading", { name: /start a new run/i }),
    ).toBeInTheDocument();
    // Domain select defaults to FMCG; CSV file input present.
    expect(screen.getByLabelText(/^domain$/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/csv file/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /^upload$/i }),
    ).toBeInTheDocument();
  });

  it("uploads a CSV and the POST /uploads call lands", async () => {
    const { fetchMock, log } = makeFetchStub({});
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const Wrapper = makeWrapper(fetchMock, "/");

    const user = userEvent.setup();
    render(<Wrapper />);

    const file = new File(["date,sku,region,demand\n"], "input.csv", {
      type: "text/csv",
    });
    const fileInput = await screen.findByLabelText(/csv file/i);
    await user.upload(fileInput, file);
    await user.click(screen.getByRole("button", { name: /^upload$/i }));

    // POST /uploads was called with FormData (multipart body).
    await waitFor(() => {
      const uploadEntry = log.find((entry) => entry.url.endsWith("/uploads"));
      expect(uploadEntry).toBeDefined();
      expect(uploadEntry?.method).toBe("POST");
    });
  });

  it("shows an error chip when the upload fails", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/cockpit-state/")) {
        return new Response("{}", { status: 200 });
      }
      if (url.endsWith("/surfaces") || url.endsWith("/api/surfaces")) {
        return new Response(
          JSON.stringify({ surfaces: ["mission_control"] }),
          { status: 200 },
        );
      }
      if (url.endsWith("/uploads")) {
        return new Response("bad csv", { status: 422 });
      }
      return new Response("{}", { status: 200 });
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const Wrapper = makeWrapper(fetchMock, "/");

    const user = userEvent.setup();
    render(<Wrapper />);

    const file = new File(["oops"], "bad.csv", { type: "text/csv" });
    const fileInput = await screen.findByLabelText(/csv file/i);
    await user.upload(fileInput, file);
    await user.click(screen.getByRole("button", { name: /^upload$/i }));

    expect(
      await screen.findByText(/upload failed/i),
    ).toBeInTheDocument();
  });

  it("sends a chat message and renders the assistant reply + user bubble", async () => {
    const { fetchMock } = makeFetchStub({
      cockpitState: () => ({
        run_id: "run-x",
        current_step: "meridian_scoping",
        active_agent: "meridian",
        phase: "meridian_scoping",
      }),
      message: () => ({
        intent: "SCOPE_RESPONSE",
        run_id: "run-x",
        reply: "Got it — noted the promo calendar.",
        possibilities: [
          { kind: "ACCEPT", label: "Continue", payload: {} },
        ],
      }),
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const Wrapper = makeWrapper(fetchMock, "/runs/run-x/console");

    const user = userEvent.setup();
    render(<Wrapper />);

    const input = await screen.findByTestId("chat-input");
    await user.type(input, "include the promo calendar");
    await user.click(screen.getByTestId("chat-send"));

    // User bubble + assistant bubble both render.
    expect(
      await screen.findByTestId("chat-bubble-user"),
    ).toHaveTextContent("include the promo calendar");
    expect(
      await screen.findByText(/got it — noted the promo calendar/i),
    ).toBeInTheDocument();
    // The possibility chip is rendered.
    expect(
      await screen.findByTestId("possibility-chip"),
    ).toHaveTextContent("Continue");
  });

  it("clicking a possibility chip submits its label as a new message", async () => {
    const { fetchMock, log } = makeFetchStub({
      cockpitState: () => ({
        run_id: "run-x",
        current_step: "meridian_scoping",
        active_agent: "meridian",
        phase: "meridian_scoping",
      }),
      message: () => ({
        intent: "SCOPE_RESPONSE",
        run_id: "run-x",
        reply: "ack",
        possibilities: [
          { kind: "ACCEPT", label: "That's the full scope", payload: {} },
        ],
      }),
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const Wrapper = makeWrapper(fetchMock, "/runs/run-x/console");

    const user = userEvent.setup();
    render(<Wrapper />);

    // First message to get a possibility chip back.
    const input = await screen.findByTestId("chat-input");
    await user.type(input, "include the promo calendar");
    await user.click(screen.getByTestId("chat-send"));

    const chip = await screen.findByTestId("possibility-chip");
    await user.click(chip);

    // Two POST /messages calls landed; the second one carries
    // the chip's label.
    await waitFor(() => {
      const messageCalls = log.filter((entry) =>
        entry.url.endsWith("/messages"),
      );
      expect(messageCalls.length).toBeGreaterThanOrEqual(2);
    });
    const lastMessageCall = log.filter((entry) =>
      entry.url.endsWith("/messages"),
    ).at(-1);
    expect(lastMessageCall?.body).toMatchObject({
      run_id: "run-x",
      user_message: "That's the full scope",
    });
  });

  it("renders the advance button and calls /advance when clicked", async () => {
    const { fetchMock, log } = makeFetchStub({
      cockpitState: () => ({
        run_id: "run-x",
        current_step: "forge_eda",
        active_agent: "conductor",
        phase: "forge_eda",
      }),
      advance: () => ({
        run_id: "run-x",
        advanced_to: "foundry_modelling",
        reply: "Forge done — running Foundry.",
        possibilities: [],
      }),
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const Wrapper = makeWrapper(fetchMock, "/runs/run-x/console");

    const user = userEvent.setup();
    render(<Wrapper />);

    // The Run tab is the second tab; click into it.
    const runTab = await screen.findByRole("tab", { name: /^run$/i });
    await user.click(runTab);

    const advanceBtn = await screen.findByTestId("advance-button");
    await user.click(advanceBtn);

    await waitFor(() => {
      expect(
        log.some((entry) => /\/runs\/run-x\/advance/.test(entry.url)),
      ).toBe(true);
    });
  });

  it("disables the advance button and shows the chat-gate message in meridian_scoping", async () => {
    const { fetchMock } = makeFetchStub({
      cockpitState: () => ({
        run_id: "run-x",
        current_step: "meridian_scoping",
        active_agent: "meridian",
        phase: "meridian_scoping",
      }),
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const Wrapper = makeWrapper(fetchMock, "/runs/run-x/console");

    const user = userEvent.setup();
    render(<Wrapper />);

    const runTab = await screen.findByRole("tab", { name: /^run$/i });
    await user.click(runTab);

    const advanceBtn = await screen.findByTestId("advance-button");
    expect(advanceBtn).toBeDisabled();
    expect(
      screen.getByText(/chat with meridian to advance/i),
    ).toBeInTheDocument();
  });

  it("renders the report links when phase is report_ready", async () => {
    const { fetchMock } = makeFetchStub({
      cockpitState: () => ({
        run_id: "run-x",
        current_step: "report_ready",
        active_agent: "conductor",
        phase: "report_ready",
      }),
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const Wrapper = makeWrapper(fetchMock, "/runs/run-x/console");

    const user = userEvent.setup();
    render(<Wrapper />);

    const runTab = await screen.findByRole("tab", { name: /^run$/i });
    await user.click(runTab);

    const links = await screen.findByTestId("report-links");
    expect(links).toHaveTextContent(/forecast review/i);
    expect(links).toHaveTextContent(/replenishment board/i);
    // The advance button is replaced — not present in this view.
    expect(screen.queryByTestId("advance-button")).not.toBeInTheDocument();
  });

  it("renders the live cockpit state in the left rail", async () => {
    const { fetchMock } = makeFetchStub({
      cockpitState: () => ({
        run_id: "run-x",
        current_step: "foundry_modelling",
        active_agent: "conductor",
        phase: "foundry_modelling",
        confidence: "medium",
        blockers: ["drift in segment G1"],
      }),
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const Wrapper = makeWrapper(fetchMock, "/runs/run-x/console");

    render(<Wrapper />);

    // The left rail's <aside aria-label="Cockpit state">
    // surfaces the phase chip + the blocker text once the
    // cockpit-state query resolves. The Run tab also renders
    // the phase text, so we scope the assertion to the rail
    // via the aria-label. ``waitFor`` waits for the fetch
    // mock's response to land and the query to settle.
    const rail = await screen.findByRole("complementary", {
      name: /cockpit state/i,
    });
    await waitFor(() => {
      expect(rail).toHaveTextContent("foundry_modelling");
    });
    expect(rail).toHaveTextContent(/drift in segment g1/i);
  });
});