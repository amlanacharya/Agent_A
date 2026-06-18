import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";

function renderWithProviders(): ReturnType<typeof render> {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <App />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("App scaffold (CB2)", () => {
  it("renders the cockpit title and subtitle", () => {
    renderWithProviders();
    expect(
      screen.getByRole("heading", { level: 1, name: /cockpit frontend/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Phase 9 scaffold\. The 9 surfaces/i),
    ).toBeInTheDocument();
  });

  it("uses design-system tokens via Tailwind classes (font-mono label, bg/text from tokens)", () => {
    renderWithProviders();
    // The JetBrains Mono label proves font-mono is wired through Tailwind.
    // The main wrapper uses bg-background + text-text-muted tokens from the
    // DESIGN.md port in tailwind.config.ts. The font link wiring is exercised
    // by opening / in a real browser (vitest's jsdom doesn't load index.html).
    const label = screen.getByText(/agent a · data intelligence cockpit/i);
    expect(label.className).toMatch(/\bfont-mono\b/);
    expect(label.className).toMatch(/\btext-label-caps\b/);
    const subtitle = screen.getByText(/Phase 9 scaffold\. The 9 surfaces/i);
    expect(subtitle.className).toMatch(/\btext-body-lg\b/);
    expect(subtitle.className).toMatch(/\btext-text-muted\b/);
    const heading = screen.getByRole("heading", { level: 1 });
    expect(heading.className).toMatch(/\btext-display-lg\b/);
    expect(heading.className).toMatch(/\btext-text-main\b/);
  });
});
