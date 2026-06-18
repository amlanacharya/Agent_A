import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { PageHeader } from "./PageHeader";

describe("<PageHeader>", () => {
  it("renders the eyebrow, title, and subtitle", () => {
    render(
      <PageHeader
        eyebrow="mission control"
        title="Run 2025-01"
        subtitle="Live cockpit state."
      />,
    );
    expect(screen.getByText(/mission control/i)).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 1, name: /Run 2025-01/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Live cockpit state\./i)).toBeInTheDocument();
  });

  it("uses the design-system typography tokens", () => {
    render(<PageHeader title="Run" />);
    const heading = screen.getByRole("heading", { level: 1 });
    expect(heading.className).toMatch(/\btext-headline-lg\b/);
    expect(heading.className).toMatch(/\btext-text-main\b/);
  });
});
