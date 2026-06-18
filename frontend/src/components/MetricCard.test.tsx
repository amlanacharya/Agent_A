import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MetricCard } from "./MetricCard";

describe("<MetricCard>", () => {
  it("renders the label, value, and caption", () => {
    render(
      <MetricCard label="Total demand" value="12,345" caption="vs 11,920 last run" />,
    );
    expect(screen.getByText(/Total demand/i)).toBeInTheDocument();
    expect(screen.getByText(/12,345/)).toBeInTheDocument();
    expect(screen.getByText(/vs 11,920 last run/i)).toBeInTheDocument();
  });

  it("uses data-mono typography for the value", () => {
    render(<MetricCard label="MAPE" value="0.082" />);
    const value = screen.getByText(/0\.082/);
    expect(value.className).toMatch(/\bfont-mono\b/);
    expect(value.className).toMatch(/\btext-data-mono\b/);
  });

  it("applies the metric-active shadow when active=true", () => {
    render(<MetricCard label="Active" value="42" active />);
    const card = screen.getByRole("article");
    expect(card.className).toMatch(/\bshadow-metric-active\b/);
  });
});
