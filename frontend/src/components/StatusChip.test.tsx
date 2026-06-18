import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { StatusChip } from "./StatusChip";

describe("<StatusChip>", () => {
  it("renders the label and applies the success tone classes", () => {
    render(<StatusChip label="Healthy" tone="success" />);
    const chip = screen.getByText(/Healthy/);
    expect(chip.className).toMatch(/bg-success-teal/);
    expect(chip.className).toMatch(/text-success-teal/);
  });

  it("uses the warning + critical palette correctly", () => {
    const { rerender } = render(<StatusChip label="Warn" tone="warning" />);
    expect(screen.getByText(/Warn/).className).toMatch(/text-warning-amber/);
    rerender(<StatusChip label="Bad" tone="critical" />);
    expect(screen.getByText(/Bad/).className).toMatch(/text-critical-rose/);
  });

  it("falls back to the neutral tone by default", () => {
    render(<StatusChip label="Idle" />);
    expect(screen.getByText(/Idle/).className).toMatch(/bg-surface-container-high/);
  });
});
