import { describe, it, expect } from "vitest";
import { colors, typography, radii, spacing, shadows } from "./tokens";

describe("design tokens", () => {
  it("exports the brand palette (Deep Indigo / Electric Cyan)", () => {
    // DESIGN.md requires #3525cd primary and #4cd7f6 Electric Cyan family.
    // The cyan family is exposed as secondaryContainer (filled) — that's
    // the saturated token, not the readable ink.
    expect(colors.primary).toBe("#3525cd");
    expect(colors.primaryContainer).toBe("#4f46e5");
    expect(colors.surfaceTint).toBe("#4d44e3");
    expect(colors.secondaryContainer).toBe("#57dffe");
    expect(colors.secondaryFixedDim).toBe("#4cd7f6");
  });

  it("exports the canvas + semantic colors", () => {
    expect(colors.background).toBe("#f7f9fb");
    expect(colors.surfaceContainerLowest).toBe("#ffffff");
    expect(colors.successTeal).toBe("#0d9488");
    expect(colors.warningAmber).toBe("#d97706");
    expect(colors.criticalRose).toBe("#e11d48");
  });

  it("exports the typography scale (DESIGN.md lines 57-106)", () => {
    expect(typography.displayLg.fontSize).toBe(48);
    expect(typography.headlineLg.fontSize).toBe(32);
    expect(typography.headlineMd.fontSize).toBe(24);
    expect(typography.bodyLg.fontSize).toBe(18);
    expect(typography.bodyMd.fontSize).toBe(16);
    expect(typography.bodySm.fontSize).toBe(14);
    expect(typography.labelCaps.letterSpacing).toBeCloseTo(0.05);
  });

  it("exports the radius scale (8px default per DESIGN.md)", () => {
    expect(radii.sm).toBe(4);
    expect(radii.DEFAULT).toBe(8);
    expect(radii.md).toBe(12);
    expect(radii.lg).toBe(16);
    expect(radii.xl).toBe(24);
    expect(radii.full).toBe(9999);
  });

  it("exports spacing + shadow tokens", () => {
    expect(spacing.gutter).toBe(24);
    expect(spacing.containerMax).toBe(1440);
    expect(spacing.stackLg).toBe(32);
    expect(shadows.card).toContain("rgba(0,0,0,0.05)");
    expect(shadows.popover).toContain("rgba(0,0,0,0.1)");
  });
});
