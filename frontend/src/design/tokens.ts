/**
 * Design tokens, ported from `prototype/data_intelligence_cockpit/DESIGN.md`.
 *
 * This module mirrors the Tailwind theme extensions in `tailwind.config.ts`
 * for non-Tailwind consumers (chart libraries that want raw hex, JSON
 * snapshots in tests, Storybook controls, etc.). The two are kept in sync
 * manually — the test in `tokens.test.ts` enforces the key surface.
 *
 * When you change a value here, change it in `tailwind.config.ts` too.
 */

export const colors = {
  surface: "#f7f9fb",
  surfaceDim: "#d8dadc",
  surfaceBright: "#f7f9fb",
  surfaceContainerLowest: "#ffffff",
  surfaceContainerLow: "#f2f4f6",
  surfaceContainer: "#eceef0",
  surfaceContainerHigh: "#e6e8ea",
  surfaceContainerHighest: "#e0e3e5",
  background: "#f7f9fb",
  surfaceVariant: "#e0e3e5",
  onSurface: "#191c1e",
  onSurfaceVariant: "#464555",
  onBackground: "#191c1e",
  inverseSurface: "#2d3133",
  inverseOnSurface: "#eff1f3",
  outline: "#777587",
  outlineVariant: "#c7c4d8",
  borderSlate: "#e2e8f0",
  primary: "#3525cd",
  onPrimary: "#ffffff",
  primaryContainer: "#4f46e5",
  onPrimaryContainer: "#dad7ff",
  surfaceTint: "#4d44e3",
  inversePrimary: "#c3c0ff",
  secondary: "#00687a",
  onSecondary: "#ffffff",
  secondaryContainer: "#57dffe",
  onSecondaryContainer: "#006172",
  secondaryFixed: "#acedff",
  secondaryFixedDim: "#4cd7f6",
  tertiary: "#7e3000",
  onTertiary: "#ffffff",
  tertiaryContainer: "#a44100",
  onTertiaryContainer: "#ffd2be",
  error: "#ba1a1a",
  onError: "#ffffff",
  errorContainer: "#ffdad6",
  onErrorContainer: "#93000a",
  successTeal: "#0d9488",
  warningAmber: "#d97706",
  criticalRose: "#e11d48",
  textMain: "#0f172a",
  textMuted: "#64748b",
} as const;

export const typography = {
  displayLg: { fontSize: 48, lineHeight: 56, fontWeight: 700, letterSpacing: -0.02 },
  headlineLg: { fontSize: 32, lineHeight: 40, fontWeight: 600, letterSpacing: -0.01 },
  headlineLgMobile: { fontSize: 24, lineHeight: 32, fontWeight: 600 },
  headlineMd: { fontSize: 24, lineHeight: 32, fontWeight: 600 },
  bodyLg: { fontSize: 18, lineHeight: 28, fontWeight: 400 },
  bodyMd: { fontSize: 16, lineHeight: 24, fontWeight: 400 },
  bodySm: { fontSize: 14, lineHeight: 20, fontWeight: 400 },
  dataMono: { fontSize: 14, lineHeight: 20, fontWeight: 500, letterSpacing: 0.02 },
  labelCaps: { fontSize: 12, lineHeight: 16, fontWeight: 600, letterSpacing: 0.05 },
} as const;

export const radii = {
  sm: 4,
  DEFAULT: 8,
  md: 12,
  lg: 16,
  xl: 24,
  full: 9999,
} as const;

export const spacing = {
  base: 4,
  gutter: 24,
  marginDesktop: 40,
  marginMobile: 16,
  stackSm: 8,
  stackMd: 16,
  stackLg: 32,
  containerMax: 1440,
} as const;

export const shadows = {
  card: "0px 4px 6px -1px rgba(0,0,0,0.05)",
  popover: "0px 10px 15px -3px rgba(0,0,0,0.1)",
  metricActive:
    "inset 0 2px 0 0 #4d44e3, 0 4px 6px -1px rgba(0,0,0,0.05)",
} as const;

export type ColorToken = keyof typeof colors;
export type TypographyToken = keyof typeof typography;
